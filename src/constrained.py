"""Grammar-constrained decoding for the tool-call envelope: force valid JSON, then measure what is left.

The README promised this comparison: *would the format failures have happened at
all if decoding had been constrained?* Constrained (grammar-guided) decoding
masks the model's distribution at every step so only tokens that keep the output
a prefix of a schema-valid call can be chosen. Formatting failures — prose,
markdown fences, the wrong envelope — become impossible **without any training**,
so whatever is still wrong afterwards is a genuine task error (wrong tool, wrong
parameter value) rather than a formatting one.

Two pieces:

* `ToolCallGrammar` — an incremental validator for a *prefix* of this task's JSON
  envelope, derived from the same Pydantic models that generate the system prompt
  (`src/schema.py`), so grammar and validator cannot drift apart. It is
  specialized to this envelope on purpose; a general JSON-Schema compiler is a
  much bigger project and would obscure the lesson.
* `GrammarProcessor` — an MLX-LM logits processor that turns the grammar into a
  token mask. The vocabulary is indexed by each token's first character and the
  mask is cached per grammar state, so a 150k-token vocabulary costs a few
  hundred cheap validations per step (measured on Qwen2.5: 424 tokens start with
  `"`, 13 are all digits).

Restrictions, all of which match how the training targets are written:
whitespace outside strings is not allowed; strings may not contain `"`, `\\` or
raw control characters; and the top-level `"tool"` key must come before
`"parameters"` so parameter keys can be validated against the chosen tool.
"""

import time
from typing import Literal, NamedTuple, get_args, get_origin

import mlx.core as mx
import numpy as np

from src.schema import PARAM_MODEL_MAP

DIGITS = "0123456789"
NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
TOP_KEYS = ("tool", "parameters")
# Which characters can legally start a value, per FieldKind.kind.
VALUE_FIRST_CHARS = {"string": '"', "enum": '"', "bool": "tf", "int": DIGITS, "string_list": "[", "object": "{"}
_FAILED = None  # sentinel: transitions return None when the character is not allowed


# --------------------------------------------------------------------------------------
# Schema-derived field kinds
# --------------------------------------------------------------------------------------

class FieldKind(NamedTuple):
    kind: str                    # "string" | "int" | "bool" | "enum" | "string_list"
    values: tuple = ()           # allowed literals for enums
    minimum: int | None = None
    maximum: int | None = None


def _bounds(field):
    minimum = maximum = None
    for constraint in getattr(field, "metadata", ()):
        name = type(constraint).__name__
        if name == "Ge":
            minimum = constraint.ge
        elif name == "Le":
            maximum = constraint.le
    return minimum, maximum


def field_kind(field) -> FieldKind:
    """Map one Pydantic field to the lexical kind the grammar enforces."""
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        return FieldKind("enum", tuple(get_args(annotation)))
    if get_origin(annotation) in (list, tuple):
        return FieldKind("string_list")
    if annotation is bool:
        return FieldKind("bool")
    if annotation is int:
        minimum, maximum = _bounds(field)
        return FieldKind("int", (), minimum, maximum)
    return FieldKind("string")


def tool_schemas():
    """{tool: {field: FieldKind}} straight from the Pydantic models."""
    return {
        tool: {name: field_kind(field) for name, field in model.model_fields.items()}
        for tool, model in PARAM_MODEL_MAP.items()
    }


def required_fields(tool):
    return tuple(name for name, field in PARAM_MODEL_MAP[tool].model_fields.items() if field.is_required())


# --------------------------------------------------------------------------------------
# The incremental grammar
# --------------------------------------------------------------------------------------

class State(NamedTuple):
    """Immutable parser state: copying it is free, which is what makes masking affordable."""

    mode: str = "start"      # start|key|in_key|colon|value|in_string|number|literal|array|in_array_string|after|after_array|done|failed
    depth: int = 0           # 0 before '{', 1 top-level object, 2 parameters object
    tool: str = ""           # chosen tool, once the "tool" value is complete
    keys: tuple = ()         # keys completed at the current depth
    outer_keys: tuple = ()   # keys completed at depth 1 while we are inside depth 2
    key: str = ""            # key whose value is being read (or just completed)
    text: str = ""           # accumulated scalar text (key name, number, string body)
    escaped: bool = False    # inside a string, the previous character was a backslash
    letters: str = ""        # accumulated literal (true/false)
    array_had: bool = False  # the current array has at least one item
    array_count: int = 0     # items completed in the current array
    after_comma: bool = False  # a comma was just consumed: `}`/`]` here would be a trailing comma


_FAILED_STATE = State(mode="failed")


class ToolCallGrammar:
    """Accepts exactly the prefixes of schema-valid calls to this task's tools."""

    def __init__(self, schemas=None):
        self.schemas = schemas or tool_schemas()
        self.reset()

    def reset(self):
        self.state = State()

    # -- introspection ------------------------------------------------------------------

    @property
    def complete(self):
        return self.state.mode == "done"

    @property
    def failed(self):
        return self.state.mode == "failed"

    def snapshot(self):
        return self.state

    def restore(self, state):
        self.state = state

    def _keys_here(self, state):
        return TOP_KEYS if state.depth == 1 else tuple(self.schemas[state.tool])

    def available_keys(self, state):
        """Keys the grammar can still reach from `state`: not yet written, and not blocked.

        `parameters` is unreachable before `tool` has been written (parameter keys are
        validated against the chosen tool), so the mask must not offer it either — otherwise
        the model can spend its whole budget spelling a key it can never close.
        """
        keys = []
        for key in self._keys_here(state):
            if key in state.keys:
                continue
            if state.depth == 1 and key == "parameters" and "tool" not in state.keys:
                continue
            keys.append(key)
        return tuple(keys)

    def _kind_for(self, state):
        if state.depth == 1:
            return FieldKind("enum", tuple(self.schemas)) if state.key == "tool" else FieldKind("object")
        return self.schemas[state.tool][state.key]

    @staticmethod
    def _is_prefix(text, literals):
        """True while `text` can still grow into one of `literals` (empty text is always fine)."""
        return any(literal.startswith(text) for literal in literals)

    # -- validation ---------------------------------------------------------------------

    def accepts(self, chunk):
        """True if `chunk` can be appended without leaving the language (state unchanged)."""
        saved = self.state
        ok = self.feed(chunk)
        self.state = saved
        return ok

    def feed(self, chunk):
        """Advance over `chunk`. Returns False (and leaves the state failed) on invalid input."""
        state = self.state
        for character in chunk:
            state = self._step(state, character)
            if state is None or state.mode == "failed":
                self.state = _FAILED_STATE
                return False
        self.state = state
        return True

    # -- the state machine --------------------------------------------------------------

    def _step(self, state, character):
        mode = state.mode
        if mode == "start":
            return state._replace(mode="key", depth=1, after_comma=False) if character == "{" else None

        if mode == "key":
            if character == '"':
                return state._replace(mode="in_key", text="", after_comma=False)
            if character == "}" and not state.after_comma:
                return self._close_object(state)
            return None

        if mode == "in_key":
            if character == '"':
                if state.text not in self._keys_here(state) or state.text in state.keys:
                    return None
                if state.text == "parameters" and "tool" not in state.keys:
                    return None  # tool-first: parameter keys are validated against the chosen tool
                return state._replace(mode="colon", key=state.text, keys=state.keys + (state.text,))
            if character in NAME_CHARS and self._is_prefix(state.text + character, self.available_keys(state)):
                return state._replace(text=state.text + character)
            return None

        if mode == "colon":
            return state._replace(mode="value") if character == ":" else None

        if mode == "value":
            return self._start_value(state, character)

        if mode == "in_string":
            if character == '"':
                return self._finish_string(state)
            if character == "\\":
                return None  # escapes are not generated by this task's targets
            kind = self._kind_for(state)
            if kind.kind == "enum" and not self._is_prefix(state.text + character, kind.values):
                return None
            return state._replace(text=state.text + character)

        if mode == "number":
            if character in DIGITS and state.text != "0":  # JSON forbids leading zeros
                return state._replace(text=state.text + character)
            return self._finish_number(state, character)

        if mode == "literal":
            letters = state.letters + character
            target = "true" if letters[0] == "t" else "false"
            if not target.startswith(letters):
                return None
            if letters == target:
                return state._replace(mode="after", letters="")
            return state._replace(letters=letters)

        if mode == "array":
            if character == '"':
                return state._replace(mode="in_array_string", text="", after_comma=False)
            if character == "]" and not state.array_had and not state.after_comma:
                return state._replace(mode="after", array_had=False, array_count=0)
            return None

        if mode == "in_array_string":
            if character == '"':
                return state._replace(mode="after_array", array_had=True,
                                      array_count=state.array_count + 1, text="")
            if character == "\\":
                return None
            return state._replace(text=state.text + character)

        if mode == "after_array":
            if character == ",":
                return state._replace(mode="array", after_comma=True)
            if character == "]":
                return state._replace(mode="after", array_had=False, array_count=0, after_comma=False)
            return None

        if mode == "after":
            if character == ",":
                # A comma is only useful if another key can still be written at this level.
                return state._replace(mode="key", after_comma=True) if self.available_keys(state) else None
            if character == "}":
                return self._close_object(state)
            return None

        return None

    def _start_value(self, state, character):
        if state.depth == 1 and state.key == "parameters":
            if character != "{":
                return None
            return state._replace(mode="key", depth=2, keys=(), outer_keys=state.keys)
        kind = self._kind_for(state)
        if kind.kind == "string_list":
            return state._replace(mode="array", array_had=False, array_count=0, after_comma=False) if character == "[" else None
        if kind.kind == "bool":
            return state._replace(mode="literal", letters=character) if character in "tf" else None
        if kind.kind == "int":
            if character not in DIGITS:
                return None
            if character == "0" and (kind.minimum or 0) > 0:
                return None  # a leading zero can never grow to reach the minimum
            return state._replace(mode="number", text=character)
        return state._replace(mode="in_string", text="") if character == '"' else None

    def _finish_string(self, state):
        kind = self._kind_for(state)
        if kind.kind == "enum" and state.text not in kind.values:
            return None
        changes = {"mode": "after", "text": ""}
        if state.depth == 1 and state.key == "tool":
            changes["tool"] = state.text
        return state._replace(**changes)

    def _finish_number(self, state, next_character):
        if not state.text:
            return None
        value = int(state.text)
        kind = self._kind_for(state)
        if kind.minimum is not None and value < kind.minimum:
            return None
        if kind.maximum is not None and value > kind.maximum:
            return None
        return self._step(state._replace(mode="after", text=""), next_character)

    def _close_object(self, state):
        if state.depth == 2:
            if not all(name in state.keys for name in required_fields(state.tool)):
                return None
            return state._replace(mode="after", depth=1, keys=state.outer_keys, outer_keys=(), key="parameters")
        if state.depth == 1 and all(name in state.keys for name in TOP_KEYS):
            return state._replace(mode="done", depth=0)
        return None


# --------------------------------------------------------------------------------------
# The logits processor
# --------------------------------------------------------------------------------------

def tokenizer_vocab_size(tokenizer):
    """Total token ids, including special/added tokens that sit beyond `vocab_size`.

    Qwen2.5 is a concrete example: `vocab_size` is 151643 while `<|im_end|>` is 151645, so
    a mask built from `vocab_size` cannot even represent the stop token.
    """
    special = [token_id for token_id in (getattr(tokenizer, "all_special_ids", None) or ())
               if isinstance(token_id, int)]
    return max([len(tokenizer), getattr(tokenizer, "vocab_size", 0) or 0,
                (max(special) + 1) if special else 0, 1])


class GrammarProcessor:
    """MLX-LM logits processor that masks every token the grammar cannot accept next.

    MLX-LM calls processors once with the prompt tokens during prefill and then
    once per generated token, so the first call's length is recorded as the
    prompt boundary and only tokens after it advance the grammar. The mask length
    is taken from the model's own logits, which is the only size that is guaranteed
    to match (tokenizer sizes disagree with model vocabularies).
    """

    def __init__(self, tokenizer, vocab_size=None, grammar=None, max_string_tokens=24, max_array_items=8):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size or tokenizer_vocab_size(tokenizer)
        self.grammar = grammar or ToolCallGrammar()
        self.max_string_tokens = max_string_tokens
        self.max_array_items = max_array_items
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.prompt_length = None
        self.output_size = None
        self.committed = 0
        self.string_run = 0
        self.invalid_tokens = 0
        self._mask_cache = {}
        self._index_vocabulary()

    @property
    def mask_size(self):
        """Logits dimension, learned from the first call (falls back to the tokenizer size)."""
        return self.output_size or self.vocab_size

    def _index_vocabulary(self):
        # Special tokens (<|im_end|>, <|endoftext|>, ...) decode to marker strings that are not
        # JSON text: feeding them to the grammar would "fail" it at the end of a good answer.
        self.special_ids = set(getattr(self.tokenizer, "all_special_ids", None) or ())
        self.token_text = {}
        self._by_first_char = {}
        self._single_char = {}
        self._text_index = {}
        for token_id in range(self.vocab_size):
            if token_id in self.special_ids:
                continue
            text = self.tokenizer.decode([token_id])
            if not text:
                continue
            self.token_text[token_id] = text
            self._by_first_char.setdefault(text[0], []).append(token_id)
            self._text_index.setdefault(text, []).append(token_id)
            if len(text) == 1:
                self._single_char.setdefault(text, token_id)
        # ASCII digits only: str.isdigit() is true for things like "²", which int() rejects.
        self._digits = [token_id for token_id, text in self.token_text.items()
                        if all(character in DIGITS for character in text)]
        # Tokens that can be appended inside a free-form JSON string without breaking it.
        self._string_safe = [
            token_id for token_id, text in self.token_text.items()
            if '"' not in text and "\\" not in text and all(ord(c) >= 0x20 for c in text)
        ]

    def _validated(self, characters):
        """Candidate tokens starting with one of `characters`, filtered by the grammar."""
        candidates = []
        for character in characters:
            candidates.extend(self._by_first_char.get(character, ()))
        allowed = [token_id for token_id in candidates if self.grammar.accepts(self.token_text[token_id])]
        if allowed:
            return allowed
        # Last resort: validated single-character tokens. Never return an unvalidated token,
        # or the mask and the grammar can disagree and the decode desynchronizes.
        fallback = []
        for character in characters:
            token_id = self._single_char.get(character)
            if token_id is not None and self.grammar.accepts(self.token_text[token_id]):
                fallback.append(token_id)
        return fallback

    def _literal_candidates(self, literals, prefix):
        """Tokens that extend `prefix` toward one of `literals` (plus the ways to close it).

        Lexical candidate generation instead of a vocabulary scan: keys and enum values
        come from a fixed list, so the candidate texts can be enumerated directly. This
        keeps every step to a handful of validations, and it is why a model cannot loop
        forever inside a key (or enum value) that no schema contains.
        """
        texts = set()
        for literal in literals:
            if not literal.startswith(prefix):
                continue
            rest = literal[len(prefix):]
            if not rest:
                texts.add('"')  # the literal is complete: closing it is the only move
                continue
            for length in range(1, len(rest) + 1):
                texts.add(rest[:length])          # extend within the literal
                texts.add(rest[:length] + '"')    # extend and close
                texts.add(rest[:length] + '":')   # extend, close, and take the colon
        candidates = [token_id for text in texts for token_id in self._text_index.get(text, ())]
        return [token_id for token_id in candidates if self.grammar.accepts(self.token_text[token_id])]

    def _state_signature(self, state):
        """A hashable key that uniquely identifies the grammar state for mask caching.

        Subclasses that use a different state shape override this; the tool-call grammar
        keys on its flat fields, while a frame-stack grammar keys on `frames`.
        """
        return (state.mode, state.depth, state.tool, state.keys, state.key, state.text,
                state.letters, state.array_had, state.after_comma, state.array_count)

    def allowed_token_ids(self):
        """Token ids the grammar can accept from its current state (cached per state)."""
        state = self.grammar.state
        signature = (self._state_signature(state), self.grammar.complete,
                     self.string_run >= self.max_string_tokens)
        if signature in self._mask_cache:
            return self._mask_cache[signature]
        mode = state.mode
        if mode == "start":
            allowed = self._validated("{")
        elif mode == "key":
            allowed = self._validated('"}')
        elif mode == "in_key":
            # Keys not yet written (and still reachable) are the only ones spellable.
            allowed = self._literal_candidates(self.grammar.available_keys(state), state.text)
        elif mode == "colon":
            allowed = self._validated(":")
        elif mode == "value":
            kind = self.grammar._kind_for(state).kind
            allowed = self._validated(VALUE_FIRST_CHARS[kind])
        elif mode in ("in_string", "in_array_string"):
            kind = self.grammar._kind_for(state)
            if kind.kind == "enum":
                allowed = self._literal_candidates(kind.values, state.text)
            elif self.string_run >= self.max_string_tokens:
                # Free-form strings are unbounded in JSON; close them before a weak model
                # wanders for the whole budget. The reference values are all short.
                allowed = list(self._text_index.get('"', ()))
            else:
                allowed = self._string_safe + list(self._text_index.get('"', ()))
        elif mode == "number":
            kind = self.grammar._kind_for(state)
            value = int(state.text) if state.text else 0
            # "0" cannot grow (JSON forbids leading zeros), and digits must respect the maximum.
            allowed = [] if state.text == "0" else [
                token_id for token_id in self._digits
                if kind.maximum is None or int(state.text + self.token_text[token_id]) <= kind.maximum]
            # Once the number is in range it must also be allowed to end, or a weak model can
            # pad digits forever: terminators are validated, which is where the bounds are checked.
            if kind.minimum is None or value >= kind.minimum:
                allowed += self._validated(",}")
        elif mode == "literal":
            target = "true" if state.letters[:1] == "t" else "false"
            allowed = self._validated(target[len(state.letters):len(state.letters) + 1])
        elif mode == "array":
            # Arrays are unbounded in JSON; cap the item count so a weak model must close them.
            allowed = self._validated('"]' if state.array_count < self.max_array_items else ']')
        elif mode == "after":
            allowed = self._validated(",}")
        elif mode == "after_array":
            allowed = self._validated(',]' if state.array_count < self.max_array_items else ']')
        elif self.grammar.complete:
            allowed = [self.eos_token_id] if self.eos_token_id is not None else []
        else:
            allowed = []
        self._mask_cache[signature] = allowed
        return allowed

    def mask(self):
        """Additive logits mask: 0 for allowed tokens, -inf for everything else."""
        signature = ("mask", self._state_signature(self.grammar.state), self.grammar.complete,
                     self.string_run >= self.max_string_tokens)
        if signature not in self._mask_cache:
            array = np.full(self.mask_size, -np.inf, dtype=np.float32)
            for token_id in self.allowed_token_ids():
                if 0 <= token_id < self.mask_size:
                    array[token_id] = 0.0
            self._mask_cache[signature] = mx.array(array)
        return self._mask_cache[signature]

    def advance(self, token_ids):
        """Feed newly generated token ids into the grammar (special tokens carry no JSON text)."""
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.special_ids or self.grammar.complete:
                continue
            if not self.grammar.feed(self.token_text.get(token_id, "")):
                self.invalid_tokens += 1

    def __call__(self, tokens, logits):
        if self.output_size is None:
            self.output_size = int(logits.shape[-1])  # the mask must match the model's logits
        generated = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens or [])
        if self.prompt_length is None:
            self.prompt_length = len(generated)  # first call carries the prompt during prefill
        generated = generated[self.prompt_length:]
        if len(generated) > self.committed:
            self.advance(generated[self.committed:])
            self.committed = len(generated)
        self.string_run = self.string_run + 1 if self.grammar.state.mode in ("in_string", "in_array_string") else 0
        return logits + self.mask()


def constrained_generate(model, tokenizer, messages, max_tokens=150, temperature=0.0, grammar=None, processor=None):
    """Decode with the grammar mask applied at every step (greedy unless temperature > 0).

    Returns the same shape as `src.inference.generate_response` so it can be swapped
    into evaluation unchanged, plus the grammar's own diagnostics. When the object is
    complete the mask allows only the end-of-turn token, so generation stops there.
    """
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    vocab_size = getattr(model, "args", None) and getattr(model.args, "vocab_size", None) or tokenizer_vocab_size(tokenizer)
    if processor is None:
        processor = GrammarProcessor(tokenizer, vocab_size, grammar)
    start = time.perf_counter()
    first_token = None
    chunks = []
    last = None
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature), logits_processors=[processor],
    ):
        if first_token is None:
            first_token = time.perf_counter() - start
        chunks.append(response.text)
        last = response
    elapsed = time.perf_counter() - start
    if last is None:
        raise RuntimeError("Generation yielded no token metadata")
    return {
        "raw_output": "".join(chunks),
        "output_tokens": last.generation_tokens,
        "prompt_tokens": last.prompt_tokens,
        "ttft_seconds": first_token,
        "latency_seconds": elapsed,
        "prefill_tokens_per_sec": last.prompt_tps,
        "decode_tokens_per_sec": last.generation_tps,
        "end_to_end_tokens_per_sec": last.generation_tokens / elapsed if elapsed else 0.0,
        "finish_reason": last.finish_reason,
        "grammar_complete": processor.grammar.complete,
        "grammar_invalid_tokens": processor.invalid_tokens,
    }
