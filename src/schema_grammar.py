"""A JSON-Schema-driven grammar and processor for grammar-constrained decoding.

`src/constrained.py` ships a grammar specialized to the five-tool envelope. This module
generalizes it to *arbitrary* function schemas, which is what running against BFCL needs:
nested objects, arrays of any item type, floats, and — for multi-function records — a
`parameters` schema that depends on which function was chosen.

Design:

* `JsonSchemaGrammar` is an incremental validator for a *prefix* of any instance of a
  given JSON object schema. It is a stack machine over two frame kinds (object, array),
  so nesting depth is unbounded. The schema is interned into a per-grammar node table and
  frames refer to nodes by index, which keeps parser state hashable (the token mask is
  cached per state).
* `SchemaGrammarProcessor` subclasses `GrammarProcessor` and only replaces the mask
  decision (`allowed_token_ids`) and the cache key: the vocabulary indexing, token
  validation, string/number candidate generation, and mask building are all inherited.

The schema subset is what BFCL v3 simple actually uses: `object` (with `properties` and
`required`), `array` (with `items`), `string`, `integer`, `float`, `boolean`, plus a
`dep` node that selects a sub-schema from a sibling's value (`parameters` depends on
`tool`). Same restrictions as the tool grammar: no whitespace outside strings, no string
escapes or control characters, and keys must be written in a stable order (a dependent
key only becomes reachable once its dependency is written).
"""

from typing import NamedTuple

from src.constrained import DIGITS, NAME_CHARS, GrammarProcessor

# Node kinds understood by the state machine.
OBJECT, ARRAY, STRING, INT, NUMBER, BOOL, DEP, ENUM = ("object", "array", "string", "int", "number", "bool", "dep", "enum")


class ObjectFrame(NamedTuple):
    props: tuple       # ((name, node_id), ...) in declaration order
    required: tuple    # required property names
    keys: tuple        # completed property names, in order
    written: tuple     # ((name, value), ...) completed scalar values, for dependency resolution
    pending: str       # property whose value is being read ("" between values)


class ArrayFrame(NamedTuple):
    item: int          # node id of the item schema
    count: int         # completed items
    had: bool


class State(NamedTuple):
    mode: str = "start"   # start|key|in_key|colon|value|in_string|number|literal|array|after|done|failed
    frames: tuple = ()    # stack of ObjectFrame / ArrayFrame, last is current
    text: str = ""        # accumulated key / string / number text
    letters: str = ""     # accumulated literal (true/false)
    after_comma: bool = False


_FAILED_STATE = State(mode="failed")


def node_object(properties, required=()):
    return {"type": OBJECT, "properties": dict(properties), "required": tuple(required)}


def node_array(items):
    return {"type": ARRAY, "items": items}


def node_dependent(on, variants):
    return {"type": DEP, "on": on, "variants": dict(variants)}


def bfcl_node(schema):
    """Convert one BFCL parameter schema (Python type names) into a grammar node.

    `enum` and `default` are ignored: the grammar enforces structure, and semantic
    constraints are the evaluator's job (BFCL ground truth is per-argument sets).
    """
    kind = schema.get("type")
    if kind in ("dict", "object"):
        return node_object(
            {name: bfcl_node(prop) for name, prop in (schema.get("properties") or {}).items()},
            schema.get("required") or (),
        )
    if kind in ("array", "list"):
        return node_array(bfcl_node(schema["items"]))
    if kind == "integer":
        return {"type": INT}
    if kind in ("float", "number"):
        return {"type": NUMBER}
    if kind == "boolean":
        return {"type": BOOL}
    if kind == "string":
        return {"type": STRING}
    raise ValueError(f"Unsupported BFCL parameter type: {kind!r}")


def bfcl_envelope(functions):
    """The {tool, parameters} envelope for one record's function list.

    One function: `parameters` is that function's schema directly. Several functions:
    `parameters` is a dependent node that resolves on the written `tool` value, so the
    grammar masks the right parameter keys for whichever function is being emitted.
    """
    tool_node = {"type": ENUM, "values": tuple(function["name"] for function in functions)}
    if len(functions) == 1:
        params_node = bfcl_node(functions[0].get("parameters") or {"type": "dict", "properties": {}})
    else:
        params_node = node_dependent(
            "tool", {function["name"]: bfcl_node(function.get("parameters") or {"type": "dict", "properties": {}})
                     for function in functions}
        )
    return node_object({"tool": tool_node, "parameters": params_node}, ("tool", "parameters"))


class JsonSchemaGrammar:
    """Incremental validator for prefixes of a JSON object schema."""

    def __init__(self, schema):
        self._nodes = []
        self._index = {}
        self.root = self._intern(schema)
        self.reset()

    # -- node interning ---------------------------------------------------------------

    def _intern(self, node):
        key = _freeze(node)
        if key in self._index:
            return self._index[key]
        node_id = len(self._nodes)
        self._nodes.append(node)
        self._index[key] = node_id
        return node_id

    def _node(self, node_id):
        return self._nodes[node_id]

    # -- introspection ------------------------------------------------------------------

    @property
    def complete(self):
        return self.state.mode == "done"

    @property
    def failed(self):
        return self.state.mode == "failed"

    def reset(self):
        self.state = State()

    def snapshot(self):
        return self.state

    def restore(self, state):
        self.state = state

    def _top(self, state):
        return state.frames[-1]

    def _written_value(self, frame, name):
        for written_name, value in frame.written:
            if written_name == name:
                return value
        return None

    def _resolve(self, node, frame):
        if node["type"] == DEP:
            value = self._written_value(frame, node["on"]) if isinstance(frame, ObjectFrame) else None
            return node["variants"].get(value)
        return node

    def _value_node(self, state):
        """The schema node of the value currently being read (object property or array item)."""
        frame = self._top(state)
        if isinstance(frame, ObjectFrame):
            for name, node_id in frame.props:
                if name == frame.pending:
                    return self._resolve(self._node(node_id), frame)
            return None
        return self._node(frame.item)

    def available_keys(self, state):
        """Property names still spellable from `state`'s top object ("" if not an object)."""
        frame = self._top(state)
        if not isinstance(frame, ObjectFrame):
            return ()
        names = []
        for name, node_id in frame.props:
            if name in frame.keys:
                continue
            node = self._node(node_id)
            if node["type"] == DEP and self._written_value(frame, node["on"]) is None:
                continue  # a dependent key is unreachable until its dependency is written
            names.append(name)
        return tuple(names)

    @staticmethod
    def _is_prefix(text, literals):
        return any(literal.startswith(text) for literal in literals)

    # -- validation ---------------------------------------------------------------------

    def accepts(self, chunk):
        saved = self.state
        ok = self.feed(chunk)
        self.state = saved
        return ok

    def feed(self, chunk):
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
            if character != "{":
                return None
            return State(mode="key", frames=(self._object_frame(self._node(self.root)),))

        if mode == "key":
            if character == '"':
                return state._replace(mode="in_key", text="", after_comma=False)
            if character == "}" and not state.after_comma:
                return self._close_frame(state)
            return None

        if mode == "in_key":
            frame = self._top(state)
            if character == '"':
                if not any(name == state.text for name, _ in frame.props) or state.text in frame.keys:
                    return None
                return state._replace(
                    mode="colon", text="",
                    frames=state.frames[:-1] + (frame._replace(keys=frame.keys + (state.text,), pending=state.text),),
                )
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
                return None
            return state._replace(text=state.text + character)

        if mode == "number":
            return self._step_number(state, character)

        if mode == "literal":
            letters = state.letters + character
            target = "true" if letters[:1] == "t" else "false"
            if not target.startswith(letters):
                return None
            if letters == target:
                return self._complete_scalar(state._replace(letters=""), target == "true")
            return state._replace(letters=letters)

        if mode == "array":
            return self._start_array_item(state, character)

        if mode == "after":
            return self._after_char(state, character)

        return None

    def _start_value(self, state, character):
        node = self._value_node(state)
        if node is None:
            return None
        kind = node["type"]
        if kind == OBJECT:
            if character != "{":
                return None
            return state._replace(mode="key", frames=state.frames + (self._object_frame(node),), after_comma=False)
        if kind == ARRAY:
            if character != "[":
                return None
            return state._replace(mode="array", frames=state.frames + (ArrayFrame(self._intern(node["items"]), 0, False),),
                                  after_comma=False)
        if kind == STRING or kind == ENUM:
            return state._replace(mode="in_string", text="") if character == '"' else None
        if kind in (INT, NUMBER):
            if character not in "-0123456789":
                return None
            if kind == INT and character == "0" and self._node_has_positive_min(node):
                return None
            return state._replace(mode="number", text=character)
        if kind == BOOL:
            return state._replace(mode="literal", letters=character) if character in "tf" else None
        return None

    def _start_array_item(self, state, character):
        node = self._node(self._top(state).item)
        kind = node["type"]
        if character == "]" and self._top(state).count == 0 and not state.after_comma:
            return self._close_frame(state)   # empty array
        if kind == OBJECT:
            if character != "{":
                return None
            return state._replace(mode="key", frames=state.frames + (self._object_frame(node),), after_comma=False)
        if kind == ARRAY:
            if character != "[":
                return None
            return state._replace(mode="array", frames=state.frames + (ArrayFrame(self._intern(node["items"]), 0, False),),
                                  after_comma=False)
        if kind == STRING or kind == ENUM:
            return state._replace(mode="in_string", text="") if character == '"' else None
        if kind in (INT, NUMBER):
            if character not in "-0123456789":
                return None
            return state._replace(mode="number", text=character)
        if kind == BOOL:
            return state._replace(mode="literal", letters=character) if character in "tf" else None
        return None

    def _step_number(self, state, character):
        text = state.text
        if character in DIGITS:
            if not _digit_after_leading_zero(text):
                return state._replace(text=text + character)
            return None
        if character in "+-":
            if text == "" and character == "-":
                return state._replace(text="-")
            if text and text[-1] in "eE" and character in "+-":
                return state._replace(text=text + character)
            return None
        if character == ".":
            if "." not in text and "e" not in text and "E" not in text and text not in ("", "-"):
                return state._replace(text=text + ".")
            return None
        if character in "eE":
            if text and text[-1] not in "eE+-." and "e" not in text and "E" not in text:
                return state._replace(text=text + character)
            return None
        return self._finish_number(state, character)

    def _finish_number(self, state, next_character):
        text = state.text
        if text in ("", "-", ".", "-.") or text.endswith(".") or text[-1:] in "eE+-":
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        node = self._value_node(state)
        if node is not None and node["type"] == INT:
            if node.get("minimum") is not None and int(value) < node["minimum"]:
                return None
            if node.get("maximum") is not None and int(value) > node["maximum"]:
                return None
        return self._step(self._complete_scalar(state._replace(text=""), value), next_character)

    def _finish_string(self, state):
        node = self._value_node(state)
        if node is not None and node["type"] == ENUM and state.text not in node.get("values", ()):
            return None
        return self._complete_scalar(state._replace(text=""), state.text)

    def _complete_scalar(self, state, value):
        frame = self._top(state)
        if isinstance(frame, ObjectFrame):
            new_frame = frame._replace(pending="", written=frame.written + ((frame.pending, value),))
            return state._replace(mode="after", frames=state.frames[:-1] + (new_frame,))
        array = frame
        new_array = array._replace(count=array.count + 1, had=True)
        return state._replace(mode="after", frames=state.frames[:-1] + (new_array,))

    def _after_char(self, state, character):
        frame = self._top(state)
        if isinstance(frame, ObjectFrame):
            if character == "," and self.available_keys(state):
                return state._replace(mode="key", after_comma=True)
            if character == "}" and not state.after_comma:
                return self._close_frame(state)
            return None
        if character == ",":
            return state._replace(mode="array", after_comma=True)
        if character == "]":
            return self._close_frame(state)
        return None

    def _close_frame(self, state):
        frame = self._top(state)
        if isinstance(frame, ObjectFrame) and not all(name in frame.keys for name in frame.required):
            return None
        below = state.frames[:-1]
        if not below:
            return State(mode="done", frames=())
        parent = below[-1]
        if isinstance(parent, ObjectFrame):
            return state._replace(mode="after", frames=below[:-1] + (parent._replace(pending=""),), after_comma=False)
        new_parent = parent._replace(count=parent.count + 1, had=True)
        return state._replace(mode="after", frames=below[:-1] + (new_parent,), after_comma=False)

    def _object_frame(self, node):
        props = tuple((name, self._intern(sub)) for name, sub in node["properties"].items())
        return ObjectFrame(props, tuple(node.get("required") or ()), (), (), "")

    @staticmethod
    def _node_has_positive_min(node):
        return node.get("minimum") is not None and node["minimum"] > 0


def _digit_after_leading_zero(text):
    """True if appending a digit would violate JSON's no-leading-zero rule.

    The rule applies only to the mantissa's integer part: once a fraction (`.`) or an
    exponent (`e`/`E`) has started, any digit is legal.
    """
    if "e" in text or "E" in text or "." in text:
        return False
    return text in ("0", "-0")


def _freeze(node):
    """A canonical hashable form of a node, used only to deduplicate within one grammar."""
    kind = node["type"]
    if kind == OBJECT:
        return (kind, tuple(sorted((name, _freeze(sub)) for name, sub in node["properties"].items())),
                tuple(node.get("required") or ()))
    if kind == ARRAY:
        return (kind, _freeze(node["items"]))
    if kind == DEP:
        return (kind, node["on"], tuple(sorted((name, _freeze(sub)) for name, sub in node["variants"].items())))
    return (kind, tuple(node.get("values") or ()), node.get("minimum"), node.get("maximum"))


# --------------------------------------------------------------------------------------
# The processor: same masking, driven by the schema grammar
# --------------------------------------------------------------------------------------

NODE_FIRST_CHARS = {
    STRING: '"', ENUM: '"', INT: "-0123456789", NUMBER: "-0123456789", BOOL: "tf", OBJECT: "{", ARRAY: "[",
}


class SchemaGrammarProcessor(GrammarProcessor):
    """A `GrammarProcessor` whose mask is driven by a `JsonSchemaGrammar`.

    Only the mask decision and the cache key differ from the base class; the vocabulary
    indexing, token validation, string/number candidate generation, and mask building are
    inherited unchanged.
    """

    def _state_signature(self, state):
        return (state.mode, state.frames, state.text, state.letters, state.after_comma)

    def allowed_token_ids(self):
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
            allowed = self._literal_candidates(self.grammar.available_keys(state), state.text)
        elif mode == "colon":
            allowed = self._validated(":")
        elif mode == "value":
            node = self.grammar._value_node(state)
            allowed = self._validated(NODE_FIRST_CHARS[node["type"]]) if node is not None else []
        elif mode == "in_string":
            node = self.grammar._value_node(state)
            if node is not None and node["type"] == ENUM:
                allowed = self._literal_candidates(node.get("values", ()), state.text)
            elif self.string_run >= self.max_string_tokens:
                allowed = list(self._text_index.get('"', ()))   # close a runaway free string
            else:
                allowed = self._string_safe + list(self._text_index.get('"', ()))
        elif mode == "number":
            node = self.grammar._value_node(state)
            allowed = self._validated("-0123456789.")
            if node is not None and node["type"] == INT:
                value = _number_text(state.text)
                if node.get("maximum") is None or (value is not None and value <= node["maximum"]):
                    allowed += self._validated(self._close_chars(state))
            else:
                allowed += self._validated(self._close_chars(state))
        elif mode == "literal":
            target = "true" if state.letters[:1] == "t" else "false"
            allowed = self._validated(target[len(state.letters):len(state.letters) + 1])
        elif mode == "array":
            node = self.grammar._node(self.grammar._top(state).item)
            allowed = self._validated(NODE_FIRST_CHARS[node["type"]])
            if self.grammar._top(state).count == 0 and not state.after_comma:
                allowed += self._validated("]")
        elif mode == "after":
            allowed = self._validated(self._close_chars(state))
        elif self.grammar.complete:
            allowed = [self.eos_token_id] if self.eos_token_id is not None else []
        else:
            allowed = []
        self._mask_cache[signature] = allowed
        return allowed

    def _close_chars(self, state):
        frame = state.frames[-1]
        if isinstance(frame, ObjectFrame):
            return ",}"
        return "," if frame.count < self.max_array_items else "]"


def _number_text(text):
    try:
        return float(text)
    except ValueError:
        return None
