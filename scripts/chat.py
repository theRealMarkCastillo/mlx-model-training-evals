"""Ask a served model for a tool call, using this project's system prompt.

`s main.py serve` runs MLX-LM's OpenAI-compatible server, which does **not** add the
system prompt this task was trained with, so this script sends it explicitly —
that is the whole point of the example.

    # terminal 1: serve the latest fused model (or add --base)
    uv run python main.py serve --preset 3b

    # terminal 2
    uv run python scripts/chat.py "Ship v2.3.1 of auth-api into prod with 3 instances"

Uses only the standard library, so it needs no extra dependency. Set MODEL
or PORT to override the defaults.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import SYSTEM_PROMPT  # noqa: E402

URL = f"http://localhost:{os.environ.get('PORT', '8080')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")


def ask(prompt, max_tokens=150):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


def main(argv):
    prompt = " ".join(argv) or "Restart pod redis-cluster-cache-3 in us-east-1 due to OOM spike."
    print(f"request:   {prompt}")
    try:
        completion = ask(prompt)
    except urllib.error.URLError as error:
        print(f"could not reach {URL}: {error.reason}")
        print("Start the server first:  uv run python main.py serve --preset 3b")
        return 1
    print(f"completion: {completion}")
    try:
        print("parsed:     " + json.dumps(json.loads(completion), separators=(",", ":")))
    except json.JSONDecodeError as error:
        print(f"parsed:     not valid JSON ({error})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
