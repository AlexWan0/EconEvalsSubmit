from typing import TypeAlias, Mapping, Sequence, TypeVar, Callable, Literal, TypedDict, Any
import orjson
import json
import re

class Turn(TypedDict):
    role: Literal['user', 'assistant']
    content: str

# From: https://github.com/python/typing/issues/182#issuecomment-1320974824
JSONType: TypeAlias = Mapping[str, "JSONType"] | Sequence["JSONType"] | str | int | float | bool | None

def fast_json_dumps(obj: JSONType) -> str:
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode(encoding='utf-8')

def fast_json_loads(serialized: str) -> JSONType:
    return orjson.loads(serialized.encode('utf-8'))

def str_to_convo(user_input: str) -> list[Turn]:
    return [
        {'role': 'user', 'content': user_input}
    ]


RE_SUFFIX = re.compile(r"^(?P<base>[^@]+)(?:@(?P<kv>.+))?$")

def parse_model_with_params(model_with_suffix: str) -> tuple[str, dict]:
    '''
    Example:
        Input: 'o4-mini@temperature=0.2,logprobs=true,tools=[{"type":"function","function":{"name":"foo"}}]'
        Output: ('o4-mini', {'temperature': 0.2, 'logprobs': True, 'tools': [...]})
    '''
    m = RE_SUFFIX.match(model_with_suffix.strip())
    if not m:
        return model_with_suffix, {}

    base = m.group("base")
    kv = m.group("kv")
    if not kv:
        return base, {}

    # split on commas while respecting brackets/braces
    parts = []
    buf, depth = [], 0
    for ch in kv:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))

    out: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = k.strip()
        raw = v.strip()
        # attempt JSON parse first
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            # fallback: booleans/numbers, else string
            low = raw.lower()
            if low in ("true", "false"):
                val = (low == "true")
            else:
                try:
                    # int or float
                    val = int(raw) if raw.isdigit() else float(raw)
                except ValueError:
                    # bareword string (e.g., high, minimal)
                    val = raw
        out[key] = val
    return base, out
