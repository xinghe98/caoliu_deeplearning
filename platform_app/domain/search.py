import re

_MAX_QUERY_LEN = 100
_MAX_TOKENS = 8
# Keep letters/digits (including CJK); drop whitespace, punctuation, underscore.
_NON_WORD_RE = re.compile(r'[^\w]+', re.UNICODE)


def normalize_title_text(value: str | None) -> str:
    if not value:
        return ''
    cleaned = _NON_WORD_RE.sub('', value.casefold())
    return cleaned.replace('_', '')


def parse_search_tokens(q: str | None) -> list[str]:
    if not q:
        return []
    raw = q.strip()[:_MAX_QUERY_LEN]
    if not raw:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for part in raw.split():
        token = normalize_title_text(part)
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= _MAX_TOKENS:
            break
    return tokens
