import hashlib
import re


INFOHASH_RE = re.compile(r'urn:btih:([a-fA-F0-9]{40}|[a-zA-Z2-7]{32})', re.I)


def normalize_info_hash(value: str | None) -> str:
    if not value:
        return ''
    cleaned = ''.join(character for character in value.lower() if character in '0123456789abcdef')
    return cleaned if len(cleaned) == 40 else ''


def extract_info_hash_from_magnet(magnet_uri: str | None) -> str:
    if not magnet_uri:
        return ''
    match = INFOHASH_RE.search(magnet_uri)
    if not match:
        return ''
    token = match.group(1)
    if len(token) == 40 and all(character in '0123456789abcdefABCDEF' for character in token):
        return token.lower()
    return ''


def canonical_key(source_url: str = '', info_hash: str = '', magnet_uri: str = '') -> str:
    clean_hash = normalize_info_hash(info_hash) or extract_info_hash_from_magnet(magnet_uri)
    if clean_hash:
        return f'btih:{clean_hash}'
    url = (source_url or '').strip().lower()
    if not url:
        raise ValueError('content_key 需要 info_hash 或 source_url')
    return f'url:{hashlib.sha256(url.encode("utf-8")).hexdigest()}'
