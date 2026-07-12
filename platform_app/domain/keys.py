import hashlib
import re
from urllib.parse import unquote, urlparse


INFOHASH_RE = re.compile(r'urn:btih:([a-fA-F0-9]{40}|[a-zA-Z2-7]{32})', re.I)


def normalize_info_hash(value: str | None) -> str:
    if not value:
        return ''
    cleaned = ''.join(character for character in value.lower() if character in '0123456789abcdef')
    if len(cleaned) == 40:
        return cleaned
    return ''


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


def content_group_id(
    info_hash: str = '',
    magnet_uri: str = '',
    download_link: str = '',
    title: str = '',
    content_key: str = '',
) -> str:
    clean_hash = normalize_info_hash(info_hash) or extract_info_hash_from_magnet(magnet_uri or download_link)
    if clean_hash:
        return f'btih:{clean_hash}'
    link = (download_link or magnet_uri or '').strip()
    if link:
        return f'link:{link}'
    normalized_title = re.sub(r'\s+', '', (title or '').strip().lower())
    if normalized_title:
        return f'title:{normalized_title}'
    if content_key:
        return content_key
    return f'row:{hashlib.sha256((title or "").encode("utf-8")).hexdigest()[:16]}'


def validate_magnet_uri(magnet_uri: str) -> str:
    value = (magnet_uri or '').strip()
    if not value:
        return ''
    if not value.lower().startswith('magnet:'):
        raise ValueError('磁力链接必须以 magnet: 开头')
    return value


def normalize_source_url(url: str) -> str:
    raw = (url or '').strip()
    if not raw:
        return ''
    parsed = urlparse(raw)
    if not parsed.scheme:
        return raw.lower()
    path = unquote(parsed.path or '')
    return f'{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}'
