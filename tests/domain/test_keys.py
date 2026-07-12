from platform_app.domain.keys import canonical_key, content_group_id, extract_info_hash_from_magnet, validate_magnet_uri
from platform_app.domain.splits import stable_split


def test_canonical_key_prefers_infohash():
    key = canonical_key(source_url='http://example.com/a', info_hash='ABCDEF0123456789ABCDEF0123456789ABCDEF01')
    assert key == 'btih:abcdef0123456789abcdef0123456789abcdef01'


def test_canonical_key_falls_back_to_url():
    key = canonical_key(source_url='http://Example.com/Post')
    assert key.startswith('url:')
    assert canonical_key(source_url='http://example.com/post') == key


def test_magnet_validation_and_extract():
    magnet = 'magnet:?xt=urn:btih:ABCDEF0123456789ABCDEF0123456789ABCDEF01&dn=x'
    assert validate_magnet_uri(magnet).startswith('magnet:')
    assert extract_info_hash_from_magnet(magnet) == 'abcdef0123456789abcdef0123456789abcdef01'


def test_validate_magnet_rejects_http():
    try:
        validate_magnet_uri('http://evil.example/x')
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_stable_split_external_locked():
    assert stable_split('any', 'external_test') == 'external_test'
    a = stable_split('group-a')
    b = stable_split('group-a')
    assert a == b
    assert a in {'train', 'validation', 'production_shadow_test'}


def test_content_group_id_link_priority():
    group = content_group_id(download_link='magnet:?xt=urn:btih:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    assert group.startswith('btih:')
