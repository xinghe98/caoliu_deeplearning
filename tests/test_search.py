from platform_app.domain.search import normalize_title_text, parse_search_tokens


def test_normalize_title_strips_space_punct_and_case():
    assert normalize_title_text('xxxab crrrs') == 'xxxabcrrrs'
    assert normalize_title_text('ABC') == 'abc'
    assert normalize_title_text('a-b_c') == 'abc'
    assert normalize_title_text(None) == ''


def test_parse_search_tokens_splits_and_dedupes():
    assert parse_search_tokens('  Foo  bar foo ') == ['foo', 'bar']
    assert parse_search_tokens('a-b c') == ['ab', 'c']
    assert parse_search_tokens('   ') == []
    assert parse_search_tokens(None) == []
