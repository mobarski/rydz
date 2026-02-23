import pytest
from rydz.client import BASE_URL, QUIRKS, register_provider, register_alias


def test_register_provider():
    register_provider("testprov", "https://test.example.com/v1")
    assert BASE_URL["testprov"] == "https://test.example.com/v1"
    assert "testprov" not in QUIRKS


def test_register_provider_with_quirks():
    register_provider("testprov2", "https://test2.example.com/v1", quirks={"top_logprobs": 10})
    assert BASE_URL["testprov2"] == "https://test2.example.com/v1"
    assert QUIRKS["testprov2"] == {"top_logprobs": 10}


def test_register_alias():
    register_alias("openai_copy", "openai")
    assert BASE_URL["openai_copy"] == BASE_URL["openai"]
    assert QUIRKS["openai_copy"] == QUIRKS["openai"]


def test_register_alias_with_quirks_override():
    register_alias("openai_custom", "openai", quirks={"max_tokens": 4})
    assert BASE_URL["openai_custom"] == BASE_URL["openai"]
    assert QUIRKS["openai_custom"]["max_tokens"] == 4


def test_register_alias_with_quirks_merge():
    register_alias("xai2", "xai", quirks={"max_tokens": 2})
    assert QUIRKS["xai2"]["top_logprobs"] == 8  # inherited from xai
    assert QUIRKS["xai2"]["max_tokens"] == 2    # added by alias


def test_register_alias_no_quirks_source():
    register_provider("bare", "https://bare.example.com/v1")
    register_alias("bare2", "bare")
    assert BASE_URL["bare2"] == "https://bare.example.com/v1"
    assert "bare2" not in QUIRKS


def test_register_alias_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        register_alias("bad", "nonexistent")
