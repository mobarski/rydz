import pytest
from rydz.client import BASE_URL, QUIRKS, register_provider, register_alias, set_quirk, _get_api_key


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


# --- set_quirk ---

def test_set_quirk():
    register_provider("test_prov_q", "https://q.example.com/v1", quirks={"max_tokens": 4})
    set_quirk("test_prov_q", "top_logprobs", 10)
    assert QUIRKS["test_prov_q"]["top_logprobs"] == 10
    assert QUIRKS["test_prov_q"]["max_tokens"] == 4  # unchanged


def test_set_quirk_overwrite():
    register_provider("test_prov_q2", "https://q2.example.com/v1", quirks={"max_tokens": 4})
    set_quirk("test_prov_q2", "max_tokens", 8)
    assert QUIRKS["test_prov_q2"]["max_tokens"] == 8


def test_set_quirk_unknown_provider():
    with pytest.raises(AssertionError):
        set_quirk("nonexistent", "key", "value")


# --- get_api_key quirk ---

def test_get_api_key_from_env(monkeypatch):
    register_provider("test_prov_env", "https://env.example.com/v1", quirks={})
    monkeypatch.setenv("TEST_PROV_ENV_API_KEY", "sk-from-env")
    assert _get_api_key("test_prov_env:some-model") == "sk-from-env"


def test_get_api_key_default_when_no_env():
    register_provider("test_prov_no", "https://no.example.com/v1", quirks={})
    assert _get_api_key("test_prov_no:some-model") == "NONE"


def test_get_api_key_quirk():
    register_provider("test_prov_vault", "https://vault.example.com/v1", quirks={})
    set_quirk("test_prov_vault", "get_api_key", lambda model: f"secret-for-{model}")
    assert _get_api_key("test_prov_vault:my-model") == "secret-for-test_prov_vault:my-model"


def test_get_api_key_quirk_overrides_env(monkeypatch):
    register_provider("test_prov_both", "https://both.example.com/v1", quirks={})
    monkeypatch.setenv("TEST_PROV_BOTH_API_KEY", "sk-from-env")
    set_quirk("test_prov_both", "get_api_key", lambda model: "sk-from-vault")
    assert _get_api_key("test_prov_both:some-model") == "sk-from-vault"
