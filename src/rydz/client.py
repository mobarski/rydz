import os
import threading

import openai

BASE_URL = {
    'lmstudio':    "http://localhost:1234/v1",
    'openrouter':  "https://openrouter.ai/api/v1",
    'hyperbolic':  "https://api.hyperbolic.xyz/v1",
    'fireworks':   "https://api.fireworks.ai/inference/v1",
    'together':    "https://api.together.xyz/v1",
    'openai':      "https://api.openai.com/v1",
    'xai':         "https://api.x.ai/v1/",
    'google':      "https://generativelanguage.googleapis.com/v1beta/",
    'cerebras':    "https://api.cerebras.ai/v1",
    'novita':      "https://api.novita.ai/openai",
    'groq':        "https://api.groq.com/openai/v1", # NO LOGPROBS
    'baseten':     "https://inference.baseten.co/v1", # NO LOGPROBS
    'siliconflow': "https://api.siliconflow.com/v1", # NO LOGPROBS
    'deepinfra':   "https://api.deepinfra.com/v1/openai", # NO LOGPROBS (streaming api only, one logprob per token)
    'huggingface': "https://router.huggingface.co/v1", # NO LOGPROBS
    'nebius':      "https://api.tokenfactory.nebius.com/v1/", # ??? ugly credit card input
}
# TODO: novita, nebius
QUIRKS = {
    'lmstudio':    {'endpoint': 'responses', 'max_tokens': 2},
    'openai':      {'max_tokens': 16},
    'xai':         {'top_logprobs': 8},
    'fireworks':   {'top_logprobs': 5},
    'cerebras':    {'temperature': 1e-8},
    'huggingface': {'get_api_key': lambda model: os.getenv('HF_TOKEN')},
}

_client_cache = {}
_client_lock = threading.Lock()


def _get_client(model):
    with _client_lock:
        if model not in _client_cache:
            _client_cache[model] = openai.OpenAI(
                api_key=_get_api_key(model),
                base_url=_get_base_url(model),
            )
        return _client_cache[model]


def _get_base_url(model):
    provider = model.partition(':')[0]
    if provider not in BASE_URL:
        raise ValueError(f"Unsupported provider: {provider}")
    return BASE_URL[provider]


def _get_api_key(model, default='NONE'):
    provider = model.partition(':')[0]
    if 'get_api_key' in QUIRKS.get(provider, {}):
        fun = QUIRKS[provider]['get_api_key']
        return fun(model)
    else:
        env_var = f"{provider.upper()}_API_KEY"
        return os.getenv(env_var, default)


def register_provider(provider, base_url, quirks=None):
    BASE_URL[provider] = base_url
    if quirks is not None:
        QUIRKS[provider] = quirks


def register_alias(alias, provider, quirks=None):
    """Register alias for an existing provider (copies base_url and quirks). Useful for multiple API keys."""
    if provider not in BASE_URL:
        raise ValueError(f"Unknown provider: {provider}")
    BASE_URL[alias] = BASE_URL[provider]
    merged = {**QUIRKS.get(provider, {}), **(quirks or {})}
    if merged:
        QUIRKS[alias] = merged


def set_quirk(provider, key, value):
    """Register a quirk for an existing provider. Useful for custom providers."""
    assert provider in QUIRKS, f"Provider {provider} not registered"
    QUIRKS[provider][key] = value


def model_name(model):
    return model.split(':')[1]


def model_aux_str(model):
    parts = model.split(':')
    return parts[2] if len(parts) > 2 else ''

