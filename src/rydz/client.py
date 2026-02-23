import os
import threading

import openai

BASE_URL = {
    'lmstudio':   "http://localhost:1234/v1",
    'openrouter': "https://openrouter.ai/api/v1",
    'hyperbolic': "https://api.hyperbolic.xyz/v1",
    'fireworks':  "https://api.fireworks.ai/inference/v1",
    'together':   "https://api.together.xyz/v1",
    'openai':     "https://api.openai.com/v1",
    'xai':        "https://api.x.ai/v1/",
    'google':     "https://generativelanguage.googleapis.com/v1beta/",
}
QUIRKS = {
    'lmstudio':  {'endpoint': 'responses', 'max_tokens': 2},
    'openai':    {'max_tokens': 16},
    'xai':       {'top_logprobs': 8},
    'fireworks': {'top_logprobs': 5},
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
    env_var = f"{provider.upper()}_API_KEY"
    return os.getenv(env_var, default)


def register_provider(provider, base_url, quirks=None):
    BASE_URL[provider] = base_url
    if quirks is not None:
        QUIRKS[provider] = quirks

def model_name(model):
    return model.split(':')[1]
