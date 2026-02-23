from math import exp
import os
import threading
import time
import types

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


def _get_response_from_responses(prompt, model):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model_name(model),
        input=prompt,
        temperature=0.0,
        top_logprobs=quirks.get('top_logprobs', 20),
        max_output_tokens=quirks.get('max_tokens', 1),
        include=['message.output_text.logprobs'],
        #reasoning_effort='none',
    )
    resp.aux = types.SimpleNamespace()
    resp.aux.rtt = time.perf_counter() - t0
    resp.aux.input_tokens  = resp.usage.input_tokens
    resp.aux.output_tokens = resp.usage.output_tokens
    resp.aux.logprobs      = resp.output[0].content[0].logprobs[0].top_logprobs
    return resp


def _get_response_from_chat(prompt, model):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_name(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=quirks.get('max_tokens', 1),
        logprobs=True,
        top_logprobs=quirks.get('top_logprobs', 20),
        #reasoning_effort='none',
        #verbosity='low',
        #logit_bias={200005: -100, 200003: -100, 200008: -100},
    )
    resp.aux = types.SimpleNamespace()
    resp.aux.rtt = time.perf_counter() - t0
    resp.aux.input_tokens  = resp.usage.prompt_tokens
    resp.aux.output_tokens = resp.usage.completion_tokens
    resp.aux.logprobs      = resp.choices[0].logprobs.content[0].top_logprobs
    return resp


def register_provider(provider, base_url, quirks=None):
    BASE_URL[provider] = base_url
    if quirks is not None:
        QUIRKS[provider] = quirks

def model_name(model):
    return model.split(':')[1]


def get_logprobs_response(model, prompt):
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    if quirks.get('endpoint') == 'responses':
        return _get_response_from_responses(prompt, model)
    else:
        return _get_response_from_chat(prompt, model)


def get_probability(resp, answer):
    p_total = 0.0
    for x in resp.aux.logprobs:
        t = x.token.lstrip().upper()
        if answer.upper().startswith(t):
            p_total += exp(x.logprob)
    return p_total

