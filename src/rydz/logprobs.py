from math import exp
import time
import types

from .client import _get_client, model_name, QUIRKS


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
