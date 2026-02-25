from math import exp
import time
import types

from .client import _get_client, model_name, QUIRKS


# REF: https://developers.openai.com/api/reference/resources/responses/methods/create
def _get_response_from_responses(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    quirks.update(kwargs)
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model_name(model),
        input=prompt,
        temperature=quirks.get('temperature', 0.0),
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


def _get_response_from_chat(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    quirks.update(kwargs)
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


def get_logprobs_response(model, prompt, **kwargs):
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    if quirks.get('endpoint') == 'responses':
        return _get_response_from_responses(model, prompt, **kwargs)
    else:
        return _get_response_from_chat(model, prompt, **kwargs)


def get_probability(resp, answer):
    p_total = 0.0
    for x in resp.aux.logprobs:
        t = x.token.lstrip().upper()
        if answer.upper().startswith(t):
            p_total += exp(x.logprob)
    return p_total

### WIP - THINKING MODELS ##############################################################

# REF: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
def _get_response_from_chat_thinking(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {})
    quirks.update(kwargs)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_name(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=quirks.get('temperature', 0.0),
        #max_tokens=1,
        logprobs=True,
        top_logprobs=3, #quirks.get('top_logprobs', 20), # XXX
        #reasoning_effort='none',
        #verbosity='low',
    )
    # TODO: total tokens (input + output + thinking[hidden])
    resp.aux = types.SimpleNamespace()
    resp.aux.all_logprobs = resp.choices[0].logprobs.content
    _skip_thinking_logprobs(resp)
    return resp


# TODO: quirks, aux
# REF: # REF: https://developers.openai.com/api/reference/resources/responses/methods/create
def _get_response_from_responses_thinking(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    quirks = QUIRKS.get(provider, {}) # TODO: add model quirks
    quirks.update(kwargs)
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model_name(model),
        input=prompt,
        temperature=quirks.get('temperature', 0.0),
        top_logprobs=quirks.get('top_logprobs', 20),
        #max_output_tokens=10,
        include=['message.output_text.logprobs'],
        text={"verbosity": "low", "format": {"type": "text"}},
        reasoning={"effort": "minimal", "summary": "concise"},
    )
    resp.aux = types.SimpleNamespace()
    resp.aux.all_logprobs = resp.output[0].content[0].logprobs
    _skip_thinking_logprobs(resp)
    return resp


def _skip_thinking_logprobs(resp):
    logprobs = resp.aux.all_logprobs
    tokens = [t.token for t in logprobs]
    # TODO: quirk
    #anchor = quirks.get('anchor', '</think>')
    anchor = '</think>'
    anchor = '<|message|>' # gpt-oss
    anchor = None # xai
    #
    if anchor:
        assert anchor in tokens, "END-OF-THINKING not found"
        i_anchor = len(tokens) - 1 - tokens[::-1].index(anchor) # index_right
        i_resp = i_anchor + 1
    else:
        i_resp = 0
    #
    for i in range(i_resp, len(logprobs)):
        if not logprobs[i].token.strip(): continue # skip empty tokens
        resp.aux.logprobs = logprobs[i].top_logprobs
        break
    else:
        resp.aux.logprobs = []
    del resp.aux.all_logprobs
