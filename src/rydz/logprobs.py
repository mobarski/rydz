from math import exp
import time
import types

from .client import _get_client, model_name, model_aux_str, QUIRKS


# REF: https://developers.openai.com/api/reference/resources/responses/methods/create
def _get_response_from_responses(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    aux_str = model_aux_str(model)
    quirks = QUIRKS.get(provider, {}).copy()
    for tag in ['thinking']:
        quirks[tag] = tag in aux_str
    quirks.update(kwargs)
    client_kwargs = dict(
        model=model_name(model),
        input=prompt,
        temperature=quirks.get('temperature', 0.0),
        top_logprobs=quirks.get('top_logprobs', 20),
        max_output_tokens=quirks.get('max_tokens', 4096 if quirks['thinking'] else 1), # TODO: thinking vs provider quirk
        include=['message.output_text.logprobs'],
    )
    if quirks['thinking']:
        client_kwargs['text'] = {"verbosity": "low", "format": {"type": "text"}}
        client_kwargs['reasoning'] = {"effort": "minimal", "summary": "concise"}
    t0 = time.perf_counter()
    resp = client.responses.create(**client_kwargs)
    resp.aux = types.SimpleNamespace()
    resp.aux.rtt = time.perf_counter() - t0
    # usage
    resp.aux.input_tokens  = resp.usage.input_tokens
    resp.aux.output_tokens = resp.usage.output_tokens
    # TODO: thinking tokens
    #
    all_logprobs = resp.output[-1].content[0].logprobs
    resp.aux.logprobs = _get_top_logprobs_skipping_thinking_tokens(all_logprobs)
    return resp


# REF: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
def _get_response_from_chat(model, prompt, **kwargs):
    client = _get_client(model)
    provider = model.partition(':')[0]
    aux_str = model_aux_str(model)
    quirks = QUIRKS.get(provider, {}).copy()
    for tag in ['thinking']:
        quirks[tag] = tag in aux_str
    quirks.update(kwargs)
    client_kwargs = dict(
        model=model_name(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=quirks.get('temperature', 0.0),
        max_completion_tokens=quirks.get('max_tokens', 4096 if quirks['thinking'] else 1), # TODO: thinking vs provider quirk
        logprobs=quirks.get('logprobs', True),
        top_logprobs=quirks.get('top_logprobs', 20),
        #reasoning_effort='low',
        #verbosity='low',
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**client_kwargs)
    resp.aux = types.SimpleNamespace()
    resp.aux.rtt = time.perf_counter() - t0
    # usage
    resp.aux.input_tokens  = resp.usage.prompt_tokens
    resp.aux.output_tokens = resp.usage.completion_tokens
    # TODO: thinking tokens from usage ???
    #
    all_logprobs = resp.choices[0].logprobs.content
    resp.aux.logprobs = _get_top_logprobs_skipping_thinking_tokens(all_logprobs)
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
    return min(1.0, p_total)


def _get_top_logprobs_skipping_thinking_tokens(logprobs):
    tokens = [t.token for t in logprobs]
    # detect presence of end-of-thinking anchor token
    for a in ['</think>', '<|message|>']: # TODO: ability to extend this list
        if a in tokens:
            anchor = a
            break
    else:
        anchor = None
    # get index of end-of-thinking anchor token
    if anchor:
        assert anchor in tokens, "END-OF-THINKING not found"
        i_anchor = len(tokens) - 1 - tokens[::-1].index(anchor) # index_right
        i_resp = i_anchor + 1
    else:
        i_resp = 0
    # get logprobs for first non-empty token after end-of-thinking anchor token
    for i in range(i_resp, len(logprobs)):
        if not logprobs[i].token.strip(): continue # skip empty tokens
        return logprobs[i].top_logprobs
    return []

# TODO: seed
# TODO: detect thinking model whem no thinking=True is provided ???
