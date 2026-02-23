from rydz import get_logprobs_response, get_probability


def test_openai():
    P = "Answer the question below. Generate only the answer in ALL CAPS "\
        "and nothing else (no spaces, tabs, new lines and markup). " 
    Q = "What is the capital of Poland?"
    A = "WARSAW"
    model = 'openai:gpt-4.1-nano'
    resp = get_logprobs_response(model, P+Q)
    assert get_probability(resp, A) > 0.9

