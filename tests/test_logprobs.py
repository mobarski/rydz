from rydz import get_logprobs_response, get_probability
import pytest


def test_openai():
    P = "Answer the question below. Generate only the answer in ALL CAPS "\
        "and nothing else (no spaces, tabs, new lines and markup). " 
    Q = "What is the capital of Poland?"
    A = "WARSAW"
    model = 'openai:gpt-4.1-nano'
    resp = get_logprobs_response(model, P+Q)
    assert get_probability(resp, A) > 0.9


THINKING_KWARGS_MODELS = [
    "fireworks:accounts/fireworks/models/gpt-oss-120b",
    "xai:grok-4-1-fast",
    "together:MiniMaxAI/MiniMax-M2.5",
]
@pytest.mark.parametrize("model", THINKING_KWARGS_MODELS)
def test_thinking_kwargs(model):
    P = "Is 42 > 24? Answer YES or NO."
    A = "YES"
    resp = get_logprobs_response(model, P, thinking=True)
    assert get_probability(resp, A) > 0.9


def test_thinking_model_name():
    P = "Is 42 > 24? Answer YES or NO."
    A = "YES"
    model = "fireworks:accounts/fireworks/models/gpt-oss-120b:thinking"
    resp = get_logprobs_response(model, P)
    assert get_probability(resp, A) > 0.9


def manual_test_thinking(model):
    P = "Is 42 > 24? Answer YES or NO."
    A = "YES"
    resp  = get_logprobs_response(model, P, thinking=True)
    assert get_probability(resp, A) > 0.9


if __name__ == "__main__":
    models = [
        "hyperbolic:openai/gpt-oss-120b",
        "together:MiniMaxAI/MiniMax-M2.5",
        "together:moonshotai/Kimi-K2.5",
        "together:Qwen/Qwen3.5-397B-A17B",
        "together:ServiceNow-AI/Apriel-1.6-15b-Thinker",
        "fireworks:accounts/fireworks/models/minimax-m2p5",
        "fireworks:accounts/fireworks/models/glm-5",
        "fireworks:accounts/fireworks/models/deepseek-v3p2",
        "xai:grok-4-1-fast",
        "fireworks:accounts/fireworks/models/glm-4p7",
        "novita:qwen/qwen3.5-397b-a17b",
        "fireworks:accounts/fireworks/models/gpt-oss-120b",
        "hyperbolic:Qwen/Qwen3-Next-80B-A3B-Thinking",
    ]
    for model in models:
        print(f"Testing {model}...")
        manual_test_thinking(model)
