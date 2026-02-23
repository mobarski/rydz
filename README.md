# Rydz

> *"Lepszy rydz niż nic"*

Build and deploy LLM-based classifiers in minutes, not days.

Rydz uses **logprobs** to extract classification probabilities from LLMs in a single API call — no fine-tuning, no training data, no ML pipeline. Just a prompt and a model.

## Why?

A perfect classifier you don't have time to build is worth less than a good-enough one you can deploy right now.
Rydz gives you the latter — and leaves the door open for the former.

> **Fun fact:** Contrary to popular belief, "rydz" in the Polish proverb doesn't refer to a mushroom.
> It's [*Camelina sativa*](https://www.muzeum-radom.pl/turystyka/historia-znanego-porzekadla-lepszy-rydz-niz-nic/1886) — a humble oil plant that thrives where nothing else will grow.
> Seemed like a fitting name for a library that gets the job done when fancier solutions aren't an option.

## Quick start

```python
from rydz import get_logprobs_response, get_probability

prompt = """Classify the sentiment of this review as POSITIVE or NEGATIVE.

Review: "The battery lasts forever and the screen is gorgeous!"

Sentiment:"""

resp = get_logprobs_response("openai:gpt-4o-mini", prompt)
print(f"positive: {get_probability(resp, 'POSITIVE'):.1%}")
print(f"negative: {get_probability(resp, 'NEGATIVE'):.1%}")
# positive: 99.2%  negative: 0.8%
```

## Parallel classification

```python
from rydz import get_logprobs_response, get_probability, tmap_unordered

def classify(text):
    prompt = f"Is this spam? Answer YES or NO.\n\n{text}\n\nAnswer:"
    resp = get_logprobs_response("together:meta-llama/Llama-3-70b-chat-hf", prompt)
    return get_probability(resp, "YES")

results = list(tmap_unordered(classify, texts, workers=16))
```

## How it works

1. You craft a prompt that frames the classification task
2. Rydz sends it to the LLM and requests **logprobs** for the first output token
3. Probabilities of your target labels are extracted directly — no sampling, no repeated calls
4. One API call → one classification with confidence scores

This means: **high throughput, low cost, low latency**. Thousands of input tokens, 1 output token.

## Features

- **One thing, done well** — logprobs-based classification
- **Multiple providers** — OpenAI, xAI, Together, Fireworks, Hyperbolic, OpenRouter, LM Studio
- **Provider quirks handled** — different APIs, limits, endpoints — all behind one interface
- **Parallel processing** — classify thousands of items across providers in seconds
- **Minimal dependencies** — just `openai`

## Installation

```bash
pip install git+https://github.com/mobarski/rydz
```

## Model format

Models use the `provider:model_name` convention:

```python
"openai:gpt-4o-mini"
"xai:grok-2"
"together:meta-llama/Llama-3-70b-chat-hf"
"lmstudio:bielik-11b"
```

## Supported providers

### local
| provider | env variable | notes |
| - | - | - |
| lmstudio | LMSTUDIO_API_KEY | |

### cloud
| provider | env variable | notes |
| - | - | - |
| openai | OPENAI_API_KEY | |
| xai | XAI_API_KEY | |
| together | TOGETHER_API_KEY | |
| hyperbolic | HYPERBOLIC_API_KEY | |
| fireworks | FIREWORKS_API_KEY | |
| openrouter | OPENROUTER_API_KEY | most inference providers = no logprobs |
| google | GOOGLE_API_KEY | no logprobs |

### custom provider

```python
from rydz import register_provider
register_provider("myprovider", "https://api.example.com/v1")
```

## Use cases

- sentiment analysis in reviews / social media
- content moderation
- support ticket classification
- ad profile matching to articles
- content complexity vs target audience
- contract risk clause detection
- scene / emotion tagging in narratives
- safety analysis of AI-generated code

## License

MIT
