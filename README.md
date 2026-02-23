# Rydz

> *"Lepszy rydz niż nic"* — Polish proverb meaning "better something than nothing"

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

## Why logprobs?

| approach | calls per classification | confidence score | training data | setup time |
| - | - | - | - | - |
| **Rydz (logprobs)** | **1** | **yes, native** | **none** | **minutes** |
| LLM + text parsing | 1 | no | none | minutes |
| LLM + repeated sampling | 5–20 | approximate | none | minutes |
| fine-tuned model | 1 | yes | hundreds+ | days–weeks |
| classical ML | 1 | yes | thousands+ | days–weeks |

Logprobs give you calibrated confidence in a single call. No repeated sampling, no parsing "yes"/"no" from free text, no training data collection. You get a probability distribution over your labels — directly from the model's internals.

**[Real-world benchmark:](https://www.linkedin.com/feed/update/urn:li:activity:7431396065020583936/)** 280 text fragments × 8 classification criteria (~1.5M input tokens, 2000+ data points) — processed in **36 seconds for $0.25** using two cloud providers in parallel, or **4 minutes** using a local model (Bielik).

## Beyond naive classifiers

A single model with a hand-crafted prompt is just the starting point. Rydz's low cost and high throughput make it practical to build more advanced systems:

- **Ensemble / majority voting** — score the same input with multiple models and aggregate results for higher accuracy and resilience
- **Automatic prompt optimization** — integrate with frameworks like [DSPy](https://github.com/stanfordnlp/dspy) to optimize prompts systematically instead of relying on intuition alone

You don't have to pick one — start simple, scale up when needed.

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
# uses MYPROVIDER_API_KEY env variable, model string: "myprovider:model-name"
```

This also lets you use **multiple API keys from the same provider** — useful for higher rate limits or separate billing:

```python
register_provider("openai2", "https://api.openai.com/v1")
# uses OPENAI2_API_KEY env variable, model string: "openai2:gpt-4.1-nano"
```

## Use cases

**E-commerce & Marketing** — sentiment analysis in product reviews, matching ad profiles to article content, brand monitoring on social media

**Customer Support** — automatic ticket categorization and priority routing, intent detection in customer messages

**Media & Publishing** — content moderation, scene and emotion tagging in narratives, content complexity matching to target audience

**Legal & Compliance** — contract risk clause detection, document confidentiality classification

**Software & AI** — safety analysis of AI-generated code, detecting policy violations in LLM outputs

## License

MIT
