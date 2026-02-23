# Rydz

Features:
- focused on only one thing
- multiple API providers (with all their quirks) hidden behind a simple API
- opinionated but elastic design (ie. using multiple keys with a single provider)
- minimal dependencies (openai)

# License

MIT

# Installation

Run `pip install rydz` or `uv add rydz`. (TODO)

You can also install it directly from github: `pip install git+https://github.com/mobarski/rydz`.

# Supported providers

## local
| provider | environmental variables | notes |
| - | - | - |
| lmstudio | LMSTUDIO_API_KEY |

## cloud
| provider | environmental variables | notes |
| - | - | - |
| openai | OPENAI_API_KEY |
| xai | XAI_API_KEY | 
| together | TOGETHER_API_KEY |
| hyperbolic | HYPERBOLIC_API_KEY |
| fireworks | FIREWORKS_API_KEY |
| openrouter | OPENROUTER_API_KEY | most inference providers = no logprobs |
| google | GOOGLE_API_KEY | no logprobs |


# References

...
