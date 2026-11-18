from app.adapters.anthropic import AnthropicAdapter
from app.adapters.cloudflare import CloudflareAdapter
from app.adapters.coarena import CoarenaAdapter
from app.adapters.cohere import CohereAdapter
from app.adapters.gemini import GeminiAdapter
from app.adapters.groq import GroqAdapter
from app.adapters.mock import MockAdapter
from app.adapters.nvidia import NvidiaAdapter
from app.adapters.openai import OpenAIAdapter
from app.adapters.openrouter import OpenRouterAdapter
from app.adapters.sambanova import SambaNovaAdapter

ADAPTERS = {
    "mock": MockAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "groq": GroqAdapter(),
    "nvidia": NvidiaAdapter(),
    "openrouter": OpenRouterAdapter(),
    "gemini": GeminiAdapter(),
    "coarena": CoarenaAdapter(),
    "cloudflare": CloudflareAdapter(),
    "cohere": CohereAdapter(),
    "sambanova": SambaNovaAdapter(),
}
