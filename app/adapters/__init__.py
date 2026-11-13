from app.adapters.anthropic import AnthropicAdapter
from app.adapters.gemini import GeminiAdapter
from app.adapters.groq import GroqAdapter
from app.adapters.mock import MockAdapter
from app.adapters.nvidia import NvidiaAdapter
from app.adapters.openai import OpenAIAdapter
from app.adapters.openrouter import OpenRouterAdapter

ADAPTERS = {
    "mock": MockAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "groq": GroqAdapter(),
    "nvidia": NvidiaAdapter(),
    "openrouter": OpenRouterAdapter(),
    "gemini": GeminiAdapter(),
}
