import os
import openai
try:
    import anthropic
except ImportError:
    anthropic = None

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# OpenAI LLM call
async def call_openai(prompt, model="gpt-3.5-turbo", max_tokens=256):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    openai.api_key = OPENAI_API_KEY
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"

# Anthropic/Claude LLM call
async def call_anthropic(prompt, model="claude-3-opus-20240229", max_tokens=256):
    if not ANTHROPIC_API_KEY or not anthropic:
        return "[Anthropic API not available]"
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"[Anthropic error: {e}]" 