import logging
from typing import List, Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

_LLM_INPUT_PER_1M  = 0.15   # USD per 1M input tokens  (gemini-2.5-flash, non-thinking)
_LLM_OUTPUT_PER_1M = 0.60   # USD per 1M output tokens

SYSTEM_PROMPT = """You are an expert real estate document analyst.
You are given context chunks extracted from property documents (deeds, agreements, listings, survey reports).
The documents may be in English, Hindi, Bengali, or other Indian languages.

Answer the user's question using ONLY the provided context.
- Be precise about property dimensions, prices, boundaries, and legal terms.
- If the answer mentions specific page numbers or document names, include them.
- If you cannot find the answer in the context, say: "I could not find this information in the uploaded documents."
- Never guess or hallucinate property details.
- Respond in the same language the user asked in.
"""


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['pdf_name']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    query: str,
    chunks: List[Dict],
    provider: Optional[str] = None,
) -> tuple[str, str]:
    """
    Returns (answer_text, provider_used).
    provider can be 'gemini' | 'openai' | 'anthropic' — falls back to settings.
    """
    provider = provider or settings.llm_provider
    context = _build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    if provider == "gemini":
        return await _gemini(user_msg), "gemini"
    elif provider == "openai":
        return await _openai(user_msg), "openai"
    elif provider == "anthropic":
        return await _anthropic(user_msg), "anthropic"
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _get_gemini_client():
    import os
    from google import genai

    if settings.google_application_credentials:
        os.environ.setdefault(
            "GOOGLE_APPLICATION_CREDENTIALS", settings.google_application_credentials
        )
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


async def _gemini(user_msg: str) -> str:
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    usage = response.usage_metadata
    in_tok  = usage.prompt_token_count or 0
    out_tok = usage.candidates_token_count or 0
    cost = (in_tok / 1_000_000) * _LLM_INPUT_PER_1M + (out_tok / 1_000_000) * _LLM_OUTPUT_PER_1M
    logger.info(
        "LLM cost [%s] — input: %d tokens | output: %d tokens | ~$%.6f",
        settings.gemini_model, in_tok, out_tok, cost,
    )

    return response.text


async def _openai(user_msg: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


async def _anthropic(user_msg: str) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text
