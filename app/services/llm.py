import logging
from typing import List, Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

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


async def _gemini(user_msg: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(user_msg)
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
