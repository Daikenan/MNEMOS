"""
Self-Consistency 验证模块 (Verifier)

在 Linguist 输出答案后，反向验证答案是否在检索到的 Context 中有直接支撑。
若无支撑，强制返回"信息不足"类回复，防止幻觉。

主要用于 Adversarial Abstention 任务：
- 遇到记忆中不存在的信息时，果断拒绝而非编造。
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from loguru import logger


VERIFICATION_SYSTEM_PROMPT = """You are a strict fact-verification judge. Your ONLY job is to determine whether a proposed answer is DIRECTLY supported by the provided source text.

## Rules (MUST follow):
1. Check if EVERY factual claim in the Answer can be traced back to a specific sentence or phrase in the Source Text.
2. If the Answer contains ANY detail, name, event, or claim that does NOT appear in the Source Text, mark it as UNSUPPORTED.
3. If the Answer correctly states that the information is not available or insufficient, mark it as SUPPORTED.
4. Do NOT consider inferences, educated guesses, or "reasonable assumptions" as supported — only explicit textual evidence counts.

## Output:
Respond with EXACTLY one word:
- SUPPORTED — if all claims in the Answer are directly backed by the Source Text
- UNSUPPORTED — if any claim lacks direct textual evidence"""


VERIFICATION_USER_TEMPLATE = """## Source Text:
{source_text}

## Question:
{question}

## Proposed Answer:
{answer}

Is every factual claim in the Proposed Answer directly supported by the Source Text? Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


# 不同语言的"信息不足"回复模板
_ABSTENTION_RESPONSES = {
    "en": "Based on the available text, there is insufficient information to answer this question. The text does not mention or provide enough details about the specific aspect being asked.",
    "zh": "根据现有文本，没有足够的信息来回答这个问题。文本中未提及或未提供足够的相关细节。",
}


class SelfConsistencyVerifier:
    """
    Self-Consistency 验证器：在答案生成后检查其是否有原文支撑。
    若验证失败（UNSUPPORTED），替换为安全的弃权回复。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 16,
        timeout: float = 20.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        try:
            from mnemos.env_loader import load_env
            load_env()
        except Exception:
            pass
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("未配置 API Key")
        return key

    async def verify_and_maybe_abstain(
        self,
        question: str,
        answer: str,
        source_text: str,
        language: str = "en",
        model_override: Optional[str] = None,
    ) -> str:
        """
        验证 answer 是否有 source_text 直接支撑。
        若无支撑 (UNSUPPORTED)，返回安全的弃权回复；否则原样返回 answer。

        Args:
            question: 原始问题
            answer: 模型生成的答案
            source_text: 检索到的原始原文
            language: 回复语言 ('en' 或 'zh')
            model_override: 覆盖验证模型

        Returns:
            验证后的答案（可能被替换为弃权回复）
        """
        if not source_text.strip() or not answer.strip():
            return _ABSTENTION_RESPONSES.get(language, _ABSTENTION_RESPONSES["en"])

        # 如果答案本身已经是"不知道"类回复，直接放行
        if _is_already_abstaining(answer):
            return answer

        user_content = VERIFICATION_USER_TEMPLATE.format(
            source_text=source_text[:30000],
            question=question,
            answer=answer,
        )
        model = (model_override or self.model).strip()
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("Verifier 调用失败，放行原始答案: {}", e)
            return answer

        choices = data.get("choices") or []
        if not choices:
            return answer
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            return answer
        verdict = (msg.get("content") or "").strip().upper()

        if "UNSUPPORTED" in verdict:
            logger.info("Verifier 判定 UNSUPPORTED → 弃权")
            return _ABSTENTION_RESPONSES.get(language, _ABSTENTION_RESPONSES["en"])

        return answer


def _is_already_abstaining(answer: str) -> bool:
    """检测答案是否已经是弃权/拒答类回复。"""
    lower = answer.lower().strip()
    abstention_markers = [
        "i don't know",
        "i do not know",
        "insufficient information",
        "not mentioned",
        "does not mention",
        "the text does not",
        "no information",
        "not enough information",
        "cannot determine",
        "cannot answer",
        "信息不足",
        "没有足够",
        "无法确定",
        "未提及",
        "不包含",
        "没有提到",
        "无法回答",
    ]
    return any(marker in lower for marker in abstention_markers)
