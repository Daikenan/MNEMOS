"""
反思哲学家 (Philosopher)

基于 Generative Agents 反思逻辑：从 Registrar 提取的事实生成高阶 Insight。
内含场景一致性检查：近期行为与长期目标不一致时，Insight 标记为「潜在的行为偏离」。
参考：docs/knowledge_base/2_generative_agents.html
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

import httpx
from loguru import logger

from mnemos.workers.philosopher_prompts import (
    PHILOSOPHER_SYSTEM_PROMPT,
    PHILOSOPHER_USER_TEMPLATE,
)

# 行为偏离标记（与 Prompt 中约定一致）
TAG_BEHAVIOR_DEVIATION = "潜在的行为偏离"


def _facts_to_text(facts: List[dict[str, Any]]) -> str:
    """将事实列表转为供 Prompt 使用的可读文本。"""
    if not facts:
        return "（暂无近期事实）"
    lines = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        entity = f.get("entity", "")
        attribute = f.get("attribute", "")
        value = f.get("value", "")
        tags = f.get("context_tags") or []
        if entity or attribute or value:
            line = f"{entity} {attribute} {value}"
            if tags:
                line += " [" + ", ".join(tags) + "]"
            lines.append(line)
    return "\n".join(lines) if lines else "（暂无近期事实）"


def _validate_and_parse_insights(raw: str) -> List[dict[str, Any]]:
    """解析模型输出的 JSON 数组，提取 insight / tag / related_goals。"""
    out: List[dict[str, Any]] = []
    raw = raw.strip()
    if "```json" in raw:
        raw = re.sub(r"^.*?```json\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"\s*```.*$", "", raw, flags=re.DOTALL)
    elif "```" in raw:
        raw = re.sub(r"^.*?```\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"\s*```.*$", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Philosopher JSON 解析失败: {} 原始片段: {}", e, raw[:200])
        return out
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            insight = item.get("insight") or item.get("text") or ""
            if not insight:
                continue
            tag = item.get("tag")
            if tag is not None and not isinstance(tag, str):
                tag = str(tag) if tag else None
            related_goals = item.get("related_goals")
            if not isinstance(related_goals, list):
                related_goals = []
            related_goals = [str(g) for g in related_goals if g]
            out.append({
                "insight": insight.strip(),
                "tag": tag.strip() if isinstance(tag, str) and tag.strip() else tag,
                "related_goals": related_goals,
            })
    elif isinstance(data, dict) and (data.get("insight") or data.get("text")):
        insight = (data.get("insight") or data.get("text") or "").strip()
        tag = data.get("tag")
        related_goals = data.get("related_goals") or []
        if isinstance(related_goals, list):
            related_goals = [str(g) for g in related_goals if g]
        else:
            related_goals = []
        out.append({"insight": insight, "tag": tag, "related_goals": related_goals})
    return out


class InsightPhilosopher:
    """
    反思哲学家：根据近期事实生成高阶 Insight，并执行场景一致性检查。
    当近期行为与推断的长期目标不一致时，Insight 的 tag 为「潜在的行为偏离」。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 2048,
        timeout: float = 60.0,
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
        import os
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "未配置 LLM API Key：设置 OPENROUTER_API_KEY 或传入 InsightPhilosopher(api_key=...)"
            )
        return key

    async def generate_insights(
        self,
        message: str,
        member_id: str,
        facts: Optional[List[dict[str, Any]]] = None,
        historical_facts: Optional[List[dict[str, Any]]] = None,
        model_override: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        基于近期事实生成洞察，并执行场景一致性检查。
        若近期行为与长期目标不一致，对应 Insight 的 tag 为「潜在的行为偏离」。
        支持注入 historical_facts（来自 MemOS 或本地缓存的目标/计划），解决跨轮「看不到历史目标」问题。

        Args:
            message: 当前用户消息（供上下文参考）
            member_id: 成员 ID
            facts: 近期事实列表（Registrar 提取的结构）
            historical_facts: 历史目标/计划类事实（认知上下文注入）

        Returns:
            洞察列表，每项含 insight, tag（可能为 "潜在的行为偏离"）, related_goals
        """
        facts = facts or []
        historical_facts = historical_facts or []
        facts_text = _facts_to_text(facts)
        if historical_facts:
            historical_section = (
                "已知长期目标/计划（来自历史记忆）：\n---\n"
                + _facts_to_text(historical_facts)
                + "\n---\n\n"
            )
        else:
            historical_section = ""

        user_content = PHILOSOPHER_USER_TEMPLATE.format(
            member_id=member_id,
            historical_section=historical_section,
            facts_text=facts_text,
            message=message or "",
        )
        model = (model_override or self.model).strip()
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": PHILOSOPHER_SYSTEM_PROMPT},
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
        except httpx.HTTPStatusError as e:
            logger.error(
                "Philosopher LLM 请求失败 status={} body={}",
                e.response.status_code,
                e.response.text[:500],
            )
            return []
        except Exception as e:
            logger.exception("Philosopher 请求异常: {}", e)
            return []

        choices = data.get("choices") or []
        raw_text = ""
        if choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                raw_text = (msg.get("content") or "").strip()
        if not raw_text:
            logger.debug("Philosopher 未返回内容")
            return []

        insights = _validate_and_parse_insights(raw_text)
        deviation_count = sum(
            1 for i in insights if i.get("tag") == TAG_BEHAVIOR_DEVIATION
        )
        if deviation_count:
            logger.info(
                "Philosopher 场景一致性检查 member_id={} 潜在行为偏离数={}",
                member_id,
                deviation_count,
            )
        return insights
