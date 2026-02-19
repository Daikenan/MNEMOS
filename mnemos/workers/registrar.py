"""
事实记录员 (Fact Registrar)

基于 Mem0 事实层：从对话中异步提取结构化事实，带场景化标签（context_tags）。
参考：docs/knowledge_base/1_mem0_fact_layer.html
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from mnemos.workers.registrar_prompts import REGISTRAR_SYSTEM_PROMPT, REGISTRAR_USER_TEMPLATE


# ---------------------------------------------------------------------------
# 结构化输出模型（与 Mem0 事实层 + 场景化建模一致）
# ---------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """单条提取事实，含 context_tags 与 confidence。"""

    entity: str = Field(..., description="主体/实体")
    attribute: str = Field(..., description="属性/关系")
    value: str = Field(..., description="属性值")
    context_tags: List[str] = Field(default_factory=list, description="场景标签，如 #健康 #家庭旅行 #成长")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0~1")


def _try_repair_truncated_json(raw: str) -> Any:
    """
    尝试修复被 max_tokens 截断的 JSON。
    策略：找到最后一个完整的 '}' 并自动补全所有未关闭的 '[' 和 '{'。
    """
    last_brace = raw.rfind("}")
    if last_brace <= 0:
        return None

    candidate = raw[:last_brace + 1].rstrip().rstrip(",")
    open_brackets = candidate.count("[") - candidate.count("]")
    open_braces = candidate.count("{") - candidate.count("}")
    candidate += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 回退：尝试找最后一个 ']' 并补全
    last_bracket = raw.rfind("]")
    if last_bracket > 0:
        candidate = raw[:last_bracket + 1]
        open_braces = candidate.count("{") - candidate.count("}")
        candidate += "}" * max(0, open_braces)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_parse_facts(raw: str) -> List[dict[str, Any]]:
    """
    对模型输出做基本 JSON 校验与解析。
    支持整段 JSON 数组或 ```json ... ``` 代码块。
    当 JSON 被 max_tokens 截断时，尝试修复并恢复已完成的条目。
    """
    out: List[dict[str, Any]] = []
    raw = raw.strip()
    # 尝试剥离 markdown 代码块
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
        # 尝试修复截断的 JSON
        data = _try_repair_truncated_json(raw)
        if data is not None:
            logger.info("Registrar JSON 截断已修复，恢复部分数据")
        else:
            logger.warning("Registrar JSON 解析失败: {} 原始片段: {}", e, raw[:200])
            return out
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out.append(item)
    elif isinstance(data, dict):
        out.append(data)
    return out


def _normalize_facts(raw_list: List[dict[str, Any]], member_id: str) -> List[dict[str, Any]]:
    """
    将解析后的字典列表规范为约定格式，并做基本校验（含 Pydantic 校验）；
    为兼容 Coordinator 的 confidence_score，同时保留 confidence 与 confidence_score。
    """
    result: List[dict[str, Any]] = []
    for d in raw_list:
        if not isinstance(d, dict):
            continue
        entity = d.get("entity") or d.get("subject", "")
        attribute = d.get("attribute") or d.get("predicate", "")
        value = d.get("value") or d.get("object", "")
        if not (entity and attribute and value):
            continue
        tags = d.get("context_tags")
        if not isinstance(tags, list):
            tags = []
        tags = [t for t in tags if isinstance(t, str) and t.strip()]
        try:
            conf = float(d.get("confidence", d.get("confidence_score", 0.5)))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        row = {
            "entity": str(entity).strip(),
            "attribute": str(attribute).strip(),
            "value": str(value).strip(),
            "context_tags": tags,
            "confidence": conf,
            "confidence_score": conf,
            "member_id": member_id,
        }
        # 使用 Pydantic 做一次结构化校验
        try:
            ExtractedFact(
                entity=row["entity"],
                attribute=row["attribute"],
                value=row["value"],
                context_tags=row["context_tags"],
                confidence=row["confidence"],
            )
        except Exception:
            continue
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# FactRegistrar：异步调用 LLM（OpenRouter / OpenAI 兼容），提取事实并校验
# ---------------------------------------------------------------------------


class FactRegistrar:
    """
    事实记录员：从文本中异步提取结构化事实（Mem0 事实层），
    并基于 System Prompt 中的场景分类逻辑打上 context_tags（场景化建模）。
    默认使用 OpenRouter（OPENROUTER_API_KEY），兼容任意 OpenAI-style 端点。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 1024,
        timeout: float = 120.0,
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
                "未配置 LLM API Key：设置 OPENROUTER_API_KEY 或传入 FactRegistrar(api_key=...)"
            )
        return key

    async def extract_facts(
        self,
        text: str,
        member_id: str,
        model_override: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        从文本中异步提取事实，带 context_tags 与 confidence，并对结果做基本 JSON 校验。

        Args:
            text: 待分析的对话或文本
            member_id: 成员 ID，用于隔离与写入元数据

        Returns:
            事实列表，每项含 entity, attribute, value, context_tags, confidence（及 confidence_score、member_id）
        """
        if not (text and text.strip()):
            return []

        user_content = REGISTRAR_USER_TEMPLATE.format(
            member_id=member_id,
            text=text.strip(),
        )
        # OpenAI / OpenRouter 格式：system 放在 messages 首条
        model = (model_override or self.model).strip()
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": REGISTRAR_SYSTEM_PROMPT},
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
            logger.error("Registrar LLM 请求失败 status={} body={}", e.response.status_code, e.response.text[:500])
            if e.response.status_code in (429, 500, 502, 503):
                raise
            return []
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
            logger.error("Registrar 请求超时: {}", e)
            raise
        except Exception as e:
            logger.exception("Registrar 请求异常: {}", e)
            return []

        # OpenAI-style: choices[0].message.content
        raw_text = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                raw_text = (msg.get("content") or "").strip()
        if not raw_text:
            logger.warning("Registrar LLM 未返回文本内容")
            return []

        parsed = _validate_and_parse_facts(raw_text)
        normalized = _normalize_facts(parsed, member_id)
        logger.debug("Registrar 提取事实数 member_id={} count={}", member_id, len(normalized))
        return normalized
