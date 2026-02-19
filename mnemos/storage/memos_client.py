"""
MemOS 云端客户端 (LTM)

将 Registrar 提取的事实持久化到 MemOS Cloud。
- context_tags 映射为 MemOS 标签系统
- entity / attribute 映射为元数据 (metadata)
- 使用 httpx 异步调用，从 .env 加载 MEMOS_API_KEY、MEMOS_BASE_URL
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx
from loguru import logger

if TYPE_CHECKING:
    from mnemos.workers.registrar import ExtractedFact

try:
    from mnemos.env_loader import load_env
    load_env()
except Exception:
    pass


def _fact_to_payload(
    fact: Union[ExtractedFact, Dict[str, Any]],
    *,
    member_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将 ExtractedFact（或兼容 dict）转为 MemOS add/message 请求体。
    - context_tags -> 标签（若 API 支持则放在 body，否则编码进 content）
    - entity, attribute, confidence -> metadata
    """
    if isinstance(fact, dict):
        entity = fact.get("entity", "")
        attribute = fact.get("attribute", "")
        value = fact.get("value", "")
        context_tags = fact.get("context_tags") or []
        confidence = fact.get("confidence", fact.get("confidence_score", 0.5))
        uid = fact.get("member_id") or member_id
    else:
        entity = fact.entity
        attribute = fact.attribute
        value = fact.value
        context_tags = getattr(fact, "context_tags", []) or []
        confidence = getattr(fact, "confidence", 0.5)
        uid = member_id

    # 可读的事实摘要，作为 message content 供 MemOS 抽象存储
    content_parts = [f"{entity} {attribute} {value}"]
    if context_tags:
        content_parts.append("Tags: " + ", ".join(context_tags))
    content = ". ".join(content_parts)

    # MemOS /add/message: user_id, conversation_id, messages[]
    payload: Dict[str, Any] = {
        "user_id": uid or "default_member",
        "conversation_id": conversation_id or "mnemos_facts",
        "messages": [
            {"role": "user", "content": content},
        ],
        # 元数据：entity、attribute、confidence，便于检索与过滤
        "metadata": {
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "confidence": float(confidence),
            "source": "mnemos_registrar",
        },
        # 标签系统：context_tags 映射为 MemOS 标签
        "tags": [t.strip() for t in context_tags if isinstance(t, str) and t.strip()],
    }
    return payload


class MemOSClient:
    """
    MemOS Cloud 异步客户端。

    - add_memory(fact): 将单条事实发送到 MemOS（调用 add/message 接口）
    - 从环境变量 MEMOS_API_KEY、MEMOS_BASE_URL 读取配置
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self._api_key = api_key or os.environ.get("MEMOS_API_KEY", "").strip()
        self._base_url = (base_url or os.environ.get("MEMOS_BASE_URL", "")).rstrip("/")
        self._timeout = timeout
        if not self._api_key:
            logger.warning("MEMOS_API_KEY 未配置，MemOS 持久化将不可用")
        if not self._base_url:
            logger.warning("MEMOS_BASE_URL 未配置，MemOS 持久化将不可用")

    def is_configured(self) -> bool:
        return bool(self._api_key and self._base_url)

    async def add_memory(
        self,
        fact: Union[ExtractedFact, Dict[str, Any]],
        *,
        member_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """
        将 Registrar 提取的一条事实发送到 MemOS。

        - context_tags 映射为 MemOS 的 tags
        - entity、attribute、value、confidence 放入 metadata

        Args:
            fact: ExtractedFact 实例或包含 entity/attribute/value/context_tags/confidence 的 dict
            member_id: 成员 ID（若 fact 为 dict 且含 member_id 则优先用 fact 内的）
            conversation_id: 会话 ID，用于 MemOS 会话聚合

        Returns:
            是否提交成功（HTTP 2xx 视为成功）
        """
        if not self.is_configured():
            logger.debug("MemOS 未配置，跳过 add_memory")
            return False

        payload = _fact_to_payload(
            fact,
            member_id=member_id,
            conversation_id=conversation_id,
        )
        url = f"{self._base_url}/add/message"
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.is_success:
                    logger.debug("MemOS add_memory 成功 entity={}", payload.get("metadata", {}).get("entity"))
                    return True
                logger.warning(
                    "MemOS add_memory 失败 status={} body={}",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        except Exception as e:
            logger.exception("MemOS add_memory 异常: {}", e)
            return False

    async def add_memories(
        self,
        facts: List[Union[ExtractedFact, Dict[str, Any]]],
        *,
        member_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> int:
        """
        批量将事实写入 MemOS，并发请求，不阻塞。

        Returns:
            成功条数
        """
        if not facts:
            return 0
        import asyncio
        tasks = [
            self.add_memory(f, member_id=member_id, conversation_id=conversation_id)
            for f in facts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for r in results if r is True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("MemOS add_memory 单条异常 fact_index={} err={}", i, r)
        return success

    async def search_memories(
        self,
        member_id: str,
        *,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        检索该成员的历史记忆，用于认知上下文注入（如 Philosopher 所需的目标/计划）。
        语义检索：按 query 或 tags 过滤，返回 fact 形态的列表供上游使用。

        Args:
            member_id: 成员 ID（对应 add_memory 时的 user_id）
            query: 检索查询，如 "目标 计划 想要 打算"
            tags: 可选标签过滤（MemOS 若支持），如 ["#健康"] 或 目标/计划 相关
            limit: 最多返回条数

        Returns:
            与 Registrar 事实结构兼容的列表，每项含 entity, attribute, value, context_tags 等，
            若 API 不支持或失败则返回 []。
        """
        if not self.is_configured():
            return []
        url = f"{self._base_url}/search"
        payload: Dict[str, Any] = {
            "user_id": member_id,
            "limit": limit,
        }
        if query:
            payload["query"] = query
        if tags:
            payload["tags"] = tags
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if not resp.is_success:
                    # 部分 MemOS 部署可能未开放 /search，视为无历史
                    if resp.status_code == 404 or resp.status_code == 501:
                        logger.debug("MemOS search 端点不可用，跳过历史检索")
                    else:
                        logger.warning("MemOS search 失败 status={} body={}", resp.status_code, resp.text[:300])
                    return []
                data = resp.json()
        except Exception as e:
            logger.debug("MemOS search 异常（可能未实现）: {}", e)
            return []
        return self._parse_search_response_to_facts(data, limit=limit)

    def _parse_search_response_to_facts(self, data: Any, limit: int = 10) -> List[Dict[str, Any]]:
        """
        将 MemOS 搜索返回的结构解析为与 Registrar 事实兼容的 dict 列表。
        兼容多种常见形态：{ "memories": [ { "content"|"message"|"text": "...", "metadata": {...} } ] } 或
        [ { "entity", "attribute", "value" } ]。
        """
        out: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = data[:limit]
        elif isinstance(data, dict):
            items = data.get("memories") or data.get("results") or data.get("data") or []
            if isinstance(items, list):
                items = items[:limit]
            else:
                items = []
        else:
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            # 已是 fact 形态
            if item.get("entity") is not None and (item.get("attribute") is not None or item.get("value") is not None):
                out.append({
                    "entity": item.get("entity", ""),
                    "attribute": item.get("attribute", ""),
                    "value": item.get("value", ""),
                    "context_tags": item.get("context_tags", []),
                })
                continue
            # 从 content + metadata 还原
            content = item.get("content") or item.get("message") or item.get("text") or ""
            meta = item.get("metadata") or {}
            entity = meta.get("entity") or ""
            attribute = meta.get("attribute") or ""
            value = meta.get("value") or ""
            if not content and not (entity or attribute or value):
                continue
            if not entity and not attribute and not value and content:
                # 纯文本记忆：整段作为 value，entity 用 member 占位
                out.append({
                    "entity": "成员",
                    "attribute": "历史记忆",
                    "value": content[:200].strip(),
                    "context_tags": item.get("tags", []),
                })
            else:
                out.append({
                    "entity": entity or "成员",
                    "attribute": attribute or "相关",
                    "value": value or content[:150].strip(),
                    "context_tags": item.get("tags", meta.get("context_tags", [])),
                })
        return out
