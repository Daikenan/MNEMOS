"""
反思层 (Reflector)：Philosopher + Psychologist

- Philosopher：从近期/历史事实做场景一致性检查，生成行为偏离类 Insight（见 philosopher.py）。
- Psychologist：从长期事实推断成员的「核心价值观」与「行为动机」，对应 KnowMe-Bench 的心理洞察维度。
  参考：docs/knowledge_base/2_generative_agents.html（反思与动机）
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

import httpx
from loguru import logger

from mnemos.workers.philosopher import InsightPhilosopher

# ---------------------------------------------------------------------------
# Psychologist 专用 Prompt：从长期事实推断核心价值观与行为动机
# ---------------------------------------------------------------------------

PSYCHOLOGIST_SYSTEM_PROMPT = """你是一位家庭记忆系统中的心理洞察专家（Psychologist），擅长**共情捕捉**与**潜台词分析**。你的任务是根据成员的**长期事实**（多轮对话中积累的目标、行为、选择、偏好）以及**当前用户消息**，推断该成员的「核心价值观」「行为动机」与「潜台词/未言明需求」，供系统做个性化回应与心理洞察评估。

## 共情捕捉与潜台词（必须执行）

- **潜台词 (subtext)**：结合「长期事实」与「当前用户消息」，推断用户没有直接说出口的需求、恐惧或期待。例如：表面问「我该不该去旅行」可能隐含「我害怕被说逃避」「我需要被允许休息」「我在寻求认同」等。
- **情绪底色**：从事实与措辞中捕捉持续存在的情绪基调（如焦虑、内疚、渴望被认可、倦怠），并在动机描述中体现。
- 在 behavioral_motivations 中，可增加与「当前消息」直接相关的动机条目，标注其与潜台词的关联。

## 隐喻词库增强（意象 → 心理学隐喻，必须执行）

若文中出现以下**典型意象**，必须结合上下文给出该意象在心理学/文学意义上的隐喻解读，并纳入动机或潜台词分析：
- **自然/物体**：蜘蛛、冰、墙、门、窗、镜子、绳索、笼子、阴影、光、火焰、水、坠落、溶解、边界、裂缝
- **身体/空间**：窒息、窒息感、边界消融、身体解体、被吞噬、被困住
若文本中出现上述意象（或类似隐喻），在 behavioral_motivations 或 supporting_summary 中明确写出：该意象在当下语境中象征何种心理状态或防御机制（如：蜘蛛→控制/恐惧；冰→隔离/冷漠；墙→边界/防御）。不要只做字面复述，要给出**隐喻层面的心理含义**。

## 任务说明

1. **核心价值观 (core_values)**  
   从事实中归纳成员稳定持有的价值取向，例如：健康优先、家庭团聚、事业成就、被认可、学习成长、经济稳健等。每条用简短短语，并给出 0–1 的置信度。

2. **行为动机 (behavioral_motivations)**  
   从事实与当前消息中推断行为背后的动机；**特别关注与当前提问相关的潜台词**（例如：问「是休息还是逃避」时，动机中应体现「对评判的担忧」「对自我正当性的寻求」等）。每条包含「动机描述」与「支撑事实摘要」，以及 0–1 的置信度。

## 输出格式

只输出一个 JSON 对象，不要其他解释。格式如下：

{
  "core_values": [
    {"value": "简短描述", "confidence": 0.0–1.0}
  ],
  "behavioral_motivations": [
    {
      "motivation": "动机描述",
      "supporting_summary": "支撑事实的一句话摘要",
      "confidence": 0.0–1.0
    }
  ]
}

- 若从给定事实中无法可靠推断，对应数组可为 []。
- confidence 表示你对该条推断的确信程度，避免虚构。"""

PSYCHOLOGIST_USER_TEMPLATE = """成员ID：{member_id}

请基于以下「长期事实」与**当前用户消息**，推断该成员的**核心价值观**与**行为动机**；并特别结合当前消息做**潜台词分析**（用户未直接说出的需求、担忧或期待）。只输出 JSON 对象，不要其他内容。

长期事实（Facts）：
---
{long_term_facts_text}
---

当前用户消息（必读，用于共情捕捉与潜台词推断）：{message}
"""


def _facts_to_text(facts: List[dict[str, Any]]) -> str:
    """将事实列表转为供 Prompt 使用的可读文本。"""
    if not facts:
        return "（暂无长期事实）"
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
    return "\n".join(lines) if lines else "（暂无长期事实）"


def _try_repair_truncated_json(raw: str):
    """
    尝试修复被 max_tokens 截断的 JSON。
    策略：找到最后一个完整的 '}' 并自动补全所有未关闭的 '[' 和 '{'。
    对于 Psychologist 输出（嵌套对象含数组），能恢复截断数组中已完成的元素。
    """
    # 通用策略：找最后一个 '}'，然后补全所有未关闭的括号
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


def _validate_and_parse_psychologist_output(raw: str) -> dict[str, Any]:
    """
    解析 Psychologist 模型输出的 JSON，提取 core_values 与 behavioral_motivations。
    当 JSON 被 max_tokens 截断时，尝试修复并恢复已完成的条目。
    """
    default = {"core_values": [], "behavioral_motivations": []}
    raw = raw.strip()
    if "```json" in raw:
        raw = re.sub(r"^.*?```json\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"\s*```.*$", "", raw, flags=re.DOTALL)
    elif "```" in raw:
        raw = re.sub(r"^.*?```\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"\s*```.*$", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    if not raw:
        return default
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        data = _try_repair_truncated_json(raw)
        if data is not None:
            logger.info("Psychologist JSON 截断已修复，恢复部分数据")
        else:
            logger.warning("Psychologist JSON 解析失败: {} 原始片段: {}", e, raw[:200])
            return default
    if not isinstance(data, dict):
        return default
    core_values = data.get("core_values")
    behavioral_motivations = data.get("behavioral_motivations")
    if not isinstance(core_values, list):
        core_values = []
    if not isinstance(behavioral_motivations, list):
        behavioral_motivations = []
    # 规范化每条结构
    out_cv = []
    for item in core_values:
        if isinstance(item, dict) and item.get("value"):
            out_cv.append({
                "value": str(item["value"]).strip(),
                "confidence": float(item.get("confidence", 0.5))
                if isinstance(item.get("confidence"), (int, float)) else 0.5,
            })
        elif isinstance(item, str) and item.strip():
            out_cv.append({"value": item.strip(), "confidence": 0.5})
    out_bm = []
    for item in behavioral_motivations:
        if isinstance(item, dict) and item.get("motivation"):
            out_bm.append({
                "motivation": str(item["motivation"]).strip(),
                "supporting_summary": str(item.get("supporting_summary", "")).strip(),
                "confidence": float(item.get("confidence", 0.5))
                if isinstance(item.get("confidence"), (int, float)) else 0.5,
            })
    return {"core_values": out_cv, "behavioral_motivations": out_bm}


class Psychologist:
    """
    心理洞察专家：从长期事实推断成员的「核心价值观」与「行为动机」。
    对应 KnowMe-Bench 的 Psychological Insight 维度（Generalize to new scenarios, Suggest new ideas）。
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
                "未配置 LLM API Key：设置 OPENROUTER_API_KEY 或传入 Psychologist(api_key=...)"
            )
        return key

    async def infer_values_and_motivations(
        self,
        member_id: str,
        long_term_facts: List[dict[str, Any]],
        message: str = "",
        model_override: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        从长期事实推断该成员的「核心价值观」与「行为动机」。

        Args:
            member_id: 成员 ID（家庭场景隔离）
            long_term_facts: 长期事实列表（来自 MemOS 检索或跨轮聚合）
            message: 当前用户消息（可选，供上下文参考）

        Returns:
            含 core_values 与 behavioral_motivations 的字典；每项含 confidence。
        """
        long_term_facts = long_term_facts or []
        facts_text = _facts_to_text(long_term_facts)

        user_content = PSYCHOLOGIST_USER_TEMPLATE.format(
            member_id=member_id,
            long_term_facts_text=facts_text,
            message=message or "",
        )
        model = (model_override or self.model).strip()
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": PSYCHOLOGIST_SYSTEM_PROMPT},
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
                "Psychologist LLM 请求失败 status={} body={}",
                e.response.status_code,
                e.response.text[:500],
            )
            if e.response.status_code in (429, 500, 502, 503):
                raise
            return {"core_values": [], "behavioral_motivations": []}
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
            logger.error("Psychologist 请求超时: {}", e)
            raise
        except Exception as e:
            logger.exception("Psychologist 请求异常: {}", e)
            return {"core_values": [], "behavioral_motivations": []}

        choices = data.get("choices") or []
        raw_text = ""
        if choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                raw_text = (msg.get("content") or "").strip()
        if not raw_text:
            logger.debug("Psychologist 未返回内容")
            return {"core_values": [], "behavioral_motivations": []}

        result = _validate_and_parse_psychologist_output(raw_text)
        if result["core_values"] or result["behavioral_motivations"]:
            logger.info(
                "Psychologist member_id={} core_values={} motivations={}",
                member_id,
                len(result["core_values"]),
                len(result["behavioral_motivations"]),
            )
        return result


# 反思层统一入口：保留对现有 Philosopher 的引用，并暴露 Psychologist
__all__ = [
    "InsightPhilosopher",
    "Psychologist",
]
