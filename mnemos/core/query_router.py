"""
查询路由器 (Query Router)

根据任务类型或查询内容，将请求分发到不同的处理流水线：
- 事实提取 (fact_extraction)：跳过语义抽象，直供原始原文
- 时序排列 (temporal_ordering)：按时间线排序证据后回答
- 对抗弃权 (adversarial_abstention)：启用 Self-Consistency 验证
- 时间推理 (temporal_reasoning)：专注时间计算与区间推理
- 记忆触发 (mnestic_trigger)：专注记忆唤醒与关联分析
- 深度分析 (deep_analysis)：场景化记忆模型 + Psychologist 心理洞察（Level III）

路由策略：
1. Benchmark 模式：直接使用 task_type 映射（无需推断）
2. 生产模式：基于关键词/正则轻量判定
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryCategory(str, Enum):
    """查询分类枚举"""
    FACT_EXTRACTION = "fact_extraction"
    TEMPORAL_ORDERING = "temporal_ordering"
    ADVERSARIAL_ABSTENTION = "adversarial_abstention"
    TEMPORAL_REASONING = "temporal_reasoning"
    MNESTIC_TRIGGER = "mnestic_trigger"
    DEEP_ANALYSIS = "deep_analysis"


# Benchmark task_type -> QueryCategory 映射
_BENCHMARK_TASK_MAP = {
    "Information Extraction": QueryCategory.FACT_EXTRACTION,
    "Logical Event Ordering": QueryCategory.TEMPORAL_ORDERING,
    "Adversarial Abstention": QueryCategory.ADVERSARIAL_ABSTENTION,
    "Temporal Reasoning": QueryCategory.TEMPORAL_REASONING,
    "Mnestic Trigger Analysis": QueryCategory.MNESTIC_TRIGGER,
    "Mind-Body Interaction": QueryCategory.DEEP_ANALYSIS,
    "Expert-Annotated Psychoanalysis": QueryCategory.DEEP_ANALYSIS,
}


@dataclass
class RoutingDecision:
    """路由决策结果"""
    category: QueryCategory
    skip_registrar: bool = False
    skip_psychologist: bool = False
    use_raw_source: bool = False
    sort_by_timeline: bool = False
    enable_self_consistency: bool = False
    system_prompt_key: str = "default"
    extra: dict = field(default_factory=dict)


# 生产模式关键词规则（中英文混合）
_FACT_KEYWORDS = re.compile(
    r"(what\s+(is|was|were|did)|who\s+(is|was|did)|"
    r"which|how\s+many|how\s+much|name\s+of|"
    r"是什么|叫什么|谁是|哪个|几个|多少|"
    r"specific(ally)?|exactly|brand|model|color|name)",
    re.IGNORECASE,
)
_TEMPORAL_KEYWORDS = re.compile(
    r"(how\s+long|duration|how\s+many\s+(minutes|hours|days|years)|"
    r"before\s+or\s+after|earlier|later|between|"
    r"多长时间|多久|之前还是之后|先后|时长)",
    re.IGNORECASE,
)
_ORDERING_KEYWORDS = re.compile(
    r"(order|sequence|first.*then|chronolog|"
    r"list.*events|arrange|排列|顺序|先后|时间线|事件排序)",
    re.IGNORECASE,
)
_ABSTENTION_KEYWORDS = re.compile(
    r"(specific\s+brand|exact\s+model|precise\s+number|"
    r"what\s+color\s+was\s+the|specific\s+type)",
    re.IGNORECASE,
)


def route_benchmark_task(task_type: str) -> RoutingDecision:
    """
    Benchmark 模式：根据已知 task_type 直接返回路由决策。
    """
    category = _BENCHMARK_TASK_MAP.get(task_type, QueryCategory.DEEP_ANALYSIS)
    return _build_decision(category)


def route_query(query: str, task_type: Optional[str] = None) -> RoutingDecision:
    """
    通用路由：优先使用 task_type（Benchmark），否则基于关键词推断。
    """
    if task_type:
        category = _BENCHMARK_TASK_MAP.get(task_type)
        if category:
            return _build_decision(category)

    # 关键词推断（生产模式）
    if _ORDERING_KEYWORDS.search(query):
        return _build_decision(QueryCategory.TEMPORAL_ORDERING)
    if _TEMPORAL_KEYWORDS.search(query):
        return _build_decision(QueryCategory.TEMPORAL_REASONING)
    if _FACT_KEYWORDS.search(query):
        return _build_decision(QueryCategory.FACT_EXTRACTION)
    return _build_decision(QueryCategory.DEEP_ANALYSIS)


def _build_decision(category: QueryCategory) -> RoutingDecision:
    """根据分类构建完整路由决策。"""
    if category == QueryCategory.FACT_EXTRACTION:
        return RoutingDecision(
            category=category,
            skip_registrar=True,
            skip_psychologist=True,
            use_raw_source=True,
            system_prompt_key="fact_extraction",
        )
    if category == QueryCategory.TEMPORAL_ORDERING:
        return RoutingDecision(
            category=category,
            skip_registrar=True,
            skip_psychologist=True,
            use_raw_source=True,
            sort_by_timeline=True,
            system_prompt_key="temporal_ordering",
        )
    if category == QueryCategory.ADVERSARIAL_ABSTENTION:
        return RoutingDecision(
            category=category,
            skip_registrar=True,
            skip_psychologist=True,
            use_raw_source=True,
            enable_self_consistency=True,
            system_prompt_key="adversarial_abstention",
        )
    if category == QueryCategory.TEMPORAL_REASONING:
        return RoutingDecision(
            category=category,
            skip_registrar=True,
            skip_psychologist=True,
            use_raw_source=True,
            system_prompt_key="temporal_reasoning",
        )
    if category == QueryCategory.MNESTIC_TRIGGER:
        return RoutingDecision(
            category=category,
            skip_registrar=True,
            skip_psychologist=True,
            use_raw_source=True,
            system_prompt_key="mnestic_trigger",
        )
    # DEEP_ANALYSIS: Level III 保持原有完整流程
    return RoutingDecision(
        category=category,
        skip_registrar=False,
        skip_psychologist=False,
        use_raw_source=False,
        system_prompt_key="deep_analysis",
    )
