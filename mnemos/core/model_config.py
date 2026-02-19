"""
混合专家架构模型配置

- 决策层（Philosopher, Psychologist, Linguist）：Claude 4.5，侧重共情与推理。
- 数据层（Registrar）：默认 gpt-4o（OpenRouter 可用），侧重结构化抽取。
可通过环境变量覆盖：MNEMOS_MODEL_LINGUIST, MNEMOS_MODEL_PHILOSOPHER, MNEMOS_MODEL_PSYCHOLOGIST, MNEMOS_MODEL_REGISTRAR。
"""

from __future__ import annotations

import os
from typing import Dict

# OpenRouter 模型 ID
# 可通过环境变量 MNEMOS_MODEL_LINGUIST 等覆盖
DEFAULT_MODEL_DECISION = "anthropic/claude-opus-4.5"  # Claude Opus 4.5（200K ctx）
DEFAULT_MODEL_DATA = "openai/gpt-4o"  # 数据层抽取


def get_model_config() -> Dict[str, str]:
    """返回各 Worker 使用的模型 ID；决策层用 Claude Sonnet 4.5，数据层用 GPT-4o。"""
    return {
        "linguist": os.environ.get("MNEMOS_MODEL_LINGUIST", DEFAULT_MODEL_DECISION),
        "philosopher": os.environ.get("MNEMOS_MODEL_PHILOSOPHER", DEFAULT_MODEL_DECISION),
        "psychologist": os.environ.get("MNEMOS_MODEL_PSYCHOLOGIST", DEFAULT_MODEL_DECISION),
        "registrar": os.environ.get("MNEMOS_MODEL_REGISTRAR", DEFAULT_MODEL_DATA),
    }
