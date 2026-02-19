#!/usr/bin/env python3
"""
KnowMe-Bench 打榜评测脚本

1. 从 KnowmeBench/dataset1 加载题目与 input 上下文，合并 reference_answer
2. 对每道题：Registrar(上下文) -> Facts；Psychologist(Facts，Level III 潜台词模式)；Linguist(Claude 4.5) -> model_answer
3. 生成 model_outputs.json，调用官方 run_eval.py 判分
4. 解析 results.json，打印 7 任务得分并与 Level III 冠军线 41.2% 对比

依赖：
- 判分步骤需在能 import openai 的环境中执行（可 `uv add openai tqdm`），并设置 OPENAI_API_KEY。
- Registrar 默认 GPT-4.5；若 OpenRouter 无该模型可设 MNEMOS_MODEL_REGISTRAR=openai/gpt-4o。

全量打榜（无 --max_per_task）：
  uv run python scripts/run_benchmarking.py --use_registrar
完成后会生成 data/model_outputs.json、data/results.json，并在 data/ 下自动保存冠军战报
champion_report_YYYYMMDD_HHMM.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 统一从项目根加载 .env 且 override=True，确保 .env 覆盖 shell 中旧 key
def _load_dotenv() -> None:
    try:
        from mnemos.env_loader import load_env
        load_env()
    except Exception:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env", override=True)
        except Exception:
            pass
_load_dotenv()

# KnowMeBench 路径（与 clone 后的目录一致）
KNOWME_ROOT = PROJECT_ROOT / "KnowMeBench"
KNOWME_DATASET = KNOWME_ROOT / "KnowmeBench" / "dataset1"
KNOWME_EVALUATE = KNOWME_ROOT / "evaluate"

# 7 个任务类型（与 evaluate prompt.md 中 # type 一致）
TASK_TYPES = [
    "Information Extraction",
    "Adversarial Abstention",
    "Temporal Reasoning",
    "Logical Event Ordering",
    "Mnestic Trigger Analysis",
    "Mind-Body Interaction",
    "Expert-Annotated Psychoanalysis",
]

# Level III：必须开启 Psychologist 潜台词分析
LEVEL_III_TASKS = {"Mind-Body Interaction", "Expert-Annotated Psychoanalysis"}

# MemBrain 1.0 Level III 冠军线（README/论文）
LEVEL_III_CHAMPION_PCT = 41.2

# 判分器偏好（与 evaluate prompt.md 一致）：先结论后支撑，击中关键概念
ANSWER_FORMAT_INSTRUCTION = (
    "【回答格式要求】先给出明确结论（或核心隐喻/心理映射），再简要列出支撑事实或推理步骤。"
    "避免冗长与模糊表述，击中关键概念即可。"
)
# Level III 专用：判分器看重 External→Internal 映射、核心隐喻、原文支撑；强制四段式（含反思校验）
LEVEL_III_ANSWER_FORMAT_INSTRUCTION = (
    "【Level III 心理分析题 - 必须采用四段式（含反思校验）】\n"
    "1. **结论**：先写清核心隐喻或心理映射（如「边界消融」「补偿机制」「沉默的耐心」「内在想象力」「自主性」「免于审视的自由」等），点明「外部行为→内心」的对应关系。必须使用**具体且独特**的心理学隐喻，避免泛化。\n"
    "   - 重要：当问题涉及\"力量\"或\"可靠\"时，优先考虑**内在品质**（如沉默的耐心、内在想象力、观察力）而非外在工具。\n"
    "   - 当问题涉及\"底线\"（bottom line）或\"不愿跨越\"时，聚焦于与**核心权威人物（尤其是父亲）的关系边界**。底线通常是关于\"公开挑战或质疑权威\"（publicly challenging authority），而非个人价值观或生活方式选择。\n"
    "   - 当问题涉及\"根本驱动力\"（fundamental driving force）或\"行为选择的根源\"时，必须穿透表面行为（如写作、创作），找到更深层的**心理防御动机**——例如：写作的底层驱动可能不是\"自我表达\"，而是\"逃避因失控而引发的深层恐惧\"（escaping deep fears triggered by loss of control）。\n"
    "2. **心理学原理解释**：用具体心理概念解释该映射。优先使用深层概念如认知锚定、回溯性一致化、沉默的耐心、内在想象力、身体知识、社会校准、自主性、免于审视的自由、失控恐惧、逃避机制、权威挑战、关系边界、补偿机制、反向形成等。避免泛化表述，要写出复杂动机、防御机制与意象。\n"
    "3. **原文事实引用**：用原文中的具体情节/台词/描写做支撑，确保不虚构。**引用时必须带上时间戳**，判分器对带时间戳的证据信任度显著更高。\n"
    "4. **反思校验**：检查你的分析是否建立在正确的事件时间线上。若存在多个事件，确认它们的先后关系是否支持你的心理推断。如有矛盾，修正你的结论。"
)

# 限速与重试
RATE_LIMIT_SLEEP_SEC = 1.2
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_SEC = 2.0
RETRY_MAX_DELAY_SEC = 15.0  # 退避上限


def _question_filename(task_type: str) -> str:
    """dataset1 题目文件名：Information Extraction_questions.json 等"""
    base = task_type.replace(" ", " ")
    return f"{base}_questions.json"


def _answer_filename(task_type: str) -> str:
    """dataset1 答案文件名：Information Extraction_answers.json 等"""
    base = task_type.replace(" ", " ")
    return f"{base}_answers.json"


def load_input_context(dataset_path: Path) -> List[Dict[str, Any]]:
    """加载 dataset1/input/dataset1.json，返回 id -> chunk 的列表（按 id 索引）。"""
    input_path = dataset_path / "input" / "dataset1.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # list of chunks with "id"


def chunks_to_text(
    chunks: List[Dict[str, Any]],
    max_chunks: int = 500,
    prioritize_inner_thought: bool = True,
) -> str:
    """
    将 input chunks 转为供 Registrar 使用的连续文本。
    当 prioritize_inner_thought=True 时，将 inner_thought 置于最前（Level III 答案常藏在内心冲突中）。
    """
    inner_lines: List[str] = []
    rest_lines: List[str] = []
    for i, c in enumerate(chunks):
        if i >= max_chunks:
            break
        if prioritize_inner_thought and c.get("inner_thought"):
            inner_lines.append(str(c["inner_thought"]).strip())
        parts = []
        if c.get("timestamp"):
            parts.append(f"[{c['timestamp']}]")
        if c.get("location"):
            parts.append(c["location"])
        if c.get("action"):
            parts.append(c["action"])
        if c.get("dialogue"):
            parts.append(c["dialogue"])
        if c.get("environment"):
            parts.append(c["environment"])
        if c.get("background"):
            parts.append(c["background"])
        if c.get("inner_thought"):
            parts.append(c["inner_thought"])
        if parts:
            rest_lines.append(" ".join(str(p) for p in parts))
    if prioritize_inner_thought and inner_lines:
        return "【内心独白，优先抽取】\n" + "\n".join(inner_lines) + "\n\n【情境与行为】\n" + "\n".join(rest_lines)
    return "\n".join(rest_lines) if rest_lines else ""


def get_evidence_ids_from_answer(answer_item: Dict[str, Any]) -> List[int]:
    """从单条 answer 中取出 evidence（可能是 int 或 list）。"""
    ev = answer_item.get("evidence")
    if ev is None:
        return []
    if isinstance(ev, list):
        return [int(x) for x in ev if isinstance(x, (int, float))]
    return [int(ev)]


def load_all_questions_and_references(
    dataset_path: Path,
    task_types: List[str],
    max_per_task: Optional[int],
) -> List[Dict[str, Any]]:
    """
    加载所有题目，合并 reference_answer，并关联 evidence 用于截取上下文。
    返回列表，每项: {task_type, id, question, reference_answer, evidence_ids}.
    """
    input_chunks = load_input_context(dataset_path)
    id_to_chunk = {c["id"]: c for c in input_chunks if isinstance(c.get("id"), (int, float))}
    rows = []
    for task_type in task_types:
        q_path = dataset_path / "question" / _question_filename(task_type)
        a_path = dataset_path / "answer" / _answer_filename(task_type)
        if not q_path.exists() or not a_path.exists():
            print(f"  [skip] {task_type}: missing question or answer file")
            continue
        with open(q_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        with open(a_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
        # 按 question id 聚合 reference；evidence 聚合
        if task_type == "Information Extraction":
            # 多个 answer 行对应同一 question_id，reference 合并
            qid_to_ref = {}
            qid_to_evidence = {}
            for a in answers:
                qid = a.get("question_id", a.get("id"))
                ref = a.get("answer", "")
                if qid not in qid_to_ref:
                    qid_to_ref[qid] = []
                    qid_to_evidence[qid] = []
                qid_to_ref[qid].append(ref)
                qid_to_evidence[qid].extend(get_evidence_ids_from_answer(a))
            q_list = {q["id"]: q["question"] for q in questions}
            for qid, refs in qid_to_ref.items():
                if qid not in q_list:
                    continue
                evidence_ids = list(dict.fromkeys(qid_to_evidence.get(qid, [])))
                rows.append({
                    "task_type": task_type,
                    "id": qid,
                    "question": q_list[qid],
                    "reference_answer": " | ".join(refs),
                    "evidence_ids": evidence_ids,
                })
        else:
            # 一般情况：answer 与 question 同 id 一一对应
            a_by_id = {}
            for a in answers:
                aid = a.get("id")
                if aid is not None:
                    a_by_id[aid] = a
            for q in questions:
                qid = q.get("id")
                if qid is None:
                    continue
                a = a_by_id.get(qid)
                ref = a.get("answer", "") if a else ""
                evidence_ids = get_evidence_ids_from_answer(a) if a else []
                rows.append({
                    "task_type": task_type,
                    "id": qid,
                    "question": q.get("question", ""),
                    "reference_answer": ref,
                    "evidence_ids": evidence_ids,
                })
    if max_per_task is not None:
        by_task: Dict[str, List[Dict]] = {}
        for r in rows:
            by_task.setdefault(r["task_type"], []).append(r)
        rows = []
        for tt in task_types:
            rows.extend((by_task.get(tt) or [])[:max_per_task])
    return rows


def build_context_from_evidence(
    id_to_chunk: Dict[int, Dict],
    evidence_ids: List[int],
    full_chunks: List[Dict],
    max_fallback_chunks: int = 400,
    prioritize_inner_thought: bool = True,
) -> str:
    """用 evidence ids 取 chunk；若无则用前 max_fallback_chunks 条。inner_thought 优先。"""
    if evidence_ids:
        chunks = [id_to_chunk[e] for e in evidence_ids if e in id_to_chunk]
    else:
        chunks = full_chunks[:max_fallback_chunks]
    return chunks_to_text(chunks, max_chunks=500, prioritize_inner_thought=prioritize_inner_thought)


def build_evidence_with_timestamps(
    id_to_chunk: Dict[int, Dict],
    evidence_ids: List[int],
    full_chunks: List[Dict],
    max_chunks: int = 120,
    max_snippet_len: int = 200,
) -> str:
    """构建带时间戳的原文证据摘要，供 Linguist 引用时保留时间戳（判分器信任度更高）。"""
    if evidence_ids:
        chunks = [id_to_chunk[e] for e in evidence_ids if e in id_to_chunk]
    else:
        chunks = full_chunks[:max_chunks]
    lines: List[str] = []
    for c in chunks[:max_chunks]:
        if not isinstance(c, dict):
            continue
        ts = c.get("timestamp") or c.get("time") or ""
        parts = []
        for key in ("inner_thought", "dialogue", "action", "environment", "background"):
            val = c.get(key)
            if val and str(val).strip():
                parts.append(str(val).strip()[:max_snippet_len])
        snippet = " | ".join(parts) if parts else "(无摘录)"
        prefix = f"[{ts}] " if ts else ""
        lines.append(prefix + snippet)
    return "\n".join(lines) if lines else ""


# 事实类任务：需「原始信息直通车」直供原文，跳过语义抽象
FACT_EXTRACTION_TASKS = {"Information Extraction"}

# 所有非 Level III 任务：统一使用路由分发 + 专用 prompt
NON_DEEP_TASKS = {
    "Information Extraction",
    "Adversarial Abstention",
    "Temporal Reasoning",
    "Logical Event Ordering",
    "Mnestic Trigger Analysis",
}


def build_raw_evidence_text(
    id_to_chunk: Dict[int, Dict],
    evidence_ids: List[int],
    full_chunks: List[Dict],
    max_chunks: int = 200,
    max_char_per_chunk: Optional[int] = 8000,
) -> str:
    """
    构建供事实类题目使用的原始原文（Raw-Fact Bypass）。
    不做语义压缩、不截断为 200 字摘要，按 chunk 保留完整字段结构，便于模型逐字提取。
    """
    if evidence_ids:
        chunks = [id_to_chunk[e] for e in evidence_ids if e in id_to_chunk]
    else:
        chunks = full_chunks[:max_chunks]
    block_lines: List[str] = []
    for c in chunks[:max_chunks]:
        if not isinstance(c, dict):
            continue
        ts = c.get("timestamp") or c.get("time") or ""
        prefix = f"[{ts}] " if ts else ""
        parts: List[str] = []
        for key in ("location", "action", "dialogue", "environment", "background", "inner_thought"):
            val = c.get(key)
            if val is not None and str(val).strip():
                raw = str(val).strip()
                if max_char_per_chunk and len(raw) > max_char_per_chunk:
                    raw = raw[:max_char_per_chunk] + "…"
                parts.append(f"  {key}: {raw}")
        if parts:
            block_lines.append(prefix + "\n" + "\n".join(parts))
    return "\n\n---\n\n".join(block_lines) if block_lines else ""


def build_full_evidence_text(
    id_to_chunk: Dict[int, Dict],
    evidence_ids: List[int],
    full_chunks: List[Dict],
    max_chunks: int = 500,
) -> str:
    """
    构建包含所有相关 chunk 的完整原文（用于需要广泛上下文的任务）。
    当 evidence_ids 为空时，使用全部 chunks 以避免信息遗漏。
    """
    if evidence_ids:
        chunks = [id_to_chunk[e] for e in evidence_ids if e in id_to_chunk]
        if not chunks:
            chunks = full_chunks[:max_chunks]
    else:
        chunks = full_chunks[:max_chunks]
    return build_raw_evidence_text(id_to_chunk, [c.get("id", -1) for c in chunks], id_to_chunk, max_chunks=max_chunks)


async def _sleep_rate_limit() -> None:
    """每题或每次调用后限速。"""
    await asyncio.sleep(RATE_LIMIT_SLEEP_SEC)


def _is_retryable(e: Exception) -> bool:
    """是否为可重试的限流/临时错误（429 限流 + 500/502/503 服务端临时故障 + 超时）。"""
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503)
    if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout)):
        return True
    return False


async def _with_retry(coro_fn):
    """执行异步调用（coro_fn 为无参可调用，返回 awaitable），遇可重试错误时指数退避重试（带上限）。"""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn()
        except Exception as e:
            last_err = e
            if _is_retryable(e) and attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BACKOFF_BASE_SEC ** (attempt + 1), RETRY_MAX_DELAY_SEC)
                if isinstance(e, httpx.HTTPStatusError):
                    label = f"HTTP {e.response.status_code}"
                elif isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout)):
                    label = f"Timeout ({type(e).__name__})"
                else:
                    label = type(e).__name__
                print(f"    [{label}, {delay:.0f}s 后重试 ({attempt+1}/{MAX_RETRIES})]")
                await asyncio.sleep(delay)
                continue
            raise
    raise last_err


async def run_mnemos_for_one(
    question_row: Dict[str, Any],
    context_text: str,
    evidence_with_timestamps: str,
    registrar: Any,
    psychologist: Any,
    linguist: Any,
    model_config: Dict[str, str],
    member_id: str = "knowme_bench",
    raw_source_text: str = "",
    timeline_text: str = "",
    verifier: Any = None,
    routing_decision: Any = None,
) -> str:
    """
    路由分发模式：根据 routing_decision 走不同处理管线。

    - 事实提取 / 时序排列 / 对抗弃权 / 时间推理 / 记忆触发：
      跳过 Registrar & Psychologist，直接用原始原文 + 专用 system prompt
    - 深度分析 (Level III)：
      保持 Registrar -> Psychologist -> Linguist (Jarvis) 完整流程
    """
    from mnemos.core.query_router import route_benchmark_task, QueryCategory

    task_type = question_row["task_type"]
    question = question_row["question"]

    if routing_decision is None:
        routing_decision = route_benchmark_task(task_type)
    rd = routing_decision

    # ========== 事实类 / 时序类 / 弃权类 / 推理类 / 触发类：专用管线 ==========
    if rd.category != QueryCategory.DEEP_ANALYSIS:
        context: Dict[str, Any] = {
            "system_prompt_key": rd.system_prompt_key,
            "insights": [],
            "psychologist_result": None,
            "answer_format_instruction": None,
        }
        if rd.sort_by_timeline and timeline_text.strip():
            context["timeline_text"] = timeline_text.strip()[:180_000]
        elif raw_source_text.strip():
            context["raw_source_text"] = raw_source_text.strip()[:180_000]
        elif evidence_with_timestamps.strip():
            context["source_with_timestamps"] = evidence_with_timestamps.strip()[:180_000]

        try:
            if linguist:
                model_answer = await _with_retry(lambda: linguist.generate_response(
                    message=question,
                    member_id=member_id,
                    context=context,
                    model_override=model_config.get("linguist"),
                ))
            else:
                model_answer = "(Linguist 未配置)"
        except Exception as e:
            print(f"  [Linguist error] {e}")
            model_answer = f"(生成失败: {e})"

        if rd.enable_self_consistency and verifier and model_answer.strip():
            try:
                source_for_verify = raw_source_text or evidence_with_timestamps
                model_answer = await verifier.verify_and_maybe_abstain(
                    question=question,
                    answer=model_answer,
                    source_text=source_for_verify[:30_000],
                    language="en",
                )
            except Exception as e:
                print(f"  [Verifier error] {e}")

        return model_answer.strip() or "(无输出)"

    # ========== 深度分析 (Level III)：保持原有完整流程 ==========
    # 1. Registrar: 上下文 -> Facts
    facts: List[Dict[str, Any]] = []
    if registrar and context_text.strip():
        try:
            facts = await _with_retry(lambda: registrar.extract_facts(
                text=context_text[:120000],
                member_id=member_id,
                model_override=model_config.get("registrar"),
            )) or []
        except Exception as e:
            print(f"  [Registrar error] {e}")
    if not facts and context_text.strip():
        # 无 Registrar 或抽取失败时，构造一条“原始上下文”事实供 Psychologist 使用
        facts = [{
            "entity": "narrator",
            "attribute": "context",
            "value": context_text[:8000],
            "context_tags": ["#narrative"],
            "confidence_score": 0.5,
        }]

    psychologist_result: Optional[Dict[str, Any]] = None
    if psychologist and facts:
        try:
            msg = "[本题为 Level III 心理洞察题，请强化潜台词与未言明需求分析。]\n" + question
            psychologist_result = await _with_retry(lambda: psychologist.infer_values_and_motivations(
                member_id=member_id,
                long_term_facts=facts,
                message=msg,
                model_override=model_config.get("psychologist"),
            ))
        except Exception as e:
            print(f"  [Psychologist error] {e}")

    context_l3: Dict[str, Any] = {
        "psychologist_result": psychologist_result,
        "insights": [],
        "answer_format_instruction": LEVEL_III_ANSWER_FORMAT_INSTRUCTION,
        "source_with_timestamps": evidence_with_timestamps,
    }
    try:
        if linguist:
            model_answer = await _with_retry(lambda: linguist.generate_response(
                message=question,
                member_id=member_id,
                context=context_l3,
                model_override=model_config.get("linguist"),
            ))
        else:
            model_answer = "(Linguist 未配置)"
    except Exception as e:
        print(f"  [Linguist error] {e}")
        model_answer = f"(生成失败: {e})"
    return model_answer.strip() or "(无输出)"


async def main_async(args: argparse.Namespace) -> None:
    from mnemos.core.model_config import get_model_config
    from mnemos.workers.registrar import FactRegistrar
    from mnemos.workers.reflector import Psychologist
    from mnemos.core.linguist import JarvisLinguist
    from mnemos.core.query_router import route_benchmark_task
    from mnemos.workers.verifier import SelfConsistencyVerifier
    from mnemos.utils.timeline import sort_chunks_by_timeline, build_timeline_text, select_relevant_chunks

    model_config = get_model_config()
    registrar = FactRegistrar(model=model_config["registrar"], max_tokens=2048) if args.use_registrar else None
    psychologist = Psychologist(model=model_config["psychologist"], max_tokens=2048)
    linguist = JarvisLinguist(model=model_config["linguist"], max_tokens=1536)
    verifier = SelfConsistencyVerifier()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    input_chunks = load_input_context(dataset_path)
    id_to_chunk = {c["id"]: c for c in input_chunks if isinstance(c.get("id"), (int, float))}
    full_chunks = input_chunks

    task_types = (
        ["Mind-Body Interaction", "Expert-Annotated Psychoanalysis"]
        if args.level3_only
        else TASK_TYPES
    )
    rows = load_all_questions_and_references(
        dataset_path,
        task_types,
        args.max_per_task,
    )
    print(f"加载题目数: {len(rows)}")

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.parent / (out_path.stem + "_checkpoint.json")
    run_config = {
        "output_json": str(out_path.resolve()),
        "dataset_dir": args.dataset_dir,
        "max_per_task": args.max_per_task,
        "level3_only": getattr(args, "level3_only", False),
    }

    results_for_eval: List[Dict[str, Any]] = []
    rerun_ids_set: Optional[set] = None
    if getattr(args, "rerun_ids_file", None):
        # 仅重跑额度不足导致的失败题：从现有 model_outputs 合并，正确结果不重复跑
        rerun_path = Path(args.rerun_ids_file)
        if not rerun_path.is_absolute():
            rerun_path = PROJECT_ROOT / rerun_path
        if not rerun_path.exists():
            print(f"错误：--rerun_ids_file 不存在: {rerun_path}")
            return
        with open(rerun_path, "r", encoding="utf-8") as f:
            rerun_ids = json.load(f)
        if not isinstance(rerun_ids, list):
            rerun_ids = []
        rerun_ids_set = set(int(x) for x in rerun_ids if isinstance(x, (int, float)))
        if not out_path.exists():
            print(f"错误：仅重跑时需已有 {out_path}，请先全量跑完或提供已有 model_outputs.json")
            return
        with open(out_path, "r", encoding="utf-8") as f:
            results_for_eval = json.load(f)
        if len(results_for_eval) != len(rows):
            print(f"错误：现有 model_outputs 题数 {len(results_for_eval)} 与当前题目数 {len(rows)} 不一致")
            return
        print(f"仅重跑：共 {len(rerun_ids_set)} 题（额度不足/占位答案），其余 {len(rows)-len(rerun_ids_set)} 题沿用原结果")
    elif getattr(args, "resume", False) and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            if isinstance(ck.get("config"), dict) and ck["config"] == run_config:
                results_for_eval = list(ck.get("results") or [])
                print(f"断点续传：已加载 {len(results_for_eval)} 题，从第 {len(results_for_eval)+1} 题继续")
            else:
                print("断点续传：checkpoint 配置与本次运行不一致，将从头跑")
        except Exception as e:
            print(f"断点续传：读取 checkpoint 失败 ({e})，将从头跑")
        rerun_ids_set = None

    if rerun_ids_set is not None:
        # 仅重跑 rerun_ids_set 中的题目
        pbar = tqdm(total=len(rerun_ids_set), desc="重跑进度", unit="题", ncols=80)
        done = 0
        for i, row in enumerate(rows):
            if (i + 1) not in rerun_ids_set:
                continue
            pbar.set_postfix_str(f"{row['task_type']} #{row.get('id', i+1)}")
            evidence_ids = row.get("evidence_ids") or []
            rd = route_benchmark_task(row["task_type"])

            # 智能 chunk 选择
            relevant_chunks = select_relevant_chunks(
                query=row["question"],
                all_chunks=full_chunks,
                evidence_ids=evidence_ids,
                id_to_chunk=id_to_chunk,
                margin_days=3,
                max_chunks=800,
                fallback_max=800,
            )
            rel_ids = [c.get("id", -1) for c in relevant_chunks]

            context_text = build_context_from_evidence(
                id_to_chunk, rel_ids, full_chunks, max_fallback_chunks=400
            )
            evidence_ts = build_evidence_with_timestamps(
                id_to_chunk, rel_ids, full_chunks, max_chunks=300
            )
            raw_source_text = (
                build_raw_evidence_text(id_to_chunk, rel_ids, full_chunks, max_chunks=500)
                if row.get("task_type") in NON_DEEP_TASKS
                else ""
            )
            tl_text = ""
            if rd.sort_by_timeline:
                tl_text = build_timeline_text(relevant_chunks, max_chunks=600)
            try:
                model_answer = await run_mnemos_for_one(
                    row, context_text, evidence_ts, registrar, psychologist, linguist, model_config,
                    raw_source_text=raw_source_text,
                    timeline_text=tl_text,
                    verifier=verifier,
                    routing_decision=rd,
                )
                await _sleep_rate_limit()
                results_for_eval[i] = {
                    "id": i + 1,
                    "task_type": row["task_type"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "model_answer": model_answer,
                }
                done += 1
            except Exception as e:
                print(f"\n  [重跑失败 id={i+1}] {e!r}，保留原答案，继续下一题")
            pbar.update(1)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"config": run_config, "results": results_for_eval}, f, ensure_ascii=False, indent=2)
        pbar.close()
        print(f"重跑完成：成功 {done}/{len(rerun_ids_set)} 题")
        # 写出仍为占位/失败的 id，便于下次只重跑未成功的题（省 API）
        still_bad = []
        for idx, item in enumerate(results_for_eval):
            ans = (item.get("model_answer") or "").strip()
            if not ans or "收到你的消息：" in ans or "(生成失败" in ans or ans == "(无输出)":
                still_bad.append(idx + 1)
        if still_bad:
            remaining_path = out_path.parent / "rerun_ids_remaining.json"
            with open(remaining_path, "w", encoding="utf-8") as f:
                json.dump(still_bad, f, ensure_ascii=False)
            print(f"仍有 {len(still_bad)} 题为占位/失败，已写入 {remaining_path}，下次可 --rerun_ids_file {remaining_path}")
    else:
        # 全量或断点续传
        pbar = tqdm(
            total=len(rows),
            initial=len(results_for_eval),
            desc="打榜进度",
            unit="题",
            ncols=80,
        )
        for i, row in enumerate(rows):
            if i < len(results_for_eval):
                continue
            pbar.set_postfix_str(f"{row['task_type']} #{row.get('id', i+1)}")
            evidence_ids = row.get("evidence_ids") or []
            rd = route_benchmark_task(row["task_type"])

            # 智能 chunk 选择：优先 evidence_ids → 日期过滤 → 关键词过滤 → fallback
            relevant_chunks = select_relevant_chunks(
                query=row["question"],
                all_chunks=full_chunks,
                evidence_ids=evidence_ids,
                id_to_chunk=id_to_chunk,
                margin_days=3,
                max_chunks=800,
                fallback_max=800,
            )
            rel_ids = [c.get("id", -1) for c in relevant_chunks]

            context_text = build_context_from_evidence(
                id_to_chunk, rel_ids, full_chunks, max_fallback_chunks=400
            )
            evidence_ts = build_evidence_with_timestamps(
                id_to_chunk, rel_ids, full_chunks, max_chunks=300
            )
            raw_source_text = (
                build_raw_evidence_text(id_to_chunk, rel_ids, full_chunks, max_chunks=500)
                if row.get("task_type") in NON_DEEP_TASKS
                else ""
            )
            tl_text = ""
            if rd.sort_by_timeline:
                tl_text = build_timeline_text(relevant_chunks, max_chunks=600)
            model_answer = await run_mnemos_for_one(
                row, context_text, evidence_ts, registrar, psychologist, linguist, model_config,
                raw_source_text=raw_source_text,
                timeline_text=tl_text,
                verifier=verifier,
                routing_decision=rd,
            )
            await _sleep_rate_limit()
            results_for_eval.append({
                "id": i + 1,
                "task_type": row["task_type"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "model_answer": model_answer,
            })
            pbar.update(1)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"config": run_config, "results": results_for_eval}, f, ensure_ascii=False, indent=2)
        pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_for_eval, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out_path}")

    # 4. 执行官方判分（增量模式：rerun 时只判分被重跑的题）
    eval_dir = KNOWME_EVALUATE
    if not eval_dir.exists():
        print(f"未找到 evaluate 目录: {eval_dir}，跳过判分")
        return
    results_path = out_path.parent / "results.json"
    judge_model = args.judge_model
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
        judge_model = "openai/gpt-4o"
        print("\n使用 OPENROUTER_API_KEY 判分，模型: openai/gpt-4o")

    if rerun_ids_set and results_path.exists():
        # 增量判分：只对被重跑的题调用判分器，其余复用旧 results.json
        print(f"\n增量判分：仅重判 {len(rerun_ids_set)} 题，其余复用已有 results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            old_eval = json.load(f)
        old_details = old_eval.get("details", [])
        old_by_id = {d["id"]: d for d in old_details}

        # 构造只包含重跑题的临时 input
        rerun_items = [item for item in results_for_eval if item.get("id") in rerun_ids_set]
        if rerun_items:
            tmp_input = out_path.parent / "_rerun_eval_input.json"
            tmp_output = out_path.parent / "_rerun_eval_output.json"
            with open(tmp_input, "w", encoding="utf-8") as f:
                json.dump(rerun_items, f, ensure_ascii=False, indent=2)
            cmd = [
                sys.executable, "run_eval.py",
                "--input_file", str(tmp_input.resolve()),
                "--output_file", str(tmp_output.resolve()),
                "--judge_model", judge_model,
            ]
            print(f"执行增量判分: {' '.join(cmd)}")
            proc = subprocess.run(cmd, cwd=str(eval_dir), env={**os.environ})
            if proc.returncode != 0:
                print("增量判分脚本返回非 0")
            elif tmp_output.exists():
                with open(tmp_output, "r", encoding="utf-8") as f:
                    new_eval = json.load(f)
                for d in new_eval.get("details", []):
                    old_by_id[d["id"]] = d
                    print(f"  ID={d['id']}: score={d.get('score')} ({d.get('status')})")
            # 清理临时文件
            tmp_input.unlink(missing_ok=True)
            tmp_output.unlink(missing_ok=True)

        # 合并：按原顺序重建 details
        merged_details = []
        for item in results_for_eval:
            iid = item.get("id")
            if iid in old_by_id:
                merged_details.append(old_by_id[iid])
            else:
                merged_details.append({"id": iid, "task_type": item.get("task_type"), "score": 0,
                                       "reasoning": "No evaluation available", "status": "error"})
        valid_scores = [d["score"] for d in merged_details
                        if d.get("status") == "success" and isinstance(d.get("score"), (int, float))]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        final_eval = {
            "meta": {
                "judge_model": judge_model,
                "total_items": len(results_for_eval),
                "evaluated_items": len(valid_scores),
                "average_score": round(avg_score, 4),
            },
            "details": merged_details,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_eval, f, ensure_ascii=False, indent=2)
        print(f"增量判分完成，已合并写入 {results_path}")
    else:
        # 全量判分
        input_abs = str(out_path.resolve())
        cmd = [
            sys.executable, "run_eval.py",
            "--input_file", input_abs,
            "--output_file", str(results_path.resolve()),
            "--judge_model", judge_model,
        ]
        print(f"\n执行判分: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(eval_dir), env={**os.environ})
        if proc.returncode != 0:
            print("判分脚本返回非 0。请确保已安装 openai、tqdm，并设置 OPENAI_API_KEY 或 OPENROUTER_API_KEY。")
            print("model_outputs.json 已生成，可手动执行: cd KnowMeBench/evaluate && python run_eval.py --input_file <path>/model_outputs.json --output_file <path>/results.json --judge_model gpt-4.5")
            return

    # 5. 汇总报告并保存冠军战报
    if not results_path.exists():
        print("未找到 results.json")
        return
    with open(results_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    meta = eval_data.get("meta", {})
    details = eval_data.get("details", [])
    by_task: Dict[str, List[float]] = {}
    for d in details:
        if d.get("status") != "success":
            continue
        s = d.get("score")
        if isinstance(s, (int, float)):
            tt = d.get("task_type", "Unknown")
            by_task.setdefault(tt, []).append(float(s))
    level3_scores: List[float] = []
    report_lines: List[str] = []
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("KnowMe-Bench 评测结果汇总 · Mnemos 冠军战报")
    report_lines.append("=" * 60)
    report_lines.append(f"总题数: {meta.get('total_items', 0)}  有效判分: {meta.get('evaluated_items', 0)}")
    report_lines.append(f"总体平均分 (0-5): {meta.get('average_score', 0):.2f}")
    report_lines.append("")
    report_lines.append("各任务得分 (0-5 平均 → 百分制):")
    report_lines.append("-" * 50)
    report_task_list = ["Mind-Body Interaction", "Expert-Annotated Psychoanalysis"] if args.level3_only else TASK_TYPES
    for tt in report_task_list:
        scores = by_task.get(tt, [])
        avg = sum(scores) / len(scores) if scores else 0.0
        pct = (avg / 5.0) * 100.0
        if tt in LEVEL_III_TASKS:
            level3_scores.extend(scores)
            report_lines.append(f"  {tt}: {avg:.2f}  ({pct:.1f}%)  [Level III]")
        else:
            report_lines.append(f"  {tt}: {avg:.2f}  ({pct:.1f}%)")
    report_lines.append("-" * 50)
    if level3_scores:
        l3_avg = sum(level3_scores) / len(level3_scores)
        l3_pct = (l3_avg / 5.0) * 100.0
        report_lines.append(f"")
        report_lines.append(f"Level III 平均: {l3_avg:.2f}  ({l3_pct:.1f}%)")
        report_lines.append(f"MemBrain 1.0 Level III 冠军线: {LEVEL_III_CHAMPION_PCT}%")
        if l3_pct >= LEVEL_III_CHAMPION_PCT:
            report_lines.append("  ✅ 已达到 Level III 冠军线")
        else:
            report_lines.append(f"  ⚠ 距冠军线差 {LEVEL_III_CHAMPION_PCT - l3_pct:.1f}%")
    report_lines.append("=" * 60)
    report_lines.append(f"详细结果: {results_path}")
    report_lines.append(f"生成时间: {datetime.now().isoformat()}")
    report_text = "\n".join(report_lines)
    print(report_text)

    # 自动保存冠军战报
    report_dir = out_path.parent
    report_name = f"champion_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    report_path = report_dir / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Mnemos KnowMe-Bench 冠军战报\n\n")
        f.write(report_text.replace("=" * 60, "---").replace("-" * 50, "---"))
    print(f"\n冠军战报已保存: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KnowMe-Bench 打榜：加载题目 → Mnemos 生成答案 → 判分 → 汇总")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="KnowMeBench/KnowmeBench/dataset1",
        help="dataset1 路径（相对项目根）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/model_outputs.json",
        help="输出的 model_outputs.json 路径",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.5",
        help="判分模型 (run_eval.py --judge_model)，如 gpt-4o / gpt-4.5",
    )
    parser.add_argument(
        "--max_per_task",
        type=int,
        default=None,
        help="每类任务最多跑几题（用于快速试跑）",
    )
    parser.add_argument(
        "--use_registrar",
        action="store_true",
        help="是否调用 Registrar 将上下文转为 Facts（否则用整段上下文作单条 fact）",
    )
    parser.add_argument(
        "--level3_only",
        action="store_true",
        help="仅跑 Level III 题目（Mind-Body Interaction + Expert-Annotated Psychoanalysis），可与 --max_per_task 配合做小规模抽测",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：从上次 checkpoint 继续（OpenRouter 中途挂掉时重跑可加此参数）",
    )
    parser.add_argument(
        "--rerun_ids_file",
        type=str,
        default=None,
        help="仅重跑指定 id 的题目（JSON 数组，如 data/rerun_ids.json），从现有 model_outputs 合并，不重跑正确结果",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
