"""
时间线排序工具 (Timeline Utilities)

为 Logical Event Ordering 等需要跨场景时间线排序的任务提供支持：
- 解析 chunk 中的 timestamp 字段为 unix_timestamp
- 按时间线排序 chunk 列表
- 生成带编号的时间线文本（供模型按序推理）
- 从 Query 中提取日期关键词，智能选择相关 chunk
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


# 常见 timestamp 格式（KnowMeBench dataset）
_TS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"), "%Y-%m-%dT%H:%M:%S"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$"), "%Y-%m-%d %H:%M"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    # 处理 "August 15, 1969" 等英文格式
    (re.compile(r"^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}$"), None),
]

# 英文月份简写
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_timestamp(ts_str: str) -> Optional[float]:
    """
    将 timestamp 字符串解析为 unix_timestamp（秒）。
    返回 None 表示无法解析。
    """
    if not ts_str or not isinstance(ts_str, str):
        return None
    ts_str = ts_str.strip()
    if not ts_str:
        return None

    for pattern, fmt in _TS_PATTERNS:
        if pattern.match(ts_str):
            if fmt:
                try:
                    dt = datetime.strptime(ts_str, fmt)
                    return dt.timestamp()
                except (ValueError, OSError):
                    continue
            else:
                # 英文日期格式
                return _parse_english_date(ts_str)

    # 尝试通用解析
    return _try_generic_parse(ts_str)


def _parse_english_date(s: str) -> Optional[float]:
    """解析 'August 15, 1969' 等英文日期。"""
    s = s.replace(",", "").strip()
    parts = s.split()
    if len(parts) < 2:
        return None
    month_name = parts[0].lower()
    month = _MONTH_MAP.get(month_name)
    if not month:
        return None
    try:
        if len(parts) == 3:
            day, year = int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            year = int(parts[1])
            day = 1
        else:
            return None
        dt = datetime(year, month, day)
        return dt.timestamp()
    except (ValueError, OSError):
        return None


def _try_generic_parse(s: str) -> Optional[float]:
    """尝试从字符串中提取日期信息。"""
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            time_m = re.search(r"(\d{1,2}):(\d{2}):?(\d{2})?", s[m.end():])
            if time_m:
                dt = dt.replace(
                    hour=int(time_m.group(1)),
                    minute=int(time_m.group(2)),
                    second=int(time_m.group(3) or 0),
                )
            return dt.timestamp()
        except (ValueError, OSError):
            pass
    return None


def sort_chunks_by_timeline(
    chunks: List[Dict[str, Any]],
    timestamp_key: str = "timestamp",
) -> List[Dict[str, Any]]:
    """
    按 timestamp 字段对 chunk 列表排序。
    无法解析的 chunk 置于末尾（保持原序）。
    每个 chunk 会被添加 _unix_timestamp 字段。
    """
    annotated: List[Tuple[float, int, Dict[str, Any]]] = []
    no_ts: List[Tuple[int, Dict[str, Any]]] = []

    for idx, chunk in enumerate(chunks):
        ts_raw = chunk.get(timestamp_key) or chunk.get("time") or ""
        unix_ts = parse_timestamp(str(ts_raw))
        chunk_copy = dict(chunk)
        if unix_ts is not None:
            chunk_copy["_unix_timestamp"] = unix_ts
            annotated.append((unix_ts, idx, chunk_copy))
        else:
            chunk_copy["_unix_timestamp"] = None
            no_ts.append((idx, chunk_copy))

    annotated.sort(key=lambda x: (x[0], x[1]))
    result = [item[2] for item in annotated]
    result.extend(item[1] for item in no_ts)
    return result


def build_timeline_text(
    chunks: List[Dict[str, Any]],
    max_chunks: int = 200,
    max_char_per_field: int = 500,
    include_event_numbers: bool = True,
) -> str:
    """
    将按时间线排序后的 chunks 构建为编号时间线文本，
    供模型识别事件先后顺序。

    格式示例：
    [Event 1] [1969-08-15 14:00:30]
      location: A gravel road leading to a farm
      action: The family walked along the road
      ...

    [Event 2] [1969-08-15 14:30:00]
      ...
    """
    sorted_chunks = sort_chunks_by_timeline(chunks)
    lines: List[str] = []
    event_num = 0

    for chunk in sorted_chunks[:max_chunks]:
        if not isinstance(chunk, dict):
            continue
        event_num += 1
        ts = chunk.get("timestamp") or chunk.get("time") or ""
        prefix = f"[Event {event_num}]" if include_event_numbers else ""
        ts_label = f" [{ts}]" if ts else ""
        header = f"{prefix}{ts_label}".strip()
        if header:
            lines.append(header)

        for key in ("location", "action", "dialogue", "environment",
                     "background", "inner_thought"):
            val = chunk.get(key)
            if val and str(val).strip():
                raw = str(val).strip()
                if len(raw) > max_char_per_field:
                    raw = raw[:max_char_per_field] + "…"
                lines.append(f"  {key}: {raw}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 智能 Chunk 选择：从 Query 中提取日期，按日期范围过滤相关 chunks
# ---------------------------------------------------------------------------

# 匹配多种日期格式
_QUERY_DATE_PATTERNS = [
    # "July 15, 1975" / "August 18, 1975"
    re.compile(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}",
        re.IGNORECASE,
    ),
    # "1975-08-18" ISO format
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    # "September 30, 1975" as part of a sentence
    re.compile(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}",
        re.IGNORECASE,
    ),
    # "July 1998" / "November 2003" (month + year, no day)
    re.compile(
        r"(?:early|late|mid-?)?\s*(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}",
        re.IGNORECASE,
    ),
    # "1990s" / "summer of 2003"
    re.compile(r"\b(?:summer|winter|spring|autumn|fall)\s+(?:of\s+)?\d{4}\b", re.IGNORECASE),
    # "January 20, 1976" / "February 15, 1985" etc
    re.compile(r"\b\d{4}\b"),
]

_MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def extract_dates_from_query(query: str) -> List[str]:
    """
    从查询文本中提取日期字符串，返回 YYYY-MM-DD 格式列表。
    对于只有月份的日期（如 "July 1998"），返回 YYYY-MM 格式。
    对于只有年份的（如 "1990s"），返回 YYYY 格式。
    """
    dates: List[str] = []

    # Pattern 0: date range "Month Day-Day, Year" or "Month Day to Day, Year"
    for m in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})\s*[-–to]+\s*(\d{1,2}),?\s+(\d{4})",
        query, re.IGNORECASE
    ):
        month_num = _MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if month_num:
            start_day = int(m.group(2))
            end_day = int(m.group(3))
            year = m.group(4)
            for day in range(start_day, end_day + 1):
                d = f"{year}-{month_num:02d}-{day:02d}"
                if d not in dates:
                    dates.append(d)

    # Pattern 0b: "Month Day and Day, Year"
    for m in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})\s+and\s+(\d{1,2}),?\s+(\d{4})",
        query, re.IGNORECASE
    ):
        month_num = _MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if month_num:
            start_day = int(m.group(2))
            end_day = int(m.group(3))
            year = m.group(4)
            for day in range(start_day, end_day + 1):
                d = f"{year}-{month_num:02d}-{day:02d}"
                if d not in dates:
                    dates.append(d)

    # Pattern 1: "Month Day, Year"
    for m in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),?\s+(\d{4})",
        query, re.IGNORECASE
    ):
        month_num = _MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if month_num:
            d = f"{m.group(3)}-{month_num:02d}-{int(m.group(2)):02d}"
            if d not in dates:
                dates.append(d)

    # Pattern 2: ISO dates
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", query):
        if m.group(0) not in dates:
            dates.append(m.group(0))

    # Pattern 3: "Month Year" (no day)
    for m in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})",
        query, re.IGNORECASE
    ):
        month_num = _MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if month_num:
            ym = f"{m.group(2)}-{month_num:02d}"
            if not any(d.startswith(ym) for d in dates):
                dates.append(ym)

    # Pattern 4: "summer/winter/spring/fall of YYYY"
    for m in re.finditer(
        r"(?:summer|winter|spring|autumn|fall)\s+(?:of\s+)?(\d{4})",
        query, re.IGNORECASE
    ):
        year = m.group(1)
        if not any(d.startswith(year) for d in dates):
            dates.append(year)

    # Pattern 5: "early/late/mid YYYY" or "in YYYY"
    for m in re.finditer(
        r"(?:early|late|mid-?)\s+(\d{4})\b",
        query, re.IGNORECASE
    ):
        year = m.group(1)
        if not any(d.startswith(year) for d in dates):
            dates.append(year)

    # Pattern 6: "New Year's Eve YYYY" → YYYY-12-31
    for m in re.finditer(r"New\s+Year'?s?\s+Eve\s+(\d{4})", query, re.IGNORECASE):
        d = f"{m.group(1)}-12-31"
        if d not in dates:
            dates.append(d)

    # Pattern 7: "the 1990s" → 1990
    for m in re.finditer(r"\b(\d{4})s\b", query):
        decade = m.group(1)
        if not any(d.startswith(decade[:3]) for d in dates):
            dates.append(decade)

    # Pattern 8: "from Month YYYY to Month YYYY" → both months
    for m in re.finditer(
        r"from\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})\s+to\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})",
        query, re.IGNORECASE
    ):
        m1 = _MONTH_NAME_TO_NUM.get(m.group(1).lower())
        m2 = _MONTH_NAME_TO_NUM.get(m.group(3).lower())
        y1, y2 = int(m.group(2)), int(m.group(4))
        if m1 and m2:
            cur_y, cur_m = y1, m1
            while (cur_y, cur_m) <= (y2, m2):
                ym = f"{cur_y}-{cur_m:02d}"
                if not any(d.startswith(ym) for d in dates):
                    dates.append(ym)
                cur_m += 1
                if cur_m > 12:
                    cur_m = 1
                    cur_y += 1

    # Pattern 9: standalone 4-digit year in context (e.g., "between July 1 and 4, 1999")
    # Only add if no dates found yet
    if not dates:
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", query):
            year = m.group(1)
            if year not in dates:
                dates.append(year)

    return list(dict.fromkeys(dates))


def filter_chunks_by_dates(
    all_chunks: List[Dict[str, Any]],
    date_strings: List[str],
    margin_days: int = 3,
    max_chunks: int = 800,
) -> List[Dict[str, Any]]:
    """
    根据日期字符串列表，从 all_chunks 中过滤出时间戳匹配的 chunk。

    匹配策略：
    1. 精确日期 (YYYY-MM-DD)：选中该日 ± margin_days 的 chunk
    2. 月份 (YYYY-MM)：选中该月全部 chunk
    3. 年份 (YYYY)：选中该年全部 chunk

    返回去重且按原始顺序排列的 chunk 列表。
    """
    if not date_strings:
        return []

    target_date_prefixes: Set[str] = set()
    target_exact_dates: List[Tuple[datetime, datetime]] = []

    for ds in date_strings:
        parts = ds.split("-")
        if len(parts) == 3:
            try:
                center = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                start = center - timedelta(days=margin_days)
                end = center + timedelta(days=margin_days)
                target_exact_dates.append((start, end))
            except ValueError:
                target_date_prefixes.add(ds)
        elif len(parts) == 2:
            target_date_prefixes.add(ds)
        elif len(parts) == 1 and len(ds) == 4:
            target_date_prefixes.add(ds)

    selected: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()

    for chunk in all_chunks:
        ts = str(chunk.get("timestamp") or "").strip()
        if not ts:
            continue

        chunk_id = chunk.get("id")
        if chunk_id is not None and chunk_id in seen_ids:
            continue

        matched = False
        # Check prefix matches (YYYY-MM or YYYY)
        for prefix in target_date_prefixes:
            if ts.startswith(prefix):
                matched = True
                break

        # Check exact date ranges
        if not matched and target_exact_dates:
            chunk_date = _try_generic_parse(ts)
            if chunk_date is not None:
                chunk_dt = datetime.fromtimestamp(chunk_date)
                for start, end in target_exact_dates:
                    if start <= chunk_dt <= end:
                        matched = True
                        break

        if matched:
            selected.append(chunk)
            if chunk_id is not None:
                seen_ids.add(chunk_id)
            if len(selected) >= max_chunks:
                break

    return selected


def _extract_keywords_from_query(query: str) -> List[str]:
    """
    从查询中提取有意义的关键词（用于无日期题的 chunk 过滤）。
    去除常见停用词，返回小写关键词列表。
    """
    _STOP_WORDS = {
        "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "down",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "it", "its", "he", "she", "they", "them", "his", "her", "their",
        "my", "your", "our", "i", "me", "we", "you", "him",
        "based", "rank", "events", "event", "list", "three", "following",
        "describe", "specific", "scene", "scenes", "sorted", "highest",
        "lowest", "most", "least", "level", "degree", "stages", "key",
        "leading", "gradual", "escalation", "narrator", "text", "source",
    }
    words = re.findall(r"[a-zA-Z]{3,}", query.lower())
    return [w for w in words if w not in _STOP_WORDS]


def filter_chunks_by_keywords(
    all_chunks: List[Dict[str, Any]],
    keywords: List[str],
    min_keyword_hits: int = 2,
    max_chunks: int = 800,
) -> List[Dict[str, Any]]:
    """
    根据关键词从 all_chunks 中过滤出语义相关的 chunk。
    一个 chunk 必须包含至少 min_keyword_hits 个不同的关键词才被选中。
    """
    if not keywords:
        return []

    selected: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()

    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        if chunk_id is not None and chunk_id in seen_ids:
            continue

        chunk_text = " ".join(
            str(chunk.get(k, ""))
            for k in ("action", "dialogue", "environment", "background", "inner_thought", "location")
        ).lower()

        hits = sum(1 for kw in keywords if kw in chunk_text)
        if hits >= min_keyword_hits:
            selected.append(chunk)
            if chunk_id is not None:
                seen_ids.add(chunk_id)
            if len(selected) >= max_chunks:
                break

    return selected


def select_relevant_chunks(
    query: str,
    all_chunks: List[Dict[str, Any]],
    evidence_ids: List[int],
    id_to_chunk: Dict[int, Dict[str, Any]],
    margin_days: int = 3,
    max_chunks: int = 800,
    fallback_max: int = 800,
) -> List[Dict[str, Any]]:
    """
    智能选择与 Query 相关的 chunks。

    优先级：
    1. 如有 evidence_ids 且能匹配到 chunk → 直接使用
    2. 从 Query 提取日期，按日期过滤 → 日期相关 chunk
    3. 关键词匹配过滤 → 语义相关 chunk
    4. 以上都失败 → fallback 到前 fallback_max 条
    """
    # 1. evidence_ids
    if evidence_ids:
        matched = [id_to_chunk[eid] for eid in evidence_ids if eid in id_to_chunk]
        if matched:
            return matched

    # 2. date-based filtering
    dates = extract_dates_from_query(query)
    if dates:
        filtered = filter_chunks_by_dates(all_chunks, dates, margin_days=margin_days, max_chunks=max_chunks)
        if filtered:
            return filtered

    # 3. keyword-based filtering
    keywords = _extract_keywords_from_query(query)
    if keywords:
        kw_filtered = filter_chunks_by_keywords(
            all_chunks, keywords, min_keyword_hits=2, max_chunks=max_chunks
        )
        if kw_filtered:
            return kw_filtered

    # 4. fallback (increased from 500 to 800)
    return all_chunks[:fallback_max]
