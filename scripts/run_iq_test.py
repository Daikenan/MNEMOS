#!/usr/bin/env python3
"""
记忆能力测试脚本：三天场景（目标设定 → 首次违背 + 家庭变动 → 连环违背 + 社交）

验证：Registrar 事实抽取、Philosopher 场景一致性检查（潜在的行为偏离）、
认知上下文注入（历史目标/计划）、Cartographer 图谱更新。

运行方式（在项目根目录）：
  uv run python scripts/run_iq_test.py
  若提示找不到 mnemos，请先：PYTHONPATH=. uv run python scripts/run_iq_test.py
"""

import asyncio
from loguru import logger

from mnemos.core.coordinator import MnemosCoordinator, CoordinatorInput
from mnemos.core.linguist import JarvisLinguist
from mnemos.workers import FactRegistrar, InsightPhilosopher, Cartographer


async def run_iq_test():
    # 初始化：注入 Linguist（根据 insights 调整语气）、Registrar、Philosopher、Cartographer
    coord = MnemosCoordinator(
        linguist=JarvisLinguist(),
        registrar=FactRegistrar(),
        philosopher=InsightPhilosopher(),
        cartographer=Cartographer(),
        memos_client=None,
    )

    test_member = "user_wang_001"

    # --- DAY 1: 设定目标 ---
    print("\n--- DAY 1: Setting Goals ---")
    input_d1 = CoordinatorInput(
        message="从今天开始我要认真减肥了，计划每周跑三次步，晚上绝对不吃主食！",
        member_id=test_member,
    )
    res_d1 = await coord.process(input_d1)
    print(f"Registrar 提取事实数: {len(res_d1.facts or [])}")
    if res_d1.facts:
        for f in (res_d1.facts or [])[:5]:
            print(f"  事实: {f.get('entity')} {f.get('attribute')} {f.get('value')} [{f.get('context_tags')}]")

    # --- DAY 2: 家庭变动 + 首次违背 ---
    print("\n--- DAY 2: Family Change & First Slip ---")
    input_d2 = CoordinatorInput(
        message="昨天加班太累没去跑步，深夜还点了个烧烤... 对了，我爷爷下周要搬来跟我一起住，得收拾个房间出来。",
        member_id=test_member,
    )
    res_d2 = await coord.process(input_d2)
    print(f"Registrar 提取事实数: {len(res_d2.facts or [])}")
    for insight in res_d2.insights or []:
        if insight.get("tag") == "潜在的行为偏离":
            print(f"🚩 警报捕捉成功: {insight['insight']}")
        else:
            print(f"Insight: {insight.get('insight')} [Tag: {insight.get('tag')}]")

    # --- DAY 3: 连环违背 + 复杂社交（看 Linguist 是否因「行为偏离」加入关怀提醒）---
    print("\n--- DAY 3: Multi-slip & Social Context ---")
    input_d3 = CoordinatorInput(
        message="今天表妹找我吃火锅，又没运动。不过爷爷终于到了，家里现在好热闹啊。",
        member_id=test_member,
    )
    res_d3 = await coord.process(input_d3)
    print(f"Registrar 提取事实数: {len(res_d3.facts or [])}")
    print("\n🎙️ Day 3 回复（Linguist 根据 insights 调整语气，若有行为偏离应带关怀提醒）：")
    print(f"   「{res_d3.response}」")

    # 检查图谱更新
    updates = res_d3.graph_updates or []
    print(f"图谱更新条数: {len(updates)}")
    if updates:
        add_nodes = [u for u in updates if u.get("action") == "add_node"]
        add_edges = [u for u in updates if u.get("action") == "add_edge"]
        print(f"  本轮新增节点: {len(add_nodes)}, 新增边: {len(add_edges)}")

    # 检查最终反思是否成功关联了 Day 1 的减肥目标
    print("\n--- Final Insights (Day 3) ---")
    for insight in res_d3.insights or []:
        print(f"Final Insight: {insight['insight']} [Tag: {insight.get('tag')}]")

    # 可选：若注入了 Cartographer，可打印图规模
    if coord.cartographer and hasattr(coord.cartographer, "node_count"):
        print(f"\n图谱规模: 节点数={coord.cartographer.node_count()}, 边数={coord.cartographer.edge_count()}")


if __name__ == "__main__":
    asyncio.run(run_iq_test())
