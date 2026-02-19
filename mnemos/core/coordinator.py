"""
Mnemos 协调中枢 (Coordinator)

负责接收输入，利用 asyncio.gather 分发任务，协调各个 Worker 的异步执行。
遵循"主对话流与后台认知流分离"原则，确保低延迟响应。
事实提取完成后自动持久化到 MemOS（若已配置 MemOSClient）。
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from mnemos.core.model_config import get_model_config


@dataclass
class CoordinatorInput:
    """协调器输入数据结构"""
    message: str
    member_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class CoordinatorOutput:
    """协调器输出数据结构"""
    response: str
    facts: Optional[List[Dict[str, Any]]] = None
    insights: Optional[List[Dict[str, Any]]] = None
    psychologist_result: Optional[Dict[str, Any]] = None
    graph_updates: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class MnemosCoordinator:
    """
    异步协调中枢
    
    职责：
    1. 接收用户输入
    2. 使用 asyncio.gather 并发调度各个 Worker
    3. 确保主对话流（Linguist）优先返回，后台任务不阻塞
    """
    
    def __init__(
        self,
        linguist: Optional[Any] = None,
        registrar: Optional[Any] = None,
        philosopher: Optional[Any] = None,
        psychologist: Optional[Any] = None,
        cartographer: Optional[Any] = None,
        memos_client: Optional[Any] = None,
        model_config: Optional[Dict[str, str]] = None,
    ):
        """
        初始化协调器
        
        Args:
            linguist: 对话专家实例（负责快速生成回复）
            registrar: 事实记录员实例（异步提取实体与事实）
            philosopher: 反思哲学家实例（生成高阶 Insight）
            psychologist: 心理洞察专家实例（从长期事实推断核心价值观与行为动机，可选）
            cartographer: 制图师实例（负责图谱节点连线）
            memos_client: MemOS 客户端（事实提取后自动写入云端 LTM，可选）
            model_config: 各 Worker 模型 ID，如 {"linguist": "...", "philosopher": "...", "psychologist": "...", "registrar": "..."}。
                         未传时使用 get_model_config()（决策层 Claude 4.5，数据层 GPT-4.5）。
        """
        self.linguist = linguist
        self.registrar = registrar
        self.philosopher = philosopher
        self.psychologist = psychologist
        self.cartographer = cartographer
        self.memos_client = memos_client
        self.model_config = model_config if model_config is not None else get_model_config()
        
        # 后台任务追踪
        self._background_tasks: set = set()
        # 认知上下文注入：每成员近期「目标/计划」类事实缓存（MemOS 不可用时的回退）
        self._goal_plan_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._goal_plan_cache_max = 20
        self._goal_like_attributes = ("目标", "计划", "希望", "打算", "想要", "约定")
        # 长期事实缓存：供 Psychologist 按需使用（测试可经 context["long_term_facts_override"] 注入）
        self._long_term_facts_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._long_term_facts_cache_max = 80
    
    async def process(self, input_data: CoordinatorInput) -> CoordinatorOutput:
        """
        处理输入，协调各个 Worker 的异步执行
        
        执行流程：
        1. 立即启动 Linguist 生成回复（主对话流）
        2. 并发启动 Registrar、Cartographer 等后台任务（不阻塞对话）
        3. 使用 asyncio.gather 等待所有任务完成
        4. 返回结果
        
        Args:
            input_data: 协调器输入数据
            
        Returns:
            CoordinatorOutput: 协调器输出数据
        """
        # 验证 member_id（家庭场景隔离性要求）
        if not input_data.member_id:
            raise ValueError("member_id 是必需的，用于防止跨成员记忆污染")
        
        # 先跑 Registrar，再跑 Philosopher，以便 Linguist 能根据 insights 调整语气
        registrar_task = asyncio.create_task(
            self._run_registrar(input_data)
        )
        results = await asyncio.gather(registrar_task, return_exceptions=True)
        facts = results[0] if not isinstance(results[0], Exception) else None
        graph_updates = None
        
        # 事实持久化：将 Registrar 提取的事实自动写入 MemOS（后台执行，不阻塞）
        if self.memos_client and facts and getattr(self.memos_client, "is_configured", lambda: False)():
            persist_task = asyncio.create_task(
                self._persist_facts_to_memos(facts, input_data)
            )
            self._track_background_task(persist_task)
        
        # 认知上下文注入：在调用 Philosopher 前检索该成员的历史目标/计划（MemOS + 本地缓存）
        historical_facts = await self._gather_historical_goals_or_plans(input_data.member_id)
        if facts:
            self._update_goal_plan_cache(input_data.member_id, facts)
        
        # 反思触发：Philosopher 生成 insights（注入历史目标/计划以解决「看不到历史目标」）
        insights = None
        if self._should_trigger_philosopher(input_data, facts):
            try:
                insights = await self._run_philosopher(
                    input_data, facts, historical_facts=historical_facts,
                    model_override=self.model_config.get("philosopher"),
                )
            except Exception as e:
                logger.warning("Philosopher 执行失败: {}", e)
        
        # 反思环节按需调用 Psychologist：用长期事实推断核心价值观与行为动机
        psychologist_result = None
        long_term_facts = await self._get_long_term_facts_for_psychologist(input_data)
        if self.psychologist and long_term_facts and self._should_trigger_psychologist(long_term_facts):
            try:
                psychologist_result = await self._run_psychologist(
                    input_data, long_term_facts,
                    model_override=self.model_config.get("psychologist"),
                )
            except Exception as e:
                logger.warning("Psychologist 执行失败: {}", e)
        if facts:
            self._update_long_term_facts_cache(input_data.member_id, facts)
        
        # 对话专家：传入 insights 与 psychologist_result，以便深度共情与反思型回答
        try:
            response = await self._run_linguist(
                input_data, facts=facts, insights=insights, psychologist_result=psychologist_result,
                model_override=self.model_config.get("linguist"),
            )
        except Exception as e:
            logger.warning("Linguist 执行失败: {}", e)
            response = "抱歉，生成回复时出现错误"
        
        # 制图师：用 facts 与 insights 织网，确保每次交互后的记忆在图谱中有位置
        try:
            graph_updates = self._run_cartographer(input_data, facts, insights)
        except Exception as e:
            logger.warning("Cartographer 更新图谱失败: {}", e)
            graph_updates = None
        
        return CoordinatorOutput(
            response=response,
            facts=facts,
            insights=insights,
            psychologist_result=psychologist_result,
            graph_updates=graph_updates,
            metadata={
                "member_id": input_data.member_id,
                "timestamp": datetime.now().isoformat(),
                "session_id": input_data.session_id
            }
        )
    
    async def _get_long_term_facts_for_psychologist(self, input_data: CoordinatorInput) -> List[Dict[str, Any]]:
        """
        获取供 Psychologist 使用的长期事实。
        优先使用 context 中的 long_term_facts_override（模拟/测试用），否则用本地缓存 + 可选 MemOS。
        """
        ctx = input_data.context or {}
        override = ctx.get("long_term_facts_override")
        if override is not None and isinstance(override, list) and len(override) > 0:
            return override
        member_id = input_data.member_id
        out: List[Dict[str, Any]] = []
        if self.memos_client and getattr(self.memos_client, "search_memories", None):
            if getattr(self.memos_client, "is_configured", lambda: False)():
                try:
                    from_memos = await self.memos_client.search_memories(
                        member_id, query="", limit=30
                    )
                    out.extend(from_memos)
                except Exception as e:
                    logger.debug("MemOS 检索长期事实失败: {}", e)
        cached = self._long_term_facts_cache.get(member_id, [])
        for f in cached:
            if isinstance(f, dict) and f not in out:
                out.append(f)
        return out[: self._long_term_facts_cache_max]

    def _update_long_term_facts_cache(self, member_id: str, facts: List[Dict[str, Any]]) -> None:
        """将本批事实追加到该成员的长期事实缓存，供 Psychologist 使用。"""
        for f in facts:
            if not isinstance(f, dict):
                continue
            self._long_term_facts_cache.setdefault(member_id, [])
            self._long_term_facts_cache[member_id].append({
                "entity": f.get("entity", ""),
                "attribute": f.get("attribute", ""),
                "value": f.get("value", ""),
                "context_tags": f.get("context_tags", []),
            })
        if member_id in self._long_term_facts_cache:
            self._long_term_facts_cache[member_id] = self._long_term_facts_cache[member_id][
                -self._long_term_facts_cache_max :
            ]

    def _should_trigger_psychologist(self, long_term_facts: List[Dict[str, Any]]) -> bool:
        """按需触发 Psychologist：当存在足够长期事实时触发。"""
        return len(long_term_facts) >= 5

    async def _run_psychologist(
        self,
        input_data: CoordinatorInput,
        long_term_facts: List[Dict[str, Any]],
        model_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """运行 Psychologist，返回 core_values 与 behavioral_motivations。"""
        if not self.psychologist or not hasattr(self.psychologist, "infer_values_and_motivations"):
            return None
        return await self.psychologist.infer_values_and_motivations(
            member_id=input_data.member_id,
            long_term_facts=long_term_facts,
            message=input_data.message,
            model_override=model_override,
        )

    async def _run_linguist(
        self,
        input_data: CoordinatorInput,
        facts: Optional[List[Dict[str, Any]]] = None,
        insights: Optional[List[Dict[str, Any]]] = None,
        psychologist_result: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        运行 Linguist（对话专家）。传入 facts、insights 与 psychologist_result，
        以便根据「潜在的行为偏离」调整语气，并根据心理洞察做深度共情回答。
        """
        context = dict(input_data.context or {})
        context["facts"] = facts or []
        context["insights"] = insights or []
        context["psychologist_result"] = psychologist_result
        if self.linguist and hasattr(self.linguist, "generate_response"):
            return await self.linguist.generate_response(
                message=input_data.message,
                member_id=input_data.member_id,
                context=context,
                model_override=model_override,
            )
        return f"[Linguist 占位] 收到消息: {input_data.message}"
    
    async def _run_registrar(self, input_data: CoordinatorInput) -> Optional[List[Dict[str, Any]]]:
        """
        运行 Registrar（事实记录员）
        
        异步提取实体与事实（Mem0 逻辑），不得阻塞对话。
        
        Args:
            input_data: 输入数据
            
        Returns:
            Optional[List[Dict[str, Any]]]: 提取的事实列表，包含 confidence_score
        """
        if self.registrar:
            # 如果 Registrar 已实现，调用其方法
            if hasattr(self.registrar, 'extract_facts'):
                return await self.registrar.extract_facts(
                    text=input_data.message,
                    member_id=input_data.member_id,
                    model_override=self.model_config.get("registrar"),
                )
        
        # 占位实现：返回空列表
        return []
    
    async def _gather_historical_goals_or_plans(self, member_id: str) -> List[Dict[str, Any]]:
        """
        为该成员聚合「历史目标/计划」类事实，供 Philosopher 做场景一致性检查。
        优先从 MemOS 检索（#目标/#计划 或语义 query），再合并本地缓存。
        """
        historical: List[Dict[str, Any]] = []
        if self.memos_client and getattr(self.memos_client, "search_memories", None):
            if getattr(self.memos_client, "is_configured", lambda: False)():
                try:
                    from_memos = await self.memos_client.search_memories(
                        member_id,
                        query="目标 计划 想要 打算 希望 约定",
                        limit=10,
                    )
                    historical.extend(from_memos)
                except Exception as e:
                    logger.debug("MemOS 检索历史目标失败: {}", e)
        cached = self._goal_plan_cache.get(member_id, [])
        for f in cached:
            if f not in historical and isinstance(f, dict):
                historical.append(f)
        return historical[:15]

    def _update_goal_plan_cache(self, member_id: str, facts: List[Dict[str, Any]]) -> None:
        """从本批 facts 中筛出目标/计划类事实，追加到该成员的本地缓存。"""
        for f in facts:
            if not isinstance(f, dict):
                continue
            attr = (f.get("attribute") or "").strip()
            if not attr:
                continue
            if attr not in self._goal_like_attributes and not any(k in attr for k in self._goal_like_attributes):
                continue
            self._goal_plan_cache.setdefault(member_id, [])
            self._goal_plan_cache[member_id].append({
                "entity": f.get("entity", ""),
                "attribute": f.get("attribute", ""),
                "value": f.get("value", ""),
                "context_tags": f.get("context_tags", []),
            })
        if member_id in self._goal_plan_cache:
            self._goal_plan_cache[member_id] = self._goal_plan_cache[member_id][-self._goal_plan_cache_max :]

    async def _run_philosopher(
        self,
        input_data: CoordinatorInput,
        facts: Optional[List[Dict[str, Any]]] = None,
        historical_facts: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        运行 Philosopher（反思哲学家），注入历史目标/计划以支持跨轮一致性检查。
        """
        if self.philosopher and hasattr(self.philosopher, "generate_insights"):
            return await self.philosopher.generate_insights(
                message=input_data.message,
                member_id=input_data.member_id,
                facts=facts,
                historical_facts=historical_facts or [],
                model_override=model_override,
            )
        return []
    
    def _run_cartographer(
        self,
        input_data: CoordinatorInput,
        facts: Optional[List[Dict[str, Any]]],
        insights: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        运行 Cartographer（制图师）：将 facts 与 insights 织入图谱。
        由 process() 在获得 facts 与 insights 后同步调用。
        """
        if not self.cartographer or not hasattr(self.cartographer, "update_graph"):
            return None
        return self.cartographer.update_graph(
            facts=facts or [],
            insights=insights or [],
            member_id=input_data.member_id,
        )
    
    def _should_trigger_philosopher(
        self,
        input_data: CoordinatorInput,
        facts: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """
        判断是否需要触发 Philosopher（反思触发机制）
        
        触发器示例：
        - 发现情感波动
        - 检测到生活变动
        - 重要事实提取成功
        
        Args:
            input_data: 输入数据
            facts: 已提取的事实列表
            
        Returns:
            bool: 是否需要触发 Philosopher
        """
        # 占位实现：简单判断是否有事实提取
        if facts and len(facts) > 0:
            # 检查是否有高置信度的事实
            high_confidence_facts = [
                f for f in facts 
                if isinstance(f, dict) and f.get('confidence_score', 0) > 0.8
            ]
            return len(high_confidence_facts) > 0
        
        return False
    
    async def _persist_facts_to_memos(
        self,
        facts: List[Dict[str, Any]],
        input_data: CoordinatorInput,
    ) -> None:
        """
        将提取的事实持久化到 MemOS 云端（LTM）。
        在后台任务中执行，不阻塞主对话流。
        """
        if not facts or not hasattr(self.memos_client, "add_memories"):
            return
        try:
            count = await self.memos_client.add_memories(
                facts,
                member_id=input_data.member_id,
                conversation_id=input_data.session_id,
            )
            if count > 0:
                logger.debug("MemOS 持久化事实数 member_id={} count={}", input_data.member_id, count)
        except Exception as e:
            logger.warning("MemOS 持久化失败: {}", e)

    def _track_background_task(self, task: asyncio.Task) -> None:
        """
        追踪后台任务，确保任务完成
        
        Args:
            task: 后台任务
        """
        self._background_tasks.add(task)
        
        # 任务完成后自动清理
        task.add_done_callback(self._background_tasks.discard)
    
    async def shutdown(self) -> None:
        """
        关闭协调器，等待所有后台任务完成
        
        用于优雅关闭，确保所有后台任务都完成后再退出。
        """
        if self._background_tasks:
            # 等待所有后台任务完成
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
