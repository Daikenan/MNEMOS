# Mnemos 项目开发进度看板

## 🚀 当前状态
- **环境**: uv + Python 3.12+
- **核心理论**: Mem0, GraphRAG, Generative Agents, Zep, Memary
- **基础设施**: MemOS Cloud 已对接（MemOSClient + Coordinator 持久化挂载）
- **项目状态**: 点（事实）线（洞察）网（图谱）就绪；Cartographer 本地 MultiDiGraph + JSON/GraphML 持久化

## 🛠️ 任务清单 (Roadmap)

### 第零阶段：理论准备 (Preparation)
- [x] **Task 0: 理论资料与链接对照整理**
    - [x] 建立知识库引用规范
    - [x] 完成理论参考映射（Mem0、Generative Agents、GraphRAG、Zep）

### 第一阶段：核心骨架搭建 (Backbone)
- [x] **Task 1: 异步协调中枢 (Coordinator)**
    - [x] 定义 `MnemosCoordinator` 类
    - [x] 实现 `asyncio.gather` 并发调度
    - [x] 编写 Linguist/Registrar/Cartographer 的占位接口
- [x] **Task 2: 基础依赖安装**
    - [x] `uv add httpx pydantic loguru python-dotenv`

### 第二阶段：记忆摄入流水线 (Ingestion)
- [x] **Task 3: 事实记录员 (Registrar)**
    - [x] 编写结构化事实提取 Prompt（含场景分类 #健康/#家庭旅行/#成长）
    - [x] 实现异步提取逻辑（FactRegistrar + Claude API + context_tags + JSON 校验）
- [x] **Task 4: MemOS 云端对接**
    - [x] 实现 `MemOSClient`（`mnemos/storage/memos_client.py`）
    - [x] 对接 `add_message` 接口（context_tags→tags，entity/attribute→metadata）
    - [x] Coordinator 后台任务挂载，事实提取后自动持久化到 MemOS

### 第三阶段：反思与图谱 (Cognitive & Structure)
- [x] **Task 5: 哲学家 (Philosopher) 反思引擎**
    - [x] 异步 Insights 提取逻辑（InsightPhilosopher + OpenRouter）
    - [x] generate_insights Prompt 含「场景一致性检查」：近期行为与长期目标不一致时标记为「潜在的行为偏离」
- [x] **Task 6: 制图师 (Cartographer) 关系建模**
    - [x] 基于 NetworkX 的 MultiDiGraph（entity/value 为节点，attribute 为有向边 relation_type）
    - [x] `update_graph(facts, insights, member_id)`：织入事实与洞察，insights 挂到节点高阶属性
    - [x] `save_graph()` / `load_graph()`：JSON（node-link）与 GraphML 持久化
    - [x] Coordinator 在获得 facts 与 insights 后调用制图师，每次交互后记忆入网
- [x] **Task 7: 认知上下文注入 (Memory Re-injection)**
    - [x] MemOSClient.search_memories(member_id, query=, tags=, limit=)：按 member 与语义/标签检索历史记忆，解析为 fact 列表
    - [x] Coordinator 在调用 Philosopher 前：_gather_historical_goals_or_plans（MemOS 检索 + 本地目标/计划缓存），并传入 historical_facts
    - [x] 本地 _goal_plan_cache：从本批 facts 中筛出 attribute 含 目标/计划/希望/打算/想要/约定 的事实，供 MemOS 不可用时的回退
    - [x] Philosopher.generate_insights(..., historical_facts=)：Prompt 中增加「已知长期目标/计划（来自历史记忆）」区块，支持跨轮一致性检查
    [x] Task 8: Linguist 语言学家 (Jarvis 人格) (基于洞察生成带有关怀提醒的温和回复)

### 第四阶段: 服务化与端侧接入 (进行中 - 核心任务)
[ ] Task 9: Mnemos API 服务化 (Server-side)

[ ] 使用 FastAPI 封装 MnemosCoordinator。

[ ] 接口设计：/chat 接收消息，返回 reply 和 deviation_flag。

[ ] 异常处理：模型请求失败时的回退机制（Fallback）。

[ ] Task 10: Android Root 手表端接入 (Client-side)

[ ] 编写轻量级 Shell 脚本 (mnemos_client.sh) 用于测试 API。

[ ] (可选) 利用 Root 权限在手表端实现语音采集上传。

[ ] (进阶) 针对手表窄屏优化 Linguist 的回复长度。

### 第五阶段: 进化与工程化 (待启动 - 长期目标)
[ ] Task 11: 数据工厂 (Data Collector)

[ ] 自动记录 (Input, Facts, Insights, Output) 用于未来 10B 模型微调。

[ ] Task 12: 图谱深度增强 (GraphRAG)

[ ] 让 Linguist 具备“多跳”查询能力（例如由“爷爷”联想到“搬家计划”）。

[ ] Task 13: 性能优化

[ ] 优化 Philosopher 的检索算法，减少 Token 消耗。

---

## 📝 开发笔记
- 所有的 Worker 必须保持异步非阻塞。
- 检索权重公式：Similarity, Importance, Recency。