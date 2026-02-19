# Mnemos 逻辑完整性审视：复杂场景推演与模型依赖分析

## 一、测试场景设计

**设定**：一位用户（member_id = `user_health`) 在三天内与家庭记忆系统对话，表现出：
1. **健康承诺的违背**：先表达减肥/运动目标，随后行为与之相悖；
2. **家庭结构变化**：提到两位新家庭成员（爷爷搬来同住、表妹来玩）。

| 时间 | 用户消息（简化） |
|------|------------------|
| **Day 1** | 「我今年一定要减肥，打算每周跑步三次，晚上少吃主食。」 |
| **Day 2** | 「昨天没跑成，加班到很晚就点了宵夜。爷爷下周从老家搬来跟我们住，得收拾一间房。」 |
| **Day 3** | 「今天又没运动，中午和表妹吃了火锅。爷爷到了，家里热闹不少。」 |

---

## 二、逐轮推演：各模块如何处理

### 2.1 整体数据流（单轮）

```
用户消息 → Coordinator.process()
    ├─ Linguist (并发)     → response
    ├─ Registrar (并发)    → facts
    ├─ [若有 facts 且 MemOS 配置] → 后台 MemOS add_memories(facts)
    ├─ _should_trigger_philosopher(facts)? → 若 True：Philosopher → insights
    ├─ Cartographer.update_graph(facts, insights, member_id) → graph_updates
    └─ 返回 CoordinatorOutput(response, facts, insights, graph_updates)
```

以下按**单轮**说明各模块的输入/输出与逻辑要点。

---

### 2.2 Day 1：用户表达健康承诺

**输入**：`message = "我今年一定要减肥，打算每周跑步三次，晚上少吃主食。"`, `member_id = "user_health"`

| 模块 | 输入 | 处理逻辑 | 输出（示例） |
|------|------|----------|--------------|
| **Coordinator** | CoordinatorInput(message, member_id) | 并发跑 Linguist + Registrar；无 MemOS 时跳过持久化；本轮有高置信度事实 → 触发 Philosopher；最后跑 Cartographer | - |
| **Linguist** | message, member_id | 占位或调用 generate_response；当前为占位 | `response = "[Linguist 占位] 收到消息: ..."` |
| **Registrar** | text=message, member_id | 调用 OpenRouter，用 REGISTRAR_SYSTEM_PROMPT 做实体-属性-值抽取 + context_tags (#健康) + confidence | **facts** 示例：<br>• entity=我, attribute=目标/计划, value=减肥, context_tags=[#健康], confidence≈0.9<br>• entity=我, attribute=运动计划, value=每周跑步三次, context_tags=[#健康], confidence≈0.85<br>• entity=我, attribute=饮食计划, value=晚上少吃主食, context_tags=[#健康], confidence≈0.85 |
| **MemOS** | facts, member_id | 若配置：每条 fact 转 add_memory（tags=context_tags, metadata=entity/attribute/confidence） | LTM 中写入 3 条记忆 |
| **Philosopher 触发** | - | `_should_trigger_philosopher`: facts 非空且存在 confidence_score > 0.8 → **True** | 进入 Philosopher |
| **Philosopher** | message, member_id, **facts=本轮 3 条** | 将 facts 转成「近期事实」文本；模型推断长期目标（减肥、运动、饮食控制）；本轮事实均为「目标/计划」→ 与目标一致 | **insights** 示例：<br>• insight="用户设定了明确的减肥与运动计划，与健康目标一致。", tag=null, related_goals=[减肥] |
| **Cartographer** | facts, insights, member_id | 节点：我、减肥、每周跑步三次、晚上少吃主食；事实边（目标、运动计划、饮食计划）；同会话 + 同 #健康 共现边权重 +1；insights 挂到上述节点 | graph_updates: add_node/add_edge/attach_insights/increment_weight |

**Day 1 小结**：目标类事实被正确抽取并打上 #健康；Philosopher 仅看到「计划」，判为一致；图中出现「我」与多个健康相关实体及共现边。

---

### 2.3 Day 2：违背 + 新成员（爷爷）

**输入**：`message = "昨天没跑成，加班到很晚就点了宵夜。爷爷下周从老家搬来跟我们住，得收拾一间房。"`

| 模块 | 处理要点 | 输出（示例） |
|------|----------|--------------|
| **Registrar** | 从一句中同时抓「行为」与「家庭」 | **facts**：<br>• 我 - 未执行/行为 - 昨天没跑步 [#健康], confidence≈0.85<br>• 我 - 行为 - 加班到很晚 [#日常] 等<br>• 我 - 行为 - 点了宵夜 [#健康]<br>• 爷爷 - 计划/安排 - 下周从老家搬来同住 [#家庭]/#日常<br>• 我家 - 待办 - 收拾一间房 等 |
| **MemOS** | 同上，按条写入 | 新记忆与 Day 1 的「目标」在 LTM 中共存（当前无跨轮检索注入） |
| **Philosopher** | **仅接收本轮的 facts**，不包含 Day 1 的「减肥/跑步计划」 | 若本轮事实里没有再次出现「目标=减肥」，模型只能从**本批**推断目标；可能推断出「规律运动」「健康饮食」并看到「没跑步」「宵夜」→ **insight** 可标记为 "潜在的行为偏离"，related_goals=[减肥/规律运动]；**依赖本批是否足够表达目标** |
| **Cartographer** | 新增节点：爷爷、老家、一间房 等；边：我→爷爷（家庭关系）、我→宵夜 等；同会话 + 同 #健康 共现加强 | 图中「我」与「爷爷」等建立联系；共现边权重累加 |

**Day 2 关键点**：  
- 行为偏离能否被标记，取决于 **Philosopher 本批事实** 里是否包含或隐含「长期目标」。当前实现**不**把 MemOS 历史或上一轮目标注入 Philosopher，因此若 Day 2 的 Registrar 没有抽到「用户有减肥目标」类事实，Philosopher 可能只能从「宵夜」「没跑步」推断目标再判偏离，或漏判。  
- 爷爷作为新实体被 Registrar 抽出并进入图谱，家庭关系得以扩展。

---

### 2.4 Day 3：再次违背 + 表妹出现

**输入**：`message = "今天又没运动，中午和表妹吃了火锅。爷爷到了，家里热闹不少。"`

| 模块 | 处理要点 | 输出（示例） |
|------|----------|--------------|
| **Registrar** | 「又没运动」「和表妹吃火锅」「爷爷到了」 | **facts**：<br>• 我 - 行为 - 今天又没运动 [#健康]<br>• 我 - 与…一起/行为 - 和表妹吃火锅 [#健康]/#家庭<br>• 表妹 - 出现/关系 - 来玩 等<br>• 爷爷 - 状态 - 到了/已搬来 [#家庭]<br>• 家里 - 状态 - 热闹 等 |
| **Philosopher** | 仍仅本批事实 | 若本批中有「目标」类事实可推断，或模型从「又没运动」「火锅」反推目标，可再次输出 "潜在的行为偏离"；表妹、爷爷作为家庭成员被纳入上下文 |
| **Cartographer** | 新节点：表妹、火锅、家里 等；我↔表妹、爷爷 等共现与事实边；同会话、同 #健康/#家庭 共现边权重再增 | 三天下来，「我」与 减肥、跑步、宵夜、火锅、爷爷、表妹 等形成多跳与加权共现，可支撑「谁和谁常一起出现」的联想 |

**Day 3 小结**：  
- 新成员（表妹）和已有成员（爷爷）的状态变化被记录；  
- 健康承诺违背在**单轮**内可被识别，但跨多天的「承诺 vs 连续违背」的连贯性，仍受限于 Philosopher 只看到当日事实。

---

## 三、当前逻辑下的缺口与假设

1. **Philosopher 仅用本轮 facts**  
   - 长期目标若只在 Day 1 说过，Day 2/3 的 Philosopher 看不到 Day 1 的 MemOS 记忆。  
   - 若要「跨会话一致性检查」，需要：在调用 Philosopher 前，从 MemOS（或 LTM）按 member_id 检索与当前事实相关的历史事实/目标，拼进 Philosopher 的「近期事实」或单独「已知目标」字段。

2. **Registrar 的实体与关系质量**  
   - 新家庭成员（爷爷、表妹）能否稳定被识别为 entity，以及关系（与…同住、与…吃饭）是否落在 attribute/value 上，直接决定图谱与洞察质量。  
   - 若「我」有时被抽成「用户」或省略，图谱中可能出现多个等价节点，需后续做实体对齐或归一化。

3. **Cartographer 不区分子图**  
   - 当前按 member_id 在边/点上打标，但同一图中多成员混在一起；若要做「仅看某成员相关子图」的检索，需在查询时按 member_id 过滤。

4. **Linguist 未用记忆**  
   - 回复未接入 MemOS/图谱，因此不会出现「您上次说打算每周跑三次……」这类基于记忆的回应；若要，需在 Linguist 的 context 中注入检索结果。

---

## 四、哪一步最依赖高性能模型（微调重点）

| 模块 | 模型角色 | 难度与瓶颈 |
|------|----------|------------|
| **Registrar** | 实体-属性-值 + 场景标签 + 置信度 | **结构化抽取**：输出格式固定，易用 JSON/function calling 约束；难点在于边界 case（隐含主体、代词指代、多义「目标」）。适合用**数据微调**提升 entity/attribute/value 的稳定性和 context_tags 一致性。 |
| **Philosopher** | 推断长期目标 + 判断行为与目标是否一致 + 生成洞察并打 tag | **推理与判断**：需要从零散事实中归纳「目标」、再对「行为」做一致性判断，并生成自然语言洞察。当前 Prompt 已明确规则，但「目标推断」和「是否偏离」仍依赖模型的世界知识与推理能力，且**仅能看到本批事实**，信息不完整时易漏判或误判。 |

**结论**：  
**最依赖高性能模型、且最值得作为微调重点的，是 Philosopher（反思哲学家）。**

理由简述：

1. **任务本质**：做的是「目标推断 + 一致性判断 + 自然语言概括」，属于高层认知任务，对推理质量和稳定性要求高。  
2. **错误代价**：漏标「潜在的行为偏离」会削弱家庭记忆的提醒/反思价值；误标则可能造成不当干预或用户反感。  
3. **数据可得性**：可基于「事实列表 + 人工标注的 insight/tag/related_goals」构造微调数据，或从对话日志中筛出「明显违背/明显一致」的片段做弱监督。  
4. **与架构的契合**：Registrar 输出已是结构化事实，可作为 Philosopher 的稳定输入；微调后的 Philosopher 可在不改变接口的前提下，提升「场景一致性检查」的准确率与可解释性。

**建议的微调方向**：  
- 收集/构造 (facts, insights, tag, related_goals) 配对数据，尤其包含「目标仅在历史会话出现」「多轮行为与单次承诺」的样本。  
- 若后续将 MemOS 历史注入 Philosopher，微调数据中应包含「当前事实 + 历史目标/历史事实 → insight/tag」的样本，以训练模型在信息更完整时的表现。

---

## 五、场景验证清单（可落地的测试用例）

可用以下清单在集成/端到端测试中验证逻辑完整性：

- [ ] **Day 1**：Registrar 输出中至少包含 1 条「目标/计划」类事实且带 #健康；Philosopher 被触发且 insight 的 tag 非 "潜在的行为偏离"；图中存在节点「我」及与减肥/跑步/主食相关的边。  
- [ ] **Day 2**：Registrar 输出中包含「没跑步」「宵夜」及「爷爷」「搬来」相关事实；Philosopher 若本批含目标或可推断目标，至少 1 条 insight 的 tag 为 "潜在的行为偏离"；图中出现节点「爷爷」及与「我」的边。  
- [ ] **Day 3**：Registrar 输出中包含「没运动」「表妹」「火锅」「爷爷到了」；图中出现「表妹」；共现边（如 我–表妹、我–火锅）权重随轮次增加。  
- [ ] **跨轮**（若已实现 MemOS→Philosopher 注入）：用 Day 1 目标 + Day 2/3 事实一起喂给 Philosopher，应稳定得到 "潜在的行为偏离" 类 insight。

上述文档可直接用于评审、测试设计以及后续 MemOS 检索注入与 Philosopher 微调规划。
