# KnowMe-Bench（PersonaMem）本地复刻测试环境步骤

本文档列出在本地复刻 **Know Me, Respond to Me**（PersonaMem / KnowMe-Bench）测试环境、并对接 Mnemos 记忆流水线所需的步骤。

---

## 一、KnowMe-Bench 与 Mnemos 逻辑对照

### 1.1 三级评估框架（PersonaMem）

| 层级 | 含义 | 典型 Query 类型 | 对应 Mnemos 模块 |
|------|------|----------------|------------------|
| **Fact Retrieval** | 回忆用户共享事实、承认最新偏好 | Recall user shared facts, Acknowledge latest user preference | Registrar + MemOS/LTM 检索 |
| **Temporal Reasoning** | 跟踪偏好演变、重访偏好更新原因 | Track full preference evolution, Revisit reasons behind preference updates | 跨轮 historical_facts 注入 + Philosopher |
| **Psychological Insight** | 在新场景中泛化原因、提出新想法 | Generalize to new scenarios, Suggest new ideas | **Psychologist**（核心价值观 / 行为动机推断） |

### 1.2 与 logic_integrity_scenario 的对比

- **logic_integrity_scenario.md**：侧重单轮/跨轮「目标 vs 行为」一致性检查（Philosopher 行为偏离标记）。
- **KnowMe-Bench**：侧重「用户画像内部化 → 偏好演变追踪 → 个性化回应」的端到端评测；心理洞察对应我们新增的 **Psychologist** 从长期事实推断核心价值观与行为动机。

---

## 二、本地复刻 KnowMe-Bench 测试环境所需步骤

### 步骤 1：克隆 PersonaMem 官方仓库

```bash
git clone https://github.com/bowen-upenn/PersonaMem.git
cd PersonaMem
```

### 步骤 2：创建虚拟环境并安装依赖

项目使用 `uv` 时，可在 Mnemos 外单独为 PersonaMem 建环境：

```bash
# 若使用 venv
python -m venv .venv_personamem
source .venv_personamem/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

若需跑 **Gemini** 模型（与 OpenAI 依赖冲突），建议用 Conda：

```bash
conda create -n persona_mem python=3.9
conda activate persona_mem
pip install -r requirements.txt
pip install -q -U google-genai
```

### 步骤 3：准备 API Keys

在 PersonaMem 项目根目录下创建 `api_tokens/` 目录，并按需创建下列文件并写入对应 API Key（纯文本）：

- `openai_key.txt` — OpenAI 模型
- `gemini_key.txt` — Google Gemini
- `claude_key.txt` — Anthropic Claude
- `lambda_key.txt` — Lambda Cloud（如 Llama、DeepSeek 等）

### 步骤 4：下载 Benchmark 数据（HuggingFace）

PersonaMem 数据托管在 HuggingFace：

- 数据集：<https://huggingface.co/datasets/bowen-upenn/PersonaMem>

可按上下文长度选择：

- **32k**：`questions_32k.csv`，`shared_contexts_32k.jsonl`
- **128k**：`questions_128k.csv`，`shared_contexts_128k.jsonl`
- **1M**：`questions_1M.csv`，`shared_contexts_1M.jsonl`

使用 `datasets` 或网页下载后，将 CSV/JSONL 放到 PersonaMem 的 `data/` 下（或按仓库 README 指定路径配置）。

### 步骤 5：运行官方推理脚本（复现榜单）

在 PersonaMem 目录下执行对应模型的脚本，例如：

```bash
# 以 GPT-4o 为例
bash scripts/inference_gpt_4o.sh
```

脚本内可修改 `BENCHMARK_SIZE` 为 `32k`、`128k` 或 `1M`。结果会写入 `data/results/`。

### 步骤 6：理解评测输入输出格式（便于对接 Mnemos）

- **输入**：每条样本包含 `persona_id`、`user_question_or_message`、`all_options`（多选）、`correct_answer`，以及通过 `shared_context_id` + `end_index_in_shared_context` 从 `shared_contexts_*.jsonl` 切片得到的**长对话上下文**。
- **评测方式**：给定上下文 + 当前用户 in-situ 问题，模型从 `all_options` 中选出最佳回复；与 `correct_answer` 对比算准确率。
- **7 类 Query**：Recall user shared facts, Suggest new ideas, Acknowledge latest user preference, Track full preference evolution, Revisit reasons behind preference updates, Provide preference aligned recommendations, Generalize to new scenarios.

### 步骤 7：将 Mnemos 接入 PersonaMem 评测（可选）

若要使用 Mnemos 的「Registrar → MemOS → Philosopher/Psychologist」流水线参与评测，需要：

1. **上下文注入**：用 PersonaMem 的 `shared_contexts_*` 作为「历史对话」，按会话顺序逐条经过 Coordinator，将产生 facts 写入 MemOS（或本地 LTM 模拟）。
2. **检索与反思**：在回答某条 `user_question_or_message` 前，用 MemOS 检索该 persona 的历史事实；调用 Philosopher（场景一致性）+ Psychologist（核心价值观/行为动机）生成洞察。
3. **回答生成**：Linguist 在 context 中注入检索到的 facts + insights + psychologist 输出，生成回复；再将该回复与 `all_options` 做匹配（或用 PersonaMem 的 generative 评测接口，用联合序列概率选最佳选项）。
4. **评估脚本**：可 fork PersonaMem 的 `inference.py`，将其中 `query_llm` 改为：先跑 Mnemos 的 Coordinator（或仅 Registrar + MemOS + Philosopher + Psychologist），再调用 Linguist 或外部 LLM 生成答案，最后与原版一样汇总准确率。

### 步骤 8：（可选）自建数据与 pipeline

若要从头生成 Persona 与多轮对话（而非仅跑现成 benchmark）：

1. 下载 [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub)，放到 `data/source/Persona_Hub_20000.jsonl`。
2. 运行对话生成：`bash scripts/run_all_prepare_data.sh`（可指定 `--model`、`--topics`、`--n_persona` 等）。
3. 生成 QA 对：`bash scripts/run_all_prepare_qa.sh`。
4. 构建长上下文：`bash scripts/run_generate_benchmark.sh large`（small/medium/large 对应 32k/128k/1M）。

---

## 三、Mnemos 侧已做的对齐

- **Philosopher**：继续负责「目标推断 + 行为一致性检查」，对应 Temporal Reasoning 中的「偏好与行为是否一致」。
- **Psychologist**（`mnemos/workers/reflector.py`）：从长期事实推断「核心价值观」与「行为动机」，专门服务 **Psychological Insight**（Generalize to new scenarios, Suggest new ideas），便于后续在生成回复时注入价值/动机信息以冲击心理洞察类题目。
- **Coordinator**：已支持 `historical_facts` 注入 Philosopher；后续可增加「按需调用 Psychologist」的触发逻辑（如定时或按会话数），并将 `core_values` / `behavioral_motivations` 写入 context 供 Linguist 使用。

---

## 四、简要检查清单

- [ ] 克隆 PersonaMem 并安装依赖
- [ ] 配置 `api_tokens/` 下的 API keys
- [ ] 下载 HuggingFace 上的 PersonaMem 数据到本地
- [ ] 成功跑通至少一个 `scripts/inference_*.sh` 并得到 `data/results/` 结果
- [ ] （可选）在 Mnemos 中为 Coordinator 增加 Psychologist 调用与 context 注入
- [ ] （可选）编写 Mnemos 版 inference 脚本，输出与 PersonaMem 题目格式兼容的答案并计算准确率

完成以上步骤后，即可在本地复刻 KnowMe-Bench 测试环境，并在此基础上对接 Mnemos 记忆与反思流水线冲击榜单。
