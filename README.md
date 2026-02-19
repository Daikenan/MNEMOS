# Mnemos

类人记忆系统，具备异步协作能力（Coordinator / Linguist / Registrar / Philosopher / Psychologist / Cartographer），支持 KnowMe-Bench 打榜评测。

---

## 环境与依赖

- **Python**: ≥ 3.12  
- **包管理**: 使用 `uv`，勿用 `pip install`。

```bash
# 克隆后
cd Mnemos
uv sync
```

环境变量：复制 `env.example` 为 `.env` 并填入真实值。

```bash
cp env.example .env
```

| 变量 | 说明 |
|------|------|
| `OPENROUTER_API_KEY` | 决策层、数据层与**判分器** LLM 调用（必填）；仅此一项即可完成打榜与判分 |
| `OPENAI_API_KEY` | KnowMe-Bench 判分器用（可选）；不设则自动用 OpenRouter 判分（openai/gpt-4o） |
| `MEMOS_API_KEY` / `MEMOS_BASE_URL` | MemOS 云端 LTM（可选） |

判分步骤需可 `import openai`，若未安装：

```bash
uv add openai tqdm
```

---

## 模型配置（混合专家）

- **决策层**（Philosopher, Psychologist, Linguist）：默认 **Claude 4.5**（`anthropic/claude-4.5-opus-20251124`），侧重共情与推理。  
- **数据层**（Registrar）：默认 **gpt-4o**（`openai/gpt-4o`），侧重结构化事实抽取（OpenRouter 可用；若使用其他网关可改）。

通过环境变量覆盖各 Worker 模型：

```bash
MNEMOS_MODEL_LINGUIST=...
MNEMOS_MODEL_PHILOSOPHER=...
MNEMOS_MODEL_PSYCHOLOGIST=...
MNEMOS_MODEL_REGISTRAR=...   # 例如 openai/gpt-4o
```

配置逻辑见 `mnemos/core/model_config.py`。

---

## KnowMe-Bench 打榜操作手册

### 1. 准备评测仓

若尚未克隆 KnowMe-Bench 官方仓，在项目根执行：

```bash
git clone https://github.com/QuantaAlpha/KnowMeBench
```

数据与评测脚本将位于 `KnowMeBench/KnowmeBench/dataset1` 与 `KnowMeBench/evaluate/`。

### 2. 试跑（每任务少量题）

建议先试跑，确认环境与模型均正常：

```bash
uv run python scripts/run_benchmarking.py --use_registrar --max_per_task 2
```

- `--use_registrar`：用 Registrar 将上下文转为 Facts，再交给 Psychologist / Linguist。  
- `--max_per_task 2`：每个任务只跑 2 题，便于快速验证。

### 3. 全量打榜

不限制题数即跑全部 7 个任务、全量题目：

```bash
uv run python scripts/run_benchmarking.py --use_registrar
```

脚本会依次：

1. 从 `KnowmeBench/dataset1/question/` 与 `input/` 加载题目与上下文（**内心独白**在上下文中优先排前）；  
2. 对每题：Registrar → Psychologist（Level III 题自动潜台词分析）→ Linguist，生成 `model_answer`；  
3. 写出 `data/model_outputs.json`，并调用 `KnowMeBench/evaluate/run_eval.py` 判分；  
4. 解析 `data/results.json`，在终端打印各任务得分；  
5. 在 `data/` 下自动保存 **冠军战报** `champion_report_YYYYMMDD_HHMM.md`。

### 4. 输出文件说明

| 文件 | 说明 |
|------|------|
| `data/model_outputs.json` | 每题 `id`、`task_type`、`question`、`reference_answer`、`model_answer` |
| `data/results.json` | 官方 run_eval 的判分详情（每题得分与 reasoning） |
| `data/champion_report_YYYYMMDD_HHMM.md` | 本次运行的汇总战报（各任务得分、Level III 是否达标） |

### 5. 可选参数

```bash
uv run python scripts/run_benchmarking.py --help
```

常用：

- `--dataset_dir KnowMeBench/KnowmeBench/dataset1`：指定 dataset 路径（默认即此）。  
- `--output_json data/model_outputs.json`：指定 model 输出 JSON 路径。  
- `--judge_model gpt-4o`：判分模型（默认 gpt-4.5，若不可用可改为 gpt-4o）。  
- `--max_per_task N`：每任务最多 N 题（不传则全量）。  
- `--use_registrar`：启用 Registrar 做上下文→Facts（推荐）。

### 6. 限速与重试

脚本内已做：

- 每题结束后 `asyncio.sleep(1.2)` 限速；  
- Registrar / Psychologist / Linguist 调用遇 **429** 时指数退避重试（最多 3 次）。

若仍遇限流，可适当调大间隔或减少并发（当前为单题顺序执行）。

**单独执行判分**（已有 `model_outputs.json` 时）：脚本会从项目根加载 `.env`，子进程可拿到 `OPENAI_API_KEY`。若在终端手动执行 `cd KnowMeBench/evaluate && uv run python run_eval.py ...`，请先导出密钥，例如在项目根执行：

```bash
set -a && source .env && set +a
cd KnowMeBench/evaluate && uv run python run_eval.py --input_file ../../data/model_outputs.json --output_file ../../data/results.json --judge_model gpt-4o
```

---

## 其他

- **API 服务**：`mnemos.api.app` 提供 FastAPI，可由 `uv run uvicorn mnemos.api.app:app --reload` 启动，供对话/记忆接口调用。  
- **Level III 心理洞察测试**：`uv run python tests/psychologist_test.py`，模拟一月记忆流并生成 `data/hybrid_benchmark_results.jsonl`。  
- 理论与场景说明见 `docs/`（如 `logic_integrity_scenario.md`、`knowme_bench_local_setup.md`）。

