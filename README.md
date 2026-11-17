# AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale

<p align="center">
| <a href="https://arxiv.org/abs/2511.17190"><b>arXiv</b></a> |
</p>

## Overview
![AutoLink](assets/overview.png)

For industrial-scale text-to-SQL, supplying the entire database schema to Large Language Models (LLMs) is impractical due to context window limits and irrelevant noise. Schema linking, which filters the schema to a relevant subset, is therefore critical. However, existing methods incur prohibitive costs, struggle to trade off recall and noise, and scale poorly to large databases. We present **AutoLink**, an autonomous agent framework that reformulates schema linking as an iterative, agent-driven process. Guided by an LLM, AutoLink dynamically explores and expands the linked schema subset, progressively identifying necessary schema components without inputting the full database schema. Our experiments demonstrate AutoLink's superior performance, achieving state-of-the-art strict schema linking recall of **97.4%** on Bird-Dev and **91.2%** on Spider-2.0-Lite, with competitive execution accuracy, i.e., **68.7** EX on Bird-Dev (better than CHESS) and **34.9** EX on Spider-2.0-Lite (ranking 2nd on the official leaderboard). Crucially, AutoLink exhibits **exceptional scalability**, **maintaining high recall**, **efficient token consumption**, and **robust execution accuracy** on large schemas (e.g., over 3,000 columns) where existing methods severely degrade—making it a highly scalable, high-recall schema linking solution for industrial text-to-SQL systems.

## Folder Structure  
```
- linking_results/                     -- Results after schema linking used to SQL generation
- run/  
  - bigquery_credentials/              -- Place bigquery credentials
  - documents/                         -- Constructed column-level documents
  - embeddings/                        -- Document embeddings
  - log_path/                          -- Results of schema linking and sql generation
    - final_schema_prompts/            -- Prompts after AutoLink's schema linking process used to SQL generation
    - schema_prompts/                  -- Prompts of schema used to AutoLink's schema linking process
    - sql_gen/                         -- Candidate SQLs and execution results
    - sql_revise/                      -- Revision result of every candidates
    - sql_selection/                   -- The final SQL after cadidate selection
    - merge_candidates.json            -- The final schema linking results, used for calculating recall(The difference from unfilled_schema.json lies in whether nested columns in BigQuery are expanded)
    - unfilled_schema.json             -- The final schema linking results, used for constructing the SQL generation prompt
    - ...                              -- Other intermediate results
  - resource/                          -- Copied from Spider 2.0-Lite Repo
  - snowflake_credential/              -- Place snowflake credential
  - add_id.py                          -- Primary and foreign key rule processing
  - complete_schema.py                 -- Iterative, agent-driven schema linking
  - config.py                          -- Prompts 
  - embedding_docs.py                  -- Embedding documents
  - generate_docs.py                   -- Generate documents  
  - generate_schema.py                 -- Generate schema  
  - main.sh                            -- Main script of schema linking
  - model_manager.py                   -- Embedding model manager
  - postprocess.py                     -- Postprocess after schema linking
  - retrieve_topk_schema.py            -- Retrieve script
  - spdier2_data.json                  -- Spider 2.0-Lite test set
  - sql_execution.py                   -- Execute SQL
  - sql_gen.sh                         -- The script to generate SQL
  - sql_generation.py                  -- Generate candidate SQLs
  - sql_revise.py                      -- Revise the SQL with error execution after candidate SQL generation
  - sql_selection.py                   -- Select the final SQL via self-consistency based on SQL execution results
  - utils.py                           -- Utility functions
```

## Settint Up Environment
1. Clone the repository:
  ```bash
  git clone https://github.com/wzy416/AutoLink.git
  cd AutoLink
  ```
2. Create conda environment:
  ```bash
  conda create -n AutoLink python=3.12
  conda activate AutoLink
  pip install -r requirements.txt
  ```
3. Modify the api key:

  To use your own LLM, modify the OPENAI_API_KEY and OPENAI_BASE_URL in `run/main.sh`.


4. Copy repository from [`Spider 2.0-lite`](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite)

  You should copy [`spdier2-lite/resource`](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite) from [`Spider 2.0`](https://github.com/xlang-ai/Spider2) repo to `run/resource/`. Follow the [`bigquery guideline`](https://github.com/xlang-ai/Spider2/blob/main/assets/Bigquery_Guideline.md) and  [`snowflake guideline`](https://github.com/xlang-ai/Spider2/blob/main/assets/Snowflake_Guideline.md) to sign up accounts, then put the credential json files under `run/bigquery_credentials/` and `run/snowflake_credential/` separately.

## Running the code
1. Schema Linking
  ```bash
  cd ./run
  bash main.sh
  ```
  It will gradually complete the entire process from document construction, document embedding, initial schema retrieval, to schema exploration and expansion.

2. SQL Generation
  ```bash
  bash sql_gen.sh
  ```
  It will progressively complete candidate SQL generation, SQL revision, and final SQL selection.

# Citation
If you find this repo helpful, please cite our work:
```bibtex
@misc{wang2025autolinkautonomousschemaexploration,
      title={AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale}, 
      author={Ziyang Wang and Yuanlei Zheng and Zhenbiao Cao and Xiaojin Zhang and Zhongyu Wei and Pei Fu and Zhenbo Luo and Wei Chen and Xiang Bai},
      year={2025},
      eprint={2511.17190},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.17190}, 
}
```




## 本地模型服务

在已激活且安装 vLLM 的环境中，从 AutoLink 根目录启动：

```bash
bash scripts/start_models/Qwen3-8B.sh
```

默认模型 `/root/autodl-tmp/models/Qwen3-8B`，服务名 `Qwen3-8B`，监听 `127.0.0.1:8000`，GPU 0、单卡张量并行、上下文 65536、最多32条并发序列、显存比例0.90，模型精度为 `auto`。保留默认思考模式，使用 `qwen3` reasoning parser。AutoLink 自行解析文本工具指令，故不启用 vLLM 原生工具调用参数。退出只清理本服务进程组。

可通过环境变量指定 GPU、Python、监听地址和端口，例如：

```bash
CUDA_VISIBLE_DEVICES=1 HOST=0.0.0.0 PORT=8000 bash scripts/start_models/Qwen3-8B.sh
```

每次实验使用单独命名的 run 脚本，集中设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MODEL_NAME`、数据路径、预算及输出目录，并直接调用所需 Python 阶段。不要调用会覆盖环境变量的原始通用 shell 入口。本地 Qwen3-8B 实验的接口地址为 `http://127.0.0.1:8000/v1`，模型名为 `Qwen3-8B`。Mini-Dev 独立实验入口见下文；该入口只完成静态与数据准备检查，尚未执行模型实验。

## 统一数据层与实验入口

数据集由 `run/data_layer.py` 统一解析，入口为 `run_experiment.py`。各原生阶段直接读取数据层，使用明确的 `dialect` 字段选择数据库引擎。未设置数据层配置时保留原 Spider 2.0-Lite 入口的路径和ID前缀兼容。原模型名默认值不变，实验脚本通过环境变量指定模型与接口。

当前新增数据集：

| `--dataset` | `--data_root` | `--database_root` |
| --- | --- | --- |
| `bird_minidev` | 相邻 `datasets/bird/MINIDEV` | 可省略，默认 `data_root/dev_databases` |
| `spider2_lite_sqlite` | 相邻 `Spider2/spider2-lite` | 相邻 `datasets/spider2/lite/local_sqlite` |

Spider2 使用官方问题文件、SQLite元数据和题目引用的文档；实际数据库按 `local-map.jsonl` 解析，并处理 `DB_IMDB/Db-IMDB`、`SQLITE_SAKILA/sqlite-sakila` 元数据目录别名。Mini-Dev 使用问题、evidence、CSV字段说明和每表前三行样例。在线对象不包含 gold SQL 或 difficulty；评估答案不由在线数据层加载。

```bash
# 使用已激活的 AutoLink 环境；只做全量只读校验
python run_experiment.py --dataset bird_minidev \
  --data_root ../datasets/bird/MINIDEV --stage validate
python run_experiment.py --dataset spider2_lite_sqlite \
  --data_root ../Spider2/spider2-lite \
  --database_root ../datasets/spider2/lite/local_sqlite --stage validate

# 独立实验脚本集中设置模型、端点、预算、输出目录
bash exp_bird.sh --check
bash run_spider2_lite_sqlite_qwen3_8b.sh --check

# 无模型的数据与字段文档准备
OUTPUT_DIR=outputs/spider2_smoke bash run_spider2_lite_sqlite_qwen3_8b.sh --limit 2 --stage prepare

# 以下显式执行才会请求嵌入服务/LLM
OUTPUT_DIR=outputs/spider2_smoke bash run_spider2_lite_sqlite_qwen3_8b.sh --limit 2 --stage all
```

默认执行 `--stage all`（完整实验及自动续跑）；`validate` 仅检查数据；`prepare` 生成在线数据、元数据和字段文档，不下载模型、不建向量索引；`link` 继续构建索引并完成Schema探索；`sql` 要求同输出目录已有完整link阶段，再执行候选生成、执行、修订与投票；`all` 执行完整流程。`--check`、`--prepare-only` 分别等价于 validate、prepare；不再提供 `--run` 开关。默认 `limit=0`，即读取当前数据集全量样本，脚本不固定135或500。需要小样本时显式传 `--limit 2`（也兼容环境变量 `LIMIT`），并指定独立 `OUTPUT_DIR`；不能把子集结果视为全量。

独立脚本配置模型、接口、数据路径和输出目录，并显式列出 `TEMPERATURE=0`、`MAX_TOKENS=16384`；其他参数使用统一入口默认值，可按需通过环境变量覆盖。脚本使用现有环境，不创建或安装环境。Qwen3-8B端点默认为 `http://127.0.0.1:8000/v1`；模型需另行启动。实验脚本恢复原版初始检索数量：BIRD 和 Spider2 均为 `TOP_N=100`，均支持环境变量覆盖。直接调用Python入口时 `TOP_N` 默认仍为100。补充检索数量固定为3。其余默认5候选、16384输出token，不显式设置随机种子；SQL候选生成、探索、修订和选择阶段统一使用 `TEMPERATURE=0`；客户端不再传入 `top_p` 或采样 `top_k`，保留默认思考；原探索10轮、修订5轮。嵌入通过独立vLLM服务调用，文档和查询共用 `EMBEDDING_*` 配置，GPU由服务端指定。统一入口按阶段调度原生函数以明确传播阶段错误，不改变提示词或新增模式修复算法。

嵌入服务配置集中在 `exp_bird.sh`，默认使用本地 BGE-M3 服务：

| 参数 | BIRD脚本默认值 | 用途 |
| --- | --- | --- |
| `EMBEDDING_BASE_URL` | `http://127.0.0.1:9000/v1` | 独立嵌入服务地址 |
| `EMBEDDING_API_KEY` | `EMPTY` | 嵌入服务密钥，不写入清单 |
| `EMBEDDING_MODEL` | `/root/autodl-tmp/models/bge-m3` | 与服务的 served-model-name 一致 |
| `EMBEDDING_DIMENSIONS` | `0` | 0保留原生维度，不传 dimensions；正数仅适用于支持Matryoshka的模型 |
| `EMBEDDING_MAX_TOKENS` | `8192` | 每条嵌入输入的token上限，通过 truncate_prompt_tokens 由服务端从右侧截断，保留前N个token |

BGE-M3不支持任意修改输出维度，因此应保持 `EMBEDDING_DIMENSIONS=0`；客户端不会自行截断或填充向量。输入token上限应不超过服务的 `--max-model-len`。这些参数与LLM生成的 `MAX_TOKENS` 相互独立。[vLLM嵌入接口说明](https://docs.vllm.ai/en/v0.19.0/models/pooling_models/embed/)

```bash
# 单独启动嵌入服务，GPU编号在服务端指定；不会由实验脚本自动启动。
CUDA_VISIBLE_DEVICES=0 bash scripts/start_models/bge-m3.sh

# 在另一终端启动实验（LLM服务也需已启动）
RUN_ID=run1 bash exp_bird.sh
```

文档构建及问题/补充检索均通过 `/v1/embeddings` 获取向量，本地只构建和查询FAISS索引，不再加载SentenceTransformer。返回结果按index还原顺序，并校验数量、有限数值和维度；每个索引保存 `embedding_config.json`，查询配置不一致时拒绝复用。更换嵌入模型、服务配置或维度后需新建运行ID并重建索引。`EMBEDDING_DEVICE` 已移除，服务端通过 `CUDA_VISIBLE_DEVICES` 选择GPU。Python入口及Spider2脚本保留原模型名 `BAAI/bge-large-en-v1.5`、输入上限512作为默认配置，使用时需启动对应服务或覆盖为现有服务配置。

BIRD 数据集与划分分开配置：`DATASET=bird`、`SPLIT=minidev`（默认）或 `SPLIT=dev`。`DATA_ROOT` 指向 BIRD 总目录，入口自动读取 `MINIDEV/mini_dev_sqlite.json` 或 `dev/dev.json`，数据库默认位于对应划分的 `dev_databases/`。当前只支持 SQLite；`--database_root` 可单独覆盖数据库路径。旧参数 `--dataset bird_minidev` 仍接受直接指向 Mini-Dev 的路径，Spider2 入口保持兼容。实验清单记录 dataset 和 split，防止不同划分混用检查点。

```bash
DATA_ROOT=/home/linhx/workspace/kouan/SAGE-SQL/datasets/bird \
    bash exp_bird.sh --split minidev --check
DATA_ROOT=/home/linhx/workspace/kouan/SAGE-SQL/datasets/bird \
    RUN_ID=run1 bash exp_bird.sh --split dev
```

BIRD 脚本默认 `WORKERS=16`，可用 `WORKERS=4 bash exp_bird.sh` 或 `bash exp_bird.sh --workers 4` 调整，`--workers 1` 恢复串行；直接调用 Python 入口默认仍为1。并发作用于探索、SQL生成、执行、修订和选择阶段的不同题目；阶段之间和同题候选之间保持顺序。文档构建、嵌入、初始检索和合并/导出仍顺序执行，原探索/修订中的SQLite锁保留。请求预算预留与共享选择日志受锁保护；基础设施错误停止提交新任务，等待已开始任务结束后向外传播，完成单元仍可续跑。`WORKERS` 记录在实验清单中，改变并发数时需使用新的运行ID或输出目录。

BIRD 脚本通过显式 `RUN_ID` 区分实验，默认 `run1`，输出目录为 `outputs/autolink_bird_<split>_<模型>_<RUN_ID>/`。每次新实验修改脚本中的默认ID或通过环境变量指定不同ID；相同ID会复用检查点续跑。模型名中的 `/` 会替换为 `_`。需要续跑或分阶段执行时，指定相同的 `RUN_ID`、`OUTPUT_DIR` 或 `--output_dir`，并保持模型、参数及输入一致；命令行 `--output_dir` 优先于环境变量。

```bash
# 一次启动，顺序执行run1、run2、run3
bash exp_bird.sh --runs 3

# 从run4开始，执行run4、run5（也可在脚本里修改RUNS）
RUN_ID=run4 RUNS=2 bash exp_bird.sh

# 指定不同ID，启动两次独立实验
RUN_ID=run1 bash exp_bird.sh
RUN_ID=run2 bash exp_bird.sh

# 为实验命名；再次运行相同命令即可续跑
RUN_ID=trial_1 bash exp_bird.sh

# 分阶段执行必须使用同一目录
bash exp_bird.sh --output_dir outputs/bird_trial_2 --stage link
bash exp_bird.sh --output_dir outputs/bird_trial_2 --stage sql
```

多次运行使用 `RUNS`（默认1）或 `--runs` 指定次数，递增 `RUN_ID` 末尾的数字；多次运行的ID必须以数字结尾。每个run使用独立目录，run之间顺序执行，题目并发仍由 `WORKERS` 控制。如果显式传入 `OUTPUT_DIR` 或 `--output_dir`，单次运行直接使用该目录，多次运行则分别使用其下的 `<RUN_ID>/` 子目录。任一run失败后停止后续run，重新执行相同命令可利用检查点续跑；已完成的run会跳过已完成阶段。

产物位于 `outputs/`：`data/` 保存白名单在线数据、规范化元数据和运行配置；`documents/`、`embeddings/` 保存可复用产物；`logs/` 保存原生中间结果；`calls/` 保存无密钥的请求响应；`stages/` 为完成标记；最终 `predictions.json` 保留原始question_id、题目顺序与预测SQL。`manifest.json` 校验配置、输入、源码和关键依赖版本，变化时要求新输出目录，且通过文件锁防止同一实验并行覆盖。

### 实验日志与成本统计

每个运行目录会保留原始记录，并在每个执行阶段结束、整次启动正常结束或捕获失败后自动生成 `metrics/` 汇总。记录中的 `run_id` 标识实验，`session_id` 区分同一目录的多次启动；`stage_attempt_id`、`unit_attempt_id` 区分阶段和题目/候选的每次尝试。

| 文件 | 内容 |
| --- | --- |
| `calls/<stage>-<id>.json` | LLM请求/响应、服务返回的usage、请求耗时、题目ID、候选编号、调用序号、轮次、结束原因和状态 |
| `embedding_calls/<stage>-<id>.json` | 嵌入调用usage、耗时、输入数量/哈希、数据库及批次；不重复保存输入文本或向量 |
| `telemetry/sessions/` | 每次启动的起止时间、耗时、完成/失败/中断状态 |
| `telemetry/stages/` | 阶段尝试、耗时和续跑跳过记录 |
| `telemetry/units/` | 每题/候选的执行尝试、耗时和检查点跳过记录 |
| `telemetry/calls/` | 不含大段请求/响应的调用统计副本，用于快速汇总 |
| `metrics/summary.json` | 全run的LLM与嵌入调用次数、已知token总量、耗时分布及统计口径 |
| `metrics/calls.csv` | 每次API尝试一行，方便按题目/阶段/候选/启动筛选 |
| `metrics/questions.csv` | 每道登记题目一行，包含失败及未完成题目的成本与耗时 |
| `metrics/stages.csv` | 各阶段的累计耗时、LLM/嵌入调用和token统计 |
| `metrics/sessions.csv`、`metrics/units.csv` | 启动及题目任务尝试明细 |

LLM的 `candidate_index` 从0开始，`call_index` 从1开始；探索/修订的 `round_index` 是该任务尝试内模型调用的顺序，续跑重试用新的 `unit_attempt_id` 区分。选择阶段还记录 `pair_candidates`。嵌入的查询调用关联题目；共享文档嵌入只关联数据库/批次，不强行分摊给某一道题。

统计口径：

- Call数表示已记录的客户端尝试，分开统计 `returned`、`request_failure`、`interrupted` 和 `started`。`started` 可能仍在执行或被强制中断，不等于服务已完成；当前SDK重试关闭。
- Token字段按服务返回值保存，缺失为JSON `null` / CSV空格，不按0补齐。`reported_*_tokens` 只汇总已知部分，结合 `missing_total_tokens` 判断覆盖；reasoning tokens通常已包含在completion tokens中，不重复相加。
- `latency_seconds` 是客户端API往返时间，包含网络和服务排队；不代表GPU计算时间。P50/P95使用nearest-rank定义。
- `sessions.finalized_wall_seconds_sum` 是各次已结束启动的真实耗时之和，含校验、准备、阶段执行和续跑检查，排除两次启动之间的停机时间及最后一次汇总写出。强制中断而未结束的启动时长未知。
- 逐题 `active_seconds_sum` 累计任务尝试耗时；`observed_span_seconds_sum` 累计该题在每次启动中从首个任务开始到最后一个已结束任务的时间跨度，包含阶段间等待。两者均不包含共享预处理的成本分摊，不能将并发请求耗时相加冒充端到端时间。
- 当前请求非流式，`ttft_seconds=null`，不能测量首token延迟。失败、输出超限和重跑产生的已知成本均保留；检查点跳过不会产生新的API调用。
- 旧日志缺少的题目归属、嵌入usage和启动耗时无法补算，汇总会标记未归属/旧格式记录。为了取得完整统计，应使用新的运行ID。

强制中断后或需要重新导出统计时，无需调用模型：

```bash
python summarize_metrics.py outputs/<运行目录>
```

执行与探索、修订均通过数据层的只读SQLite连接，仅可访问登记数据库；执行/修订有SQL时间限制，探索不设置查询时限。保留原方法把空结果交给修订的行为。模型输出超限（`finish_reason="length"`）在单题处理最外层捕获：记录 `errors/<instance_id>.json`，该题最终 SQL 置空，跳过该题剩余处理并继续其他题；续跑也跳过这些已标记题。错误记录关联 `calls/` 中的响应、时间和 token 用量。网络/服务异常或非超限的无最终内容仍中止阶段，保留续跑能力，不把未处理尾部评分为错误。未将模型响应当作正确性标签，当前入口不自动计算EX。


验证：

```bash
python -m compileall -q run run_experiment.py summarize_metrics.py
bash -n exp_bird.sh run_spider2_lite_sqlite_qwen3_8b.sh
git diff --check
```


### 全量默认与自动续跑

API请求超时使用OpenAI SDK默认值，通过SDK的 `DefaultHttpxClient` 保留默认超时配置；不再使用 `API_TIMEOUT`。整个阶段不设置超时，不再使用 `STAGE_TIMEOUT`。`SQL_TIMEOUT=600` 控制SQLite执行及修订中的查询超时；Schema探索不设置查询时限。已完成单元通过检查点续跑。配置和代码均记录在清单中，修改后应使用新的输出目录。

默认输出目录不含固定题数：`outputs/autolink_bird_minidev_Qwen3-8B/` 与 `outputs/autolink_spider2_lite_sqlite_Qwen3-8B/`。保持配置不变，重新执行相同命令即可自动续跑，无需额外恢复开关：

```bash
bash run_spider2_lite_sqlite_qwen3_8b.sh
# 中断后，再次执行同一条命令
bash run_spider2_lite_sqlite_qwen3_8b.sh
```

完整阶段通过 `stages/` 跳过；初始检索、Schema探索、候选生成、执行、修订和最终选择通过 `checkpoints/<stage>/` 逐题/逐候选记录完成状态，并检查产物SHA-256。仅有SQL或原始响应文件存在不视为完整单元；半写、缺失、被修改的单元会重新执行。中断的当前单元从该单元起点重跑，未声称恢复LLM内部推理状态或探索的某一轮；此前物理调用日志仍保留并计入调用上限。嵌入等未完成预处理阶段可重新执行。模型、参数、题目、输入文件、代码或依赖变化时拒绝混用原目录，需新 `OUTPUT_DIR`。旧版本结果不会绕过配置检查强行迁移。

### 数据集对应的官方预测与评估格式

最终导出仍保留通用 `predictions.json`，同时根据数据集生成：

| 数据集 | 官方评估输入 | 评估入口 |
| --- | --- | --- |
| BIRD Mini-Dev | `predict_dev.json`，按输入顺序编号，值为 `SQL\t----- bird -----\tdb_id` | 相邻 `mini_dev/evaluation/` 官方评估器 |
| Spider2-Lite SQLite | `submission_sql/<instance_id>.sql`，平铺目录，包含全部目标题目 | `Spider2/spider2-lite/evaluation_suite/evaluate.py` |

导出前检查ID、数据库、顺序与完整性；执行失败的已完成候选仍提交其SQL，不仅导出成功题。输出超限题保留题目 ID 并提交空 SQL（BIRD 保留数据库分隔符，Spider2 保留空 `.sql` 文件），计入评估范围；其他缺失/重复预测拒绝导出。`export_manifest.json` 记录格式、题目范围与文件哈希。使用官方评估器前应确认提交覆盖全部目标题目；官方脚本不会读取该清单，而是按提交交集评分。

```bash
# 从工作区根目录进入官方评估目录
cd Spider2/spider2-lite/evaluation_suite
python evaluate.py --mode sql \
  --result_dir /root/autodl-tmp/workspace/kouan/AutoLink/outputs/autolink_spider2_lite_sqlite_Qwen3-8B/submission_sql
```

BIRD预测已由实验末尾导出为官方格式 `predict_dev.json`，直接使用相邻 `mini_dev` 的评估脚本。可参考 `mini_dev/run_evaluation_direct_bird_dev_Qwen2.5-Coder-7B-Instruct.sh`，将预测、数据库、gold SQL、难度文件与日志路径改为本次 Mini-Dev 实验路径；该参考脚本原本指向完整 BIRD Dev，不能原样用于 Mini-Dev。子集必须自行准备同题目、同顺序的离线gold和难度文件。AutoLink不再提供BIRD评估包装脚本，也不自动准备这些离线评估文件；Gold不进入在线状态。

Spider2直接使用官方 `evaluate.py --mode sql`，不再提供额外评估包装脚本。运行前需按照官方README，将SQLite数据库放入或链接至 `Spider2/spider2-lite/resource/databases/spider2-localdb/`；当前数据库实际存放于 `datasets/spider2/lite/local_sqlite/`，官方评估器不会自动读取AutoLink的数据层配置。需保留官方问题清单对应的数据库文件名，并在官方 `evaluation_suite/` 目录执行，以满足其相对路径要求。

对SQLite子集采用官方输出的 `Final score`；其另一个 `Real score` 固定除以547，不是135题子集指标。按题数明确标注子集，不能据此报告完整Lite成绩。评估需使用具备官方评估依赖的Python环境；本次仅清理包装脚本，未建立数据库链接或运行评估。

### 显式模型参数

两份推理脚本显式配置：

```bash
export TEMPERATURE="${TEMPERATURE:-0}"       # SQL生成、探索、修订和选择
export MAX_TOKENS="${MAX_TOKENS:-16384}"        # 单次生成上限，包含思考token
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
```

实验脚本仅设置客户端生成上限 `max_tokens=16384`，不设置或要求服务端返回 `max_model_len`。总上下文上限只由模型启动脚本控制；当前启动配置为65536。模型预检仅核对服务模型名，已完成全部模型阶段的续跑不要求服务仍在线。修改生成预算后需使用新的实验输出目录。
