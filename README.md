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

默认模型 `/root/autodl-tmp/models/Qwen3-8B`，服务名 `Qwen3-8B`，监听 `127.0.0.1:8000`，GPU 0、单卡张量并行、上下文 32768、最多4条并发序列、显存比例0.90，模型精度为 `auto`。保留默认思考模式，使用 `qwen3` reasoning parser。AutoLink 自行解析文本工具指令，故不启用 vLLM 原生工具调用参数。退出只清理本服务进程组。

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
bash run_bird_minidev_qwen3_8b.sh --check
bash run_spider2_lite_sqlite_qwen3_8b.sh --check

# 无模型的数据与字段文档准备
OUTPUT_DIR=outputs/spider2_smoke bash run_spider2_lite_sqlite_qwen3_8b.sh --limit 2 --stage prepare

# 以下显式执行才会加载嵌入模型/请求LLM
OUTPUT_DIR=outputs/spider2_smoke bash run_spider2_lite_sqlite_qwen3_8b.sh --limit 2 --stage all
```

默认执行 `--stage all`（完整实验及自动续跑）；`validate` 仅检查数据；`prepare` 生成在线数据、元数据和字段文档，不下载模型、不建向量索引；`link` 继续构建索引并完成Schema探索；`sql` 要求同输出目录已有完整link阶段，再执行候选生成、执行、修订与投票；`all` 执行完整流程。`--check`、`--prepare-only` 分别等价于 validate、prepare；不再提供 `--run` 开关。默认 `limit=0`，即读取当前数据集全量样本，脚本不固定135或500。需要小样本时显式传 `--limit 2`（也兼容环境变量 `LIMIT`），并指定独立 `OUTPUT_DIR`；不能把子集结果视为全量。

独立脚本配置模型、接口、数据路径和输出目录，并显式列出 `TEMPERATURE=0`、`SQL_GENERATION_TEMPERATURE=1.0`、`MAX_TOKENS=16384`；其他参数使用统一入口默认值，可按需通过环境变量覆盖。脚本使用现有环境，不创建或安装环境。Qwen3-8B端点默认为 `http://127.0.0.1:8000/v1`；模型需另行启动。实验脚本按论文主实验设置初始检索数量：BIRD `TOP_N=30`，Spider2 `TOP_N=100`，均支持环境变量覆盖。直接调用Python入口时 `TOP_N` 默认仍为100。补充检索数量固定为3。其余默认5候选、16384输出token、seed=42；SQL候选生成阶段使用 `SQL_GENERATION_TEMPERATURE=1.0`，探索、修订和选择阶段使用 `TEMPERATURE=0`；客户端不再传入 `top_p` 或采样 `top_k`，保留默认思考；原探索10轮、修订5轮。两份实验脚本默认通过 `EMBEDDING_DEVICE=cpu` 让BGE使用CPU；可用 `EMBEDDING_DEVICE=cuda:1 bash run_bird_minidev_qwen3_8b.sh` 显式指定GPU，编号相对于 `CUDA_VISIBLE_DEVICES`。可通过 `EMBEDDING_MODEL` 覆盖嵌入模型。直接调用Python入口且未设置设备时仍回退到CPU。统一入口顺序调度原生函数以明确传播阶段错误，不改变提示词或新增模式修复算法。

产物位于 `outputs/`：`data/` 保存白名单在线数据、规范化元数据和运行配置；`documents/`、`embeddings/` 保存可复用产物；`logs/` 保存原生中间结果；`calls/` 保存无密钥的请求响应；`stages/` 为完成标记；最终 `predictions.json` 保留原始question_id、题目顺序与预测SQL。`manifest.json` 校验配置、输入、源码和关键依赖版本，变化时要求新输出目录，且通过文件锁防止同一实验并行覆盖。

执行与探索、修订均通过数据层的只读SQLite连接，仅可访问登记数据库并具有SQL时间限制。保留原方法把空结果交给修订的行为。模型输出超限（`finish_reason="length"`）在单题处理最外层捕获：记录 `errors/<instance_id>.json`，该题最终 SQL 置空，跳过该题剩余处理并继续其他题；续跑也跳过这些已标记题。错误记录关联 `calls/` 中的响应、时间和 token 用量。网络/服务异常或非超限的无最终内容仍中止阶段，保留续跑能力，不把未处理尾部评分为错误。未将模型响应当作正确性标签，当前入口不自动计算EX。

原 `scripts/minidev/experiment.py` 和 `stage.py` 已移除；不再复制源码到输出目录，不再通过替换 `sqlite3.connect` 注入数据访问。数据读取、路径与方言适配已接入原生程序。

验证：

```bash
python -m unittest discover -s tests -v
python -m compileall -q run run_experiment.py tests
bash -n run_bird_minidev_qwen3_8b.sh run_spider2_lite_sqlite_qwen3_8b.sh
git diff --check
```

数据层测试使用合成只读数据库与模型桩，覆盖字段白名单、别名、缺失文档、重复题目、数据集切换、越界访问、写入拒绝、超时，以及原生文档、Schema提示、SQL生成方言、执行和修订接口。全量数据资源校验：Spider2 SQLite 135题/30库/432个元数据对象，Mini-Dev 500题/11库/75表。尚未进行真实LLM端到端实验，也未验证BigQuery/Snowflake远程运行。

### 全量默认与自动续跑

单阶段默认超时为 `STAGE_TIMEOUT=518400`（144小时），可通过环境变量覆盖；`API_TIMEOUT=1200` 控制单次请求超时，`SQL_TIMEOUT=60` 控制SQLite执行超时。三项默认时限均为此前的两倍。超时按每次启动的单个阶段计时，已完成单元通过检查点续跑。`STAGE_TIMEOUT` 仍属于清单配置，已有实验不能直接改变该环境变量后复用原目录。

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
export TEMPERATURE="${TEMPERATURE:-0}"       # 探索、修订和选择
export SQL_GENERATION_TEMPERATURE="${SQL_GENERATION_TEMPERATURE:-1.0}"
export MAX_TOKENS="${MAX_TOKENS:-16384}"        # 单次生成上限，包含思考token
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
```

实验脚本仅设置客户端生成上限 `max_tokens=16384`，不设置或要求服务端返回 `max_model_len`。总上下文上限只由模型启动脚本控制；当前启动配置仍为32768。模型预检仅核对服务模型名，已完成全部模型阶段的续跑不要求服务仍在线。修改生成预算后需使用新的实验输出目录。
