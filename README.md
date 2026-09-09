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

## Qwen3-8B / BIRD Mini-Dev 独立实验

从 AutoLink 根目录、使用已激活的 AutoLink 环境运行：

```bash
# 只读预检（不带参数时也仅预检）
bash run_bird_minidev_qwen3_8b.sh --check

# 两题链路测试：显式启动后才会请求模型
LIMIT=2 bash run_bird_minidev_qwen3_8b.sh --run

# 全部500题
bash run_bird_minidev_qwen3_8b.sh --run
```

默认数据目录为 `/root/autodl-tmp/workspace/kouan/datasets/bird/MINIDEV`，模型名 `Qwen3-8B`，服务地址 `http://127.0.0.1:8000/v1`。先在另一个终端启动模型服务。默认输出目录 `outputs/autolink_bird_minidev_Qwen3-8B_n500/`；LIMIT=2 对应独立 n2 目录。`--prepare-only` 仅转换数据，不调用模型。可通过 `PYTHON_BIN` 指定当前环境的 Python；脚本不创建或安装环境。

独立 run 脚本集中设置数据、模型、端点、初始 top-100 列、5 个 SQL 候选、每次最大8192输出token、模型请求超时600秒、SQLite执行时间限制30秒、单阶段最长12小时及单阶段模型调用上限30000次。采样 temperature=0.6、top_p=0.95、top_k=20、seed=42，保留Qwen3默认思考。原生探索最多10轮、错误修订最多5轮；本入口顺序执行各题以便暴露失败，未改原程序默认并发值。HTTP超时是客户端超时，阶段外层另有墙钟限制。

Mini-Dev 适配代码位于 `scripts/minidev/`：将问题映射到原生 local 实例，使用官方CSV字段说明与每表前三行数据，evidence 转为外部说明；在隔离工作目录生成原生 `spider2_data.json`，不覆盖原仓库数据。Gold SQL、difficulty 不写入在线数据。SQLite 在适配进程中限定到登记数据库、只读连接并设置查询超时；保留原流程对空结果触发修订的处理。

阶段依次为字段文档、BGE向量索引、初始检索、补标识符列、初始Schema、探索、Schema合并、最终Schema、五候选生成、执行、修订、投票和导出。复用原生Python函数及提示词，`postprocess.merge(is_preprocess=True)` 与原shell入口一致；模型接口、CPU嵌入与只读执行限制仅在独立适配进程中生效。BGE默认 `BAAI/bge-large-en-v1.5`，首次可能需要下载；可用 `EMBEDDING_MODEL` 指定本地模型目录。

`manifest.json` 保存不含密钥的配置、输入与代码哈希；`stages/` 保存完成标记；各阶段日志及模型请求/响应分别存储，原生中间结果位于 `work/logs/`。同配置重跑会跳过完整阶段；发生模型调用错误或无最终内容时中止，不无限重试，不将未完成尾部统计为错误。配置或输入变化必须使用新的 `OUTPUT_DIR`。缺失结果不导出部分汇总。最终 `predictions.json` 保持题目顺序和 question_id，用于后续离线评估；本脚本不自动计算EX或声称复现论文成绩。

检查：`bash -n run_bird_minidev_qwen3_8b.sh`、Python AST语法检查、`--check`；两题 `--prepare-only` 已验证字段白名单和资源格式。端到端模型执行尚未验证。
