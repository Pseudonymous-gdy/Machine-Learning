# Binary Classification with a Bank Dataset

Website: https://www.kaggle.com/competitions/playground-series-s5e8/

---

## English

### A brief overview

This repository contains an end-to-end training/inference pipeline for the bank dataset used in the linked Kaggle competition. The code expects the dataset columns listed below (the training set includes the target column `y`; the test set usually does not).

- age: Age of the client (numeric)
- job: Type of job (categorical: "admin.", "blue-collar", "entrepreneur", etc.)
- marital: Marital status (categorical: "married", "single", "divorced")
- education: Level of education (categorical: "primary", "secondary", "tertiary", "unknown")
- default: Has credit in default? (categorical: "yes", "no")
- balance: Average yearly balance in euros (numeric)
- housing: Has a housing loan? (categorical: "yes", "no")
- loan: Has a personal loan? (categorical: "yes", "no")
- contact: Type of communication contact (categorical: "unknown", "telephone", "cellular")
- day: Last contact day of the month (numeric, 1-31)
- month: Last contact month of the year (categorical: "jan", "feb", "mar", …, "dec")
- duration: Last contact duration in seconds (numeric)
- campaign: Number of contacts performed during this campaign (numeric)
- pdays: Number of days since the client was last contacted from a previous campaign (numeric; -1 means no previous contact)
- previous: Number of contacts performed before this campaign (numeric)
- poutcome: Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")
- y: Target variable — whether the client subscribed (binary: "yes", "no")

### Citation
Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset. https://kaggle.com/competitions/playground-series-s5e8, 2025. Kaggle.

---

### Quick start (English)

Prerequisites

- Python 3.10+
- Recommended: virtual environment or dev container

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn xgboost seaborn joblib
```

Data placement

Place the CSV files under the project folder:

- `Project/Binary Classification with a Bank Dataset/train.csv`
- `Project/Binary Classification with a Bank Dataset/test.csv`

Run modes

1) Safe (default) — fast smoke test / CI-friendly:

```bash
python "Project/Binary Classification with a Bank Dataset/main.py"

# faster debug run with smaller sample
python "Project/Binary Classification with a Bank Dataset/main.py" --select 0.01
```

Output: by default the script writes `Project/Binary Classification with a Bank Dataset/submission.csv`; use `--out` to change the path.

2) Full run (`--full`) — uses full dataset and larger models (high resource usage):

```bash
python "Project/Binary Classification with a Bank Dataset/main.py" --full
```

Warning: `--full` enables bigger models and parallelism by default and may cause out-of-memory (OOM) or worker termination on machines with limited RAM — test first with `--select`.

Monitoring and troubleshooting

- Background run and live logs:

```bash
PYTHONUNBUFFERED=1 python "Project/Binary Classification with a Bank Dataset/main.py" --full > run_full.log 2>&1 &
tail -f run_full.log
```

- If resources are limited:
	- Use `--select 0.05` or smaller first.
	- Reduce `n_estimators` and `n_jobs` in `main.py` or in model wrappers.
	- Run on a machine/cluster with more RAM/CPU.

- If the process is killed or you see `TerminatedWorkerError`, check kernel logs:

```bash
dmesg | grep -i -E 'killed process|oom|Out of memory' -A3
```

Outputs & validation

- `main.py` prints validation ROC AUC after training on the validation fold.
- In full mode the script prints the first 10 rows of the submission (submission head) and writes the CSV to `--out` (default `Project/Binary Classification with a Bank Dataset/submission.csv`).

Common issues & fixes

- KeyError: "['contacted', 'total_contacts'] not in index" — Cause: derived columns must be created before transforming test data. Ensure `dl.merge_columns()` runs before `dl.encode_and_scale()`; `main.py` calls these in the correct order.
- TypeError in `sklearn.clone` for XGBoost wrapper — fixed in repo: `XgBoost.py` accepts `**kwargs` and is clone-compatible.

Debug & CI suggestions

- For CI or quick local reproducibility, run a smoke script using `--select 0.01`. If you want, I can add `dev/run_smoke.py` that executes the smoke flow for CI.

---

## 中文（Chinese）

### 项目简介

本仓库实现了针对 Kaggle 银行数据竞赛的端到端训练/推理流水线。代码期望如下列的字段（训练集包含目标列 `y`；测试集通常不包含 `y`）：

- age: 客户年龄（数值型）
- job: 职业类型（类别型，例如："admin.", "blue-collar", "entrepreneur" 等）
- marital: 婚姻状况（类别型："married", "single", "divorced"）
- education: 教育程度（类别型："primary", "secondary", "tertiary", "unknown"）
- default: 是否有违约（类别型："yes", "no"）
- balance: 年均余额（欧元，数值型）
- housing: 是否有房贷（类别型："yes", "no"）
- loan: 是否有个人贷款（类别型："yes", "no"）
- contact: 联系方式（类别型："unknown", "telephone", "cellular"）
- day: 最后一次联系的日（数值，1-31）
- month: 最后一次联系的月（类别型："jan", "feb", …, "dec"）
- duration: 最后一次联系的通话时长（秒，数值型）
- campaign: 本次活动联系次数（数值型）
- pdays: 距离上一次活动的天数（-1 表示未曾联系过）
- previous: 之前的联系次数（数值型）
- poutcome: 上次活动结果（类别型："unknown", "other", "failure", "success"）
- y: 目标变量——是否订阅（"yes"/"no"）

### 引用
Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset. https://kaggle.com/competitions/playground-series-s5e8, 2025. Kaggle.

---

### 运行指引（中文）

先决条件

- Python 3.10+
- 建议在虚拟环境或 dev container 中运行

安装依赖：

```bash
pip install numpy pandas scipy scikit-learn xgboost seaborn joblib
```

数据位置

将数据放在：

- `Project/Binary Classification with a Bank Dataset/train.csv`
- `Project/Binary Classification with a Bank Dataset/test.csv`

运行模式

1）安全模式（默认） — 快速 smoke 测试 / CI 友好：

```bash
python "Project/Binary Classification with a Bank Dataset/main.py"

# 指定更小采样以便更快调试
python "Project/Binary Classification with a Bank Dataset/main.py" --select 0.01
```

脚本会把提交文件写到 `Project/Binary Classification with a Bank Dataset/submission.csv`（或使用 `--out` 指定路径）。

2）全量训练（`--full`） — 使用全量数据与更大模型（资源消耗高）：

```bash
python "Project/Binary Classification with a Bank Dataset/main.py" --full
```

注意：`--full` 会使用更大的模型与并行化，可能导致内存不足或系统杀进程（OOM）。建议先用 `--select` 近似运行，或者在大内存机器/集群上执行。

监控与故障排查

- 后台运行并记录日志：

```bash
PYTHONUNBUFFERED=1 python "Project/Binary Classification with a Bank Dataset/main.py" --full > run_full.log 2>&1 &
tail -f run_full.log
```

- 资源受限时建议：
	- 使用 `--select 0.05` 或更小的采样先跑；
	- 在 `main.py` 中将学习器的 `n_estimators` 与 `n_jobs` 调小；
	- 在有更多内存的机器上运行全量训练。

- 若进程被系统杀死或出现 `TerminatedWorkerError`，请检查内核日志：

```bash
dmesg | grep -i -E 'killed process|oom|Out of memory' -A3
```

输出与验证

- 训练完成后，`main.py` 会打印验证集上的 ROC AUC（Validation ROC AUC）。
- 全量运行完成后，脚本会打印 submission 文件的前 10 行（submission head），并把 CSV 写入 `--out` 指定路径（默认 `Project/Binary Classification with a Bank Dataset/submission.csv`）。

常见问题与解决办法

- KeyError: "['contacted', 'total_contacts'] not in index" — 原因：在对测试集做变换前需要先运行 `dl.merge_columns()` 生成派生列；`main.py` 已按正确顺序处理，但若单独调用 DataLoader，请确保先合并列。
- sklearn.clone 对 XGBoost wrapper 抛出 TypeError — 已在仓库修复：`XgBoost.py` 接受 `**kwargs`，支持 clone。

调试与 CI 建议

- 若用于 CI 或本地快速复现，建议添加一个 smoke 脚本（例如 `dev/run_smoke.py`）并在 CI 中只运行 `--select 0.01` 的流程。如需我可以帮你添加该脚本。

---

如果还需要我把该 README 的中英双语格式改为并列对照（每段中英文并排）或生成 `dev/run_smoke.py`，请告诉我（我可以继续修改）。

---

## Citation
Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset. https://kaggle.com/competitions/playground-series-s5e8, 2025. Kaggle.

---

## 运行指引 — Binary Classification with a Bank Dataset

本节为 `Project/Binary Classification with a Bank Dataset` 子目录下的运行说明（包含依赖安装、数据位置、快速验证与全量运行建议）。

### 概要
本项目包含一个小型端到端训练/推理流水线，入口脚本为 `main.py`（位于 `Project/Binary Classification with a Bank Dataset/`）。仓库内已经做了若干工程化改动，旨在：

- 修复模块相对导入问题，让模块既能作为包导入也能作为单文件脚本运行；
- 在 `utils/Data_loader/Data_loader.py` 中实现了训练时拟合变换器（encoder/scale/selector）并在测试时复用的 API（避免数据泄露）；
- 将原始手写 stacking 替换为安全的 `sklearn.ensemble.StackingClassifier`，并在 `main.py` 提供了安全与全量运行模式的开关。

### 先决条件 / 依赖
- Python 3.10+
- 建议在虚拟环境或 dev container 中运行

安装基础 Python 包：

```bash
pip install numpy pandas scipy scikit-learn xgboost seaborn joblib
```

（如果你在公司/受限网络环境，请依照内部包管理策略安装相应包）

### 数据位置
请把数据放在子目录下：

- `Project/Binary Classification with a Bank Dataset/train.csv`
- `Project/Binary Classification with a Bank Dataset/test.csv`

注意：训练集应包含 `y` 列；测试集通常不包含 `y`。

### 运行模式

1) 安全（默认） — 快速 smoke-test / CI 友好：

```bash
# 在仓库根目录下运行（默认使用小样本/较小模型以便快速验证）
python "Project/Binary Classification with a Bank Dataset/main.py"

# 指定更小样本以加速验证（用于调试）
python "Project/Binary Classification with a Bank Dataset/main.py" --select 0.01
```

运行结束后（默认行为），脚本会把提交文件写入 `Project/Binary Classification with a Bank Dataset/submission.csv`，你也可以用 `--out` 指定输出路径。

2) 全量训练（`--full`） — 使用全量数据与更大模型（资源消耗高）：

```bash
python "Project/Binary Classification with a Bank Dataset/main.py" --full
```

注意：`--full` 会启用更大的模型与并行化（默认），有可能导致内存不足或进程被系统终止（OOM）。强烈建议先用 `--select` 做一次近似运行，或在大内存机器/集群上运行。

### 监控与故障排查

- 实时输出日志并后台运行：

```bash
# 将输出重定向到日志文件并后台执行
PYTHONUNBUFFERED=1 python "Project/Binary Classification with a Bank Dataset/main.py" --full > run_full.log 2>&1 &
tail -f run_full.log
```

- 当资源受限时的建议：
	- 使用 `--select 0.05` 或更小的采样比例先跑一遍；
	- 在 `main.py` 中把基础学习器的 `n_estimators` 与 `n_jobs` 调小；
	- 在内存紧张时，用 `--full` 前先确保机器/节点有足够内存与 CPU；

- 如果遇到进程被系统杀死或 `TerminatedWorkerError`（Joblib worker 被干掉）：

```bash
dmesg | grep -i -E 'killed process|oom|Out of memory' -A3
```

或使用 `top` / `htop` 实时查看内存/CPU 使用情况。常见解决方案是降低 `--select`、减小 `n_estimators`、或在更大的机器上运行。

### 输出与验证

- `main.py` 在训练完成后会打印验证集上的 ROC AUC（Validation ROC AUC）以供参考；
- 在全量运行完成后，脚本还会打印 submission 文件的前 10 行（submission head），并把文件写入 `--out` 指定的路径（默认 `Project/Binary Classification with a Bank Dataset/submission.csv`）。

### 常见问题与解决办法

- Problem: KeyError: "['contacted', 'total_contacts'] not in index" — 原因：在对测试数据做变换前必须先运行 `dl.merge_columns()` 生成派生列；`main.py` 已处理该顺序但本地调用时请确保先合并列。
- Problem: TypeError in `sklearn.clone` for XGBoost wrapper — 仓库中已修复，`XgBoost.py` 接受 `**kwargs` 以支持 clone 操作。

### 调试与复现建议

- 为 CI 或本地快速复现，可把一份 smoke 脚本放到 `dev/run_smoke.py`，在 CI 中只运行 `--select 0.01` 的检查。若需要我可以帮你添加该脚本。

---

