运行指南 — Binary Classification with a Bank Dataset
===============================================

概述
----
本项目包含一个小型训练/推理流水线，主要入口是 `main.py`（位于项目子目录 `Project/Binary Classification with a Bank Dataset/`）。
本文档说明如何准备环境、运行“安全模式”快速测试以及运行“全量模式”来生成提交文件（submission.csv）。

先决条件 / 依赖
-----------------
- Python 3.10+（推荐使用虚拟环境或 dev container）
- 建议安装依赖：

```bash
pip install numpy pandas scipy scikit-learn xgboost seaborn joblib
```

（如果你在企业/离线环境，请根据本机包管理策略安装相应包。）

数据位置
---------
本仓库假定数据文件在相对路径：

- `Project/Binary Classification with a Bank Dataset/train.csv`
- `Project/Binary Classification with a Bank Dataset/test.csv`

请确保上述文件存在且列格式与代码预期一致（训练集须包含 `y` 列，测试集未必包含）。

主要文件说明
----------------
- `utils/Data_loader/Data_loader.py`：负责数据清洗、合并派生列、编码/缩放与特征选择。已实现训练/测试可复用的 transformer。 
- `utils/Models/*.py`：模型封装（RandomForest、LogisticRegression、XGBoost 等），已做少量兼容性改动。
- `main.py`：入口脚本，提供命令行参数 `--full`、`--select`、`--out`。

运行模式说明
----------------
1) 安全模式（默认） — 推荐用于本地、CI 快速验证：

```bash
# 在仓库根目录运行（安全模式：小样本、较小模型、串行训练）
python "Project/Binary Classification with a Bank Dataset/main.py"

# 指定更小样本以更快完成（用于调试）：
python "Project/Binary Classification with a Bank Dataset/main.py" --select 0.01
```

运行结束后，默认会在 `Project/Binary Classification with a Bank Dataset/submission.csv`（或 `--out` 指定路径）写入结果。

2) 全量模式（--full） — 使用全量数据与较大模型（高资源消耗）：

```bash
python "Project/Binary Classification with a Bank Dataset/main.py" --full
```

注意：`--full` 会使用更大的模型与并行（默认），可能需要大量内存/CPU；在资源有限的机器上可能被操作系统杀掉（OOM）。建议先用 `--select` 做一次近似试跑。

常用运行技巧
----------------
- 输出实时日志并后台运行：

```bash
PYTHONUNBUFFERED=1 python "Project/Binary Classification with a Bank Dataset/main.py" --full > run_full.log 2>&1 &
tail -f run_full.log
```

- 在资源有限时：
  - 使用 `--select 0.05` 或更小先跑小批量；
  - 或者在 `main.py` 中把基学习器的 `n_estimators` / `n_jobs` 降低；
  - 将 `--full` 去掉以使用安全默认。

监控与故障排查
-------------------
- 如果进程被终止（exit code 143 或 Joblib 的 TerminatedWorkerError），常见原因是内存耗尽：检查系统日志：

```bash
dmesg | grep -i -E 'killed process|oom|Out of memory' -A3
```

- 使用 `top` / `htop` 实时监控内存/CPU：

```bash
top
# 或
htop
```

- 若发现 OOM：
  - 降低 `--select` 采样；
  - 降低模型复杂度（在 `main.py` 或模型类中调整 `n_estimators`、`n_jobs` 等）；
  - 在更大内存的机器/集群上运行全量。

输出与验证
---------------
- `main.py` 在训练完成后会输出验证集的 ROC AUC（Validation ROC AUC）。
- 全量模式完成后会额外打印 submission 的前 10 行（submission head），并把提交文件写到 `--out` 指定路径。

常见问题与解决
------------------
- Problem: KeyError: "['contacted', 'total_contacts'] not in index" — 原因：在对 test 数据做 transform 前必须先调用 `dl.merge_columns()` 生成派生列；`main.py` 已处理该顺序。
- Problem: TypeError in sklearn.clone for XGBoost wrapper — 已在 `XgBoost.py` 修复以接收 `**kwargs`。

如果需要我代为跑 full（注意资源与风险），请先确认执行环境的内存/CPU，否则建议按上面步骤逐步扩大样本。

如需更多：我可以把一份更详细的运行资源估计（预估内存/时间）加入 README，也可以把 smoke 脚本保存到 `dev/run_smoke.py` 供 CI / 本地复现。

---
（自动生成的运行指南；如需把内容合并入项目根 README 或其他位置告诉我具体路径）
