# hcsw-audit-model-trainer — 快速上手

银行流水科目模型独立训练工具，负责从只读业务库读取标注数据、Fine-tune BGE 语义模型，并将训练产物 SCP 上传到云端 Web 服务。

---

## 环境要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.11 | 3.13 |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 5 GB（含模型权重） | SSD 10 GB+ |

---

## 1. 创建私有虚拟环境

```powershell
# 在项目根目录执行
cd hcsw-audit-model-trainer
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

> `sentence-transformers` 和 `torch` 体积较大，首次安装约需 5–10 分钟。  
> 有 NVIDIA 显卡时，删除 `requirements.txt` 中的 `--index-url` 行，改为 `torch>=2.3`。

---

## 3. 配置 .env

```powershell
copy .env.example .env
# 编辑 .env，填入数据库密码、部署服务器信息
```

关键配置项：

| 变量 | 说明 |
|------|------|
| `DB_HOST` | 只读业务库地址（`172.16.14.125`） |
| `DB_PASSWORD` | 只读库密码 |
| `BASE_MODEL_PATH` | 基础预训练模型目录（见第 4 步） |
| `DEPLOY_HOST` | 云端服务器 IP |
| `DEPLOY_USER` | SSH 用户名 |
| `DEPLOY_REMOTE_DIR` | 云端模型存放目录 |
| `DEPLOY_KEY_PATH` | SSH 私钥路径（留空时交互输入密码） |

---

## 4. 放置基础预训练模型

将 BGE 模型文件夹复制或软链接到 `models/base/`（对应 `.env` 中的 `BASE_MODEL_PATH`）。

**从原项目复制（推荐）：**

```powershell
# 将 bge-base-zh-v1.5 复制过来
Copy-Item -Recurse ..\hcsw-audit-banl-statement\models\bge-base-zh-v1.5 .\models\base
```

**或指定绝对路径（无需复制）：**

在 `.env` 中设置：
```ini
BASE_MODEL_PATH=E:\project\hcsw-audit\hcsw-audit-bank-statement\hcsw-audit-banl-statement\models\bge-base-zh-v1.5
```

---

## 5. 执行训练

```powershell
# 训练一批（默认 BATCH_LIMIT 条），完成后退出
python train.py

# 循环训练所有未训批次，直到没有新数据
python train.py --all

# 强制训练（忽略最小样本数，数据量少时调试用）
python train.py --force

# 从指定流水 ID 重新开始（覆盖 checkpoint）
python train.py --from-id 0

# 限制每批读取条数
python train.py --limit 10000

# 查看当前版本状态
python train.py --status
```

---

## 6. 部署到云端

```powershell
# 上传最新 finetuned 版本
python scripts/deploy.py

# 上传指定版本
python scripts/deploy.py --version v3

# 上传后重启云端服务
python scripts/deploy.py --reload

# 仅打印将要执行的操作，不实际上传
python scripts/deploy.py --dry-run
```

---

## 训练流程详解

### 整体架构

```
train.py (CLI 入口)
    └─ training/trainer.py     ← 版本管理、断点、训练主循环
         ├─ training/data_builder.py  ← 数据拉取与三元组构建
         ├─ training/evaluator.py     ← 验证集 Top-1 准确率评估
         └─ db/reader.py              ← 只读库 SQL 查询（aiomysql）
```

---

### 第一阶段：数据读取（`db/reader.py`）

通过 `aiomysql` 异步连接只读业务库，执行以下 SQL：

```sql
SELECT
    bas.id, bas.type, bas.money, ba.currency,
    bas.summary, bas.trade_type, ac.name AS correct_subject
FROM bank_account_statement bas
INNER JOIN account_chart ac
    ON ac.id = bas.account_chart_id AND ac.is_deleted = 0 AND ac.company_id = 0
INNER JOIN bank_account ba
    ON ba.id = bas.bank_account_id AND ba.is_deleted = 0
WHERE bas.is_deleted = 0
  AND bas.id > {last_id}          -- 断点续训，避免重复
  AND bas.account_chart_id > 0   -- 仅含科目标注的记录
  AND bas.summary IS NOT NULL AND bas.summary != ''
ORDER BY bas.id ASC
LIMIT {BATCH_LIMIT}              -- 默认 50000 条
```

过滤规则：
- `company_id = 0`：只取公共科目（共享科目树），防止跨公司数据污染
- `id > last_id`：断点恢复，避免重复训练同一批数据

---

### 第二阶段：Anchor 归一化（`data_builder._normalize_anchor`）

每条流水记录被转换为一段带语义前缀的文本作为 **anchor**：

```
{收支方向} {币种标签} {金额档位} {summary} {trade_type}
```

| 字段 | 规则 |
|------|------|
| 收支方向 | `type=1` → `[收入]`；`type=2` → `[支出]` |
| 币种标签 | 非人民币（CNY/RMB）则附加 `[USD]`、`[HKD]` 等 |
| 金额档位 | `≥500000` → `[大额]`；`≥10000` → `[中额]`；`>0` → `[小额]` |
| 正文 | `{summary} {trade_type}` 拼接，合并多余空白 |

示例：
```
[支出] [HKD] [大额] 代付港币电汇 SWIFT转账
```

---

### 第三阶段：数据集划分与均衡（`data_builder.build_train_val`）

1. 将原始记录随机打乱，按 **val_ratio=20%** 切分为训练集与验证集
2. **训练集** 经过两步均衡处理：
   - **按科目精确去重**：相同 anchor 文本在同一科目下只保留一条，消除重复噪声
   - **超 cap 随机截断**：每个科目最多保留 `MAX_SAMPLES_PER_SUBJECT`（默认 500 条），防止高频科目垄断
3. **验证集** 不去重不截断，保留原始频率分布，使评估结果更贴近生产
4. 为每条训练/验证记录随机采样一个**负例（negative）**：从其余科目中随机取一个科目名

最终产出 `TrainingTriplet(anchor, positive, negative)` 三元组列表。

---

### 第四阶段：Fine-tune（`trainer._run_one_batch`）

#### 模型加载策略

- 若 `models/finetuned/` 下已存在历史版本（`v1`、`v2` 等），则加载**最新版本**继续训练（增量 Fine-tune）
- 若无历史版本，则从 `models/base/`（BGE 预训练模型）开始

#### 训练超参

| 参数 | 值 |
|------|----|
| 损失函数 | `MultipleNegativesRankingLoss`（MNR Loss） |
| Batch size | 128 |
| Epochs | 2 |
| Learning rate | 2e-6 |
| Warmup steps | `ceil(steps_per_epoch × 10%)` |

`MultipleNegativesRankingLoss` 将批内其他样本的 positive 作为额外负例，无需显式构造难负例，对语义相似度任务效果显著。

---

### 第五阶段：验证与版本决策（`evaluator.evaluate`）

评估指标为 **Top-1 余弦相似度准确率**：

1. 对验证集中所有唯一科目名用模型编码，得到科目嵌入矩阵（`normalize=True`）
2. 对所有验证 anchor 编码
3. 计算 `anchor_embs @ subject_embs.T`，取 `argmax` 预测科目
4. 统计命中正确科目的比例

训练前后各评估一次：

```
baseline_acc  →  （Fine-tune）  →  new_acc
```

**版本保存规则**：
- `new_acc > baseline_acc`：保存为新版本 `vN`，写入 `versions.json`，设为 `active_version`
- `new_acc ≤ baseline_acc`：模型准确率未提升，丢弃本次训练结果，保留旧路径

---

### 第六阶段：Checkpoint 与版本文件管理

#### `checkpoints/training_checkpoint.json`

记录断点，防止意外中断后重复训练相同数据：

```json
// 批次进行中（训练完成前）
{
  "last_id": 0,           // 本批起始 ID（前一批结束位）
  "pending_last_id": 696130,  // 本批目标结束 ID
  "pending_since": "2026-04-11T09:15:21"
}

// 批次训练完成后
{
  "last_id": 696130       // 已完成到此 ID，下批从此继续
}
```

若启动时 checkpoint 同时含 `last_id` 和 `pending_last_id`，说明上次训练未正常完成，系统会自动**重新训练该批次**（从 `last_id` 而非 `pending_last_id` 开始）。

写入采用**原子操作**（先写 `.tmp` 再 `os.replace`），防止写入中断造成文件损坏。

#### `models/finetuned/versions.json`

```json
{
  "active_version": "v2",
  "versions": [
    {
      "version": "v1",
      "model_path": "models/finetuned/v1",
      "accuracy": 0.923100,
      "train_count": 18432,
      "trained_at": "2026-04-10T14:23:00",
      "note": ""
    },
    {
      "version": "v2",
      "model_path": "models/finetuned/v2",
      "accuracy": 0.941500,
      "train_count": 22018,
      "trained_at": "2026-04-11T09:55:00",
      "note": ""
    }
  ]
}
```

云端 Web 服务启动时读取此文件，加载 `active_version` 对应的模型目录。

---

### 完整流程时序图

```
train.py
  │
  ├─ 读取 checkpoint（last_id）
  │
  ├─ [loop] _run_one_batch(last_id)
  │    │
  │    ├─ db/reader: SELECT 流水（id > last_id, LIMIT batch_limit）
  │    ├─ data_builder: anchor 归一化 + 训练/验证集切分 + 三元组构建
  │    │
  │    ├─ checkpoint 写入 pending（记录"训练中"状态）
  │    │
  │    ├─ 加载模型（最新 finetuned 版本 or base）
  │    ├─ evaluator: 计算 baseline_acc
  │    │
  │    ├─ SentenceTransformer.fit（MNR Loss, 2 epochs）
  │    │
  │    ├─ evaluator: 计算 new_acc
  │    │
  │    ├─ [new_acc > baseline] 保存 vN，更新 versions.json
  │    │   [否则] 丢弃
  │    │
  │    └─ checkpoint 更新为 last_id=new_last_id
  │
  └─ [--all] 继续下一批，直到无数据；[单批] 退出
```

---

### 关键配置参数速查

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BATCH_LIMIT` | 50000 | 每批最多读取的流水条数 |
| `MIN_TRAIN_SAMPLES` | 200 | 低于此数跳过训练（`--force` 可绕过） |
| `MAX_SAMPLES_PER_SUBJECT` | 500 | 单科目最大训练样本数（均衡截断） |
| `BASE_MODEL_PATH` | `models/base/` | 初始 BGE 预训练模型目录 |
| `FINETUNED_MODEL_DIR` | `models/finetuned/` | Fine-tune 产物存放目录 |

训练完成后，产物写入：

| 文件/目录 | 说明 |
|-----------|------|
| `models/finetuned/vN/` | 微调模型权重 |
| `models/finetuned/versions.json` | 版本清单（Web 服务读取此文件激活版本） |
| `checkpoints/training_checkpoint.json` | 断点记录（last\_id，防重复训练） |

---

## 6. 上传模型到云端服务器

```powershell
# 上传最新 finetuned 版本
python scripts/deploy.py

# 上传指定版本
python scripts/deploy.py --version v3

# 上传并重启云端服务（需 DEPLOY_SERVICE_NAME 配置）
python scripts/deploy.py --reload
```

**deploy.py 执行流程：**

1. 通过 SSH/SFTP 连接云端服务器（使用私钥或交互密码）
2. 上传 `models/finetuned/vN/` 目录（按文件大小跳过未变化文件）
3. 上传 `models/finetuned/versions.json`（Web 服务据此切换激活版本）
4. （可选）执行 `systemctl restart <service>`

---

## 7. 云端 Web 服务接入约定

Web 服务只需读取云端的 `versions.json`，根据 `active_version` 字段加载对应版本：

```json
{
  "active_version": "v3",
  "versions": [
    {
      "version": "v3",
      "model_path": "/data/hcsw-audit/models/finetuned/v3",
      "accuracy": 0.8714,
      "train_count": 49800,
      "trained_at": "2026-04-09T10:15:00",
      "note": ""
    }
  ]
}
```

Web 服务启动时读取此文件，加载 `model_path` 指向的模型（无需重新部署代码）。

---

## 版本回滚

如需回滚到旧版本，只需编辑云端的 `versions.json`，将 `active_version` 改为目标版本号，然后重启 Web 服务。

```powershell
# 远端操作示例
# 编辑 /data/hcsw-audit/models/finetuned/versions.json
# 将 "active_version": "v3" 改为 "active_version": "v2"
# 然后重启服务
systemctl restart subject-matcher
```

---

## 重要规范

| 规范 | 说明 |
|------|------|
| 只读库 | `172.16.14.125/hcsw_audit`，**严禁写入** |
| 训练触发 | **仅手动执行**，无定时任务 |
| 版本管理 | 全部通过 `versions.json` 记录，不依赖数据库 |
| 模型上传 | 训练完成后手动执行 `python scripts/deploy.py` |
