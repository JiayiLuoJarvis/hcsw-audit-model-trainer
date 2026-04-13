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
