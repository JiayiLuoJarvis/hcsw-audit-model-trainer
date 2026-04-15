# hcsw-audit-model-trainer

银行流水科目 BGE 模型独立训练工具。从只读业务库读取标注流水，Fine-tune BGE 语义模型，通过 SCP 将训练产物部署到云端 Web 服务。

---

## 环境初始化（首次运行）

```powershell
# 1. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编辑 .env，填入数据库密码及云端服务器信息
#    （.env 已从 .env.example 复制，只需修改对应值）
notepad .env
```

**BGE 基础模型**：将 `bge-base-zh-v1.5`（或 `bge-small-zh-v1.5`）放到 `.env` 中 `BASE_MODEL_PATH` 指定的目录，默认为 `models/base/`。

---

## 训练

```powershell
# 训练一批（默认 BATCH_LIMIT 条），完成后退出
python train.py

# 循环训练所有未训批次，直到没有新数据
python train.py --all

# 强制训练（忽略最小样本数，数据量少时调试用）
python train.py --force

# 从指定流水 ID 重新开始（覆盖 checkpoint）
python train.py --from-id 0

# 限制本次读取条数
python train.py --limit 10000

# 查看当前版本状态
python train.py --status
```

训练成功后产物：

| 路径 | 说明 |
|------|------|
| `models/finetuned/vN/` | 微调模型权重 |
| `models/finetuned/versions.json` | 版本清单，Web 服务读此文件切换模型 |
| `checkpoints/training_checkpoint.json` | 断点记录（last_id），防重复训练 |

---

## 部署到云端服务器

```powershell
# 上传最新 finetuned 版本（自动识别本地最新版本）
python scripts/deploy.py

# 上传指定版本
python scripts/deploy.py --version v3

source venv/bin/activate && python scripts/deploy.py

# 上传后重启云端服务（需 .env 中配置 DEPLOY_SERVICE_NAME）
python scripts/deploy.py --reload

# 预览操作，不实际上传
python scripts/deploy.py --dry-run
```

**deploy.py 执行流程：**
1. SSH/SFTP 连接云端（使用 `DEPLOY_KEY_PATH` 私钥，或交互输入密码）
2. 上传 `models/finetuned/vN/` → `DEPLOY_REMOTE_DIR/vN/`（按文件大小跳过未变化文件）
3. 上传 `models/finetuned/versions.json` → `DEPLOY_REMOTE_DIR/versions.json`
4. 可选：执行 `sudo systemctl restart <DEPLOY_SERVICE_NAME>`

---

## 版本回滚

无需重新训练，直接编辑云端 `versions.json`，将 `active_version` 改为目标版本号，重启服务即生效：

```bash
# 在云端服务器执行
vim /data/hcsw-audit/models/finetuned/versions.json
# 将 "active_version": "v3" 改为 "active_version": "v2"
sudo systemctl restart subject-matcher
```

---

## 云端 Web 服务接入

Web 服务启动时读取 `versions.json`，加载 `active_version` 对应目录的模型权重，无需修改代码：

```json
{
  "active_version": "v3",
  "versions": [
    {
      "version": "v3",
      "model_path": "/data/hcsw-audit/models/finetuned/v3",
      "accuracy": 0.8714,
      "train_count": 49800,
      "trained_at": "2026-04-09T10:15:00"
    }
  ]
}
```

---

## 重要规范

| 规范 | 说明 |
|------|------|
| 只读库 | `172.16.14.125/hcsw_audit`，**严禁写入** |
| 训练触发 | **仅手动执行**，无定时任务 |
| 版本管理 | 全部通过 `versions.json`，不依赖数据库 |
| 代码规范 | 见 [.github/CODE_STANDARDS.md](.github/CODE_STANDARDS.md) |
| 详细配置 | 见 [docs/setup.md](docs/setup.md) |
