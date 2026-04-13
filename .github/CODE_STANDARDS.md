# hcsw-audit-model-trainer — 代码规范

## 1. Python 版本与环境

| 项目 | 规范 |
|------|------|
| Python 版本 | **3.11+**，语法须兼容 3.11 |
| 虚拟环境 | 项目根目录 `.venv/`，**不允许使用全局 Python 环境** |
| 类型检查 | Pyright `typeCheckingMode: standard`，提交前须通过 |

---

## 2. 代码风格

### 文件头

所有 `.py` 文件第一个非空行必须为：

```python
from __future__ import annotations
```

### 命名规则

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/函数/变量 | `snake_case` | `build_triplets`, `last_id` |
| 类 | `PascalCase` | `TrainingTriplet`, `Settings` |
| 常量 | `UPPER_SNAKE_CASE` | `_CHECKPOINT_FILE` |
| 私有函数/变量 | 单下划线前缀 | `_read_checkpoint()` |
| 类型别名 | `PascalCase` | `RawPair = dict[str, str]` |

### 导入顺序（isort 规则）

```python
# 1. __future__
from __future__ import annotations

# 2. 标准库
import json
import os

# 3. 第三方库（空一行）
import aiomysql
from sentence_transformers import SentenceTransformer

# 4. 项目内模块（空一行）
from config.settings import settings
from training.data_builder import TrainingTriplet
```

---

## 3. 类型标注

- 所有**公开函数**（无下划线前缀）须标注参数和返回值类型
- 允许使用 `X | Y` 联合类型（Python 3.10+ 语法）
- `list[dict]`、`dict[str, int]` 等小写泛型（PEP 585），不用 `List`、`Dict`

```python
# Good
def build_triplets(raw_pairs: list[dict]) -> list[TrainingTriplet]: ...

# Bad
def build_triplets(raw_pairs: List[Dict]) -> List[TrainingTriplet]: ...
```

---

## 4. 日志规范

- 使用 `logging`，**禁止使用 `print()` 打印业务日志**（`show_status` 等 CLI 输出除外）
- Logger 在模块顶层声明：`logger = logging.getLogger(__name__)`
- 使用 `%s` 占位符，**不使用 f-string 拼接日志**（避免在 DEBUG 关闭时产生字符串开销）

```python
# Good
logger.info("读取流水 %d 条，起始 ID: %d", len(rows), last_id)

# Bad
logger.info(f"读取流水 {len(rows)} 条，起始 ID: {last_id}")
```

---

## 5. 数据库操作规范

### 只读库（`172.16.14.125/hcsw_audit`）

- **严禁执行 INSERT / UPDATE / DELETE**
- 所有查询只能在 `db/reader.py` 中实现，其他模块通过调用此模块函数访问
- 连接用完立即 `conn.close()`，不使用全局连接

### 可写库

本项目**不连接可写库**，版本管理改为 JSON 文件。

---

## 6. 文件原子写入

训练状态（`training_checkpoint.json`）和版本清单（`versions.json`）必须使用原子写入，防止进程中断导致文件损坏：

```python
# 正确做法
tmp = target_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, target_path)  # 原子替换
```

---

## 7. 异常处理

- 不要静默吞掉所有异常，必须至少 `logger.warning(...)` 记录
- IO 操作（读写文件、数据库）须有 `try/except`，失败时记录日志并决定是否继续
- 训练主流程（`trainer.py/run()`）中，数据库连接失败应直接抛出，由调用方决策

---

## 8. 禁止事项

| 禁止 | 替代方案 |
|------|----------|
| 直接写死 IP / 密码 | 从 `settings` 读取 `.env` 配置 |
| `time.sleep()` 轮询 | 不涉及定时，无此场景 |
| 修改只读库数据 | 只允许 SELECT |
| 在 `training/` 之外导入 `torch` | 仅训练模块依赖 torch |
| 提交 `.env` 文件 | `.gitignore` 已排除 |
| 提交模型权重（`models/`） | `.gitignore` 已排除 |

---

## 9. 提交规范

Commit message 格式（参考 Conventional Commits）：

```
<type>(<scope>): <subject>

type:
  feat     新功能
  fix      Bug 修复
  refactor 重构（不改变功能）
  docs     仅文档变更
  chore    依赖更新、环境配置等
  perf     性能优化

示例:
  feat(trainer): 支持 --from-id 参数重置训练起点
  fix(deploy): 修复私钥路径为空时连接失败
  docs(setup): 更新云端部署步骤
```

---

## 10. 目录职责说明

| 目录/文件 | 职责 | 修改规则 |
|-----------|------|----------|
| `config/settings.py` | 配置（仅从 .env 读取） | 变量名全大写，提供默认值 |
| `db/reader.py` | 只读库查询 | 仅 SELECT，严禁写操作 |
| `training/data_builder.py` | 三元组构建 | 不依赖 IO 以外的外部状态 |
| `training/evaluator.py` | 模型评估 | 纯函数，无副作用 |
| `training/trainer.py` | 训练主流程 | 版本记录写 JSON，不写数据库 |
| `train.py` | CLI 入口 | 仅解析参数，不含业务逻辑 |
| `scripts/deploy.py` | SCP 上传 | 仅传输，不修改本地状态 |
