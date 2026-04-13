"""
训练入口脚本

用法:
  python train.py                  # 训练一批（默认 BATCH_LIMIT 条），完成后退出
  python train.py --all            # 循环训练，直到没有新数据为止
  python train.py --force          # 忽略最小样本数限制，强制训练
  python train.py --from-id 0      # 从指定流水 ID 重新训练（覆盖 checkpoint）
  python train.py --limit 10000    # 限制每批读取条数
  python train.py --status         # 查看当前版本状态，不执行训练
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# 将项目根目录加入 sys.path，支持直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from training.trainer import run, show_status

logging.basicConfig(
    level=logging.DEBUG if settings.LOG_DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# 同时写文件日志
_log_dir = settings.LOG_DIR
os.makedirs(_log_dir, exist_ok=True)
_file_handler = logging.FileHandler(
    os.path.join(_log_dir, "training.log"), encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="银行流水科目 BGE Fine-tune 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 MIN_TRAIN_SAMPLES 限制，强制训练（调试用）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多从数据库读取的流水条数（默认取 .env BATCH_LIMIT）",
    )
    parser.add_argument(
        "--from-id",
        type=int,
        default=None,
        dest="from_id",
        help="从指定流水 ID 开始读取（覆盖 checkpoint 的 last_id）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="循环训练所有未训批次，直到没有新数据为止",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查看当前版本状态和 checkpoint，不执行训练",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    logger.info(
        "启动训练 | force=%s | limit=%s | from_id=%s | all=%s",
        args.force,
        args.limit,
        args.from_id,
        args.run_all,
    )
    run(force=args.force, limit=args.limit, from_id=args.from_id, run_all=args.run_all)


if __name__ == "__main__":
    main()
