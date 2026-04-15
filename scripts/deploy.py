"""
将训练好的模型通过 SCP 上传到云端服务器

用法:
  python scripts/deploy.py                  # 上传最新 finetuned 版本
  python scripts/deploy.py --version v3     # 上传指定版本
  python scripts/deploy.py --dry-run        # 仅打印将要执行的操作，不实际上传

上传内容:
  1. models/finetuned/vN/  → DEPLOY_REMOTE_DIR/vN/
  2. models/finetuned/versions.json → DEPLOY_REMOTE_DIR/versions.json

云端 Web 服务启动时读取 versions.json，据此加载 active_version 对应的模型目录。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERSIONS_FILE = os.path.join(settings.FINETUNED_MODEL_DIR, "versions.json")


def _latest_local_version() -> str | None:
    base = settings.FINETUNED_MODEL_DIR
    if not os.path.isdir(base):
        return None
    dirs = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    if not dirs:
        return None
    return f"v{max(int(d[1:]) for d in dirs)}"


def _scp(src: str, dest: str, recursive: bool = False, dry_run: bool = False) -> None:
    """调用系统 scp 命令，密码由 SSH 直接在终端提示输入。"""
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if recursive:
        cmd.append("-r")
    cmd += [src, dest]
    print(f"  执行: {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"错误：scp 上传失败（exit={result.returncode}）")
            sys.exit(result.returncode)


def deploy(version: str, dry_run: bool = False) -> None:
    local_model_dir = os.path.join(settings.FINETUNED_MODEL_DIR, version)
    if not os.path.isdir(local_model_dir):
        print(f"错误：本地模型目录不存在: {local_model_dir}")
        sys.exit(1)

    if not os.path.exists(_VERSIONS_FILE):
        print(f"错误：versions.json 不存在: {_VERSIONS_FILE}")
        sys.exit(1)

    if not settings.DEPLOY_HOST or not settings.DEPLOY_USER or not settings.DEPLOY_REMOTE_DIR:
        print("错误：请在 .env 中配置 DEPLOY_HOST / DEPLOY_USER / DEPLOY_REMOTE_DIR")
        sys.exit(1)

    target = f"{settings.DEPLOY_USER}@{settings.DEPLOY_HOST}"
    remote_dir = settings.DEPLOY_REMOTE_DIR

    print(f"\n目标服务器: {target}")
    print(f"远端目录  : {remote_dir}")
    print(f"上传版本  : {version}")
    if dry_run:
        print("模式      : dry-run（不实际上传）\n")

    # 上传模型目录
    print(f"\n[1/2] 上传模型权重 -> {remote_dir}/{version}")
    _scp(local_model_dir, f"{target}:{remote_dir}/", recursive=True, dry_run=dry_run)

    # 上传 versions.json
    print(f"\n[2/2] 上传 versions.json -> {remote_dir}/versions.json")
    _scp(_VERSIONS_FILE, f"{target}:{remote_dir}/versions.json", dry_run=dry_run)

    print(f"\n完成！版本 {version} 已部署到 {settings.DEPLOY_HOST}")


def main() -> None:
    parser = argparse.ArgumentParser(description="上传训练模型到云端服务器（SCP）")
    parser.add_argument(
        "--version",
        default=None,
        help="要上传的版本号（如 v3），默认取本地最新版本",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="仅打印操作，不实际上传",
    )
    args = parser.parse_args()

    version = args.version or _latest_local_version()
    if version is None:
        print("错误：本地没有任何 finetuned 版本，请先执行 python train.py")
        sys.exit(1)

    deploy(version=version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
