"""
将训练好的模型通过 SCP（SFTP）上传到云端服务器

用法:
  python scripts/deploy.py                  # 上传最新 finetuned 版本
  python scripts/deploy.py --version v3     # 上传指定版本
  python scripts/deploy.py --reload         # 上传后重启云端服务
  python scripts/deploy.py --dry-run        # 仅打印将要执行的操作，不实际上传

上传内容:
  1. models/finetuned/vN/  → DEPLOY_REMOTE_DIR/vN/
  2. models/finetuned/versions.json → DEPLOY_REMOTE_DIR/versions.json

云端 Web 服务启动时读取 versions.json，据此加载 active_version 对应的模型目录。
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
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


def _sftp_upload_dir(sftp, local_dir: str, remote_dir: str, dry_run: bool = False) -> None:
    """递归上传目录，按文件大小跳过未变化的文件。"""
    if dry_run:
        print(f"  [dry-run] 上传目录: {local_dir} -> {remote_dir}")
        return

    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        sftp.mkdir(remote_dir)

    entries = sorted(os.listdir(local_dir))
    for idx, name in enumerate(entries, 1):
        local_path = os.path.join(local_dir, name)
        remote_path = f"{remote_dir}/{name}"
        if os.path.isdir(local_path):
            _sftp_upload_dir(sftp, local_path, remote_path)
        else:
            local_size = os.path.getsize(local_path)
            try:
                remote_size = sftp.stat(remote_path).st_size
                if remote_size == local_size:
                    print(f"  [{idx}/{len(entries)}] 跳过（未变化）: {name}")
                    continue
            except FileNotFoundError:
                pass
            size_mb = local_size / 1024 / 1024
            print(f"  [{idx}/{len(entries)}] 上传: {name}  ({size_mb:.1f} MB)")
            sftp.put(local_path, remote_path)


def _sftp_upload_file(sftp, local_path: str, remote_path: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"  [dry-run] 上传文件: {local_path} -> {remote_path}")
        return
    print(f"  上传: {os.path.basename(local_path)}")
    sftp.put(local_path, remote_path)


def _get_ssh_client(password: str | None):
    try:
        import paramiko
    except ImportError:
        print("错误：缺少 paramiko，请执行：pip install paramiko")
        sys.exit(1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507

    connect_kwargs: dict = {
        "hostname": settings.DEPLOY_HOST,
        "username": settings.DEPLOY_USER,
        "timeout": 30,
    }
    key_path = settings.DEPLOY_KEY_PATH
    if key_path and os.path.exists(key_path):
        connect_kwargs["key_filename"] = key_path
        print(f"使用私钥连接: {key_path}")
    elif password:
        connect_kwargs["password"] = password
    else:
        connect_kwargs["password"] = getpass.getpass(
            f"SSH 密码 ({settings.DEPLOY_USER}@{settings.DEPLOY_HOST}): "
        )

    client.connect(**connect_kwargs)
    return client


def deploy(version: str, reload: bool = False, dry_run: bool = False) -> None:
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

    print(f"\n目标服务器: {settings.DEPLOY_USER}@{settings.DEPLOY_HOST}")
    print(f"远端目录  : {settings.DEPLOY_REMOTE_DIR}")
    print(f"上传版本  : {version}")
    if dry_run:
        print("模式      : dry-run（不实际上传）\n")

    client = _get_ssh_client(password=None)
    try:
        sftp = client.open_sftp()

        # 确保远端根目录存在
        if not dry_run:
            try:
                sftp.stat(settings.DEPLOY_REMOTE_DIR)
            except FileNotFoundError:
                sftp.mkdir(settings.DEPLOY_REMOTE_DIR)

        # 上传模型目录
        remote_version_dir = f"{settings.DEPLOY_REMOTE_DIR}/{version}"
        print(f"\n[1/2] 上传模型权重 -> {remote_version_dir}")
        _sftp_upload_dir(sftp, local_model_dir, remote_version_dir, dry_run)

        # 上传 versions.json
        remote_versions = f"{settings.DEPLOY_REMOTE_DIR}/versions.json"
        print(f"\n[2/2] 上传 versions.json -> {remote_versions}")
        _sftp_upload_file(sftp, _VERSIONS_FILE, remote_versions, dry_run)

        sftp.close()

        # 可选：重启云端服务
        if reload and not dry_run:
            service = settings.DEPLOY_SERVICE_NAME
            print(f"\n重启服务: systemctl restart {service}")
            _, stdout, stderr = client.exec_command(
                f"sudo systemctl restart {service}"
            )
            exit_code = stdout.channel.recv_exit_status()
            if exit_code == 0:
                print(f"服务 {service} 已重启")
            else:
                err = stderr.read().decode().strip()
                print(f"警告：重启服务失败（exit={exit_code}）: {err}")

    finally:
        client.close()

    print(f"\n完成！版本 {version} 已部署到 {settings.DEPLOY_HOST}")
    print("云端 Web 服务将在下次启动（或重载）时加载新版本。")


def main() -> None:
    parser = argparse.ArgumentParser(description="上传训练模型到云端服务器（SCP/SFTP）")
    parser.add_argument(
        "--version",
        default=None,
        help="要上传的版本号（如 v3），默认取本地最新版本",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="上传完成后重启云端 systemd 服务（需 DEPLOY_SERVICE_NAME 配置）",
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

    deploy(version=version, reload=args.reload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
