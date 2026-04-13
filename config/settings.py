from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# 计算项目根目录（config/ 的上一级）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 只读库（训练数据来源，银行流水 / 科目树，严禁写入）──────
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "hcsw_audit"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""

    # ── 模型路径 ─────────────────────────────────────────────────
    BASE_MODEL_PATH: str = os.path.join(_ROOT, "models", "base")
    FINETUNED_MODEL_DIR: str = os.path.join(_ROOT, "models", "finetuned")

    # ── 训练参数 ─────────────────────────────────────────────────
    BATCH_LIMIT: int = 50000
    MIN_TRAIN_SAMPLES: int = 200
    MAX_SAMPLES_PER_SUBJECT: int = 500

    # ── 日志 ─────────────────────────────────────────────────────
    LOG_DIR: str = os.path.join(_ROOT, "logs")
    LOG_DEBUG: bool = False

    # ── SCP 部署 ─────────────────────────────────────────────────
    DEPLOY_HOST: str = ""
    DEPLOY_USER: str = ""
    DEPLOY_REMOTE_DIR: str = ""
    DEPLOY_KEY_PATH: str = ""
    DEPLOY_SERVICE_NAME: str = "subject-matcher"


settings = Settings()
