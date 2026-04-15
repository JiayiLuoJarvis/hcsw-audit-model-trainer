"""
Fine-tune 主训练逻辑

版本管理全部通过 models/finetuned/versions.json 记录，不依赖数据库。
断点恢复通过 checkpoints/training_checkpoint.json 实现。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from math import ceil

from typing import cast

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset

from config.settings import settings
from training.data_builder import TrainingTriplet, build_train_val_from_db
from training.evaluator import evaluate

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_FILE = os.path.join(_ROOT, "checkpoints", "training_checkpoint.json")
_VERSIONS_FILE = os.path.join(settings.FINETUNED_MODEL_DIR, "versions.json")


#  Checkpoint 


def _read_checkpoint() -> dict:
    if not os.path.exists(_CHECKPOINT_FILE):
        return {}
    try:
        with open(_CHECKPOINT_FILE, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("读取 checkpoint 失败: %s", exc)
        return {}


def _write_checkpoint(data: dict) -> None:
    """原子写入（tmp -> rename）防止写一半损坏。"""
    try:
        os.makedirs(os.path.dirname(_CHECKPOINT_FILE), exist_ok=True)
        tmp = _CHECKPOINT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CHECKPOINT_FILE)
    except Exception as exc:
        logger.warning("写入 checkpoint 失败: %s", exc)


#  Versions JSON 


def _read_versions() -> dict:
    if not os.path.exists(_VERSIONS_FILE):
        return {"active_version": "base", "versions": []}
    try:
        with open(_VERSIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("读取 versions.json 失败: %s", exc)
        return {"active_version": "base", "versions": []}


def _write_versions(data: dict) -> None:
    """原子写入 versions.json。"""
    try:
        os.makedirs(os.path.dirname(_VERSIONS_FILE), exist_ok=True)
        tmp = _VERSIONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _VERSIONS_FILE)
    except Exception as exc:
        logger.warning("写入 versions.json 失败: %s", exc)


def _record_version(
    version: str, model_path: str, accuracy: float, train_count: int
) -> None:
    """将新版本写入 versions.json 并设为 active_version。"""
    data = _read_versions()
    existing = {v["version"]: v for v in data.get("versions", [])}
    existing[version] = {
        "version": version,
        "model_path": model_path,
        "accuracy": round(accuracy, 6),
        "train_count": train_count,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "note": "",
    }
    data["active_version"] = version
    data["versions"] = list(existing.values())
    _write_versions(data)
    logger.info("versions.json 已更新，active_version -> %s", version)


#  版本号管理 


def _latest_version() -> str | None:
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


def _next_version() -> str:
    latest = _latest_version()
    if latest is None:
        return "v1"
    return f"v{int(latest[1:]) + 1}"


#  训练主流程 


def _run_one_batch(
    last_id: int,
    min_samples: int | None,
    batch_limit: int,
) -> tuple[bool, int]:
    """
    训练单批数据。

    Returns:
        (has_data, new_last_id)
        has_data=False 表示数据不足，调用方应停止循环。
    """
    batch_start = time.time()

    train_triplets, val_triplets, ids, new_last_id = build_train_val_from_db(
        min_samples=min_samples, limit=batch_limit, last_id=last_id
    )

    if not train_triplets:
        logger.info("无可用训练数据，退出")
        return False, last_id

    unique_subjects = len(
        {t.positive for t in train_triplets} | {t.positive for t in val_triplets}
    )
    logger.info(
        "数据准备完成：科目 %d 种 | 训练集 %d 条 | 验证集 %d 条",
        unique_subjects,
        len(train_triplets),
        len(val_triplets),
    )

    _write_checkpoint({
        "last_id": last_id,
        "pending_last_id": new_last_id,
        "pending_since": datetime.now().isoformat(timespec="seconds"),
    })

    latest = _latest_version()
    if latest is not None:
        base_path = os.path.join(settings.FINETUNED_MODEL_DIR, latest)
        logger.info("从最新 finetuned 版本 %s 开始训练: %s", latest, base_path)
    else:
        base_path = settings.BASE_MODEL_PATH
        logger.info("无 finetuned 版本，从 base 模型开始训练: %s", base_path)

    model = SentenceTransformer(base_path)

    baseline_acc = evaluate(model, val_triplets)
    logger.info("基准准确率: %.4f", baseline_acc)

    examples = [InputExample(texts=[t.anchor, t.positive, t.negative]) for t in train_triplets]
    loader = DataLoader(cast("Dataset[InputExample]", examples), shuffle=True, batch_size=128)
    loss_fn = losses.MultipleNegativesRankingLoss(model=model)

    logger.info(
        "开始 Fine-tune：%d 条样本 | batch_size=%d | %d steps/epoch x 2 epochs",
        len(train_triplets),
        batch_size,
        len(loader),
    )
    fit_start = time.time()

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=2,
        warmup_steps=ceil(len(loader) * 0.1),
        optimizer_params={"lr": 2e-6},
        show_progress_bar=True,
    )
    fit_elapsed = time.time() - fit_start
    logger.info("Fine-tune 完成，用时 %.0f 秒（%.1f 分钟）", fit_elapsed, fit_elapsed / 60)

    new_acc = evaluate(model, val_triplets)
    logger.info("准确率变化：%.4f -> %.4f（%+.4f）", baseline_acc, new_acc, new_acc - baseline_acc)

    _write_checkpoint({"last_id": new_last_id})

    if new_acc > baseline_acc:
        version = _next_version()
        save_path = os.path.join(settings.FINETUNED_MODEL_DIR, version)
        os.makedirs(save_path, exist_ok=True)
        model.save(save_path)
        _record_version(version, save_path, new_acc, len(train_triplets))
        result_desc = f"新版本 {version} 已保存"
    else:
        result_desc = "未提升，保留旧模型"
        logger.warning("新模型准确率（%.4f）未超基准（%.4f），丢弃", new_acc, baseline_acc)

    total_elapsed = time.time() - batch_start
    logger.info(
        "======== 批次训练完成 ========\n"
        "  流水 ID 范围: %d ~ %d（%d 条）\n"
        "  科目种类: %d 种\n"
        "  训练集 / 验证集: %d / %d 条\n"
        "  基准准确率: %.4f\n"
        "  新模型准确率: %.4f（%+.4f）\n"
        "  Fine-tune 用时: %.0f 秒（%.1f 分钟）\n"
        "  批次总用时: %.0f 秒（%.1f 分钟）\n"
        "  结果: %s",
        last_id, new_last_id, len(ids),
        unique_subjects,
        len(train_triplets), len(val_triplets),
        baseline_acc,
        new_acc, new_acc - baseline_acc,
        fit_elapsed, fit_elapsed / 60,
        total_elapsed, total_elapsed / 60,
        result_desc,
    )
    return True, new_last_id


def run(
    force: bool = False,
    limit: int | None = None,
    from_id: int | None = None,
    run_all: bool = False,
) -> None:
    """
    训练入口。

    run_all=False（默认）：只训练一批，完成后退出。
    run_all=True（--all）：循环训练，直到没有新数据为止。
    """
    run_start = time.time()
    min_samples = 1 if force else None
    batch_limit = limit or settings.BATCH_LIMIT

    ckpt = _read_checkpoint()

    if from_id is not None:
        last_id = from_id
        logger.info("--from-id 指定，从流水 ID %d 开始训练", last_id)
    else:
        last_id = ckpt.get("last_id", 0)
        if "pending_last_id" in ckpt:
            logger.warning(
                "检测到上次训练未完成（批次 ID %d ~ %d，开始于 %s），将重新训练该批次",
                last_id,
                ckpt["pending_last_id"],
                ckpt.get("pending_since", "未知"),
            )

    batch_num = 0
    while True:
        batch_num += 1
        if run_all:
            logger.info("===== 开始第 %d 批训练（last_id=%d）=====", batch_num, last_id)
        has_data, last_id = _run_one_batch(last_id, min_samples, batch_limit)
        if not has_data or not run_all:
            break

    if run_all:
        total_elapsed = time.time() - run_start
        logger.info(
            "======== 全量训练完成，共 %d 批，总用时 %.0f 秒（%.1f 分钟）========",
            batch_num - (0 if has_data else 1),
            total_elapsed,
            total_elapsed / 60,
        )


def show_status() -> None:
    """打印当前版本状态和 checkpoint 信息。"""
    ckpt = _read_checkpoint()
    versions_data = _read_versions()

    print(f"active_version    : {versions_data.get('active_version', 'base')}")
    print(f"checkpoint last_id: {ckpt.get('last_id', 0)}")
    if "pending_last_id" in ckpt:
        print(f"  上次训练未完成，pending_last_id={ckpt['pending_last_id']}")

    vers = versions_data.get("versions", [])
    if vers:
        print(f"\n{'版本':<8} {'准确率':<10} {'样本数':<10} {'训练时间'}")
        for v in sorted(vers, key=lambda x: x.get("trained_at", "")):
            active_mark = " <- active" if v["version"] == versions_data.get("active_version") else ""
            print(
                f"{v['version']:<8} {v.get('accuracy', 0):<10.4f} "
                f"{v.get('train_count', 0):<10} {v.get('trained_at', '')}{active_mark}"
            )
    else:
        print("（尚无 finetuned 版本）")
