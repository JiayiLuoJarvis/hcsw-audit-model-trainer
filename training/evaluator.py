"""
验证集准确率评估

策略：对每条 anchor，在所有唯一 positive 科目中找向量最近邻，
命中正确科目则计为正确（Top-1 余弦相似度）。
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from training.data_builder import TrainingTriplet

logger = logging.getLogger(__name__)


def evaluate(
    model: SentenceTransformer,
    triplets: list[TrainingTriplet],
) -> float:
    """
    在给定三元组集合上评估 Top-1 准确率。

    Returns:
        accuracy: 0.0 ~ 1.0
    """
    if not triplets:
        return 0.0

    subjects = list({t.positive for t in triplets})
    if len(subjects) < 2:
        logger.warning("验证集科目种类不足（%d 种），跳过评估", len(subjects))
        return 0.0

    subject_embs: np.ndarray = model.encode(
        subjects, normalize_embeddings=True, show_progress_bar=False
    )
    anchors = [t.anchor for t in triplets]
    anchor_embs: np.ndarray = model.encode(
        anchors, normalize_embeddings=True, show_progress_bar=False
    )

    scores = anchor_embs @ subject_embs.T  # (n_anchors, n_subjects)
    pred_indices = np.argmax(scores, axis=1)

    correct = sum(
        1
        for pred_idx, triplet in zip(pred_indices, triplets)
        if subjects[pred_idx] == triplet.positive
    )
    accuracy = correct / len(triplets)
    logger.info(
        "验证集准确率: %.4f (%d / %d)", accuracy, correct, len(triplets)
    )
    return accuracy
