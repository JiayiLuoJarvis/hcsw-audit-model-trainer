"""
只读库连接与训练数据查询

数据来源：bank_account_statement JOIN account_chart JOIN bank_account

⚠️  严禁对此库执行任何 INSERT / UPDATE / DELETE 操作。
    此库由主业务系统维护，本项目仅允许 SELECT。
"""

from __future__ import annotations

import asyncio
import logging

import aiomysql

from config.settings import settings

logger = logging.getLogger(__name__)


async def _fetch_raw_pairs_async(limit: int, last_id: int = 0) -> list[dict]:
    """
    从只读库拉取有科目标注的流水记录。

    过滤条件（与原项目保持一致）：
      - bas.is_deleted = 0
      - bas.account_chart_id > 0
      - bas.summary IS NOT NULL AND != ''
      - ac.company_id = 0   仅公共科目，排除私有科目（避免跨公司污染）
      - ba.is_deleted = 0
    """
    conn = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        db=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        charset="utf8mb4",
    )
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT
                    bas.id          AS id,
                    bas.type        AS type,
                    bas.money       AS money,
                    ba.currency     AS currency,
                    bas.summary     AS summary,
                    bas.trade_type  AS trade_type,
                    ac.name         AS correct_subject
                FROM bank_account_statement bas
                INNER JOIN account_chart ac
                    ON ac.id = bas.account_chart_id
                    AND ac.is_deleted = 0
                    AND ac.company_id = 0
                INNER JOIN bank_account ba
                    ON ba.id = bas.bank_account_id
                    AND ba.is_deleted = 0
                WHERE bas.is_deleted = 0
                  AND bas.id > %s
                  AND bas.account_chart_id > 0
                  AND bas.summary IS NOT NULL
                  AND bas.summary != ''
                ORDER BY bas.id ASC
                LIMIT %s
                """,
                (last_id, limit),
            )
            rows = await cur.fetchall()
        logger.info(
            "从只读库读取流水 %d 条（含科目标注），起始 ID: %d",
            len(rows),
            last_id,
        )
        return list(rows)
    finally:
        conn.close()


def fetch_raw_pairs(limit: int, last_id: int = 0) -> list[dict]:
    """同步入口：从只读库拉取有标注的流水记录。"""
    return asyncio.run(_fetch_raw_pairs_async(limit, last_id))
