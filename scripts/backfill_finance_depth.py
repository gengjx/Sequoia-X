"""回补财报历史深度——增量补深模式，零删除。

对季度数 < n_quarters 的股票用 INSERT OR REPLACE 拉取更深历史。
绝不 DELETE 已有数据——_fetch_batch 天然支持增量补缺。
"""
from __future__ import annotations

from sequoia_x.core.config import Settings
from sequoia_x.data.finance_sync import FinanceSync


def main() -> None:
    settings = Settings()
    syncer = FinanceSync(settings)

    # deep_sync=True: 完全缺失 + 季度数 < n_quarters 的都采集，
    # INSERT OR REPLACE 自动补缺，不删除任何已有数据。
    result = syncer.sync_all(
        n_quarters=20, n_workers=3, deep_sync=True,
    )
    print(f"\n回补结果: {result}")


if __name__ == "__main__":
    main()
