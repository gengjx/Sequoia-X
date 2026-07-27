"""用 akshare 东财财报摘要恢复/补全财报数据（不依赖 baostock 额度）。

增量补缺：跳过已有 ≥ n_quarters 季度的股票，仅采集缺失或深度不足的。
幂等：INSERT OR REPLACE，绝不删除已有数据。
"""
from __future__ import annotations

from sequoia_x.core.config import Settings
from sequoia_x.data.finance_sync import FinanceSync


def main() -> None:
    settings = Settings()
    syncer = FinanceSync(settings)
    result = syncer.sync_all_akshare(n_quarters=20, delay=0.3)
    print(f"\n恢复结果: {result}")


if __name__ == "__main__":
    main()
