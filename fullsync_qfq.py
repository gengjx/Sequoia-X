import logging, sys, time
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", stream=sys.stdout)

if __name__ == '__main__':
    import baostock as bs
    import sqlite3
    import pandas as pd
    from sequoia_x.core.config import Settings
    
    s = Settings()
    db_path = s.db_path
    
    with sqlite3.connect(db_path) as conn:
        symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily").fetchall()]
    print(f"共{len(symbols)}只，串行+分批入库全量前复权同步...", flush=True)
    
    bs.login()
    t0 = time.time()
    success = 0
    failed = 0
    batch_rows = []
    BATCH_SIZE = 200  # 每200只入库一次
    
    def flush_batch(rows):
        if not rows:
            return
        df = pd.DataFrame(rows, columns=['symbol','date','open','high','low','close','volume','turnover','turn','pct_chg'])
        for col in ['open','high','low','close','volume','turnover','turn','pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily (symbol,date,open,high,low,close,volume,turnover,turn,pct_chg) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                df[['symbol','date','open','high','low','close','volume','turnover','turn','pct_chg']].values.tolist()
            )
            conn.commit()
    
    for i, sym in enumerate(symbols):
        bs_code = f'sh.{sym}' if sym.startswith('6') else f'sz.{sym}'
        
        for attempt in range(3):
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume,amount,turn,pctChg",
                    start_date='2024-01-01', end_date='2026-07-10',
                    frequency="d", adjustflag="2",
                )
                rows = []
                while rs.error_code == '0' and rs.next():
                    rows.append([sym] + rs.get_row_data())
                if rows or rs.error_code == '0':
                    batch_rows.extend(rows)
                    success += 1
                    break
                else:
                    if attempt == 2:
                        failed += 1
                    time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    failed += 1
                time.sleep(2)
        
        # 每200只入库一次（防中断丢数据）
        if (i + 1) % BATCH_SIZE == 0:
            flush_batch(batch_rows)
            batch_rows = []
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(symbols) - i - 1) / speed
            print(f"进度: {i+1}/{len(symbols)} ({(i+1)/len(symbols)*100:.0f}%) "
                  f"速度{speed:.1f}只/秒 ETA{eta/60:.0f}分钟 成功{success} 失败{failed}", flush=True)
    
    # 最后一批入库
    flush_batch(batch_rows)
    bs.logout()
    elapsed = time.time() - t0
    print(f"全量同步完成: 成功{success} 失败{failed} 耗时{elapsed/60:.1f}分钟", flush=True)
