import os
import requests
import datetime
import pandas as pd
import ccxt
import numpy as np
from zoneinfo import ZoneInfo

# 1. 优先从环境变量中获取飞书 Webhook 地址，确保代码开源安全!
FEISHU_WEBHOOK_URL = os.environ.get("TOM_FEISHU_WEBHOOK_URL", "")

SYMBOL = "BTC/USDT"
TIMEFRAMES = ['12h', '1d', '1w', '1m']

def get_time_str():
    try:
        now_utc = datetime.datetime.now(ZoneInfo("UTC"))
        bj_str = now_utc.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        us_str = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        now = datetime.datetime.now()
        bj_str = now.strftime("%Y-%m-%d %H:%M:%S")
        us_time = now - datetime.timedelta(hours=12)
        us_str = us_time.strftime("%Y-%m-%d %H:%M:%S")
        
    return f"[北京 {bj_str} | 美东 {us_str}]"

# 初始化交易所实例（移除代理和本地节点绑定，GitHub Actions 在云端可直连）
exchange = ccxt.okx({
    'enableRateLimit': True,
    'timeout': 15000,
})

# ==================== 技术指标计算函数 ====================
def calculate_tv_adx(df, length=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smoothed = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()
    pos_dm_smoothed = pd.Series(pos_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean()
    neg_dm_smoothed = pd.Series(neg_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean()

    pos_di = 100 * (pos_dm_smoothed / tr_smoothed)
    neg_di = 100 * (neg_dm_smoothed / tr_smoothed)

    di_sum = pos_di + neg_di
    dx = 100 * (pos_di - neg_di).abs() / di_sum.replace(0, np.nan)
    dx = dx.fillna(0)

    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']

    # RSI (6 & 12)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/6, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/6, adjust=False).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss)))

    gain12 = (delta.where(delta > 0, 0)).ewm(alpha=1/12, adjust=False).mean()
    loss12 = (-delta.where(delta < 0, 0)).ewm(alpha=1/12, adjust=False).mean()
    df['rsi12'] = 100 - (100 / (1 + (gain12 / loss12)))

    # KDJ
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    # Stoch RSI
    rsi14_delta = close.diff()
    gain14 = (rsi14_delta.where(rsi14_delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss14 = (-rsi14_delta.where(rsi14_delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rsi14 = 100 - (100 / (1 + (gain14 / loss14)))
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    stoch_rsi = (rsi14 - rsi_min) / (rsi_max - rsi_min) * 100
    df['stoch_k'] = stoch_rsi.rolling(3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ADX
    df['adx'] = calculate_tv_adx(df, length=14)

    return df

# ==================== 飞书推送函数 ====================
def send_feishu_alert(title, text):
    if not FEISHU_WEBHOOK_URL:
        print("[警告] 未设置 FEISHU_WEBHOOK_URL 环境变量，跳过推送")
        return

    time_prefix = get_time_str()
    headers = {"Content-Type": "application/json"}
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [[{"tag": "text", "text": text}]]
                }
            }
        }
    }
    try:
        response = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"{time_prefix} [推送成功] {title}")
        else:
            print(f"{time_prefix} [推送失败] 状态码: {response.status_code}")
    except Exception as e:
        print(f"{time_prefix} [推送异常] {e}")

# ==================== 主流程监控函数（单次执行） ====================
def check_signals():
    current_price = None
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        print(f"{get_time_str()} 🟢 监控运行中... {SYMBOL} 当前实时价格: ${current_price:,.2f}")
    except Exception as e:
        print(f"获取实时价格失败: {e}")

    for tf in TIMEFRAMES:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=300)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df = calculate_indicators(df)
            
            latest = df.iloc[-1]
            candle_time = latest['datetime'].strftime('%Y-%m-%d %H:%M:%S')

            rsi = latest['rsi6']
            rsi12 = latest['rsi12']
            k, d, j = latest['k'], latest['d'], latest['j']
            macd = latest['macd']
            stoch_k, stoch_d = latest['stoch_k'], latest['stoch_d']
            adx = latest['adx']
            price = latest['close']

            is_overbought = (
                adx > 38 and
                rsi > 73 and 
                rsi12 > 70 and 
                (k > 70 and d > 70 and j > 75) and 
                (stoch_k > 75 and stoch_d > 75)
            )

            is_oversold = (
                adx > 38 and
                rsi < 27 and 
                rsi12 < 30 and 
                (k < 30 and d < 30 and j < 25) and 
                (stoch_k < 25 and stoch_d < 25)
            )

            # 注意：GitHub Actions 每次启动都是全新环境，如果遇到超买/超卖信号将直接推送
            if is_overbought or is_oversold:
                signal_type = "🔴【BTC极端超买】" if is_overbought else "🟢【BTC极端超卖】"
                
                msg = (
                    f"推送时间: {get_time_str()}\n"
                    f"K线时间(UTC): {candle_time}\n"
                    f"当前价格: ${price:,.2f}\n"
                    f"------------------------\n"
                    f"• ADX(14): {adx:.2f}\n"
                    f"• RSI(6): {rsi:.2f}  RSI(12): {rsi12:.2f}\n"
                    f"• KDJ: K={k:.2f}, D={d:.2f}, J={j:.2f}\n"
                    f"• MACD (DIF): {macd:.2f}\n"
                    f"• Stoch RSI: K={stoch_k:.2f}, D={stoch_d:.2f}"
                )

                send_feishu_alert(f"{signal_type} ({tf}周期)", msg)

        except Exception as e:
            print(f"[{tf}] 数据获取或计算出错: {e}")

if __name__ == '__main__':
    # GitHub Actions 模式：执行一次即退出
    check_signals()
