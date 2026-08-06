import datetime
import json
import os
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

# 在 GitHub Actions 的 Secrets 中设置 TOM_FEISHU_WEBHOOK_URL。
FEISHU_WEBHOOK_URL = os.environ.get("TOM_FEISHU_WEBHOOK_URL")

# 如果只想监控 12 小时，请保持此配置；需要其他周期再追加，例如 "1d"。
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAMES = ["12h", "1d", "1w"]
OHLCV_LIMIT = 300
STATE_FILE = Path("state/sent_signals.json")
HEALTH_REPORT_INTERVAL = datetime.timedelta(days=2)
HEALTH_REPORT_HOUR_BEIJING = 18

exchange = ccxt.okx({"enableRateLimit": True, "timeout": 15_000})


def get_time_str():
    now_utc = datetime.datetime.now(ZoneInfo("UTC"))
    bj = now_utc.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    us = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    return f"[北京 {bj} | 美东 {us}]"


def calculate_tv_adx(df, length=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    up_move, down_move = high.diff(), -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smoothed = tr.ewm(alpha=1 / length, adjust=False).mean()
    pos_smoothed = pd.Series(pos_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
    neg_smoothed = pd.Series(neg_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
    pos_di, neg_di = 100 * pos_smoothed / tr_smoothed, 100 * neg_smoothed / tr_smoothed
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return dx.fillna(0).ewm(alpha=1 / length, adjust=False).mean()


def calculate_indicators(df):
    close, high, low = df["close"], df["high"], df["low"]
    delta = close.diff()

    def rsi(length):
        gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
        return 100 - 100 / (1 + gain / loss)

    df["rsi6"], df["rsi12"] = rsi(6), rsi(12)
    rsi14 = rsi(14)
    low_min, high_max = low.rolling(9).min(), high.rolling(9).max()
    rsv = ((close - low_min) / (high_max - low_min) * 100).replace([np.inf, -np.inf], np.nan).fillna(50)
    df["k"] = rsv.ewm(com=2, adjust=False).mean()
    df["d"] = df["k"].ewm(com=2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    stoch_rsi = (rsi14 - rsi14.rolling(14).min()) / (rsi14.rolling(14).max() - rsi14.rolling(14).min()) * 100
    df["stoch_k"] = stoch_rsi.rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["adx"] = calculate_tv_adx(df)
    return df


def load_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def send_feishu_alert(title, text):
    if not FEISHU_WEBHOOK_URL:
        print("[警告] 未设置 TOM_FEISHU_WEBHOOK_URL，跳过飞书推送")
        return False
    payload = {"msg_type": "post", "content": {"post": {"zh_cn": {"title": title, "content": [[{"tag": "text", "text": text}]]}}}}
    try:
        response = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"{get_time_str()} [推送成功] {title}")
        return True
    except requests.RequestException as exc:
        print(f"{get_time_str()} [推送失败] {exc}")
        return False


def signal_for(latest):
    overbought = latest["adx"] > 20 and latest["rsi6"] > 73 and latest["rsi12"] > 70 and latest["k"] > 70 and latest["d"] > 70 and latest["j"] > 75 and latest["stoch_k"] > 75 and latest["stoch_d"] > 75
    oversold = latest["adx"] > 38 and latest["rsi6"] < 27 and latest["rsi12"] < 30 and latest["k"] < 30 and latest["d"] < 30 and latest["j"] < 25 and latest["stoch_k"] < 25 and latest["stoch_d"] < 25
    return "极端超买" if overbought else "极端超卖" if oversold else None


def check_signals():
    state = load_state()
    errors = []
    checked_count = 0
    try:
        exchange.load_markets()
    except Exception as exc:
        errors.append(f"交易所市场加载失败：{exc}")
        print(errors[-1])
        send_health_report(state, checked_count, errors)
        return

    for symbol in SYMBOLS:
        asset = symbol.split("/")[0]
        for timeframe in TIMEFRAMES:
            try:
                # 最后一根通常尚未收线；去掉它，避免盘中假信号。
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT + 1)[:-1]
                if len(ohlcv) < 100:
                    raise ValueError("K线数量不足，无法可靠计算指标")
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                latest = calculate_indicators(df).iloc[-1]
                checked_count += 1
                signal = signal_for(latest)
                candle_time = latest["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC")
                print(f"{get_time_str()} {symbol} {timeframe} 已收线: ${latest['close']:,.4f} | 信号: {signal or '无'}")
                if not signal:
                    continue

                state_key = f"{symbol}:{timeframe}"
                candle_id = str(int(latest["timestamp"]))
                if state.get(state_key) == candle_id:
                    print(f"[{symbol} {timeframe}] 此根K线已推送，跳过重复提醒")
                    continue

                icon = "🔴" if signal == "极端超买" else "🟢"
                message = (
                    f"推送时间: {get_time_str()}\nK线时间: {candle_time}\n收盘价: ${latest['close']:,.4f}\n"
                    f"------------------------\n• ADX(14): {latest['adx']:.2f}\n"
                    f"• RSI(6): {latest['rsi6']:.2f}  RSI(12): {latest['rsi12']:.2f}\n"
                    f"• KDJ: K={latest['k']:.2f}, D={latest['d']:.2f}, J={latest['j']:.2f}\n"
                    f"• MACD (DIF): {latest['macd']:.4f}\n"
                    f"• Stoch RSI: K={latest['stoch_k']:.2f}, D={latest['stoch_d']:.2f}"
                )
                if send_feishu_alert(f"{icon}【{asset}{signal}】({timeframe})", message):
                    state[state_key] = candle_id
                    save_state(state)
            except Exception as exc:
                error_message = f"[{symbol} {timeframe}] 数据获取或计算出错: {exc}"
                errors.append(error_message)
                print(error_message)

    send_health_report(state, checked_count, errors)


def send_health_report(state, checked_count, errors):
    """每 48 小时在北京时间 18 点发送一次状态；需要 Actions 缓存 state/。"""
    now = datetime.datetime.now(ZoneInfo("UTC"))
    beijing_now = now.astimezone(ZoneInfo("Asia/Shanghai"))
    if beijing_now.hour != HEALTH_REPORT_HOUR_BEIJING:
        return
    last_report = state.get("_last_health_report_at")
    if last_report:
        try:
            if now - datetime.datetime.fromisoformat(last_report) < HEALTH_REPORT_INTERVAL:
                return
        except ValueError:
            pass

    if errors:
        title = "⚠️【监控系统运行异常】"
        details = "\n".join(f"• {error}" for error in errors[:10])
        text = (
            f"检查时间: {get_time_str()}\n"
            f"成功检查: {checked_count}/{len(SYMBOLS) * len(TIMEFRAMES)} 个币种周期\n"
            f"异常数量: {len(errors)}\n"
            f"------------------------\n{details}"
        )
    else:
        title = "✅【监控系统正常运行】"
        text = (
            f"检查时间: {get_time_str()}\n"
            f"本监控系统运行正常。\n"
            f"已成功检查 {checked_count} 个币种周期：BTC、ETH、SOL。\n"
            f"监控周期: {', '.join(TIMEFRAMES)}"
        )

    if send_feishu_alert(title, text):
        state["_last_health_report_at"] = now.isoformat()
        save_state(state)


if __name__ == "__main__":
    check_signals()

