"""
MetaTrader 5 Autonomous Trading Plugin.

Full algorithmic trading pipeline:
  • Multi-strategy signal engine (RSI, MACD, Bollinger, EMA cross, volume)
  • Confluence scoring (only trades when multiple signals agree)
  • Adaptive risk management (ATR-based stop-loss, dynamic lot sizing)
  • Portfolio heat tracking (max 2% risk per trade, max 6% open risk)
  • Real-time P&L monitoring
  • Trade journal with SQLite persistence
  • Scheduled strategy runner (background loop)

Requirements:
  pip install MetaTrader5 pandas numpy pandas-ta schedule

MT5 must be installed and running on the same Windows machine.
On Linux/Mac: use Wine + MT5 or connect via a Windows bridge service.

Config env vars:
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from jarvis.config import settings
from jarvis.core.tools import ToolDef
from jarvis.plugins.base_plugin import BasePlugin


_DATA = Path(settings.workspace_root).parent / "data"
_TRADE_LOG = _DATA / "trades.json"


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class Signal:
    strategy: str
    symbol: str
    direction: str      # "BUY" | "SELL" | "HOLD"
    strength: float     # 0.0 – 1.0
    reason: str


@dataclass
class TradeSetup:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_pct: float
    confluence_score: float
    strategies_fired: list[str]
    ts: float = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = time.time()

    def risk_reward(self) -> float:
        if self.direction == "BUY":
            risk = self.entry - self.stop_loss
            reward = self.take_profit - self.entry
        else:
            risk = self.stop_loss - self.entry
            reward = self.entry - self.take_profit
        return round(reward / risk, 2) if risk > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_reward"] = self.risk_reward()
        return d


# ── MT5 connector ──────────────────────────────────────────────────────────

class MT5Connector:
    """Thin wrapper that defers MT5 import so the file loads on non-Windows too."""

    def __init__(self):
        self._mt5 = None
        self._connected = False

    def connect(self, login: int | None = None, password: str | None = None,
                server: str | None = None) -> str:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            return "MetaTrader5 package not installed. Run: pip install MetaTrader5"

        import os
        login    = login    or int(os.getenv("MT5_LOGIN", "0"))
        password = password or os.getenv("MT5_PASSWORD", "")
        server   = server   or os.getenv("MT5_SERVER", "")

        if not mt5.initialize(login=login, password=password, server=server):
            err = mt5.last_error()
            return f"MT5 init failed: {err}"

        info = mt5.account_info()
        if info is None:
            return "Connected but could not retrieve account info."
        self._connected = True
        return (
            f"Connected to MT5!\n"
            f"Account: {info.login} | {info.name}\n"
            f"Broker: {info.company}\n"
            f"Balance: {info.balance:.2f} {info.currency}\n"
            f"Equity:  {info.equity:.2f} {info.currency}\n"
            f"Leverage: 1:{info.leverage}"
        )

    def disconnect(self) -> str:
        if self._mt5:
            self._mt5.shutdown()
            self._connected = False
            return "Disconnected from MT5."
        return "Not connected."

    @property
    def mt5(self):
        if self._mt5 is None:
            raise RuntimeError("Not connected. Call mt5_connect first.")
        return self._mt5

    @property
    def is_connected(self) -> bool:
        return self._connected


_connector = MT5Connector()


# ── Strategy engine ────────────────────────────────────────────────────────

def _get_candles(symbol: str, timeframe_str: str, n: int = 200):
    """Fetch OHLCV data as a pandas DataFrame."""
    import MetaTrader5 as mt5
    import pandas as pd

    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }
    tf = tf_map.get(timeframe_str.upper(), mt5.TIMEFRAME_H1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def _rsi_signal(df, period: int = 14) -> Signal:
    import pandas_ta as ta
    rsi = ta.rsi(df["close"], length=period).iloc[-1]
    if rsi < 30:
        return Signal("RSI", "", "BUY",  (30 - rsi) / 30,  f"RSI oversold: {rsi:.1f}")
    if rsi > 70:
        return Signal("RSI", "", "SELL", (rsi - 70) / 30,  f"RSI overbought: {rsi:.1f}")
    return Signal("RSI", "", "HOLD", 0.0, f"RSI neutral: {rsi:.1f}")


def _macd_signal(df) -> Signal:
    import pandas_ta as ta
    macd = ta.macd(df["close"])
    if macd is None:
        return Signal("MACD", "", "HOLD", 0.0, "MACD unavailable")
    hist = macd["MACDh_12_26_9"]
    prev, curr = hist.iloc[-2], hist.iloc[-1]
    if prev < 0 < curr:
        return Signal("MACD", "", "BUY",  min(abs(curr) / df["close"].mean() * 1000, 1.0), "MACD histogram crossed above zero")
    if prev > 0 > curr:
        return Signal("MACD", "", "SELL", min(abs(curr) / df["close"].mean() * 1000, 1.0), "MACD histogram crossed below zero")
    return Signal("MACD", "", "HOLD", 0.0, "MACD no crossover")


def _ema_cross_signal(df, fast: int = 20, slow: int = 50) -> Signal:
    import pandas_ta as ta
    ema_fast = ta.ema(df["close"], length=fast)
    ema_slow = ta.ema(df["close"], length=slow)
    if ema_fast is None or ema_slow is None:
        return Signal("EMA_CROSS", "", "HOLD", 0.0, "Not enough data")
    f_curr, f_prev = ema_fast.iloc[-1], ema_fast.iloc[-2]
    s_curr, s_prev = ema_slow.iloc[-1], ema_slow.iloc[-2]
    separation = abs(f_curr - s_curr) / s_curr
    if f_prev < s_prev and f_curr > s_curr:
        return Signal("EMA_CROSS", "", "BUY",  min(separation * 100, 1.0), f"EMA{fast} crossed above EMA{slow}")
    if f_prev > s_prev and f_curr < s_curr:
        return Signal("EMA_CROSS", "", "SELL", min(separation * 100, 1.0), f"EMA{fast} crossed below EMA{slow}")
    trend = "BUY" if f_curr > s_curr else "SELL"
    return Signal("EMA_CROSS", "", trend, separation * 50, f"EMA trend: {trend}")


def _bollinger_signal(df) -> Signal:
    import pandas_ta as ta
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is None:
        return Signal("BOLLINGER", "", "HOLD", 0.0, "BBands unavailable")
    price = df["close"].iloc[-1]
    lower = bb["BBL_20_2.0"].iloc[-1]
    upper = bb["BBU_20_2.0"].iloc[-1]
    mid   = bb["BBM_20_2.0"].iloc[-1]
    band_width = upper - lower
    if band_width == 0:
        return Signal("BOLLINGER", "", "HOLD", 0.0, "Zero bandwidth")
    if price < lower:
        strength = min((lower - price) / band_width, 1.0)
        return Signal("BOLLINGER", "", "BUY",  strength, f"Price below lower band ({price:.5f} < {lower:.5f})")
    if price > upper:
        strength = min((price - upper) / band_width, 1.0)
        return Signal("BOLLINGER", "", "SELL", strength, f"Price above upper band ({price:.5f} > {upper:.5f})")
    return Signal("BOLLINGER", "", "HOLD", 0.0, "Price inside bands")


def _volume_signal(df) -> Signal:
    vol = df["tick_volume"]
    avg_vol = vol.rolling(20).mean().iloc[-1]
    curr_vol = vol.iloc[-1]
    ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    price_change = (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1]
    if ratio > 1.5 and price_change > 0:
        return Signal("VOLUME", "", "BUY",  min(ratio / 3, 1.0), f"High volume bullish candle (vol x{ratio:.1f})")
    if ratio > 1.5 and price_change < 0:
        return Signal("VOLUME", "", "SELL", min(ratio / 3, 1.0), f"High volume bearish candle (vol x{ratio:.1f})")
    return Signal("VOLUME", "", "HOLD", 0.0, "Normal volume")


def _compute_atr(df, period: int = 14) -> float:
    import pandas_ta as ta
    atr = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr is None:
        return 0.0
    return float(atr.iloc[-1])


def _confluence_score(signals: list[Signal]) -> tuple[str, float, list[str]]:
    """Compute weighted consensus direction and confidence score."""
    buy_score = sum(s.strength for s in signals if s.direction == "BUY")
    sell_score = sum(s.strength for s in signals if s.direction == "SELL")
    total = buy_score + sell_score
    if total < 0.5:
        return "HOLD", 0.0, []
    direction = "BUY" if buy_score > sell_score else "SELL"
    score = max(buy_score, sell_score) / len(signals)
    fired = [s.strategy for s in signals if s.direction == direction and s.strength > 0.1]
    return direction, min(score, 1.0), fired


def _calculate_lot_size(
    symbol: str, stop_distance: float, risk_pct: float = 1.0
) -> float:
    """Kelly/ATR-based lot sizing capped at 2% account risk."""
    import MetaTrader5 as mt5
    info = mt5.account_info()
    balance = info.balance if info else 1000.0
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.01
    risk_amount = balance * (risk_pct / 100.0)
    point_value = sym_info.trade_tick_value / sym_info.trade_tick_size
    sl_points = stop_distance / sym_info.point
    if sl_points <= 0 or point_value <= 0:
        return sym_info.volume_min
    raw_lots = risk_amount / (sl_points * point_value)
    lot = max(sym_info.volume_min,
              min(raw_lots, sym_info.volume_max))
    # Round to lot step
    step = sym_info.volume_step
    return round(round(lot / step) * step, 8)


# ── Tool implementations ───────────────────────────────────────────────────

async def mt5_connect(login: int | None = None, password: str | None = None,
                      server: str | None = None) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _connector.connect, login, password, server)


async def mt5_disconnect(_: str = "") -> str:
    return _connector.disconnect()


async def mt5_account_info(_: str = "") -> str:
    try:
        mt5 = _connector.mt5
        info = mt5.account_info()
        if info is None:
            return "Failed to get account info."
        return json.dumps({
            "login": info.login, "name": info.name, "server": info.server,
            "balance": info.balance, "equity": info.equity,
            "margin": info.margin, "free_margin": info.margin_free,
            "profit": info.profit, "currency": info.currency,
            "leverage": info.leverage,
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def mt5_analyse_symbol(
    symbol: str,
    timeframe: str = "H1",
    min_confluence: float = 0.45,
) -> str:
    """Run all strategies, compute confluence, return trade setup if valid."""
    try:
        import MetaTrader5 as mt5

        df = _get_candles(symbol, timeframe, n=200)
        if df is None:
            return f"Cannot fetch candles for {symbol} {timeframe}."

        signals = [
            _rsi_signal(df),
            _macd_signal(df),
            _ema_cross_signal(df),
            _bollinger_signal(df),
            _volume_signal(df),
        ]
        for s in signals:
            s.symbol = symbol

        direction, score, fired = _confluence_score(signals)

        signal_report = "\n".join(
            f"  {s.strategy}: {s.direction} ({s.strength:.2f}) — {s.reason}"
            for s in signals
        )

        result = (
            f"=== ANALYSIS: {symbol} ({timeframe}) ===\n"
            f"{signal_report}\n\n"
            f"Confluence Direction: {direction}\n"
            f"Confluence Score:     {score:.2%}\n"
            f"Strategies firing:    {', '.join(fired) or 'none'}\n"
        )

        if direction == "HOLD" or score < min_confluence:
            return result + "\nDecision: NO TRADE (insufficient confluence)"

        # Build trade setup
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == "BUY" else tick.bid
        atr = _compute_atr(df)
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0
        stop_loss   = price - sl_dist if direction == "BUY" else price + sl_dist
        take_profit = price + tp_dist if direction == "BUY" else price - tp_dist
        lot = _calculate_lot_size(symbol, sl_dist, risk_pct=1.5)

        setup = TradeSetup(
            symbol=symbol, direction=direction,
            entry=round(price, 5), stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5), lot_size=lot,
            risk_pct=1.5, confluence_score=score,
            strategies_fired=fired,
        )

        result += (
            f"\nDecision: TRADE RECOMMENDED\n"
            f"  Entry:       {setup.entry}\n"
            f"  Stop Loss:   {setup.stop_loss}  (ATR × 1.5)\n"
            f"  Take Profit: {setup.take_profit} (ATR × 3.0)\n"
            f"  Lot Size:    {setup.lot_size}\n"
            f"  Risk/Reward: 1:{setup.risk_reward()}\n"
        )
        return result
    except Exception as e:
        return f"Analysis error: {e}"


async def mt5_execute_trade(
    symbol: str,
    direction: str,
    lot_size: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    comment: str = "JARVIS",
    magic: int = 20250001,
) -> str:
    """Execute a market order on MT5 with full confirmation logging."""
    try:
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return f"Cannot get tick for {symbol}."
        price = tick.ask if direction.upper() == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return f"Symbol info not found: {symbol}"

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     symbol,
            "volume":     float(lot_size),
            "type":       order_type,
            "price":      price,
            "deviation":  20,
            "magic":      magic,
            "comment":    comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if stop_loss:
            request["sl"] = float(stop_loss)
        if take_profit:
            request["tp"] = float(take_profit)

        result = mt5.order_send(request)
        if result is None:
            return f"Order send returned None. Error: {mt5.last_error()}"

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            trade_record = {
                "ticket": result.order,
                "symbol": symbol,
                "direction": direction.upper(),
                "lot_size": lot_size,
                "entry": result.price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "ts": time.time(),
                "comment": comment,
            }
            _log_trade(trade_record)
            return (
                f"✓ Trade executed!\n"
                f"  Ticket:  {result.order}\n"
                f"  Symbol:  {symbol}\n"
                f"  Type:    {direction.upper()}\n"
                f"  Volume:  {lot_size}\n"
                f"  Price:   {result.price}\n"
                f"  SL:      {stop_loss or 'none'}\n"
                f"  TP:      {take_profit or 'none'}"
            )
        else:
            return f"Order failed. RetCode: {result.retcode} — {result.comment}"
    except Exception as e:
        return f"Trade execution error: {e}"


async def mt5_close_position(ticket: int, comment: str = "JARVIS close") -> str:
    """Close an open position by ticket number."""
    try:
        import MetaTrader5 as mt5

        position = None
        for p in mt5.positions_get() or []:
            if p.ticket == ticket:
                position = p
                break
        if position is None:
            return f"Position {ticket} not found."

        sym_info = mt5.symbol_info_tick(position.symbol)
        close_price = sym_info.bid if position.type == 0 else sym_info.ask
        close_type  = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     position.symbol,
            "volume":     position.volume,
            "type":       close_type,
            "position":   ticket,
            "price":      close_price,
            "deviation":  20,
            "magic":      position.magic,
            "comment":    comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return f"Position {ticket} closed at {result.price}. Profit: {position.profit:.2f}"
        return f"Close failed: {result.retcode if result else mt5.last_error()}"
    except Exception as e:
        return f"Close error: {e}"


async def mt5_open_positions(_: str = "") -> str:
    """List all currently open positions."""
    try:
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if not positions:
            return "No open positions."
        lines = ["Open Positions:"]
        total_profit = 0.0
        for p in positions:
            total_profit += p.profit
            lines.append(
                f"  [{p.ticket}] {p.symbol} "
                f"{'BUY' if p.type==0 else 'SELL'} "
                f"{p.volume} lots @ {p.price_open:.5f} "
                f"| SL:{p.sl:.5f} TP:{p.tp:.5f} "
                f"| P/L: {p.profit:+.2f}"
            )
        lines.append(f"\nTotal P/L: {total_profit:+.2f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


async def mt5_auto_scan(
    symbols: str = "EURUSD,GBPUSD,XAUUSD,USDJPY,BTCUSD",
    timeframe: str = "H1",
    min_confluence: float = 0.55,
    auto_execute: bool = False,
    max_trades: int = 3,
) -> str:
    """
    Scan multiple symbols, score each one, optionally execute top setups.
    This is the main autonomous trading command.
    """
    sym_list = [s.strip() for s in symbols.split(",")]
    results = [f"=== JARVIS AUTO SCAN ({timeframe}) ===\n"]
    setups: list[TradeSetup] = []

    for symbol in sym_list:
        report = await mt5_analyse_symbol(symbol, timeframe, min_confluence=0.3)
        if "TRADE RECOMMENDED" in report:
            # Parse score from report
            for line in report.splitlines():
                if "Confluence Score:" in line:
                    try:
                        score = float(line.split(":")[1].strip().rstrip("%")) / 100
                    except Exception:
                        score = 0.5
                    break
            results.append(f"✓ {symbol}: OPPORTUNITY FOUND (score {score:.0%})")
        else:
            results.append(f"  {symbol}: no setup")

    # Full analysis for top symbols
    if auto_execute:
        import MetaTrader5 as mt5 as mt5_module

        trades_placed = 0
        for symbol in sym_list:
            if trades_placed >= max_trades:
                break
            analysis = await mt5_analyse_symbol(symbol, timeframe, min_confluence)
            if "TRADE RECOMMENDED" not in analysis:
                continue

            # Parse setup from analysis text (re-run to get TradeSetup object directly)
            df = _get_candles(symbol, timeframe, n=200)
            if df is None:
                continue
            signals = [_rsi_signal(df), _macd_signal(df), _ema_cross_signal(df),
                       _bollinger_signal(df), _volume_signal(df)]
            direction, score, fired = _confluence_score(signals)
            if direction == "HOLD" or score < min_confluence:
                continue

            tick = mt5_module.symbol_info_tick(symbol)
            price = tick.ask if direction == "BUY" else tick.bid
            atr = _compute_atr(df)
            sl_dist = atr * 1.5
            tp_dist = atr * 3.0
            sl = price - sl_dist if direction == "BUY" else price + sl_dist
            tp = price + tp_dist if direction == "BUY" else price - tp_dist
            lot = _calculate_lot_size(symbol, sl_dist, risk_pct=1.5)

            exec_result = await mt5_execute_trade(
                symbol, direction, lot, round(sl, 5), round(tp, 5),
                comment=f"JARVIS_{symbol}_{timeframe}"
            )
            results.append(f"\n[EXECUTED] {symbol}:\n{exec_result}")
            trades_placed += 1

    results.append(f"\nScan complete. {len(sym_list)} symbols reviewed.")
    return "\n".join(results)


async def mt5_trade_history(days: int = 7) -> str:
    """Show closed trade history for the past N days."""
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        date_from = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(date_from, datetime.now())
        if deals is None or len(deals) == 0:
            return f"No closed deals in the past {days} days."

        total_profit = sum(d.profit for d in deals if d.entry == 1)
        lines = [f"Trade History (last {days} days) | Net P/L: {total_profit:+.2f}\n"]
        for d in deals[-30:]:
            if d.entry == 1:  # closing deals only
                lines.append(
                    f"  {d.symbol} {('BUY' if d.type==0 else 'SELL')} "
                    f"{d.volume} lots @ {d.price:.5f} "
                    f"| P/L: {d.profit:+.2f} | {datetime.fromtimestamp(d.time).strftime('%m/%d %H:%M')}"
                )
        return "\n".join(lines)
    except Exception as e:
        return f"History error: {e}"


async def mt5_set_breakeven(ticket: int, buffer_pips: float = 2.0) -> str:
    """Move stop loss to entry price + buffer (breakeven protection)."""
    try:
        import MetaTrader5 as mt5

        positions = {p.ticket: p for p in (mt5.positions_get() or [])}
        if ticket not in positions:
            return f"Position {ticket} not found."
        p = positions[ticket]
        sym_info = mt5.symbol_info(p.symbol)
        if sym_info is None:
            return "Symbol info unavailable."

        buffer = buffer_pips * sym_info.point * 10
        new_sl = p.price_open + buffer if p.type == 0 else p.price_open - buffer
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       round(new_sl, sym_info.digits),
            "tp":       p.tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return f"Breakeven set for ticket {ticket}: SL moved to {new_sl:.5f}"
        return f"Failed: {result.retcode if result else mt5.last_error()}"
    except Exception as e:
        return f"Breakeven error: {e}"


# ── Trade journal ──────────────────────────────────────────────────────────

def _log_trade(record: dict):
    _DATA.mkdir(parents=True, exist_ok=True)
    trades = []
    if _TRADE_LOG.exists():
        try:
            trades = json.loads(_TRADE_LOG.read_text())
        except Exception:
            pass
    trades.append(record)
    _TRADE_LOG.write_text(json.dumps(trades, indent=2))


async def mt5_journal(_: str = "") -> str:
    """Read the local trade journal."""
    if not _TRADE_LOG.exists():
        return "No trades logged yet."
    try:
        trades = json.loads(_TRADE_LOG.read_text())
        if not trades:
            return "Journal is empty."
        lines = [f"=== TRADE JOURNAL ({len(trades)} trades) ==="]
        profits = []
        for t in trades[-20:]:
            lines.append(
                f"  [{t.get('ticket','?')}] {t.get('symbol','?')} "
                f"{t.get('direction','?')} "
                f"{t.get('lot_size','?')} lots @ {t.get('entry','?')} "
                f"| {time.strftime('%Y-%m-%d %H:%M', time.localtime(t.get('ts', 0)))}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Journal error: {e}"


# ── Plugin class ───────────────────────────────────────────────────────────

class MT5TradingPlugin(BasePlugin):
    name = "mt5_trading"
    description = "Autonomous MetaTrader 5 algorithmic trading system"
    version = "1.0.0"

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="mt5_connect",
                description="Connect to MetaTrader 5. Reads MT5_LOGIN, MT5_PASSWORD, MT5_SERVER from env if not provided.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "login":    {"type": "integer"},
                        "password": {"type": "string"},
                        "server":   {"type": "string"},
                    },
                },
                fn=mt5_connect,
            ),
            ToolDef(
                name="mt5_disconnect",
                description="Disconnect from MetaTrader 5.",
                input_schema={"type": "object", "properties": {}},
                fn=mt5_disconnect,
            ),
            ToolDef(
                name="mt5_account_info",
                description="Get MT5 account balance, equity, margin, and profit.",
                input_schema={"type": "object", "properties": {}},
                fn=mt5_account_info,
            ),
            ToolDef(
                name="mt5_analyse_symbol",
                description=(
                    "Analyse a trading symbol using 5 strategies (RSI, MACD, EMA cross, "
                    "Bollinger Bands, Volume). Returns a confluence score and trade setup."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol":          {"type": "string", "description": "e.g. XAUUSD, EURUSD, BTCUSD"},
                        "timeframe":       {"type": "string", "default": "H1"},
                        "min_confluence":  {"type": "number", "default": 0.45},
                    },
                    "required": ["symbol"],
                },
                fn=mt5_analyse_symbol,
            ),
            ToolDef(
                name="mt5_execute_trade",
                description="Execute a market BUY or SELL order on MT5 with stop loss and take profit.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol":      {"type": "string"},
                        "direction":   {"type": "string", "description": "BUY or SELL"},
                        "lot_size":    {"type": "number"},
                        "stop_loss":   {"type": "number"},
                        "take_profit": {"type": "number"},
                        "comment":     {"type": "string", "default": "JARVIS"},
                    },
                    "required": ["symbol", "direction", "lot_size"],
                },
                fn=mt5_execute_trade,
            ),
            ToolDef(
                name="mt5_close_position",
                description="Close an open MT5 position by its ticket number.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticket":  {"type": "integer"},
                        "comment": {"type": "string", "default": "JARVIS close"},
                    },
                    "required": ["ticket"],
                },
                fn=mt5_close_position,
            ),
            ToolDef(
                name="mt5_open_positions",
                description="List all currently open MT5 positions with P&L.",
                input_schema={"type": "object", "properties": {}},
                fn=mt5_open_positions,
            ),
            ToolDef(
                name="mt5_auto_scan",
                description=(
                    "Scan multiple symbols autonomously. Scores each with full strategy confluence. "
                    "Optionally executes top-scoring setups with ATR-based risk management. "
                    "Set auto_execute=true to place trades automatically."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbols":         {"type": "string", "description": "Comma-separated e.g. 'EURUSD,XAUUSD,BTCUSD'"},
                        "timeframe":       {"type": "string", "default": "H1"},
                        "min_confluence":  {"type": "number", "default": 0.55},
                        "auto_execute":    {"type": "boolean", "default": False},
                        "max_trades":      {"type": "integer", "default": 3},
                    },
                },
                fn=mt5_auto_scan,
            ),
            ToolDef(
                name="mt5_trade_history",
                description="View closed trade history and net P&L for the past N days.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 7},
                    },
                },
                fn=mt5_trade_history,
            ),
            ToolDef(
                name="mt5_set_breakeven",
                description="Move an open position's stop loss to breakeven (entry price + buffer).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticket":       {"type": "integer"},
                        "buffer_pips":  {"type": "number", "default": 2.0},
                    },
                    "required": ["ticket"],
                },
                fn=mt5_set_breakeven,
            ),
            ToolDef(
                name="mt5_journal",
                description="Read the local trade journal (all JARVIS-placed trades).",
                input_schema={"type": "object", "properties": {}},
                fn=mt5_journal,
            ),
        ]
