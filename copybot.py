#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copybot.py — Polymarket Copy-Trading Bot
==========================================
Identifies profitable wallets on Polymarket and generates
copy-trading signals based on their behavior.

Usage:
    python copybot.py scan       # Full pipeline: collect → analyze → rank
    python copybot.py monitor    # Live monitoring of tracked wallets
    python copybot.py leaderboard # Show top wallets
    python copybot.py signals    # Show recent signals
    python copybot.py backtest   # Backtest copy strategy
"""

import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Setup paths
COPYBOT_DIR = Path(__file__).parent / "copybot"
DATA_DIR = COPYBOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(COPYBOT_DIR.parent))

from copybot.data_collector import (
    build_trade_dataset, fetch_active_markets, fetch_market_info
)
from copybot.wallet_analyzer import (
    load_trades, analyze_all_wallets, filter_profitable_wallets, rank_wallets
)
from copybot.wallet_filter import (
    filter_bots, filter_non_copyable, analyze_trade_patterns, classify_wallet_strategy
)
from copybot.signal_engine import (
    generate_signal, simulate_delay, filter_delay_robust_wallets, RiskManager
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("copybot")

# Config
CONFIG_FILE = COPYBOT_DIR / "config.json"
DEFAULT_CONFIG = {
    "min_trades": 20,
    "min_roi": 0.05,
    "min_winrate": 0.52,
    "min_signal_confidence": 50,
    "min_trade_size": 50,
    "max_capital_pct": 0.05,
    "max_per_market": 0.10,
    "stop_loss_pct": 0.20,
    "n_markets_to_scan": 50,
    "monitor_interval": 300,
    "max_tracked_wallets": 20,
    "delay_test_seconds": [30, 120, 300],
}

TRACKED_WALLETS_FILE = DATA_DIR / "tracked_wallets.json"
SIGNALS_FILE = DATA_DIR / "signals.json"
LEADERBOARD_FILE = DATA_DIR / "leaderboard.json"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        # Merge with defaults
        merged = {**DEFAULT_CONFIG, **cfg}
        return merged
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_tracked_wallets():
    if TRACKED_WALLETS_FILE.exists():
        return json.loads(TRACKED_WALLETS_FILE.read_text())
    return []


def save_tracked_wallets(wallets):
    TRACKED_WALLETS_FILE.write_text(json.dumps(wallets, indent=2))


def load_signals():
    if SIGNALS_FILE.exists():
        return json.loads(SIGNALS_FILE.read_text())
    return []


def save_signals(signals):
    SIGNALS_FILE.write_text(json.dumps(signals, indent=2, default=str))


# =========================================================================
# STEP 1: FULL SCAN PIPELINE
# =========================================================================

def run_scan():
    """Full pipeline: collect data → analyze wallets → filter → rank → track."""
    cfg = load_config()
    
    print("=" * 60)
    print("  COPYBOT — FULL SCAN")
    print("=" * 60)
    
    # Step 1: Collect trade data
    print("\n[1/6] Collecting trade data...")
    try:
        build_trade_dataset(n_markets=cfg["n_markets_to_scan"])
        df = load_trades()  # reload with proper parsing
        if df is None or len(df) == 0:
            print("  No trade data collected. Check API access.")
            return
        print(f"  Collected {len(df)} trades across {df['market_id'].nunique()} markets")
    except Exception as e:
        print(f"  Error collecting data: {e}")
        return
    
    # Step 2: Analyze all wallets
    print("\n[2/6] Analyzing wallet profitability...")
    wallet_stats = analyze_all_wallets(df, min_trades=cfg["min_trades"])
    if wallet_stats is None or len(wallet_stats) == 0:
        print("  No wallets with enough trades found.")
        print(f"  (min_trades threshold: {cfg['min_trades']})")
        # Lower threshold and retry
        wallet_stats = analyze_all_wallets(df, min_trades=5)
        if wallet_stats is None or len(wallet_stats) == 0:
            print("  Still no wallets found even with min_trades=5.")
            return
        print(f"  Found {len(wallet_stats)} wallets with min 5 trades")
    else:
        print(f"  Found {len(wallet_stats)} wallets with {cfg['min_trades']}+ trades")
    
    # Step 3: Filter profitable wallets
    print("\n[3/6] Filtering profitable wallets...")
    profitable = filter_profitable_wallets(
        wallet_stats,
        min_roi=cfg["min_roi"],
        min_winrate=cfg["min_winrate"],
        min_trades=min(cfg["min_trades"], 5)
    )
    if profitable is None or len(profitable) == 0:
        print("  No profitable wallets passed filters.")
        print("  Relaxing criteria...")
        profitable = filter_profitable_wallets(
            wallet_stats, min_roi=0.01, min_winrate=0.45, min_trades=5
        )
        if profitable is None or len(profitable) == 0:
            print("  Still none. Showing top wallets by raw ROI instead.")
            profitable = wallet_stats.head(20)
        else:
            print(f"  Found {len(profitable)} with relaxed criteria")
    else:
        print(f"  {len(profitable)} wallets passed profitability filters")
    
    # Step 4: Filter out bots
    print("\n[4/6] Filtering out bots and non-copyable strategies...")
    try:
        filtered = filter_bots(profitable, df)
        filtered = filter_non_copyable(filtered, df)
        print(f"  {len(filtered)} wallets remain after bot/copyability filter")
    except Exception as e:
        print(f"  Filter error: {e} — using unfiltered list")
        filtered = profitable
    
    # Step 5: Rank wallets
    print("\n[5/6] Ranking wallets...")
    try:
        ranked = rank_wallets(filtered)
    except Exception:
        ranked = filtered
    
    top_n = min(cfg["max_tracked_wallets"], len(ranked))
    top_wallets = ranked.head(top_n)
    
    # Step 6: Analyze patterns and save
    print("\n[6/6] Analyzing trade patterns...")
    tracked = []
    for _, row in top_wallets.iterrows():
        wallet = row.get("wallet", row.name if isinstance(row.name, str) else "unknown")
        try:
            patterns = analyze_trade_patterns(df, wallet)
            strategy = classify_wallet_strategy(patterns)
        except Exception:
            patterns = {}
            strategy = {"strategy": "unknown", "confidence": 0}
        
        entry = {
            "wallet": wallet,
            "roi": round(float(row.get("roi", 0)), 4),
            "winrate": round(float(row.get("win_rate", row.get("winrate", 0))), 4),
            "total_trades": int(row.get("total_trades", 0)),
            "total_pnl": round(float(row.get("total_pnl", 0)), 2),
            "avg_trade_size": round(float(row.get("avg_trade_size", 0)), 2),
            "strategy": strategy.get("strategy", "unknown"),
            "strategy_confidence": strategy.get("confidence", 0),
            "tracked_since": datetime.now(timezone.utc).isoformat(),
            "signals_generated": 0,
        }
        tracked.append(entry)
        print(f"  {wallet[:10]}... | ROI: {entry['roi']:+.2%} | WR: {entry['winrate']:.0%} | "
              f"Trades: {entry['total_trades']} | Strategy: {entry['strategy']}")
    
    save_tracked_wallets(tracked)
    
    # Save leaderboard
    leaderboard = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_wallets_scanned": len(wallet_stats) if wallet_stats is not None else 0,
        "profitable_wallets": len(profitable) if profitable is not None else 0,
        "tracked_wallets": len(tracked),
        "wallets": tracked,
    }
    LEADERBOARD_FILE.write_text(json.dumps(leaderboard, indent=2, default=str))
    
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE")
    print(f"  Tracking {len(tracked)} wallets")
    print(f"  Leaderboard saved to {LEADERBOARD_FILE}")
    print(f"{'='*60}\n")


# =========================================================================
# STEP 2: LIVE MONITORING
# =========================================================================

def run_monitor():
    """Monitor tracked wallets for new trades and generate signals."""
    cfg = load_config()
    tracked = load_tracked_wallets()
    
    if not tracked:
        print("No tracked wallets. Run 'scan' first.")
        return
    
    risk_mgr = RiskManager(
        max_capital_pct=cfg["max_capital_pct"],
        max_per_market=cfg["max_per_market"],
        stop_loss_pct=cfg["stop_loss_pct"]
    )
    
    print("=" * 60)
    print("  COPYBOT — LIVE MONITOR")
    print("=" * 60)
    print(f"  Tracking {len(tracked)} wallets")
    print(f"  Check interval: {cfg['monitor_interval']}s")
    print(f"  Min confidence: {cfg['min_signal_confidence']}")
    print(f"  Ctrl+C to stop\n")
    
    signals = load_signals()
    seen_trades = set()
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            log.info("Checking for new trades...")
            
            # Fetch recent trades
            try:
                df = build_trade_dataset(n_markets=20)
            except Exception as e:
                log.error(f"Data fetch error: {e}")
                time.sleep(cfg["monitor_interval"])
                continue
            
            if df is None or len(df) == 0:
                log.info("No trades fetched")
                time.sleep(cfg["monitor_interval"])
                continue
            
            # Check each tracked wallet
            wallet_addresses = {w["wallet"].lower() for w in tracked}
            new_signals = 0
            
            for _, trade in df.iterrows():
                # Check if trade involves a tracked wallet
                maker = str(trade.get("maker", "")).lower()
                taker = str(trade.get("taker", "")).lower()
                
                wallet_match = None
                if maker in wallet_addresses:
                    wallet_match = maker
                elif taker in wallet_addresses:
                    wallet_match = taker
                
                if not wallet_match:
                    continue
                
                # Dedup
                trade_key = f"{wallet_match}_{trade.get('market_id', '')}_{trade.get('timestamp', '')}"
                if trade_key in seen_trades:
                    continue
                seen_trades.add(trade_key)
                
                # Get wallet stats
                wallet_info = next(
                    (w for w in tracked if w["wallet"].lower() == wallet_match), None
                )
                if not wallet_info:
                    continue
                
                # Generate signal
                try:
                    market_info = fetch_market_info(str(trade.get("market_id", "")))
                except Exception:
                    market_info = {}
                
                signal = generate_signal(
                    trade.to_dict(),
                    wallet_info,
                    market_info or {}
                )
                
                if signal and signal.get("confidence", 0) >= cfg["min_signal_confidence"]:
                    # Risk check
                    portfolio = {"balance": 1000, "positions": {}}  # placeholder
                    if risk_mgr.check_trade(signal, portfolio):
                        signal["generated_at"] = now.isoformat()
                        signals.append(signal)
                        new_signals += 1
                        
                        print(f"  🔔 SIGNAL: {signal.get('market_id', '?')} | "
                              f"{signal.get('outcome', '?')} @ ${signal.get('entry_price', 0):.3f} | "
                              f"Confidence: {signal.get('confidence', 0)} | "
                              f"Wallet: {wallet_match[:10]}...")
            
            if new_signals:
                save_signals(signals)
                log.info(f"{new_signals} new signals generated")
            else:
                log.info("No new signals")
            
            # Trim old signals (keep last 200)
            if len(signals) > 200:
                signals = signals[-200:]
                save_signals(signals)
            
            time.sleep(cfg["monitor_interval"])
            
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            save_signals(signals)
            break
        except Exception as e:
            log.error(f"Monitor error: {e}")
            time.sleep(60)


# =========================================================================
# STEP 3: DISPLAY
# =========================================================================

def show_leaderboard():
    """Display top wallets leaderboard."""
    if not LEADERBOARD_FILE.exists():
        print("No leaderboard data. Run 'scan' first.")
        return
    
    data = json.loads(LEADERBOARD_FILE.read_text())
    
    print(f"\n{'='*70}")
    print(f"  COPYBOT — WALLET LEADERBOARD")
    print(f"  Updated: {data.get('updated_at', 'unknown')}")
    print(f"  Scanned: {data.get('total_wallets_scanned', 0)} wallets")
    print(f"  Profitable: {data.get('profitable_wallets', 0)}")
    print(f"  Tracked: {data.get('tracked_wallets', 0)}")
    print(f"{'='*70}\n")
    
    wallets = data.get("wallets", [])
    if not wallets:
        print("  No wallets tracked.")
        return
    
    print(f"  {'#':<4} {'Wallet':<14} {'ROI':>8} {'WinRate':>8} {'Trades':>7} {'PnL':>10} {'Strategy':<15}")
    print(f"  {'-'*4} {'-'*14} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*15}")
    
    for i, w in enumerate(wallets, 1):
        wallet_short = w["wallet"][:12] + ".."
        roi = f"{w['roi']:+.1%}"
        wr = f"{w['winrate']:.0%}"
        trades = str(w["total_trades"])
        pnl = f"${w['total_pnl']:+,.2f}"
        strategy = w.get("strategy", "unknown")
        print(f"  {i:<4} {wallet_short:<14} {roi:>8} {wr:>8} {trades:>7} {pnl:>10} {strategy:<15}")
    
    print(f"\n{'='*70}\n")


def show_signals():
    """Display recent trade signals."""
    signals = load_signals()
    
    if not signals:
        print("No signals yet. Run 'monitor' to generate signals.")
        return
    
    print(f"\n{'='*70}")
    print(f"  COPYBOT — RECENT SIGNALS ({len(signals)} total)")
    print(f"{'='*70}\n")
    
    # Show last 20
    for s in signals[-20:]:
        conf = s.get("confidence", 0)
        emoji = "🟢" if conf >= 70 else "🟡" if conf >= 50 else "🔴"
        print(f"  {emoji} {s.get('generated_at', '?')[:19]} | "
              f"Market: {str(s.get('market_id', '?'))[:12]} | "
              f"{s.get('outcome', '?')} @ ${s.get('entry_price', 0):.3f} | "
              f"Size: ${s.get('suggested_size', 0):.2f} | "
              f"Conf: {conf}")
    
    print(f"\n{'='*70}\n")


# =========================================================================
# STEP 4: BACKTEST
# =========================================================================

def run_backtest():
    """Backtest the copy-trading strategy on historical data."""
    cfg = load_config()
    tracked = load_tracked_wallets()
    
    if not tracked:
        print("No tracked wallets. Run 'scan' first.")
        return
    
    print("=" * 60)
    print("  COPYBOT — BACKTEST")
    print("=" * 60)
    
    try:
        df = load_trades()
    except Exception:
        print("  No trade data. Run 'scan' first.")
        return
    
    if df is None or len(df) == 0:
        print("  No trade data available.")
        return
    
    delays = cfg.get("delay_test_seconds", [30, 120, 300])
    
    print(f"\n  Testing {len(tracked)} wallets with delays: {delays}s\n")
    
    results = []
    for w in tracked:
        wallet = w["wallet"]
        try:
            delay_results = simulate_delay(df, wallet, delays=delays)
            results.append({
                "wallet": wallet[:12] + "..",
                "strategy": w.get("strategy", "?"),
                "original_roi": w.get("roi", 0),
                "delay_results": delay_results,
            })
        except Exception as e:
            log.warning(f"  Backtest error for {wallet[:12]}: {e}")
    
    if not results:
        print("  No backtest results.")
        return
    
    # Display results
    print(f"  {'Wallet':<14} {'Strategy':<13} {'Orig ROI':>9} ", end="")
    for d in delays:
        print(f"{'%ds ROI' % d:>10} ", end="")
    print(f"{'Copyable?':>10}")
    
    print(f"  {'-'*14} {'-'*13} {'-'*9} ", end="")
    for _ in delays:
        print(f"{'-'*10} ", end="")
    print(f"{'-'*10}")
    
    for r in results:
        print(f"  {r['wallet']:<14} {r['strategy']:<13} {r['original_roi']:>+8.1%} ", end="")
        copyable = True
        for d in delays:
            dr = r["delay_results"].get(d, {})
            roi = dr.get("roi", 0)
            if roi <= 0:
                copyable = False
            print(f"{roi:>+9.1%} ", end="")
        emoji = "✅" if copyable else "❌"
        print(f"  {emoji}")
    
    # Summary
    copyable_count = sum(1 for r in results 
                        if all(r["delay_results"].get(d, {}).get("roi", 0) > 0 for d in delays))
    print(f"\n  Delay-robust wallets: {copyable_count}/{len(results)}")
    print(f"{'='*60}\n")


# =========================================================================
# CLI
# =========================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python copybot.py [scan|monitor|leaderboard|signals|backtest]")
        return
    
    cmd = sys.argv[1].lower()
    
    # Ensure config exists
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
    
    if cmd == "scan":
        run_scan()
    elif cmd == "monitor":
        run_monitor()
    elif cmd == "leaderboard":
        show_leaderboard()
    elif cmd == "signals":
        show_signals()
    elif cmd == "backtest":
        run_backtest()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python copybot.py [scan|monitor|leaderboard|signals|backtest]")


if __name__ == "__main__":
    main()
