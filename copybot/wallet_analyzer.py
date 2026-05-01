"""
Wallet Profitability Analysis Module

Analyzes wallet trading performance from trade data stored in trades.json.
Identifies consistently profitable wallets suitable for copy-trading strategies.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TRADES_FILE = DATA_DIR / "trades.json"


def load_trades(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load trades from JSON file and return as a pandas DataFrame.

    Args:
        filepath: Optional path to trades JSON file. Defaults to data/trades.json.

    Returns:
        DataFrame with trade records. Empty DataFrame if file not found.
    """
    path = Path(filepath) if filepath else TRADES_FILE
    if not path.exists():
        print(f"[wallet_analyzer] Trades file not found: {path}")
        return pd.DataFrame()

    with open(path, "r") as f:
        trades = json.load(f)

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Ensure numeric columns
    for col in ("price", "size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Normalize wallet fields: if wallet_address exists but maker/taker are empty, use wallet_address
    if "wallet_address" in df.columns:
        if "maker" not in df.columns:
            df["maker"] = ""
        if "taker" not in df.columns:
            df["taker"] = ""
        # Fill empty maker/taker with wallet_address
        empty_maker = df["maker"].fillna("").eq("")
        df.loc[empty_maker, "maker"] = df.loc[empty_maker, "wallet_address"]
        empty_taker = df["taker"].fillna("").eq("")
        df.loc[empty_taker, "taker"] = df.loc[empty_taker, "wallet_address"]

    return df


def extract_wallets(df: pd.DataFrame) -> np.ndarray:
    """Return unique wallet addresses from both maker and taker fields.

    Args:
        df: DataFrame of trades with 'maker' and 'taker' columns.

    Returns:
        Array of unique wallet address strings.
    """
    wallets = set()
    if "maker" in df.columns:
        wallets.update(df["maker"].dropna().unique())
    if "taker" in df.columns:
        wallets.update(df["taker"].dropna().unique())
    return np.array(list(wallets))


def calculate_wallet_pnl(df: pd.DataFrame, wallet: str) -> dict:
    """Calculate comprehensive PnL metrics for a given wallet.

    Examines all trades where the wallet appears as maker or taker,
    determines position sides, and computes realized/unrealized PnL.

    Args:
        df: DataFrame of all trades.
        wallet: Wallet address to analyze.

    Returns:
        Dict with keys: wallet, total_trades, total_volume, realized_pnl,
        unrealized_pnl, total_pnl, roi, win_rate, avg_trade_size,
        avg_holding_period, profit_factor, gross_profit, gross_loss,
        winning_trades, losing_trades, positions, max_single_trade_pnl,
        pnl_by_date, capital_deployed.
    """
    # Get all trades involving this wallet
    mask = (df["maker"] == wallet) | (df["taker"] == wallet)
    wallet_df = df[mask].copy().sort_values("timestamp").reset_index(drop=True)

    total_trades = len(wallet_df)
    if total_trades == 0:
        return _empty_wallet_stats(wallet)

    # Determine wallet's effective side per trade
    # If wallet is maker: they provide liquidity (take opposite side of trade's 'side')
    # If wallet is taker: they take the listed side
    wallet_df["wallet_side"] = wallet_df.apply(
        lambda row: row["side"] if row["taker"] == wallet
        else ("BUY" if row["side"] == "SELL" else "SELL"),
        axis=1,
    )

    wallet_df["notional"] = wallet_df["price"] * wallet_df["size"]
    total_volume = wallet_df["notional"].sum()
    avg_trade_size = wallet_df["notional"].mean()

    # Track positions per market
    positions = {}  # market_id -> list of open fills
    realized_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    winning_trades = 0
    losing_trades = 0
    trade_pnls = []
    capital_deployed = 0.0

    # Per-date PnL tracking
    pnl_by_date = {}

    for _, trade in wallet_df.iterrows():
        market = trade.get("market_id", "unknown")
        side = trade["wallet_side"]
        price = trade["price"]
        size = trade["size"]
        ts = trade["timestamp"]
        outcome = trade.get("outcome", None)
        if isinstance(ts, str):
            try:
                ts = pd.Timestamp(ts)
            except Exception:
                ts = pd.NaT
        trade_date = ts.date() if pd.notna(ts) else None

        if market not in positions:
            positions[market] = {"net_qty": 0.0, "avg_entry": 0.0, "fills": []}

        pos = positions[market]

        # Determine if this trade opens or closes a position
        direction = 1.0 if side == "BUY" else -1.0
        signed_qty = direction * size

        # Check if this trade reduces existing position (closing trade)
        if pos["net_qty"] != 0 and np.sign(signed_qty) != np.sign(pos["net_qty"]):
            # Closing (partially or fully)
            close_qty = min(abs(signed_qty), abs(pos["net_qty"]))
            # PnL = (exit_price - entry_price) * qty * position_direction
            pos_direction = np.sign(pos["net_qty"])
            pnl = (price - pos["avg_entry"]) * close_qty * pos_direction

            realized_pnl += pnl
            trade_pnls.append(pnl)

            if pnl > 0:
                gross_profit += pnl
                winning_trades += 1
            elif pnl < 0:
                gross_loss += abs(pnl)
                losing_trades += 1

            if trade_date:
                pnl_by_date[trade_date] = pnl_by_date.get(trade_date, 0.0) + pnl

            # Update position
            remaining = abs(pos["net_qty"]) - close_qty
            if remaining < 1e-12:
                # Fully closed; check if there's leftover from new trade
                leftover = abs(signed_qty) - close_qty
                if leftover > 1e-12:
                    pos["net_qty"] = direction * leftover
                    pos["avg_entry"] = price
                    capital_deployed += price * leftover
                else:
                    pos["net_qty"] = 0.0
                    pos["avg_entry"] = 0.0
            else:
                pos["net_qty"] = pos_direction * remaining
                # avg_entry stays the same for partial close
        else:
            # Opening or adding to position
            if pos["net_qty"] == 0:
                pos["avg_entry"] = price
                pos["net_qty"] = signed_qty
            else:
                # Average entry price
                total_cost = pos["avg_entry"] * abs(pos["net_qty"]) + price * size
                pos["net_qty"] += signed_qty
                pos["avg_entry"] = total_cost / abs(pos["net_qty"]) if pos["net_qty"] != 0 else 0

            capital_deployed += price * size

        pos["fills"].append({
            "side": side,
            "price": price,
            "size": size,
            "timestamp": str(ts) if pd.notna(ts) else None,
        })

    # Also check outcome field for resolved markets
    if "outcome" in wallet_df.columns:
        resolved = wallet_df[wallet_df["outcome"].notna()]
        # outcome-based PnL is already captured through position tracking above

    # Calculate unrealized PnL from open positions
    unrealized_pnl = 0.0
    open_positions = []
    for market, pos in positions.items():
        if abs(pos["net_qty"]) > 1e-12:
            # Use last traded price in that market as current price
            market_trades = df[df["market_id"] == market] if "market_id" in df.columns else pd.DataFrame()
            if not market_trades.empty:
                current_price = market_trades.sort_values("timestamp").iloc[-1]["price"]
            else:
                current_price = pos["avg_entry"]

            pos_direction = np.sign(pos["net_qty"])
            upnl = (current_price - pos["avg_entry"]) * abs(pos["net_qty"]) * pos_direction
            unrealized_pnl += upnl

            open_positions.append({
                "market_id": market,
                "side": "BUY" if pos["net_qty"] > 0 else "SELL",
                "size": abs(pos["net_qty"]),
                "entry_price": pos["avg_entry"],
                "current_price": current_price,
                "unrealized_pnl": upnl,
            })

    total_pnl = realized_pnl + unrealized_pnl
    roi = total_pnl / capital_deployed if capital_deployed > 0 else 0.0
    win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Average holding period
    avg_holding_period = None
    if "timestamp" in wallet_df.columns and len(wallet_df) >= 2:
        ts_sorted = wallet_df["timestamp"].dropna().sort_values()
        if len(ts_sorted) >= 2:
            total_span = (ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds()
            closed_count = winning_trades + losing_trades
            if closed_count > 0:
                avg_holding_period = total_span / closed_count  # seconds per round-trip

    max_single_trade_pnl = max(trade_pnls, key=abs) if trade_pnls else 0.0

    return {
        "wallet": wallet,
        "total_trades": total_trades,
        "total_volume": total_volume,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "roi": roi,
        "win_rate": win_rate,
        "avg_trade_size": avg_trade_size,
        "avg_holding_period": avg_holding_period,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "capital_deployed": capital_deployed,
        "max_single_trade_pnl": max_single_trade_pnl,
        "pnl_by_date": pnl_by_date,
        "open_positions": open_positions,
    }


def _empty_wallet_stats(wallet: str) -> dict:
    """Return zeroed-out stats dict for a wallet with no trades."""
    return {
        "wallet": wallet,
        "total_trades": 0,
        "total_volume": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "roi": 0.0,
        "win_rate": 0.0,
        "avg_trade_size": 0.0,
        "avg_holding_period": None,
        "profit_factor": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "capital_deployed": 0.0,
        "max_single_trade_pnl": 0.0,
        "pnl_by_date": {},
        "open_positions": [],
    }


def analyze_all_wallets(df: pd.DataFrame, min_trades: int = 20) -> pd.DataFrame:
    """Run calculate_wallet_pnl for all wallets meeting the minimum trade threshold.

    Args:
        df: DataFrame of all trades.
        min_trades: Minimum number of trades a wallet must have to be analyzed.

    Returns:
        DataFrame of wallet stats sorted by ROI descending.
    """
    wallets = extract_wallets(df)
    results = []

    for wallet in wallets:
        # Quick count check before expensive calculation
        mask = (df["maker"] == wallet) | (df["taker"] == wallet)
        if mask.sum() < min_trades:
            continue

        stats = calculate_wallet_pnl(df, wallet)
        # Drop non-scalar fields for the summary DataFrame
        row = {k: v for k, v in stats.items() if k not in ("pnl_by_date", "open_positions")}
        results.append(row)

    if not results:
        return pd.DataFrame()

    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values("roi", ascending=False).reset_index(drop=True)
    return stats_df


def filter_profitable_wallets(
    wallet_stats: pd.DataFrame,
    min_roi: float = 0.05,
    min_winrate: float = 0.52,
    min_trades: int = 20,
    all_wallet_details: Optional[dict] = None,
) -> pd.DataFrame:
    """Filter for consistently profitable wallets.

    Applies multiple quality filters beyond simple ROI/winrate thresholds:
    - Profit consistency: no single trade accounts for >40% of total PnL
    - Steady growth: positive PnL in at least 60% of trading days

    Args:
        wallet_stats: DataFrame from analyze_all_wallets().
        min_roi: Minimum ROI threshold.
        min_winrate: Minimum win rate threshold.
        min_trades: Minimum total trades.
        all_wallet_details: Optional dict mapping wallet -> full stats dict
            (including pnl_by_date). If None, consistency checks are skipped.

    Returns:
        Filtered DataFrame of qualifying wallets.
    """
    if wallet_stats.empty:
        return wallet_stats

    filtered = wallet_stats[
        (wallet_stats["roi"] >= min_roi)
        & (wallet_stats["win_rate"] >= min_winrate)
        & (wallet_stats["total_trades"] >= min_trades)
    ].copy()

    if filtered.empty:
        return filtered

    # Profit consistency: max single trade PnL should be <= 40% of total PnL
    consistency_mask = filtered.apply(
        lambda row: abs(row["max_single_trade_pnl"]) <= 0.4 * abs(row["total_pnl"])
        if row["total_pnl"] != 0 else True,
        axis=1,
    )
    filtered = filtered[consistency_mask]

    # Steady growth check: positive PnL on >= 60% of trading days
    if all_wallet_details:
        steady_wallets = []
        for _, row in filtered.iterrows():
            wallet = row["wallet"]
            details = all_wallet_details.get(wallet, {})
            pnl_by_date = details.get("pnl_by_date", {})

            if not pnl_by_date:
                # No date granularity — keep wallet (benefit of doubt)
                steady_wallets.append(True)
                continue

            total_days = len(pnl_by_date)
            positive_days = sum(1 for v in pnl_by_date.values() if v > 0)
            ratio = positive_days / total_days if total_days > 0 else 0
            steady_wallets.append(ratio >= 0.60)

        filtered = filtered[steady_wallets]

    # Add consistency score column
    if not filtered.empty and "max_single_trade_pnl" in filtered.columns:
        filtered = filtered.copy()
        filtered["consistency"] = 1.0 - filtered.apply(
            lambda row: abs(row["max_single_trade_pnl"]) / abs(row["total_pnl"])
            if row["total_pnl"] != 0 else 0.0,
            axis=1,
        )

    return filtered.reset_index(drop=True)


def rank_wallets(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Rank wallets by a composite score.

    Composite score = 0.4 * ROI_normalized + 0.3 * winrate + 0.2 * consistency + 0.1 * volume_score

    Args:
        filtered_df: Filtered DataFrame from filter_profitable_wallets().

    Returns:
        DataFrame sorted by composite_score descending with rank column.
    """
    if filtered_df.empty:
        return filtered_df

    ranked = filtered_df.copy()

    # Normalize ROI to [0, 1]
    roi_min, roi_max = ranked["roi"].min(), ranked["roi"].max()
    if roi_max > roi_min:
        ranked["roi_normalized"] = (ranked["roi"] - roi_min) / (roi_max - roi_min)
    else:
        ranked["roi_normalized"] = 1.0

    # Win rate is already in [0, 1]

    # Consistency: if not computed yet, derive it
    if "consistency" not in ranked.columns:
        ranked["consistency"] = 1.0 - ranked.apply(
            lambda row: abs(row["max_single_trade_pnl"]) / abs(row["total_pnl"])
            if row["total_pnl"] != 0 else 0.0,
            axis=1,
        )

    # Normalize volume to [0, 1]
    vol_min, vol_max = ranked["total_volume"].min(), ranked["total_volume"].max()
    if vol_max > vol_min:
        ranked["volume_score"] = (ranked["total_volume"] - vol_min) / (vol_max - vol_min)
    else:
        ranked["volume_score"] = 1.0

    # Composite score
    ranked["composite_score"] = (
        0.4 * ranked["roi_normalized"]
        + 0.3 * ranked["win_rate"]
        + 0.2 * ranked["consistency"]
        + 0.1 * ranked["volume_score"]
    )

    ranked = ranked.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    return ranked


if __name__ == "__main__":
    print("=" * 70)
    print("  Wallet Profitability Analysis")
    print("=" * 70)

    df = load_trades()
    if df.empty:
        print("\nNo trade data found. Ensure trades.json exists in data/ directory.")
        raise SystemExit(1)

    print(f"\nLoaded {len(df)} trades")
    wallets = extract_wallets(df)
    print(f"Found {len(wallets)} unique wallets")

    # Analyze all wallets
    print("\nAnalyzing wallets (min 20 trades)...")
    wallet_stats = analyze_all_wallets(df, min_trades=20)
    print(f"Wallets meeting trade threshold: {len(wallet_stats)}")

    if wallet_stats.empty:
        print("No wallets meet the minimum trade threshold.")
        raise SystemExit(0)

    # Build detail dict for consistency checks
    print("Computing detailed stats for filtering...")
    detail_map = {}
    for wallet in wallet_stats["wallet"].values:
        detail_map[wallet] = calculate_wallet_pnl(df, wallet)

    # Filter profitable wallets
    profitable = filter_profitable_wallets(wallet_stats, all_wallet_details=detail_map)
    print(f"Profitable wallets after filtering: {len(profitable)}")

    if profitable.empty:
        print("No wallets passed profitability filters. Showing top by ROI instead:")
        display_cols = ["wallet", "total_trades", "total_volume", "roi", "win_rate", "total_pnl", "profit_factor"]
        available = [c for c in display_cols if c in wallet_stats.columns]
        print(wallet_stats.head(20)[available].to_string(index=False))
        raise SystemExit(0)

    # Rank wallets
    ranked = rank_wallets(profitable)

    # Display top 20
    print(f"\n{'=' * 70}")
    print("  TOP 20 WALLETS BY COMPOSITE SCORE")
    print(f"{'=' * 70}\n")

    display_cols = [
        "rank", "wallet", "composite_score", "roi", "win_rate",
        "total_trades", "total_pnl", "profit_factor", "consistency",
    ]
    available = [c for c in display_cols if c in ranked.columns]
    top20 = ranked.head(20)[available]

    # Format for readability
    pd.set_option("display.max_colwidth", 16)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 140)

    print(top20.to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"  Total wallets analyzed: {len(wallet_stats)}")
    print(f"  Profitable wallets:     {len(profitable)}")
    print(f"  Showing top:            {min(20, len(ranked))}")
    print(f"{'=' * 70}")
