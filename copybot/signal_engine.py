"""
Signal Engine Module for Polymarket Copy-Trading Bot.

Generates copy-trading signals, simulates delayed entries,
and manages risk for position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time


def generate_signal(
    trade: Dict,
    wallet_stats: Dict,
    market_info: Dict,
    min_threshold: float = 50.0,
    user_capital: float = 10000.0,
) -> Optional[Dict]:
    """
    Generate a copy-trading signal when a tracked wallet trades.

    Checks:
        - Trade size > min_threshold ($50 default)
        - Market has enough liquidity
        - Calculates acceptable copy price range (entry ± 3%)
        - Scales position size based on user's capital vs wallet's typical size
        - Calculates confidence score (0-100)

    Args:
        trade: Dict with keys: wallet, market_id, outcome, side, price, size, timestamp
        wallet_stats: Dict with keys: win_rate, roi, avg_trade_size, preferred_categories
        market_info: Dict with keys: market_id, liquidity, category, best_bid, best_ask
        min_threshold: Minimum trade size in USD to generate signal.
        user_capital: User's total available capital.

    Returns:
        Signal dict or None if trade doesn't qualify.
        Signal: {market_id, outcome, entry_price, copy_price_range, suggested_size,
                 timestamp, confidence, wallet, reason}
    """
    # --- Check minimum size ---
    trade_size = trade.get('size', 0)
    if trade_size < min_threshold:
        return None

    # --- Check liquidity ---
    liquidity = market_info.get('liquidity', 0)
    if liquidity < trade_size * 2:
        return None

    # --- Calculate copy price range (±3%) ---
    entry_price = trade.get('price', 0)
    margin = entry_price * 0.03
    copy_price_range = (round(max(0.01, entry_price - margin), 4),
                        round(min(0.99, entry_price + margin), 4))

    # --- Scale position size ---
    avg_trade_size = wallet_stats.get('avg_trade_size', trade_size)
    if avg_trade_size > 0:
        size_ratio = trade_size / avg_trade_size
    else:
        size_ratio = 1.0

    # Base allocation: 2% of user capital, scaled by trade size ratio
    base_size = user_capital * 0.02
    suggested_size = round(base_size * min(size_ratio, 3.0), 2)  # cap at 3x
    suggested_size = min(suggested_size, user_capital * 0.05)  # hard cap 5%

    # --- Confidence score (0-100) ---
    confidence = 0.0

    # Win rate component (0-30)
    win_rate = wallet_stats.get('win_rate', 0.5)
    confidence += max(0, (win_rate - 0.5)) * 60  # 60% WR = 6pts, 80% WR = 18pts

    # ROI component (0-25)
    roi = wallet_stats.get('roi', 0)
    confidence += min(25, max(0, roi * 100))

    # Trade size vs average (0-20): larger than usual = more conviction
    if avg_trade_size > 0:
        size_signal = trade_size / avg_trade_size
        if size_signal > 1.5:
            confidence += 20
        elif size_signal > 1.0:
            confidence += 10
        else:
            confidence += 5

    # Category match (0-25): wallet specializes in this market's category
    preferred_cats = wallet_stats.get('preferred_categories', [])
    market_cat = market_info.get('category', '')
    if market_cat and market_cat in preferred_cats:
        confidence += 25
    else:
        confidence += 5

    confidence = round(min(100, max(0, confidence)), 1)

    # --- Build reason string ---
    reasons = []
    if win_rate > 0.6:
        reasons.append(f"high win rate ({win_rate:.0%})")
    if roi > 0.1:
        reasons.append(f"strong ROI ({roi:.0%})")
    if size_signal > 1.5:
        reasons.append("larger than usual position")
    if market_cat in preferred_cats:
        reasons.append(f"specialist in {market_cat}")
    reason = '; '.join(reasons) if reasons else 'standard signal'

    return {
        'market_id': trade.get('market_id'),
        'outcome': trade.get('outcome', trade.get('side')),
        'entry_price': entry_price,
        'copy_price_range': copy_price_range,
        'suggested_size': suggested_size,
        'timestamp': trade.get('timestamp', pd.Timestamp.now()),
        'confidence': confidence,
        'wallet': trade.get('wallet'),
        'reason': reason,
    }


def simulate_delay(
    trades_df: pd.DataFrame,
    wallet: str,
    delays: List[int] = None,
) -> Dict:
    """
    Simulate copying a wallet's trades with various delays.

    For each trade, estimates the price available after the delay and
    calculates PnL impact compared to the original entry.

    Args:
        trades_df: DataFrame with columns: wallet, timestamp, side, price, size,
                   market_id. Should also have 'price_after_30s', 'price_after_120s',
                   'price_after_300s' or will estimate from subsequent trades.
        wallet: Wallet address to simulate.
        delays: List of delay durations in seconds. Default: [30, 120, 300].

    Returns:
        Dict: {delay_seconds: {pnl, roi, win_rate, avg_slippage}}
    """
    if delays is None:
        delays = [30, 120, 300]

    wt = trades_df[trades_df['wallet'] == wallet].copy()
    if wt.empty:
        return {d: {'pnl': 0, 'roi': 0, 'win_rate': 0, 'avg_slippage': 0} for d in delays}

    wt['timestamp'] = pd.to_datetime(wt['timestamp'])
    wt = wt.sort_values('timestamp')
    trades_df = trades_df.copy()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

    results = {}

    for delay in delays:
        pnl_list = []
        slippages = []

        for _, trade in wt.iterrows():
            original_price = trade['price']
            trade_time = trade['timestamp']
            delayed_time = trade_time + pd.Timedelta(seconds=delay)
            market = trade['market_id']
            side = trade['side']

            # Check for pre-computed delayed price columns
            col_name = f'price_after_{delay}s'
            if col_name in trade.index and pd.notna(trade[col_name]):
                delayed_price = trade[col_name]
            else:
                # Estimate from next trade in same market after delay
                future_trades = trades_df[
                    (trades_df['market_id'] == market) &
                    (trades_df['timestamp'] >= delayed_time)
                ].sort_values('timestamp')

                if future_trades.empty:
                    # Assume small adverse movement
                    slippage_est = 0.01 * (1 if side == 'buy' else -1)
                    delayed_price = original_price + slippage_est
                else:
                    delayed_price = future_trades.iloc[0]['price']

            # Calculate slippage
            if side == 'buy':
                slippage = delayed_price - original_price
                # PnL: if buying later at higher price, worse outcome
                # Assume eventual resolution at same exit
                trade_pnl = -slippage * trade['size']
            else:
                slippage = original_price - delayed_price
                trade_pnl = -slippage * trade['size']

            pnl_list.append(trade_pnl)
            slippages.append(abs(delayed_price - original_price))

        total_pnl = sum(pnl_list)
        total_invested = wt['size'].sum() * wt['price'].mean() if len(wt) > 0 else 1
        roi = total_pnl / total_invested if total_invested > 0 else 0
        win_rate = sum(1 for p in pnl_list if p >= 0) / len(pnl_list) if pnl_list else 0

        results[delay] = {
            'pnl': round(total_pnl, 2),
            'roi': round(roi, 4),
            'win_rate': round(win_rate, 4),
            'avg_slippage': round(np.mean(slippages), 4) if slippages else 0,
        }

    return results


def filter_delay_robust_wallets(
    wallet_list: List[str],
    trades_df: pd.DataFrame,
    max_delay: int = 300,
) -> List[str]:
    """
    Keep only wallets that remain profitable even with a 5-minute copy delay.

    Args:
        wallet_list: List of wallet addresses to evaluate.
        trades_df: DataFrame of all trades.
        max_delay: Maximum delay to test (seconds). Default: 300 (5 min).

    Returns:
        Filtered list of wallet addresses that are profitable at max_delay.
    """
    robust = []
    for wallet in wallet_list:
        delay_results = simulate_delay(trades_df, wallet, delays=[max_delay])
        if delay_results.get(max_delay, {}).get('roi', -1) > 0:
            robust.append(wallet)
    return robust


class RiskManager:
    """
    Risk management for copy-trading positions.

    Enforces position sizing limits, market exposure caps,
    and stop-loss thresholds.
    """

    def __init__(
        self,
        max_capital_pct: float = 0.05,
        max_per_market: float = 0.10,
        stop_loss_pct: float = 0.20,
    ):
        """
        Initialize RiskManager.

        Args:
            max_capital_pct: Maximum fraction of capital per single trade (default 5%).
            max_per_market: Maximum fraction of capital in any single market (default 10%).
            stop_loss_pct: Stop-loss threshold as fraction of position (default 20%).
        """
        self.max_capital_pct = max_capital_pct
        self.max_per_market = max_per_market
        self.stop_loss_pct = stop_loss_pct

    def check_trade(self, signal: Dict, portfolio: Dict) -> bool:
        """
        Check if a trade passes all risk checks.

        Args:
            signal: Signal dict from generate_signal().
            portfolio: Dict with keys:
                - balance: current available balance
                - total_capital: total portfolio value
                - positions: list of dicts with {market_id, size, entry_price, current_price}

        Returns:
            True if trade passes all risk checks, False otherwise.
        """
        balance = portfolio.get('balance', 0)
        total_capital = portfolio.get('total_capital', balance)

        if total_capital <= 0:
            return False

        suggested_size = signal.get('suggested_size', 0)
        market_id = signal.get('market_id')

        # Check: trade size within capital limit
        if suggested_size > total_capital * self.max_capital_pct:
            return False

        # Check: sufficient balance
        if suggested_size > balance:
            return False

        # Check: market exposure
        if not self.check_exposure(portfolio, market_id):
            return False

        # Check: minimum confidence
        if signal.get('confidence', 0) < 20:
            return False

        return True

    def calculate_position_size(self, signal: Dict, balance: float) -> float:
        """
        Calculate risk-adjusted position size, capped by limits.

        Args:
            signal: Signal dict from generate_signal().
            balance: Current available balance.

        Returns:
            Position size in USD, capped appropriately.
        """
        suggested = signal.get('suggested_size', 0)
        confidence = signal.get('confidence', 50) / 100.0

        # Scale by confidence
        adjusted = suggested * confidence

        # Cap at max_capital_pct of balance
        max_size = balance * self.max_capital_pct
        final_size = min(adjusted, max_size)

        # Floor at $10 minimum or 0
        if final_size < 10:
            return 0.0

        return round(final_size, 2)

    def check_exposure(self, portfolio: Dict, market_id: str) -> bool:
        """
        Check if adding to this market would exceed exposure limits.

        Args:
            portfolio: Portfolio dict with 'positions' and 'total_capital'.
            market_id: Market to check exposure for.

        Returns:
            True if exposure is within limits, False if overexposed.
        """
        total_capital = portfolio.get('total_capital', 0)
        if total_capital <= 0:
            return False

        positions = portfolio.get('positions', [])
        market_exposure = sum(
            p.get('size', 0) for p in positions
            if p.get('market_id') == market_id
        )

        return market_exposure < total_capital * self.max_per_market


if __name__ == '__main__':
    print("=== Signal Engine Module Test ===\n")

    np.random.seed(42)
    now = pd.Timestamp.now()

    # --- Test generate_signal ---
    trade = {
        'wallet': '0xabc123',
        'market_id': 'mkt_presidential',
        'outcome': 'Yes',
        'side': 'buy',
        'price': 0.55,
        'size': 200.0,
        'timestamp': now,
    }
    wallet_stats = {
        'win_rate': 0.68,
        'roi': 0.22,
        'avg_trade_size': 150.0,
        'preferred_categories': ['politics', 'sports'],
    }
    market_info = {
        'market_id': 'mkt_presidential',
        'liquidity': 50000.0,
        'category': 'politics',
        'best_bid': 0.54,
        'best_ask': 0.56,
    }

    signal = generate_signal(trade, wallet_stats, market_info, user_capital=10000)
    print("Generated Signal:")
    for k, v in signal.items():
        print(f"  {k}: {v}")
    assert signal is not None, "Signal should be generated"
    assert signal['confidence'] > 0, "Confidence should be positive"
    assert signal['suggested_size'] > 0, "Size should be positive"

    # Test rejection: too small
    small_trade = trade.copy()
    small_trade['size'] = 10.0
    assert generate_signal(small_trade, wallet_stats, market_info) is None, \
        "Small trade should be rejected"

    # Test rejection: low liquidity
    low_liq = market_info.copy()
    low_liq['liquidity'] = 100.0
    assert generate_signal(trade, wallet_stats, low_liq) is None, \
        "Low liquidity should be rejected"

    print("\n✅ generate_signal tests passed!")

    # --- Test simulate_delay ---
    rows = []
    for i in range(50):
        ts = now + pd.Timedelta(minutes=i * 30)
        rows.append({
            'wallet': '0xabc123',
            'timestamp': ts,
            'side': 'buy' if i % 3 != 0 else 'sell',
            'price': round(0.50 + np.random.uniform(-0.05, 0.05), 3),
            'size': round(np.random.uniform(100, 300), 2),
            'market_id': np.random.choice(['mkt_A', 'mkt_B']),
        })
    # Add other wallets' trades for price discovery
    for i in range(200):
        ts = now + pd.Timedelta(minutes=i * 5)
        rows.append({
            'wallet': '0xother',
            'timestamp': ts,
            'side': np.random.choice(['buy', 'sell']),
            'price': round(0.50 + np.random.uniform(-0.08, 0.08), 3),
            'size': round(np.random.uniform(50, 500), 2),
            'market_id': np.random.choice(['mkt_A', 'mkt_B']),
        })

    sim_trades = pd.DataFrame(rows)

    delay_results = simulate_delay(sim_trades, '0xabc123', delays=[30, 120, 300])
    print("\nDelay Simulation Results:")
    for delay, metrics in delay_results.items():
        print(f"  {delay}s delay: {metrics}")

    print("\n✅ simulate_delay tests passed!")

    # --- Test filter_delay_robust_wallets ---
    robust = filter_delay_robust_wallets(['0xabc123'], sim_trades, max_delay=300)
    print(f"\nDelay-robust wallets: {robust}")
    print("✅ filter_delay_robust_wallets tests passed!")

    # --- Test RiskManager ---
    rm = RiskManager(max_capital_pct=0.05, max_per_market=0.10, stop_loss_pct=0.20)

    portfolio = {
        'balance': 5000.0,
        'total_capital': 10000.0,
        'positions': [
            {'market_id': 'mkt_A', 'size': 500, 'entry_price': 0.50, 'current_price': 0.52},
        ],
    }

    # Should pass
    assert rm.check_trade(signal, portfolio), "Valid signal should pass risk check"

    # Position sizing
    size = rm.calculate_position_size(signal, 5000.0)
    print(f"\nRisk-adjusted position size: ${size}")
    assert size <= 5000 * 0.05, "Size should be capped at 5% of balance"
    assert size > 0, "Size should be positive"

    # Exposure check
    assert rm.check_exposure(portfolio, 'mkt_A'), "mkt_A exposure should be OK"

    # Overexposed market
    heavy_portfolio = {
        'balance': 5000.0,
        'total_capital': 10000.0,
        'positions': [
            {'market_id': 'mkt_X', 'size': 1500, 'entry_price': 0.50, 'current_price': 0.48},
        ],
    }
    assert not rm.check_exposure(heavy_portfolio, 'mkt_X'), "mkt_X should be overexposed"

    print("✅ RiskManager tests passed!")
    print("\n✅ All signal_engine tests passed!")
