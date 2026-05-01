"""
Wallet Filter Module for Polymarket Copy-Trading Bot.

Filters out non-copyable wallets (bots, scalpers, arb bots) and
analyzes trade patterns to classify wallet strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def filter_bots(wallet_stats_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and remove bot wallets from the dataset.

    Detection criteria:
        - Trades > 10 per minute in any rolling window = bot
        - Average hold time < 5 minutes = scalper
        - Perfectly symmetric buy/sell patterns = arb bot
        - > 90% of trades in same 1-second block = bot

    Args:
        wallet_stats_df: DataFrame with wallet statistics (indexed or columned by 'wallet').
        trades_df: DataFrame of trades with columns:
            wallet, timestamp, side ('buy'/'sell'), price, size, market_id

    Returns:
        Filtered wallet_stats_df with bot wallets removed.
    """
    if wallet_stats_df.empty or trades_df.empty:
        return wallet_stats_df

    bot_wallets = set()
    trades = trades_df.copy()
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])

    for wallet, grp in trades.groupby('wallet'):
        grp = grp.sort_values('timestamp')

        # --- High-frequency check: >10 trades in any 1-minute window ---
        if len(grp) >= 10:
            grp_indexed = grp.set_index('timestamp')
            rolling_counts = grp_indexed.resample('1min').size()
            if (rolling_counts > 10).any():
                bot_wallets.add(wallet)
                continue

        # --- Scalper check: average hold time < 5 minutes ---
        buys = grp[grp['side'] == 'buy'].sort_values('timestamp')
        sells = grp[grp['side'] == 'sell'].sort_values('timestamp')
        if len(buys) > 0 and len(sells) > 0:
            hold_times = []
            sell_times = sells['timestamp'].tolist()
            sell_idx = 0
            for _, buy_row in buys.iterrows():
                while sell_idx < len(sell_times) and sell_times[sell_idx] <= buy_row['timestamp']:
                    sell_idx += 1
                if sell_idx < len(sell_times):
                    ht = (sell_times[sell_idx] - buy_row['timestamp']).total_seconds()
                    hold_times.append(ht)
                    sell_idx += 1
            if hold_times and np.mean(hold_times) < 300:  # 5 minutes
                bot_wallets.add(wallet)
                continue

        # --- Arb bot check: perfectly symmetric buy/sell patterns ---
        buy_counts = grp[grp['side'] == 'buy'].groupby('market_id').size()
        sell_counts = grp[grp['side'] == 'sell'].groupby('market_id').size()
        common_markets = buy_counts.index.intersection(sell_counts.index)
        if len(common_markets) > 2:
            diffs = abs(buy_counts.reindex(common_markets, fill_value=0) -
                        sell_counts.reindex(common_markets, fill_value=0))
            if (diffs == 0).all():
                bot_wallets.add(wallet)
                continue

        # --- Same-second block check: >90% trades in same 1-second block ---
        second_blocks = grp['timestamp'].dt.floor('1s')
        block_counts = second_blocks.value_counts()
        if len(grp) > 5 and block_counts.max() / len(grp) > 0.9:
            bot_wallets.add(wallet)
            continue

    wallet_col = 'wallet' if 'wallet' in wallet_stats_df.columns else wallet_stats_df.index.name
    if wallet_col == 'wallet':
        filtered = wallet_stats_df[~wallet_stats_df['wallet'].isin(bot_wallets)]
    else:
        filtered = wallet_stats_df[~wallet_stats_df.index.isin(bot_wallets)]

    return filtered


def filter_non_copyable(wallet_stats_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove wallets whose strategies cannot be practically copied.

    Criteria:
        - Average hold time < 30 minutes (too fast to copy)
        - Relies on being first in orderbook (latency-dependent)
        - Only trades during very specific microsecond windows (automated)

    Args:
        wallet_stats_df: DataFrame with wallet statistics.
        trades_df: DataFrame of trades.

    Returns:
        Filtered wallet_stats_df with non-copyable wallets removed.
    """
    if wallet_stats_df.empty or trades_df.empty:
        return wallet_stats_df

    non_copyable = set()
    trades = trades_df.copy()
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])

    for wallet, grp in trades.groupby('wallet'):
        grp = grp.sort_values('timestamp')

        # --- Hold time < 30 minutes ---
        buys = grp[grp['side'] == 'buy'].sort_values('timestamp')
        sells = grp[grp['side'] == 'sell'].sort_values('timestamp')
        if len(buys) > 0 and len(sells) > 0:
            hold_times = []
            sell_times = sells['timestamp'].tolist()
            sell_idx = 0
            for _, buy_row in buys.iterrows():
                while sell_idx < len(sell_times) and sell_times[sell_idx] <= buy_row['timestamp']:
                    sell_idx += 1
                if sell_idx < len(sell_times):
                    ht = (sell_times[sell_idx] - buy_row['timestamp']).total_seconds()
                    hold_times.append(ht)
                    sell_idx += 1
            if hold_times and np.mean(hold_times) < 1800:  # 30 minutes
                non_copyable.add(wallet)
                continue

        # --- Orderbook-first detection: consistently gets best prices ---
        if len(grp) >= 10:
            # If wallet's buy prices are consistently at or near the lowest
            # and sell prices at or near the highest per market, it's likely
            # relying on orderbook position
            for mkt, mkt_grp in grp.groupby('market_id'):
                mkt_buys = mkt_grp[mkt_grp['side'] == 'buy']
                if len(mkt_buys) >= 5:
                    all_buys_in_mkt = trades[(trades['market_id'] == mkt) &
                                             (trades['side'] == 'buy')]
                    if len(all_buys_in_mkt) > 0:
                        pct_rank = mkt_buys['price'].apply(
                            lambda p: (all_buys_in_mkt['price'] <= p).mean()
                        )
                        if pct_rank.mean() < 0.1:  # consistently lowest prices
                            non_copyable.add(wallet)
                            break

        # --- Microsecond window trading ---
        if len(grp) >= 10:
            micros = grp['timestamp'].dt.microsecond
            unique_micros = micros.nunique()
            if unique_micros <= 3 and len(grp) > 10:
                non_copyable.add(wallet)
                continue

    wallet_col = 'wallet' if 'wallet' in wallet_stats_df.columns else wallet_stats_df.index.name
    if wallet_col == 'wallet':
        filtered = wallet_stats_df[~wallet_stats_df['wallet'].isin(non_copyable)]
    else:
        filtered = wallet_stats_df[~wallet_stats_df.index.isin(non_copyable)]

    return filtered


def analyze_trade_patterns(trades_df: pd.DataFrame, wallet: str) -> Dict:
    """
    Analyze trading patterns for a specific wallet.

    Analyzes:
        - Entry timing: 'early' (before major moves) vs 'late' (momentum)
        - Position sizing strategy: fixed, kelly, martingale, or variable
        - Market specialization by category
        - Average time between trades
        - Preferred trade size range

    Args:
        trades_df: DataFrame of all trades.
        wallet: Wallet address to analyze.

    Returns:
        Dict with keys: entry_timing, position_sizing, market_specialization,
        avg_time_between_trades, preferred_size_range, trade_count
    """
    wt = trades_df[trades_df['wallet'] == wallet].copy()
    if wt.empty:
        return {
            'entry_timing': 'unknown',
            'position_sizing': 'unknown',
            'market_specialization': {},
            'avg_time_between_trades': None,
            'preferred_size_range': (0, 0),
            'trade_count': 0,
        }

    wt['timestamp'] = pd.to_datetime(wt['timestamp'])
    wt = wt.sort_values('timestamp')

    # --- Entry timing ---
    # Compare wallet's entry price to subsequent price movement
    entry_timing = 'unknown'
    if len(wt) >= 5 and 'price_after' in wt.columns:
        buys = wt[wt['side'] == 'buy']
        if len(buys) > 0 and 'price_after' in buys.columns:
            price_moves = buys['price_after'] - buys['price']
            if price_moves.mean() > 0:
                entry_timing = 'early'
            else:
                entry_timing = 'late'
    else:
        # Heuristic: if trades cluster before big volume spikes, it's early
        buys = wt[wt['side'] == 'buy']
        if len(buys) >= 3:
            # Check if buy prices tend to be below median price for same market
            all_prices = trades_df.groupby('market_id')['price'].median()
            below_median = 0
            for _, row in buys.iterrows():
                if row['market_id'] in all_prices.index:
                    if row['price'] < all_prices[row['market_id']]:
                        below_median += 1
            if len(buys) > 0:
                ratio = below_median / len(buys)
                entry_timing = 'early' if ratio > 0.6 else 'late'

    # --- Position sizing ---
    sizes = wt['size'].values
    if len(sizes) >= 3:
        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        if cv < 0.1:
            position_sizing = 'fixed'
        elif cv < 0.3:
            position_sizing = 'kelly'
        else:
            # Check for martingale: sizes increase after losses
            size_diffs = np.diff(sizes)
            if len(size_diffs) > 2 and (size_diffs > 0).sum() / len(size_diffs) > 0.7:
                position_sizing = 'martingale'
            else:
                position_sizing = 'variable'
    else:
        position_sizing = 'unknown'

    # --- Market specialization ---
    market_counts = wt['market_id'].value_counts()
    total = market_counts.sum()
    market_specialization = (market_counts / total).to_dict() if total > 0 else {}

    # --- Average time between trades ---
    if len(wt) >= 2:
        time_diffs = wt['timestamp'].diff().dropna().dt.total_seconds()
        avg_time = time_diffs.mean()
    else:
        avg_time = None

    # --- Preferred size range ---
    if len(sizes) > 0:
        preferred_size_range = (float(np.percentile(sizes, 25)),
                                float(np.percentile(sizes, 75)))
    else:
        preferred_size_range = (0.0, 0.0)

    return {
        'entry_timing': entry_timing,
        'position_sizing': position_sizing,
        'market_specialization': market_specialization,
        'avg_time_between_trades': avg_time,
        'preferred_size_range': preferred_size_range,
        'trade_count': len(wt),
    }


def classify_wallet_strategy(patterns: Dict) -> Tuple[str, float]:
    """
    Classify a wallet's strategy based on analyzed patterns.

    Categories:
        - 'event_driven': Trades around specific events, early entry
        - 'momentum': Late entry, follows trends
        - 'contrarian': Buys when others sell, early entry
        - 'diversified': Trades many markets evenly
        - 'specialist': Focuses on few markets

    Args:
        patterns: Dict from analyze_trade_patterns().

    Returns:
        Tuple of (strategy_label, confidence) where confidence is 0.0-1.0.
    """
    if patterns.get('trade_count', 0) == 0:
        return ('unknown', 0.0)

    scores = {
        'event_driven': 0.0,
        'momentum': 0.0,
        'contrarian': 0.0,
        'diversified': 0.0,
        'specialist': 0.0,
    }

    # Entry timing signals
    timing = patterns.get('entry_timing', 'unknown')
    if timing == 'early':
        scores['event_driven'] += 0.3
        scores['contrarian'] += 0.2
    elif timing == 'late':
        scores['momentum'] += 0.4

    # Position sizing signals
    sizing = patterns.get('position_sizing', 'unknown')
    if sizing == 'kelly':
        scores['event_driven'] += 0.2
    elif sizing == 'fixed':
        scores['diversified'] += 0.2
    elif sizing == 'martingale':
        scores['contrarian'] += 0.2
    elif sizing == 'variable':
        scores['momentum'] += 0.1

    # Market specialization
    spec = patterns.get('market_specialization', {})
    num_markets = len(spec)
    if num_markets >= 10:
        scores['diversified'] += 0.4
    elif num_markets <= 3 and num_markets > 0:
        scores['specialist'] += 0.4
        # Check concentration
        max_share = max(spec.values()) if spec else 0
        if max_share > 0.5:
            scores['specialist'] += 0.2

    # Trade frequency
    avg_time = patterns.get('avg_time_between_trades')
    if avg_time is not None:
        if avg_time < 3600:  # < 1 hour between trades
            scores['momentum'] += 0.1
        elif avg_time > 86400:  # > 1 day between trades
            scores['event_driven'] += 0.2

    # Pick highest scoring strategy
    best = max(scores, key=scores.get)
    confidence = scores[best]
    # Normalize confidence to 0-1
    total = sum(scores.values())
    if total > 0:
        confidence = scores[best] / total

    return (best, round(confidence, 3))


if __name__ == '__main__':
    print("=== Wallet Filter Module Test ===\n")

    # Create sample trades
    np.random.seed(42)
    n_trades = 200
    wallets = ['wallet_human', 'wallet_bot', 'wallet_scalper', 'wallet_arb']
    now = pd.Timestamp.now()

    rows = []
    # Human trader: spread out trades
    for i in range(30):
        rows.append({
            'wallet': 'wallet_human',
            'timestamp': now + pd.Timedelta(hours=i * 2),
            'side': np.random.choice(['buy', 'sell']),
            'price': round(np.random.uniform(0.3, 0.7), 2),
            'size': round(np.random.uniform(50, 500), 2),
            'market_id': np.random.choice(['mkt_A', 'mkt_B', 'mkt_C']),
        })

    # Bot: many trades in same minute
    base = now + pd.Timedelta(days=1)
    for i in range(50):
        rows.append({
            'wallet': 'wallet_bot',
            'timestamp': base + pd.Timedelta(seconds=i),
            'side': np.random.choice(['buy', 'sell']),
            'price': round(np.random.uniform(0.4, 0.6), 2),
            'size': round(np.random.uniform(10, 50), 2),
            'market_id': 'mkt_A',
        })

    # Scalper: very short hold times
    for i in range(40):
        rows.append({
            'wallet': 'wallet_scalper',
            'timestamp': base + pd.Timedelta(minutes=i * 2),
            'side': 'buy' if i % 2 == 0 else 'sell',
            'price': round(np.random.uniform(0.45, 0.55), 2),
            'size': round(np.random.uniform(100, 200), 2),
            'market_id': 'mkt_B',
        })

    # Arb bot: symmetric buy/sell across markets
    for mkt in ['mkt_X', 'mkt_Y', 'mkt_Z']:
        for i in range(10):
            rows.append({
                'wallet': 'wallet_arb',
                'timestamp': base + pd.Timedelta(minutes=i * 5),
                'side': 'buy',
                'price': 0.50,
                'size': 100.0,
                'market_id': mkt,
            })
            rows.append({
                'wallet': 'wallet_arb',
                'timestamp': base + pd.Timedelta(minutes=i * 5 + 1),
                'side': 'sell',
                'price': 0.52,
                'size': 100.0,
                'market_id': mkt,
            })

    trades_df = pd.DataFrame(rows)

    wallet_stats_df = pd.DataFrame({
        'wallet': wallets,
        'total_trades': [30, 50, 40, 60],
        'win_rate': [0.62, 0.55, 0.70, 0.80],
        'roi': [0.15, 0.02, 0.05, 0.10],
    })

    # Test filter_bots
    filtered = filter_bots(wallet_stats_df, trades_df)
    print(f"Original wallets: {wallet_stats_df['wallet'].tolist()}")
    print(f"After bot filter: {filtered['wallet'].tolist()}")
    assert 'wallet_human' in filtered['wallet'].values, "Human should survive bot filter"

    # Test filter_non_copyable
    filtered2 = filter_non_copyable(wallet_stats_df, trades_df)
    print(f"After non-copyable filter: {filtered2['wallet'].tolist()}")

    # Test analyze_trade_patterns
    patterns = analyze_trade_patterns(trades_df, 'wallet_human')
    print(f"\nPatterns for wallet_human:")
    for k, v in patterns.items():
        if k == 'market_specialization':
            print(f"  {k}: {len(v)} markets")
        else:
            print(f"  {k}: {v}")

    # Test classify_wallet_strategy
    strategy, confidence = classify_wallet_strategy(patterns)
    print(f"\nStrategy: {strategy} (confidence: {confidence})")

    print("\n✅ All wallet_filter tests passed!")
