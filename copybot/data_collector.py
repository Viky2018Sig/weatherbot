"""
Polymarket Data Collector Module

Fetches trade data, market info, and order book data from:
- Polymarket CLOB API (https://clob.polymarket.com)
- Polymarket Gamma API (https://gamma-api.polymarket.com)

The CLOB /trades endpoint requires API key authentication.
This module gracefully falls back to price history + order book data
when API keys are not configured.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"

DATA_DIR = Path("/root/weatherbot/copybot/data")
TRADES_FILE = DATA_DIR / "trades.json"

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
RATE_LIMIT_INTERVAL = 0.5  # 2 requests/second → 0.5s between requests

# Optional: set POLYMARKET_API_KEY env var for authenticated CLOB endpoints
API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("polymarket_collector")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_last_request_time = 0.0


def _rate_limit():
    """Enforce max 2 requests per second."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    """HTTP GET with retry logic and rate limiting."""
    _rate_limit()
    hdrs = {"Accept": "application/json"}
    if API_KEY:
        hdrs["Authorization"] = f"Bearer {API_KEY}"
    if headers:
        hdrs.update(headers)

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=15)
            if resp.status_code == 429:
                wait = RETRY_DELAY * attempt
                logger.warning("Rate limited (429). Waiting %.1fs …", wait)
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 1. fetch_recent_trades  (data-api → CLOB auth → price-history fallback)
# ---------------------------------------------------------------------------
def fetch_recent_trades(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch recent trades with real wallet addresses.

    Strategy (in order):
      1. Polymarket Data API  – GET /trades  (public, has proxyWallet)
      2. CLOB authenticated   – GET /trades  (needs API key)
      3. Price-history fallback (no wallet data – last resort)

    Returns list of trade dicts with keys:
        wallet_address, market_id, side, price, size, timestamp, outcome,
        transaction_hash, condition_id, slug, pseudonym
    """
    # --- 1. Data API (public, returns proxyWallet) ---
    trades = _fetch_from_data_api(limit=limit)
    if trades:
        return trades

    # --- 2. Try authenticated CLOB endpoint ---
    try:
        resp = _get(f"{CLOB_BASE_URL}/trades", params={"limit": min(limit, 500)})
        if resp.status_code == 200:
            raw = resp.json()
            items = raw if isinstance(raw, list) else raw.get("data", raw.get("trades", []))
            trades = []
            for t in items[:limit]:
                trades.append({
                    "wallet_address": t.get("maker", "") or t.get("taker", ""),
                    "market_id": t.get("market") or t.get("asset_id", ""),
                    "maker": t.get("maker", ""),
                    "taker": t.get("taker", ""),
                    "side": t.get("side", "").lower(),
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "timestamp": t.get("timestamp") or t.get("match_time", ""),
                    "outcome": t.get("outcome", t.get("asset_id", "")),
                    "transaction_hash": t.get("transactionHash", ""),
                    "condition_id": "",
                    "slug": "",
                    "pseudonym": "",
                })
            logger.info("Fetched %d trades from CLOB /trades", len(trades))
            return trades
        elif resp.status_code == 401:
            logger.info("CLOB /trades requires auth – falling back to price history.")
        else:
            logger.warning("CLOB /trades returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("CLOB /trades request failed: %s", exc)

    # --- 3. Fallback: build synthetic trade records from price history ---
    trades = []
    try:
        markets = fetch_active_markets(limit=20)
    except Exception:
        logger.error("Could not fetch active markets for fallback trades.")
        return trades

    for mkt in markets:
        if len(trades) >= limit:
            break
        token_ids = _extract_token_ids(mkt)
        if not token_ids:
            continue
        token_id = token_ids[0]
        try:
            resp = _get(
                f"{CLOB_BASE_URL}/prices-history",
                params={"market": token_id, "interval": "1d", "fidelity": 60},
            )
            if resp.status_code != 200:
                continue
            history = resp.json().get("history", [])
            for pt in history:
                trades.append({
                    "wallet_address": "",
                    "market_id": str(mkt.get("id", "")),
                    "maker": "",
                    "taker": "",
                    "side": "buy",
                    "price": float(pt.get("p", 0)),
                    "size": 0.0,
                    "timestamp": datetime.fromtimestamp(
                        int(pt.get("t", 0)), tz=timezone.utc
                    ).isoformat(),
                    "outcome": mkt.get("question", ""),
                    "transaction_hash": "",
                    "condition_id": "",
                    "slug": "",
                    "pseudonym": "",
                })
                if len(trades) >= limit:
                    break
        except Exception as exc:
            logger.debug("Price history fetch failed for token %s: %s", token_id, exc)

    logger.info("Built %d synthetic trade records from price history (fallback).", len(trades))
    return trades[:limit]


def _fetch_from_data_api(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch trades from the Polymarket Data API (public endpoint).
    Returns trades with real wallet addresses (proxyWallet field).

    Endpoint: GET https://data-api.polymarket.com/trades?limit=N
    Max per request appears to be ~100, so we paginate if needed.
    """
    all_trades: List[Dict[str, Any]] = []
    batch_size = min(limit, 100)
    max_pages = (limit + batch_size - 1) // batch_size

    last_timestamp = None
    for page in range(max_pages):
        if len(all_trades) >= limit:
            break

        params: dict = {"limit": batch_size}
        if last_timestamp is not None:
            # Paginate by requesting trades before the last seen timestamp
            params["before"] = last_timestamp

        try:
            resp = _get(f"{DATA_API_URL}/trades", params=params)
            if resp.status_code != 200:
                logger.warning("Data API /trades returned %d: %s", resp.status_code, resp.text[:200])
                break

            raw = resp.json()
            if not raw:
                break

            for t in raw:
                ts = t.get("timestamp", 0)
                # Convert unix timestamp to ISO format
                try:
                    ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                except (ValueError, TypeError, OSError):
                    ts_iso = str(ts)

                all_trades.append({
                    "wallet_address": t.get("proxyWallet", ""),
                    "market_id": t.get("conditionId", ""),
                    "maker": "",
                    "taker": "",
                    "side": t.get("side", "").lower(),
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "timestamp": ts_iso,
                    "outcome": t.get("outcome", ""),
                    "transaction_hash": t.get("transactionHash", ""),
                    "condition_id": t.get("conditionId", ""),
                    "slug": t.get("slug", ""),
                    "pseudonym": t.get("pseudonym", ""),
                    "asset_id": t.get("asset", ""),
                    "title": t.get("title", ""),
                })

            # Update pagination cursor
            last_ts = raw[-1].get("timestamp")
            if last_ts == last_timestamp:
                break  # No progress
            last_timestamp = last_ts

        except Exception as exc:
            logger.warning("Data API /trades request failed (page %d): %s", page, exc)
            break

    logger.info("Fetched %d trades with wallet addresses from Data API.", len(all_trades))
    return all_trades[:limit]


# ---------------------------------------------------------------------------
# 2. fetch_market_info
# ---------------------------------------------------------------------------
def fetch_market_info(market_id: str) -> Dict[str, Any]:
    """
    Fetch market details from Gamma API.

    Endpoint: GET /markets/{id}

    Returns dict with: question, outcome_prices, volume, closed, resolved, end_date.
    """
    resp = _get(f"{GAMMA_BASE_URL}/markets/{market_id}")
    if resp.status_code != 200:
        logger.error("Gamma /markets/%s returned %d", market_id, resp.status_code)
        return {}

    data = resp.json()

    # outcome_prices may be a JSON-encoded string or a list
    outcome_prices = data.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "question": data.get("question", ""),
        "outcome_prices": outcome_prices,
        "volume": float(data.get("volume", 0) or 0),
        "closed": bool(data.get("closed", False)),
        "resolved": bool(data.get("resolved", False)),
        "end_date": data.get("endDate", ""),
        # Preserve extra fields useful downstream
        "condition_id": data.get("conditionId", ""),
        "clob_token_ids": _extract_token_ids(data),
        "outcomes": data.get("outcomes", []),
        "liquidity": float(data.get("liquidity", 0) or 0),
    }


# ---------------------------------------------------------------------------
# 3. fetch_active_markets
# ---------------------------------------------------------------------------
def fetch_active_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch currently active/open markets from Gamma API.

    Endpoint: GET /markets?active=true&closed=false&limit={limit}
    """
    resp = _get(
        f"{GAMMA_BASE_URL}/markets",
        params={"active": "true", "closed": "false", "limit": limit},
    )
    if resp.status_code != 200:
        logger.error("Gamma /markets returned %d", resp.status_code)
        return []

    data = resp.json()
    markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
    logger.info("Fetched %d active markets from Gamma API.", len(markets))
    return markets[:limit]


# ---------------------------------------------------------------------------
# 4. fetch_trades_for_market
# ---------------------------------------------------------------------------
def fetch_trades_for_market(token_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch trades for a specific market token.

    Tries Data API with conditionId first, then authenticated CLOB,
    then falls back to price history.

    Returns list of trade dicts.
    """
    # --- Try Data API (has wallet addresses) ---
    try:
        resp = _get(
            f"{DATA_API_URL}/trades",
            params={"asset_id": token_id, "limit": min(limit, 100)},
        )
        if resp.status_code == 200:
            raw = resp.json()
            if raw:
                trades = []
                for t in raw[:limit]:
                    ts = t.get("timestamp", 0)
                    try:
                        ts_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError, OSError):
                        ts_iso = str(ts)
                    trades.append({
                        "wallet_address": t.get("proxyWallet", ""),
                        "market_id": t.get("conditionId", ""),
                        "maker": "",
                        "taker": "",
                        "side": t.get("side", "").lower(),
                        "price": float(t.get("price", 0)),
                        "size": float(t.get("size", 0)),
                        "timestamp": ts_iso,
                        "outcome": t.get("outcome", ""),
                        "transaction_hash": t.get("transactionHash", ""),
                        "condition_id": t.get("conditionId", ""),
                        "slug": t.get("slug", ""),
                        "pseudonym": t.get("pseudonym", ""),
                        "asset_id": t.get("asset", ""),
                        "title": t.get("title", ""),
                    })
                logger.info("Fetched %d trades for token %s from Data API", len(trades), token_id[:16])
                return trades
    except Exception as exc:
        logger.debug("Data API trade fetch failed for token %s: %s", token_id[:16], exc)

    # --- Try authenticated CLOB endpoint ---
    try:
        resp = _get(
            f"{CLOB_BASE_URL}/trades",
            params={"asset_id": token_id, "limit": min(limit, 500)},
        )
        if resp.status_code == 200:
            raw = resp.json()
            items = raw if isinstance(raw, list) else raw.get("data", raw.get("trades", []))
            trades = []
            for t in items[:limit]:
                trades.append({
                    "wallet_address": t.get("maker", "") or t.get("taker", ""),
                    "market_id": t.get("market", ""),
                    "maker": t.get("maker", ""),
                    "taker": t.get("taker", ""),
                    "side": t.get("side", "").lower(),
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "timestamp": t.get("timestamp") or t.get("match_time", ""),
                    "outcome": t.get("outcome", ""),
                    "transaction_hash": t.get("transactionHash", ""),
                    "condition_id": "",
                    "slug": "",
                    "pseudonym": "",
                })
            logger.info("Fetched %d trades for token %s", len(trades), token_id[:16])
            return trades
    except Exception:
        pass

    # --- Fallback: price history ---
    trades: List[Dict[str, Any]] = []
    try:
        resp = _get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 60},
        )
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            for pt in history[:limit]:
                trades.append({
                    "wallet_address": "",
                    "market_id": token_id,
                    "maker": "",
                    "taker": "",
                    "side": "buy",
                    "price": float(pt.get("p", 0)),
                    "size": 0.0,
                    "timestamp": datetime.fromtimestamp(
                        int(pt.get("t", 0)), tz=timezone.utc
                    ).isoformat(),
                    "outcome": "",
                    "transaction_hash": "",
                    "condition_id": "",
                    "slug": "",
                    "pseudonym": "",
                })
            logger.info(
                "Built %d price-history records for token %s (fallback).",
                len(trades), token_id[:16],
            )
    except Exception as exc:
        logger.warning("Price history fallback failed for %s: %s", token_id[:16], exc)

    return trades


# ---------------------------------------------------------------------------
# 5. fetch_market_resolution
# ---------------------------------------------------------------------------
def fetch_market_resolution(condition_id: str) -> Dict[str, Any]:
    """
    Check if a market has resolved and what the outcome was.

    Uses Gamma API to look up the market by condition_id.

    Returns dict with: resolved, closed, outcome, outcome_prices.
    """
    # Gamma API lets us search by slug or id; condition_id is in market detail.
    # We first try to find the market via the search-like endpoint.
    resp = _get(
        f"{GAMMA_BASE_URL}/markets",
        params={"condition_id": condition_id, "limit": 1},
    )
    if resp.status_code != 200:
        logger.error("Gamma resolution lookup returned %d", resp.status_code)
        return {"resolved": False, "closed": False, "outcome": None, "outcome_prices": []}

    data = resp.json()
    markets = data if isinstance(data, list) else data.get("data", [])

    if not markets:
        logger.warning("No market found for condition_id %s", condition_id[:16])
        return {"resolved": False, "closed": False, "outcome": None, "outcome_prices": []}

    mkt = markets[0]

    outcome_prices = mkt.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            pass

    outcomes = mkt.get("outcomes", [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, TypeError):
            pass

    # Determine winning outcome (price ~1.0 after resolution)
    winning_outcome = None
    if mkt.get("resolved") or mkt.get("closed"):
        try:
            prices = [float(p) for p in outcome_prices]
            max_idx = prices.index(max(prices))
            if outcomes and max_idx < len(outcomes):
                winning_outcome = outcomes[max_idx]
        except (ValueError, IndexError):
            pass

    return {
        "resolved": bool(mkt.get("resolved", False)),
        "closed": bool(mkt.get("closed", False)),
        "outcome": winning_outcome,
        "outcome_prices": outcome_prices,
        "question": mkt.get("question", ""),
    }


# ---------------------------------------------------------------------------
# 6. build_trade_dataset
# ---------------------------------------------------------------------------
def build_trade_dataset(n_markets: int = 50) -> pd.DataFrame:
    """
    Fetch trades across multiple active markets, enrich with market info,
    and save to /root/weatherbot/copybot/data/trades.json.

    Primary: Data API bulk trades (has wallet addresses).
    Fallback: per-market trade fetching via fetch_trades_for_market().

    Returns a pandas DataFrame of all collected records.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Primary: bulk fetch from Data API (has wallet addresses) ---
    logger.info("Attempting bulk trade fetch from Data API …")
    all_records = _fetch_from_data_api(limit=max(n_markets * 20, 2000))

    if all_records:
        wallet_count = sum(1 for r in all_records if r.get("wallet_address"))
        logger.info(
            "Data API returned %d trades (%d with wallet addresses).",
            len(all_records), wallet_count,
        )
    else:
        # --- Fallback: per-market fetching ---
        logger.info("Data API returned nothing. Falling back to per-market fetching.")
        logger.info("Fetching up to %d active markets …", n_markets)
        markets = fetch_active_markets(limit=n_markets)
        if not markets:
            logger.error("No active markets returned. Aborting.")
            return pd.DataFrame()

        all_records = []
        for idx, mkt in enumerate(markets):
            market_id = str(mkt.get("id", ""))
            question = mkt.get("question", "unknown")
            logger.info("[%d/%d] Processing market: %s", idx + 1, len(markets), question[:80])

            # Get enriched market info
            try:
                info = fetch_market_info(market_id)
            except Exception as exc:
                logger.warning("Could not fetch info for market %s: %s", market_id, exc)
                info = {}

            # Get token IDs for this market
            token_ids = _extract_token_ids(mkt)
            if not token_ids:
                logger.debug("No token IDs for market %s – skipping trades.", market_id)
                continue

            # Fetch trades for first token
            try:
                trades = fetch_trades_for_market(token_ids[0], limit=100)
            except Exception as exc:
                logger.warning("Trade fetch failed for market %s: %s", market_id, exc)
                trades = []

            # Enrich each trade record with market metadata
            for t in trades:
                t["market_id"] = market_id
                t["question"] = info.get("question", question)
                t["market_volume"] = info.get("volume", 0)
                t["market_liquidity"] = info.get("liquidity", 0)
                t["market_closed"] = info.get("closed", False)
                t["market_resolved"] = info.get("resolved", False)
                t["end_date"] = info.get("end_date", "")
                t["outcome_prices"] = json.dumps(info.get("outcome_prices", []))
                all_records.append(t)

    # Build DataFrame
    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No records collected.")
        return df

    # Save to JSON
    df.to_json(str(TRADES_FILE), orient="records", indent=2)
    logger.info("Saved %d records to %s", len(df), TRADES_FILE)

    return df


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _extract_token_ids(market_data: dict) -> List[str]:
    """Extract CLOB token IDs from a Gamma market object."""
    raw = market_data.get("clobTokenIds", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = [raw] if raw else []
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Polymarket Data Collector")
    print("=" * 60)

    df = build_trade_dataset(n_markets=50)

    if df.empty:
        print("\nNo data collected.")
    else:
        print(f"\n{'─' * 60}")
        print(f"Total records collected : {len(df):,}")
        print(f"Unique markets          : {df['market_id'].nunique()}")
        if "price" in df.columns:
            print(f"Price range             : {df['price'].min():.3f} – {df['price'].max():.3f}")
            print(f"Mean price              : {df['price'].mean():.3f}")
        if "market_volume" in df.columns:
            print(f"Total market volume     : ${df.groupby('market_id')['market_volume'].first().sum():,.0f}")
        if "timestamp" in df.columns:
            print(f"Time range              : {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"Output file             : {TRADES_FILE}")
        print(f"{'─' * 60}")
