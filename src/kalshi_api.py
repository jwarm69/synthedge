"""
Kalshi API wrapper for fetching crypto market data.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Fee handling is configurable; set fee_per_contract to match your schedule.
DEFAULT_FEE_PER_CONTRACT = 0.0
DEFAULT_MAX_KELLY_FRACTION = 0.25

# Ticker mappings
KALSHI_TICKERS = {
    "BTC": "KXBTC",
    "ETH": "KXETH",
}


def get_crypto_events(coin: str = "BTC", limit: int = 10) -> list:
    """
    Fetch upcoming hourly crypto events from Kalshi.

    Args:
        coin: BTC or ETH
        limit: Number of events to fetch

    Returns:
        List of event dicts with ticker, title, strike_date
    """
    series_ticker = KALSHI_TICKERS.get(coin.upper(), "KXBTC")

    url = f"{BASE_URL}/events"
    params = {
        "series_ticker": series_ticker,
        "limit": limit,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])
    except Exception as e:
        print(f"Error fetching events: {e}")
        return []


def get_markets_for_event(event_ticker: str) -> list:
    """
    Fetch all markets (price ranges) for a specific event.

    Args:
        event_ticker: e.g., "KXBTC-26JAN1321"

    Returns:
        List of market dicts with strike prices, bids, asks
    """
    url = f"{BASE_URL}/markets"
    params = {"event_ticker": event_ticker}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("markets", [])
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def parse_market_structure(markets: list) -> dict:
    """
    Parse market data into structured format.

    Returns:
        {
            "above_strikes": [{"strike": 99750, "yes_bid": 0.05, "yes_ask": 0.08, ...}],
            "below_strikes": [...],
            "ranges": [{"floor": 94000, "cap": 94250, "yes_bid": ..., ...}],
        }
    """
    result = {
        "above_strikes": [],  # Price will be above X
        "below_strikes": [],  # Price will be below X
        "ranges": [],         # Price will be between X and Y
    }

    for m in markets:
        strike_type = m.get("strike_type", "")

        market_info = {
            "ticker": m.get("ticker"),
            "subtitle": m.get("subtitle"),
            "yes_bid": m.get("yes_bid", 0) / 100,  # Convert cents to dollars
            "yes_ask": m.get("yes_ask", 0) / 100,
            "no_bid": m.get("no_bid", 0) / 100,
            "no_ask": m.get("no_ask", 0) / 100,
            "last_price": m.get("last_price", 0) / 100,
            "volume": m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
        }

        if strike_type == "greater":
            market_info["floor_strike"] = m.get("floor_strike")
            result["above_strikes"].append(market_info)
        elif strike_type == "less":
            market_info["cap_strike"] = m.get("cap_strike")
            result["below_strikes"].append(market_info)
        elif strike_type == "between":
            market_info["floor_strike"] = m.get("floor_strike")
            market_info["cap_strike"] = m.get("cap_strike")
            result["ranges"].append(market_info)

    # Sort by strike price
    result["above_strikes"].sort(key=lambda x: x.get("floor_strike", 0))
    result["below_strikes"].sort(key=lambda x: x.get("cap_strike", 0), reverse=True)
    result["ranges"].sort(key=lambda x: x.get("floor_strike", 0))

    return result


def get_next_hourly_event(coin: str = "BTC") -> Optional[dict]:
    """
    Get the next upcoming hourly event for a coin.
    Backward-compatible alias for get_next_event(coin, "1h").

    Returns:
        Event dict or None
    """
    return get_next_event(coin, timeframe="1h")


def get_next_event(coin: str = "BTC", timeframe: str = "1h") -> Optional[dict]:
    """
    Get the next upcoming event for a coin and timeframe.

    Args:
        coin: BTC or ETH
        timeframe: "1h" (hourly) or "15min" (15-minute)

    Returns:
        Event dict or None. Returns None gracefully if the timeframe
        is not available for this coin.
    """
    events = get_crypto_events(coin, limit=20)
    now = datetime.utcnow()

    for event in events:
        strike_date_str = event.get("strike_date", "")
        try:
            strike_date = datetime.fromisoformat(strike_date_str.replace("Z", "+00:00"))
            strike_date = strike_date.replace(tzinfo=None)

            if strike_date <= now:
                continue

            # Filter by timeframe
            if timeframe in ("15min", "15m"):
                # 15min events: look for settlement window <= 20 minutes
                # or title/subtitle containing "15" minute pattern
                title = (event.get("title", "") + event.get("sub_title", "")).lower()
                if "15 min" in title or "15min" in title or "15-min" in title:
                    return event
                # Also check if settlement window is roughly 15 minutes
                open_date_str = event.get("open_date", "")
                if open_date_str:
                    try:
                        open_date = datetime.fromisoformat(open_date_str.replace("Z", "+00:00"))
                        open_date = open_date.replace(tzinfo=None)
                        window_minutes = (strike_date - open_date).total_seconds() / 60
                        if 10 <= window_minutes <= 20:
                            return event
                    except Exception:
                        pass
            else:
                # 1h events: settlement window > 30 minutes or default hourly
                title = (event.get("title", "") + event.get("sub_title", "")).lower()
                # Skip if this looks like a 15-min event
                if "15 min" in title or "15min" in title or "15-min" in title:
                    continue
                return event
        except Exception:
            continue

    return None


def get_market_snapshot(coin: str = "BTC", timeframe: str = "1h") -> Optional[dict]:
    """
    Get a complete market snapshot for a coin and timeframe.

    Returns:
        {
            "event": event dict,
            "structure": parsed market structure,
            "settlement_time": datetime,
            "time_to_settlement_min": float,
            "liquidity": {"total_volume": int, "total_oi": int, "active_markets": int},
            "timeframe": str,
        }
        or None if no event available
    """
    event = get_next_event(coin, timeframe=timeframe)
    if not event:
        return None

    markets = get_markets_for_event(event.get("event_ticker"))
    if not markets:
        return None

    structure = parse_market_structure(markets)

    # Parse settlement time
    strike_date_str = event.get("strike_date", "")
    settlement_time = None
    time_to_settlement = 0
    try:
        settlement_time = datetime.fromisoformat(strike_date_str.replace("Z", "+00:00"))
        settlement_time = settlement_time.replace(tzinfo=None)
        time_to_settlement = (settlement_time - datetime.utcnow()).total_seconds() / 60
    except Exception:
        pass

    # Calculate liquidity
    total_volume = sum(m.get("volume", 0) for m in markets)
    total_oi = sum(m.get("open_interest", 0) for m in markets)
    active_markets = sum(1 for m in markets if m.get("volume", 0) > 0 or m.get("open_interest", 0) > 0)

    return {
        "event": event,
        "structure": structure,
        "settlement_time": settlement_time,
        "time_to_settlement_min": max(0, time_to_settlement),
        "liquidity": {
            "total_volume": total_volume,
            "total_oi": total_oi,
            "active_markets": active_markets,
        },
        "timeframe": timeframe,
    }


def display_event_summary(coin: str = "BTC"):
    """
    Print a summary of the next hourly event and its markets.
    """
    event = get_next_hourly_event(coin)

    if not event:
        print(f"No upcoming {coin} events found")
        return

    print(f"\n{'='*60}")
    print(f"KALSHI {coin} HOURLY CONTRACT")
    print(f"{'='*60}")
    print(f"Event: {event.get('event_ticker')}")
    print(f"Title: {event.get('title')}")
    print(f"Settlement: {event.get('strike_date')}")

    markets = get_markets_for_event(event.get("event_ticker"))
    structure = parse_market_structure(markets)

    print(f"\nTotal markets: {len(markets)}")
    print(f"  - Above strikes: {len(structure['above_strikes'])}")
    print(f"  - Below strikes: {len(structure['below_strikes'])}")
    print(f"  - Range buckets: {len(structure['ranges'])}")

    # Show some example ranges with activity
    active_ranges = [r for r in structure["ranges"] if r["volume"] > 0 or r["open_interest"] > 0]

    if active_ranges:
        print(f"\nActive ranges (with volume/OI):")
        for r in active_ranges[:5]:
            print(f"  {r['subtitle']}: last=${r['last_price']:.2f}, vol={r['volume']}")
    else:
        # Show ranges near current price (middle of range list)
        mid_idx = len(structure["ranges"]) // 2
        print(f"\nSample ranges (around middle of distribution):")
        for r in structure["ranges"][max(0, mid_idx-3):mid_idx+3]:
            print(f"  {r['subtitle']}: bid=${r['yes_bid']:.2f}, ask=${r['yes_ask']:.2f}")

    return event, structure


def get_recommended_contracts(current_price: float, prediction: str, structure: dict) -> list:
    """
    Based on model prediction, recommend which contracts to consider.

    Args:
        current_price: Current BTC price
        prediction: "UP" or "DOWN"
        structure: Output from parse_market_structure()

    Returns:
        List of recommended contracts with expected value analysis
    """
    recommendations = []

    if prediction.upper() == "UP":
        # Look for "above current price" contracts
        for strike in structure["above_strikes"]:
            if strike.get("floor_strike", 0) > current_price:
                recommendations.append({
                    "type": "above",
                    "strike": strike["floor_strike"],
                    "ticker": strike["ticker"],
                    "yes_ask": strike["yes_ask"],
                    "rationale": f"If UP, price should exceed ${strike['floor_strike']:,.0f}"
                })
                break  # Just recommend the closest above-current strike

        # Also recommend ranges just above current price
        for r in structure["ranges"]:
            floor = r.get("floor_strike", 0)
            cap = r.get("cap_strike", 0)
            if floor <= current_price < cap:
                recommendations.append({
                    "type": "range",
                    "range": r["subtitle"],
                    "ticker": r["ticker"],
                    "yes_ask": r["yes_ask"],
                    "rationale": f"Current price is in this range"
                })
            elif floor > current_price and len([rec for rec in recommendations if rec["type"] == "range"]) < 3:
                recommendations.append({
                    "type": "range",
                    "range": r["subtitle"],
                    "ticker": r["ticker"],
                    "yes_ask": r["yes_ask"],
                    "rationale": f"Price target if UP"
                })

    elif prediction.upper() == "DOWN":
        # Look for "below current price" contracts
        for strike in structure["below_strikes"]:
            if strike.get("cap_strike", float("inf")) < current_price:
                recommendations.append({
                    "type": "below",
                    "strike": strike["cap_strike"],
                    "ticker": strike["ticker"],
                    "yes_ask": strike["yes_ask"],
                    "rationale": f"If DOWN, price should fall below ${strike['cap_strike']:,.0f}"
                })
                break

    return recommendations


def expected_value_yes(
    prob_yes: float,
    yes_ask: float,
    fee_per_contract: float = DEFAULT_FEE_PER_CONTRACT
) -> float:
    """
    Expected value per contract for a YES position.

    Args:
        prob_yes: Model probability the contract resolves YES
        yes_ask: Market ask price (0-1)
        fee_per_contract: Flat fee per contract (configure to your schedule)
    """
    cost = yes_ask + fee_per_contract
    if cost <= 0:
        return 0.0

    profit_win = 1.0 - cost
    profit_loss = -cost
    return prob_yes * profit_win + (1 - prob_yes) * profit_loss


def kelly_fraction(
    prob_yes: float,
    yes_ask: float,
    fee_per_contract: float = DEFAULT_FEE_PER_CONTRACT,
    max_fraction: float = DEFAULT_MAX_KELLY_FRACTION
) -> float:
    """
    Kelly-optimal bankroll fraction for a YES position.
    """
    cost = yes_ask + fee_per_contract
    if cost <= 0 or cost >= 1:
        return 0.0

    b = (1.0 - cost) / cost
    if b <= 0:
        return 0.0

    fraction = (prob_yes * (b + 1.0) - 1.0) / b
    return max(0.0, min(fraction, max_fraction))


def position_size(
    bankroll: float,
    prob_yes: float,
    yes_ask: float,
    fee_per_contract: float = DEFAULT_FEE_PER_CONTRACT,
    max_fraction: float = DEFAULT_MAX_KELLY_FRACTION
) -> int:
    """
    Suggested contract count based on Kelly fraction.
    """
    if bankroll <= 0:
        return 0

    cost = yes_ask + fee_per_contract
    if cost <= 0:
        return 0

    fraction = kelly_fraction(
        prob_yes,
        yes_ask,
        fee_per_contract=fee_per_contract,
        max_fraction=max_fraction
    )
    stake = bankroll * fraction
    return int(stake // cost)


if __name__ == "__main__":
    print("Fetching Kalshi BTC hourly contract data...")
    display_event_summary("BTC")

    print("\n" + "="*60)
    print("Fetching Kalshi ETH hourly contract data...")
    display_event_summary("ETH")
