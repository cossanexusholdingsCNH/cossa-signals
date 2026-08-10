"""
One-off diagnostic — NOT part of the main pipeline.

paper_trade.py's live "ticks" subscription rejects RDBULL/RDBEAR as
invalid symbols, even though ticks_history (used for all historical data
fetching/backtesting) accepts them fine. This narrows down why:

  1. Control check — subscribe to R_75 (a continuous index known to work
     broadly) via the same raw "ticks" request. If THIS also fails, the
     problem is with live tick subscription in general on this
     connection/app_id, not specific to RDBULL/RDBEAR. If it succeeds,
     the problem is specific to Daily Reset Indices.
  2. Re-check active_symbols (this returned empty when tried months ago,
     during the original --all discovery). If it works now, search the
     result for anything with "bull" or "bear" in the name — the live
     symbol code may simply be different from "RDBULL"/"RDBEAR", which
     were only ever confirmed against ticks_history, not live ticks.

Usage: python check_live_symbols.py
"""
import asyncio
import json
import os
from pathlib import Path

import websockets
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / "config" / ".env"
load_dotenv(ENV_PATH)
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"


async def try_tick_subscribe(symbol: str, listen_seconds: float = 3.0):
    async with websockets.connect(DERIV_WS_URL) as ws:
        await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
        try:
            response_raw = await asyncio.wait_for(ws.recv(), timeout=listen_seconds)
            response = json.loads(response_raw)
            if "error" in response:
                print(f"  [FAIL] {symbol}: {response['error'].get('message')}")
                return False
            elif "tick" in response:
                print(f"  [OK]   {symbol}: live tick received, quote={response['tick'].get('quote')}")
                return True
            else:
                print(f"  [???]  {symbol}: unexpected response shape: {list(response.keys())}")
                return False
        except asyncio.TimeoutError:
            print(f"  [FAIL] {symbol}: no response within {listen_seconds}s (subscribed but silent, or hung)")
            return False


async def check_active_symbols():
    async with websockets.connect(DERIV_WS_URL) as ws:
        await ws.send(json.dumps({
            "active_symbols": "brief",
            "product_type": "basic",
        }))
        response_raw = await ws.recv()
        response = json.loads(response_raw)
        if "error" in response:
            print(f"  active_symbols FAILED: {response['error'].get('message')}")
            return
        symbols = response.get("active_symbols", [])
        print(f"  active_symbols returned {len(symbols)} symbols")
        matches = [
            s for s in symbols
            if "bull" in s.get("display_name", "").lower()
            or "bear" in s.get("display_name", "").lower()
            or "bull" in s.get("symbol", "").lower()
            or "bear" in s.get("symbol", "").lower()
        ]
        if matches:
            print("  Bull/Bear-related symbols found:")
            for m in matches:
                print(f"    symbol_code={m.get('symbol')!r}  display_name={m.get('display_name')!r}  "
                      f"market={m.get('market')!r}  exchange_is_open={m.get('exchange_is_open')}")
        else:
            print("  No symbol with 'bull' or 'bear' in its name or display_name was found.")


async def main():
    print("=== Control: live tick subscribe on R_75 (known-working continuous index) ===")
    await try_tick_subscribe("R_75")

    print("\n=== Live tick subscribe on RDBULL / RDBEAR ===")
    await try_tick_subscribe("RDBULL")
    await try_tick_subscribe("RDBEAR")

    print("\n=== active_symbols (searching for the real live Bull/Bear Market symbol codes) ===")
    await check_active_symbols()


if __name__ == "__main__":
    asyncio.run(main())
