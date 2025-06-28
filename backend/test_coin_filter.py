#!/usr/bin/env python3
"""
Simple test script to verify coin filtering is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bybit_client import BybitClient
from allowed_coins import ALLOWED_COINS

def test_coin_filtering():
    """Test that bybit client only returns allowed coins"""
    print("Testing coin filtering...")
    print(f"Number of allowed coins: {len(ALLOWED_COINS)}")
    print(f"First 10 allowed coins: {ALLOWED_COINS[:10]}")
    
    try:
        bybit_client = BybitClient()
        filtered_symbols = bybit_client.get_derivatives_symbols()
        
        print(f"\nFiltered symbols returned: {len(filtered_symbols)}")
        
        if filtered_symbols:
            print(f"First 5 filtered symbols: {[s['symbol'] for s in filtered_symbols[:5]]}")
            
            # Check that all returned symbols are in allowed list
            all_allowed = True
            for symbol_data in filtered_symbols:
                if symbol_data['symbol'] not in ALLOWED_COINS:
                    print(f"ERROR: {symbol_data['symbol']} is not in allowed list!")
                    all_allowed = False
            
            if all_allowed:
                print("✅ SUCCESS: All returned symbols are in the allowed list")
            else:
                print("❌ FAILED: Some symbols are not in the allowed list")
        else:
            print("❌ No symbols returned - check API credentials")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_coin_filtering()