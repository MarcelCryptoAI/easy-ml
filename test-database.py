#!/usr/bin/env python3

import requests
import json

def test_api():
    base_url = "https://easy-ml-production.up.railway.app"
    
    print("🔍 Testing API endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   - Database coins: {data.get('database_coins', 'N/A')}")
            print(f"   - Bybit connected: {data.get('bybit_connected', 'N/A')}")
            print(f"   - Account balance: {data.get('account_balance', 'N/A')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test coins
    try:
        response = requests.get(f"{base_url}/coins", timeout=10)
        print(f"✅ Coins: {response.status_code}")
        if response.status_code == 200:
            coins = response.json()
            print(f"   - Total coins available: {len(coins)}")
            if coins:
                print(f"   - First few coins: {[c['symbol'] for c in coins[:5]]}")
    except Exception as e:
        print(f"❌ Coins check failed: {e}")
    
    # Test predictions for BTCUSDT
    try:
        response = requests.get(f"{base_url}/predictions/BTCUSDT", timeout=10)
        print(f"✅ BTCUSDT Predictions: {response.status_code}")
        if response.status_code == 200:
            predictions = response.json()
            print(f"   - Predictions found: {len(predictions)}")
            for pred in predictions:
                print(f"   - {pred['model_type']}: {pred['prediction']} ({pred['confidence']:.1f}%)")
        else:
            print(f"   - No predictions yet for BTCUSDT")
    except Exception as e:
        print(f"❌ Predictions check failed: {e}")
    
    # Test training session
    try:
        response = requests.get(f"{base_url}/training/session", timeout=10)
        print(f"✅ Training Session: {response.status_code}")
        if response.status_code == 200:
            session = response.json()
            print(f"   - Current coin: {session.get('current_coin', 'N/A')}")
            print(f"   - Completed items: {session.get('completed_items', 0)}/{session.get('total_queue_items', 0)}")
    except Exception as e:
        print(f"❌ Training session check failed: {e}")

if __name__ == "__main__":
    test_api()