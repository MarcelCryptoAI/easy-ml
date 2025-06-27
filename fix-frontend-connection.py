#!/usr/bin/env python3

import requests
import json
import time

def test_connection():
    base_url = "https://easy-ml-production.up.railway.app"
    
    print("🔍 Testing backend connection...")
    
    for attempt in range(5):
        try:
            response = requests.get(f"{base_url}/health", timeout=20)
            if response.status_code == 200:
                print(f"✅ Backend is UP! Status: {response.status_code}")
                data = response.json()
                print(f"   Database coins: {data.get('database_coins', 'N/A')}")
                print(f"   Bybit connected: {data.get('bybit_connected', 'N/A')}")
                return True
            else:
                print(f"⚠️ Backend returned {response.status_code}")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}: {e}")
            
        if attempt < 4:
            print(f"   Retrying in 10 seconds...")
            time.sleep(10)
    
    print("💥 Backend is not responding")
    return False

def test_ml_data():
    base_url = "https://easy-ml-production.up.railway.app"
    
    print("\n🤖 Testing ML training data...")
    
    # Test some coins that should have training data
    test_coins = ["1000APUUSDT", "10000WENUSDT", "1000BONKPERP"]
    
    for coin in test_coins:
        try:
            response = requests.get(f"{base_url}/predictions/{coin}", timeout=15)
            if response.status_code == 200:
                predictions = response.json()
                print(f"✅ {coin}: {len(predictions)} predictions")
                for pred in predictions:
                    print(f"   - {pred['model_type']}: {pred['prediction']} ({pred['confidence']:.1f}%)")
            else:
                print(f"❌ {coin}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {coin}: {e}")

if __name__ == "__main__":
    if test_connection():
        test_ml_data()
    else:
        print("\n🛠️ Backend needs to be fixed first.")
        print("Check Railway dashboard: https://railway.app/dashboard")
        print("Look for error logs in the backend service.")