#!/usr/bin/env python3
"""Test script to verify /signals endpoint functionality"""

import requests
import json
from datetime import datetime

def test_signals_endpoint():
    """Test the /signals endpoint"""
    base_url = "http://localhost:8000"
    
    print(f"\n🔍 Testing /signals endpoint at {datetime.now()}")
    print("=" * 60)
    
    # Test /signals endpoint
    try:
        response = requests.get(f"{base_url}/signals")
        data = response.json()
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Success: {data.get('success', False)}")
        print(f"📊 Total Signals: {data.get('total_signals', 0)}")
        print(f"⚙️  Criteria: {json.dumps(data.get('criteria', {}), indent=2)}")
        
        signals = data.get('signals', [])
        if signals:
            print(f"\n🎯 Found {len(signals)} trading signals:")
            print("-" * 60)
            for i, signal in enumerate(signals[:5]):  # Show first 5
                print(f"\n{i+1}. {signal['coin_symbol']} - {signal['signal_type']}")
                print(f"   Models Agreed: {signal['models_agreed']}/{signal['total_models']}")
                print(f"   Avg Confidence: {signal['avg_confidence']}%")
                print(f"   Current Price: ${signal.get('current_price', 0):.4f}")
                print(f"   Consensus: Buy={signal.get('consensus_breakdown', {}).get('buy', 0)}, "
                      f"Sell={signal.get('consensus_breakdown', {}).get('sell', 0)}, "
                      f"Hold={signal.get('consensus_breakdown', {}).get('hold', 0)}")
        else:
            print("\n⚠️  No signals generated. Checking debug info...")
            
            # Check debug endpoint
            debug_response = requests.get(f"{base_url}/debug/signals")
            debug_data = debug_response.json()
            
            print(f"\n🔍 Debug Information:")
            print(f"   Active Coins: {debug_data.get('active_coins', 0)}")
            print(f"   Recent Predictions: {debug_data.get('recent_predictions_count', 0)}")
            print(f"   Test Signals: {len(debug_data.get('test_signals_generated', []))}")
            
            if debug_data.get('sample_predictions_by_coin'):
                print(f"\n   Sample Predictions by Coin:")
                for coin, preds in list(debug_data['sample_predictions_by_coin'].items())[:3]:
                    print(f"   - {coin}: {len(preds)} predictions")
    
    except Exception as e:
        print(f"\n❌ Error testing /signals: {e}")
        
    # Also test /debug/predictions
    try:
        print(f"\n\n🔍 Testing /debug/predictions endpoint")
        print("=" * 60)
        
        response = requests.get(f"{base_url}/debug/predictions")
        data = response.json()
        
        print(f"📊 Total Predictions: {data.get('total_predictions', 0)}")
        print(f"📈 Recent (24h): {data.get('recent_predictions_24h', 0)}")
        print(f"🪙 Coins with Predictions: {data.get('coins_with_predictions', 0)}")
        
        if data.get('sample_predictions'):
            print(f"\n   Recent Predictions:")
            for pred in data['sample_predictions']:
                print(f"   - {pred['coin']}: {pred['model']} -> {pred['prediction']} ({pred['confidence']}%)")
    
    except Exception as e:
        print(f"\n❌ Error testing /debug/predictions: {e}")

if __name__ == "__main__":
    test_signals_endpoint()