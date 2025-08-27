#!/usr/bin/env python3
"""
Simple test script for the Momentum Strategy API
"""

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_levels():
    """Test the levels endpoint with sample tickers"""
    print("\n🔍 Testing levels endpoint...")
    try:
        tickers = "RELIANCE.NS,TCS.NS,INFY.NS"
        response = requests.get(f"{BASE_URL}/levels?tickers={tickers}", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Levels endpoint passed: {len(data['levels'])} tickers processed")
            for level in data['levels']:
                print(f"   {level['Ticker']}: Close={level['Close']}, ATR={level['ATR14']}, Stop={level['Stop_2xATR']}")
            return True
        else:
            print(f"❌ Levels endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Levels endpoint error: {e}")
        return False

def test_run_simple():
    """Test the run endpoint with minimal parameters"""
    print("\n🔍 Testing run endpoint (simple)...")
    try:
        # Test with minimal parameters
        params = {
            "capital": 50000,
            "top_n": 5,
            "price_cap": 2000
        }
        
        print("   ⏳ This may take a few minutes for data download...")
        start_time = time.time()
        
        response = requests.get(f"{BASE_URL}/run", params=params, timeout=300)
        
        elapsed = time.time() - start_time
        print(f"   ⏱️  Request completed in {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Run endpoint passed:")
            print(f"   - Universe: {data['metadata']['total_universe']} tickers")
            print(f"   - Valid: {data['metadata']['valid_tickers']} tickers")
            print(f"   - Eligible: {data['metadata']['eligible_count']} tickers")
            print(f"   - Selected: {data['metadata']['top_selected']} tickers")
            
            if data['top']:
                print(f"   - Top stock: {data['top'][0]['Ticker']} (Score: {data['top'][0]['MomentumScore']:.2f})")
            
            return True
        else:
            print(f"❌ Run endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Run endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Momentum Strategy API Tests")
    print("=" * 50)
    
    # Check if server is running
    print("📡 Checking if API server is running...")
    if not test_health():
        print("\n❌ API server is not running or not accessible")
        print("   Please start the server with: python main.py")
        return
    
    # Test levels endpoint
    test_levels()
    
    # Test run endpoint
    test_run_simple()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
    print("\n💡 Tips:")
    print("   - Check the 'output' directory for generated CSV files")
    print("   - Monitor the console for detailed logs")
    print("   - Adjust config.yaml for different strategy parameters")

if __name__ == "__main__":
    main()