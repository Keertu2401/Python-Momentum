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
    print("Testing health endpoint...")
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
    print("\nTesting levels endpoint...")
    try:
        tickers = "RELIANCE.NS,TCS.NS,INFY.NS"
        response = requests.get(f"{BASE_URL}/levels?tickers={tickers}", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Levels endpoint passed: {len(data['levels'])} tickers processed")
            for level in data['levels']:
                print(f"  {level['Ticker']}: Close={level['Close']}, Stop={level['Stop_2xATR']}")
            return True
        else:
            print(f"❌ Levels endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Levels endpoint error: {e}")
        return False

def test_run_strategy():
    """Test the main strategy endpoint"""
    print("\nTesting run strategy endpoint...")
    try:
        # Test with smaller parameters for faster execution
        params = {
            "capital": 50000,
            "top_n": 5,
            "price_cap": 2000
        }
        
        print("This may take a few minutes for data download...")
        response = requests.get(f"{BASE_URL}/run", params=params, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Strategy endpoint passed!")
            print(f"  Universe: {data['metadata']['total_universe']} tickers")
            print(f"  Valid: {data['metadata']['valid_tickers']} tickers")
            print(f"  Eligible: {data['metadata']['eligible_count']} tickers")
            print(f"  Top selected: {data['metadata']['top_selected']} tickers")
            
            if data['top']:
                print("\nTop selections:")
                for stock in data['top'][:3]:  # Show first 3
                    print(f"  {stock['Ticker']}: Score={stock['MomentumScore']:.2f}, Price={stock['Price']:.2f}")
            
            return True
        else:
            print(f"❌ Strategy endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Strategy endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Momentum Strategy API")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Levels Endpoint", test_levels),
        ("Strategy Endpoint", test_run_strategy)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)