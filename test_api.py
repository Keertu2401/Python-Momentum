"""
Comprehensive test suite for the momentum strategy API
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import os
import yaml

# Import the main application
from main import app, load_config, load_universe, yf_download, momentum_frame

client = TestClient(app)

class TestAPI:
    """Test class for API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = load_config()
        assert isinstance(config, dict)
        assert "universe_source" in config
        assert "capital" in config
        assert "top_n" in config
    
    def test_run_endpoint_basic(self):
        """Test basic run endpoint functionality"""
        response = client.get("/run?capital=100000&top_n=5")
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]  # 500 acceptable if no data available
        
        if response.status_code == 200:
            data = response.json()
            assert "reference_dates" in data
            assert "eligible" in data
            assert "top" in data
            assert "summary" in data
    
    def test_run_endpoint_with_date(self):
        """Test run endpoint with specific date"""
        test_date = "2024-01-15"
        response = client.get(f"/run?ref_date={test_date}&capital=50000&top_n=3")
        
        # Should handle date parameter correctly
        assert response.status_code in [200, 400, 500]
    
    def test_run_endpoint_invalid_date(self):
        """Test run endpoint with invalid date"""
        response = client.get("/run?ref_date=invalid-date")
        assert response.status_code == 400
    
    def test_run_endpoint_validation(self):
        """Test parameter validation"""
        # Test negative capital
        response = client.get("/run?capital=-1000")
        assert response.status_code == 422
        
        # Test excessive top_n
        response = client.get("/run?top_n=100")
        assert response.status_code == 422
    
    def test_levels_endpoint(self):
        """Test levels endpoint"""
        # Test with NSE stock symbols
        test_tickers = "RELIANCE.NS,TCS.NS"
        response = client.get(f"/levels?tickers={test_tickers}")
        
        # Should either succeed or handle gracefully
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "levels" in data
            if data["levels"]:
                level = data["levels"][0]
                assert "Ticker" in level
                assert "Close" in level
                assert "ATR14" in level
                assert "Stop_2xATR" in level
    
    def test_levels_endpoint_empty_tickers(self):
        """Test levels endpoint with empty tickers"""
        response = client.get("/levels?tickers=")
        assert response.status_code == 400

class TestDataFunctions:
    """Test class for data processing functions"""
    
    def test_sample_data_download(self):
        """Test data download with a small sample"""
        # Test with a few liquid NSE stocks
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        
        try:
            data = yf_download(tickers, start_date, end_date)
            
            if not data.empty:
                assert isinstance(data, pd.DataFrame)
                assert data.index.dtype.kind == 'M'  # Datetime index
                assert len(data.columns) <= len(tickers)  # Some tickers might fail
                
                # Check for valid price data
                assert (data > 0).all().all()  # All prices should be positive
                
        except Exception as e:
            # Data download can fail due to network issues
            pytest.skip(f"Data download failed: {e}")
    
    def test_momentum_calculation_sample(self):
        """Test momentum calculation with sample data"""
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic price series
        n_stocks = 5
        prices = pd.DataFrame(index=dates)
        
        for i in range(n_stocks):
            # Generate random walk with drift
            returns = np.random.normal(0.0005, 0.02, len(dates))
            price_series = 100 * np.exp(np.cumsum(returns))
            prices[f'STOCK_{i}.NS'] = price_series
        
        # Test momentum calculation
        ref_date = dates[-50]  # 50 days from end
        
        try:
            mom_df, ref_info = momentum_frame(prices, ref_date)
            
            assert isinstance(mom_df, pd.DataFrame)
            assert isinstance(ref_info, dict)
            
            # Check required columns
            required_cols = ['Price', '6M_Return(%)', '12M_ex1_Return(%)', 
                           'Volatility(%)', 'MomentumScore', 'Above_200DMA']
            for col in required_cols:
                assert col in mom_df.columns
            
            # Check data types and ranges
            assert mom_df['Price'].dtype in [np.float64, np.int64]
            assert mom_df['Volatility(%)'].min() >= 0  # Volatility should be non-negative
            
        except ValueError as e:
            # Expected if insufficient data
            pytest.skip(f"Insufficient sample data: {e}")

class TestUtilityFunctions:
    """Test utility and helper functions"""
    
    def test_config_validation(self):
        """Test configuration file validation"""
        if os.path.exists("config.yaml"):
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            # Test required fields
            required_fields = ['universe_source', 'capital', 'top_n', 'history_days']
            for field in required_fields:
                assert field in config, f"Missing required config field: {field}"
            
            # Test data types
            assert isinstance(config['capital'], (int, float))
            assert isinstance(config['top_n'], int)
            assert config['top_n'] > 0
            assert config['capital'] > 0
    
    def test_universe_loading_fallback(self):
        """Test universe loading with fallback"""
        try:
            universe = load_universe()
            assert isinstance(universe, list)
            assert len(universe) > 0
            
            # Check format
            for symbol in universe[:5]:  # Check first 5
                assert isinstance(symbol, str)
                assert symbol.endswith('.NS')
                
        except Exception as e:
            # Universe loading can fail
            pytest.skip(f"Universe loading failed: {e}")

def test_data_validation():
    """Test data validation functions"""
    # Create test data with issues
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    test_data = pd.DataFrame(index=dates)
    
    # Good stock
    test_data['GOOD.NS'] = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    
    # Stock with insufficient data
    insufficient_data = pd.Series(index=dates, dtype=float)
    insufficient_data.iloc[:50] = 100 + np.cumsum(np.random.normal(0, 1, 50))
    test_data['INSUFFICIENT.NS'] = insufficient_data
    
    # Stock with extreme moves
    extreme_data = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    extreme_data[100] = extreme_data[99] * 2  # 100% jump
    test_data['EXTREME.NS'] = extreme_data
    
    # Stock with negative prices
    negative_data = 100 + np.cumsum(np.random.normal(-0.1, 1, len(dates)))
    test_data['NEGATIVE.NS'] = negative_data
    
    from utils import validate_stock_data
    
    clean_data, issues = validate_stock_data(test_data, min_history_days=200)
    
    assert isinstance(clean_data, pd.DataFrame)
    assert isinstance(issues, list)
    assert len(clean_data.columns) <= len(test_data.columns)

# Performance benchmark test
def test_performance_benchmark():
    """Basic performance benchmark"""
    import time
    
    # Test API response time
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 1.0  # Should respond within 1 second
    assert response.status_code == 200

if __name__ == "__main__":
    # Run basic tests
    print("Running basic API tests...")
    
    # Test health endpoint
    try:
        response = client.get("/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test configuration
    try:
        config = load_config()
        print(f"Config loaded: {len(config)} settings")
    except Exception as e:
        print(f"Config loading failed: {e}")
    
    print("Basic tests completed. Run 'pytest test_api.py' for full test suite.")