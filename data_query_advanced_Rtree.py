import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- Helper functions remain the same ---
def get_sp500_symbols():
    # ... (same as before) ...
    list = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM']
    return list

def get_stock_data(symbol, period="2y", interval="1h"): # Fetch more data to have enough for a year's lookback
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        # We need at least lookback_window + 1 data points
        if len(data) < 100:
            return None
        return data
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def create_sequential_dataset(data, lookback_window=72):
    """
    Creates a dataset where features are the raw sequence of past prices.
    Features: A window of past prices, normalized.
    Target: The very next data point's return.
    """
    X, y = [], []
    
    # Use closing prices
    close_prices = data['Close'].values
    
    # Iterate from the first possible window to the end
    for i in range(lookback_window, len(close_prices)):
        
        # --- 1. Create Features: The lookback window ---
        # The window is from i-lookback_window up to (but not including) i
        feature_window = close_prices[i - lookback_window : i]
        
        # --- NORMALIZATION ---
        # It's crucial to normalize the window. A good way is to make it relative to the
        # last price in the window, so the model learns from the *shape* of the history.
        # We calculate (price / last_price_in_window) - 1
        # The last value in the sequence will always be 0.
        last_price = feature_window[-1]
        normalized_window = (feature_window / last_price) - 1
        
        X.append(normalized_window)
        
        # --- 2. Create the Target ---
        # The target is the return of the next hour (at time `i`)
        current_price = close_prices[i]
        previous_price = close_prices[i-1] # This is the same as `last_price`
        
        target_return = (current_price - previous_price) / previous_price
        y.append(target_return)
        
    return np.array(X), np.array(y)


def build_dataset(symbols, lookback_window=72):
    """
    Builds the dataset by processing each stock and concatenating the results.
    """
    all_X, all_y = [], []
    print(f"Building sequential dataset for {len(symbols)} stocks with lookback={lookback_window}...")
    
    for symbol in tqdm(symbols, desc="Processing Stocks"):
        data = get_stock_data(symbol)
        if data is None:
            continue
        
        # Ensure there's enough data for at least one window
        if len(data) > lookback_window:
            X_stock, y_stock = create_sequential_dataset(data, lookback_window=lookback_window)
            all_X.append(X_stock)
            all_y.append(y_stock)
    
    # Concatenate results from all stocks
    if not all_X:
        return pd.DataFrame(), pd.Series()

    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    # Convert to DataFrame for easier handling
    feature_columns = [f'price_{lookback_window-i}_ago' for i in range(lookback_window)]
    X_df = pd.DataFrame(final_X, columns=feature_columns)
    y_series = pd.Series(final_y, name='target_return')
    
    return X_df, y_series

def train_model(X, y):
    """Train the model on the sequential dataset"""
    print(f"Training model on {X.shape[0]} examples with {X.shape[1]} features each...")
    
    # We must do a chronological split, not a random one.
    # Take the first 80% of the data for training and the last 20% for testing.
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # No scaling is needed because we normalized each window individually.
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=12,          # Limit tree depth to prevent overfitting
        min_samples_leaf=20,   # Require more samples in a leaf node
        max_features='sqrt',   # Consider a subset of features for each split
        random_state=42,
        n_jobs=-1              # Use all available CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance - MSE: {mse:.8f}, R²: {r2:.4f}")
    
    if r2 < 0:
        print("Warning: R² is negative. The model performs worse than a naive model predicting the average return.")
    
    return model

def main():
    """Main execution function"""
    # NOTE: A lookback of 8759 is computationally VERY expensive.
    # Start with a much smaller number like 72 (3 days) or 168 (1 week).
    LOOKBACK_WINDOW = 72 
    
    symbols = get_sp500_symbols()[:20] # Start with a small number of stocks
    
    # Build dataset using the new sequential method
    X, y = build_dataset(symbols, lookback_window=LOOKBACK_WINDOW)
    
    if X.empty:
        print("No data collected. Exiting.")
        return
        
    print(f"\nDataset created successfully.")
    print(f"Shape of feature matrix (X): {X.shape}")
    print(f"Shape of target vector (y): {y.shape}")
    print("\nSample of the first row of features:")
    print(X.head(1))
    
    # Train model
    model = train_model(X, y)
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main()