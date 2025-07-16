import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

def get_sp500_symbols():
    """Returns a list of S&P 500 symbols."""
    # (Using a smaller list for brevity in the example)
    list = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM']
    return list

def get_stock_data(symbol, period="2y", interval="1h"):
    """Fetch stock data from yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if len(data) < 100:
            return None
        return data
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def create_sequential_dataset(data, lookback_window=72):
    """
    Creates features, target, and timestamps from a single stock's data.
    """
    X, y, timestamps = [], [], []
    
    close_prices = data['Close'].values
    dates = data.index

    for i in range(lookback_window, len(close_prices)):
        # Features: The lookback window of prices
        feature_window = close_prices[i - lookback_window : i]
        
        # Normalize the window to focus on shape, not absolute price
        last_price = feature_window[-1]
        if last_price == 0: continue # Avoid division by zero
        normalized_window = (feature_window / last_price) - 1
        
        # Target: The return of the next hour
        current_price = close_prices[i]
        previous_price = close_prices[i-1]
        if previous_price == 0: continue # Avoid division by zero
        target_return = (current_price - previous_price) / previous_price
        
        X.append(normalized_window)
        y.append(target_return)
        timestamps.append(dates[i])
        
    return np.array(X), np.array(y), np.array(timestamps)

def build_dataset(symbols, lookback_window=72):
    """
    Builds and correctly sorts the entire dataset from all symbols.
    """
    all_X, all_y, all_timestamps = [], [], []
    print(f"Building sequential dataset for {len(symbols)} stocks with lookback={lookback_window}...")
    
    for symbol in tqdm(symbols, desc="Processing Stocks"):
        data = get_stock_data(symbol)
        if data is None or len(data) <= lookback_window:
            continue
        
        X_stock, y_stock, timestamps_stock = create_sequential_dataset(data, lookback_window=lookback_window)
        
        if len(X_stock) > 0:
            all_X.append(X_stock)
            all_y.append(y_stock)
            all_timestamps.append(timestamps_stock)

    if not all_X:
        return pd.DataFrame()

    # Concatenate data from all stocks
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_timestamps = np.concatenate(all_timestamps, axis=0)
    
    # Create the final DataFrame
    feature_columns = [f'price_{lookback_window-i}_ago' for i in range(lookback_window)]
    full_dataset = pd.DataFrame(final_X, columns=feature_columns)
    full_dataset['timestamp'] = final_timestamps
    full_dataset['target_return'] = final_y
    
    # Sort the ENTIRE dataset by time - THIS IS THE CRITICAL FIX
    print("Sorting the full dataset by timestamp...")
    full_dataset = full_dataset.sort_values(by='timestamp').reset_index(drop=True)
    
    return full_dataset

def train_model(full_dataset):
    """
    Trains and evaluates the model using a robust chronological split.
    """
    # Separate features and target from the sorted DataFrame
    y = full_dataset['target_return']
    X = full_dataset.drop(columns=['target_return', 'timestamp'])
    
    print(f"\nTraining model on {X.shape[0]} examples with {X.shape[1]} features each...")
    
    # Perform a strict chronological split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Training data from ~{pd.to_datetime(full_dataset['timestamp'].iloc[0]).date()} to ~{pd.to_datetime(full_dataset['timestamp'].iloc[split_index-1]).date()}")
    print(f"Testing data from  ~{pd.to_datetime(full_dataset['timestamp'].iloc[split_index]).date()} to ~{pd.to_datetime(full_dataset['timestamp'].iloc[-1]).date()}")
    
    # No scaling is needed because we normalized each window individually.
    
    # Train model
    print("Fitting RandomForestRegressor model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Backtest Results ---")
    print(f"Test Set MSE: {mse:.8f}")
    print(f"Test Set R²:  {r2:.4f}")
    
    if r2 < 0:
        print("\nWarning: R² is negative. The model is performing worse than a naive model that always predicts the average return.")
    elif r2 < 0.001:
        print("\nNote: R² is very close to zero, indicating little to no predictive power.")
    
    return model

def main():
    """Main execution function."""
    LOOKBACK_WINDOW = 168  # Using a 1-week lookback (168 hours)
    NUM_SYMBOLS = 100      # Number of stocks to process
    
    # Define a filename for the cached dataset
    csv_filename = f"sorted_stock_data_lookback_{LOOKBACK_WINDOW}_symbols_{NUM_SYMBOLS}.csv"

    # --- Check for cached data first ---
    if os.path.exists(csv_filename):
        print(f"Loading existing dataset from '{csv_filename}'...")
        full_dataset = pd.read_csv(csv_filename, parse_dates=['timestamp'])
    else:
        print(f"Dataset file not found. Building new dataset...")
        symbols = get_sp500_symbols()[:NUM_SYMBOLS]
        
        full_dataset = build_dataset(symbols, lookback_window=LOOKBACK_WINDOW)
        
        if full_dataset.empty:
            print("No data collected. Exiting.")
            return
            
        print(f"Saving newly created dataset to '{csv_filename}'...")
        full_dataset.to_csv(csv_filename, index=False)
        print("Save complete.")

    if full_dataset.empty:
        print("Dataset is empty. Exiting.")
        return
        
    print(f"\nDataset loaded successfully.")
    print(f"Total examples: {len(full_dataset)}")
    
    # Pass the full, sorted dataset to the training function
    model = train_model(full_dataset)
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main()
