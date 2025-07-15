# Stock Market ML Inefficiency Demo

This project demonstrates how simple price-based machine learning techniques are inefficient for predicting the stock market. By training models on historical price data for S&P 500 stocks, it highlights the limitations and challenges of using naive ML approaches for financial forecasting.

## Project Structure

- `data/` — Main project directory
  - `raw/` — Raw historical and fundamental data for stocks
  - `tf_ready/` — Data formatted for TensorFlow models
  - `naive_ML/` — Naive ML models and scripts
  - `data_query.py` — Functions to request and format stock data
  - `data_query_advanced.py` — S&P 500 symbol retrieval and advanced queries
  - `data_formatter.py` — Data formatting utilities
  - `data_load.py` — Data loading and downloading utilities
  - `stock_dashboard.py`, `tk_stock_dashboard.py` — Visualization dashboards
  - `train_learn_CS.py`, `naive_ML/train_learn_naive.py` — Model training scripts

## Key Features

- **S&P 500 Coverage:** Automatically fetches and processes all S&P 500 stocks.
- **Historical Data Query:** Downloads and formats historical price data for any period/interval.
- **Naive ML Models:** Trains simple models to predict future prices, demonstrating their inefficiency.
- **Dashboards:** Visualizes stock data and model predictions.

## Usage

1. **Install Requirements**
   - Ensure you have Python 3.x and `pandas` installed.
   - You may also need `yfinance` and `tensorflow` for data loading and ML.

2. **Fetch and Format Data**
   - Use `data_query_advanced.py` and `data_query.py` to download and format data for S&P 500 stocks:
     ```python
     from data_query_advanced import get_sp500_symbols
     from data_query import request_tf_formatted_data

     symbols = get_sp500_symbols()
     for symbol in symbols:
         request_tf_formatted_data(symbol, '1y', '1h')
     ```

3. **Train Naive Models**
   - Run scripts in `naive_ML/` to train and evaluate simple ML models on the data.

4. **Visualize Results**
   - Use the dashboard scripts to explore data and model predictions.

## Conclusion

This project provides hands-on evidence that naive, price-only ML models are not effective for market prediction, encouraging exploration of more sophisticated approaches and additional data sources.

---

*For educational and research purposes only. Not financial advice.* 
