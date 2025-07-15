import yfinance as yf
import os
import pandas as pd

# import data in the raw folder from Yfianance

def data_load_tech(ticker,period,interval):
    dat = yf.Ticker(ticker)
    data = dat.history(period=period, interval=interval)

    # Build folder and create it if it doesn't exist
    folder_path = os.path.join("raw", ticker)
    os.makedirs(folder_path, exist_ok=True)

    # Build filename and path
    filename = f"{ticker}_{period}_{interval}.csv"
    file_path = os.path.join(folder_path, filename)

    data.to_csv(file_path)
    print("Data saved to", file_path)

# test
# data_load_tech("AAPL","2d","1h")


def data_load_fund(ticker):
    dat = yf.Ticker(ticker)

    # Get data
    data1 = dat.info                      # dict
    data2 = dat.calendar                  # dict (despite appearing like a DataFrame)
    data3 = dat.analyst_price_targets     # DataFrame

    # Define path for fundamentals
    folder_path = os.path.join("raw", ticker, "fundamentals")
    os.makedirs(folder_path, exist_ok=True)

    pd.DataFrame([data1]).to_csv(os.path.join(folder_path, "i.csv"))

    pd.DataFrame(data2).T.to_csv(os.path.join(folder_path, "c.csv"))

    pd.DataFrame(list(data3.items()), columns=["Metric", "Value"]) \
        .to_csv(os.path.join(folder_path, "a.csv"), index=False)

    print(f"Fundamental data saved to {folder_path}")

