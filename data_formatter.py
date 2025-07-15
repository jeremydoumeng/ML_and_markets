import yfinance as yf
from data_load import data_load_tech, data_load_fund
import pandas as pd
import os
import numpy as np

# takes in argument a certain ticker
# extracts the data
# formats it in an exploitable way (depending on the strategy)

def request_data_fund(ticker):
    path1 = "raw" + "/"+ ticker  +"/" + "fundamentals/a.csv"
    path2 = "raw" + "/" + ticker  + "/" + "fundamentals/i.csv"
    path3 = "raw" + "/" + ticker +  "/" + "fundamentals/c.csv"

    paths = [path1, path2, path3]
    results = []
    for filepath in paths:
        if os.path.exists(filepath):
            print("already exists")
            results.append(filepath)
        else:
            data_load_fund(ticker)
            print("file imported")
            results.append(filepath)
    return results

def parse_tech(ticker, period, interval):
    data = request_data_tech(ticker, period, interval)
    df = pd.read_csv(data, index_col=0, parse_dates=True)
    return df

def parse_fund(ticker):
    data = request_data_fund(ticker)
    res = []
    for file in data:
        df = pd.read_csv(file)
        res.append(df)
    return res

print(parse_fund("AAPL"))

def format(ticker, period, interval):
    load = parse(ticker, period, interval)


def format_for_tensorflow(df, n_steps=50, target_col='Close'):
    """
    Formatte un DataFrame historique pour TensorFlow.
    Chaque ligne contient les n_steps dernières valeurs de Open, High, Low, Close, Volume,
    et la target (target_col) à prédire au pas suivant (n = -1).
    """
    df = df.sort_index()
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X, y = [], []
    # On s'arrête à len(df) - 1 pour que y soit la valeur suivante
    for i in range(n_steps, len(df) - 1):
        X.append(df[features].iloc[i-n_steps:i].values.flatten())
        y.append(df[target_col].iloc[i+1])
    X = np.array(X)
    y = np.array(y)
    # Crée un DataFrame pour export
    columns = []
    for step in range(n_steps, 0, -1):
        for feat in features:
            columns.append(f"{feat}_t-{step}")
    columns.append("Target_Price")
    data = np.concatenate([X, y.reshape(-1,1)], axis=1)
    df_out = pd.DataFrame(data, columns=columns)
    return df_out

def save_tf_csv(df, n_steps, target_col, out_path):
    """
    Formatte et sauvegarde le DataFrame historique au format TensorFlow-ready CSV.
    """
    df_tf = format_for_tensorflow(df, n_steps=n_steps, target_col=target_col)
    df_tf.to_csv(out_path, index=False)

# Exemple d'utilisation (à adapter selon votre flux de travail) :
# df = parse_tech('AAPL', '1y', '1d')
# df_tf = format_for_tensorflow(df, n_steps=50, target_col='Close')
# save_tf_csv(df, n_steps=50, target_col='Close', out_path='AAPL_tf_ready.csv')

