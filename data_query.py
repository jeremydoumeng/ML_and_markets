import os
from data_load import data_load_tech, data_load_fund
import pandas as pd
from data_formatter import format_for_tensorflow

# will request data_load to import data from Yfinance if necessary

def request_data_tech(ticker, period, interval):
    filename = ticker + "_" + period + "_" + interval + ".csv"
    path = "raw" +"/"+ ticker
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        print("already exists")
        return filepath
    else:
        data_load_tech(ticker, period, interval)
        print("file imported")
        return filepath

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


def request_tf_formatted_data(ticker, period, interval, n_steps=50, target_col='Close', n_cols=None):
    """
    Cherche ou crée le fichier formaté TensorFlow pour un ticker donné.
    Si le fichier existe déjà, le retourne. Sinon, le crée à partir des données brutes (téléchargées si besoin).
    n_cols : nombre de colonnes/features à conserver (None = toutes)
    target_col : colonne cible à prédire (valeur à l'instant t+1)
    """
    # Dossier de sortie
    tf_dir = os.path.join('tf_ready', ticker)
    os.makedirs(tf_dir, exist_ok=True)
    tf_filename = f"{ticker}_{period}_{interval}_tf_{n_steps}steps.csv"
    tf_filepath = os.path.join(tf_dir, tf_filename)

    if os.path.exists(tf_filepath):
        print("TF formatted file already exists")
        return tf_filepath
    # Sinon, on cherche le brut (ou on le télécharge)
    raw_dir = os.path.join('raw', ticker)
    raw_filename = f"{ticker}_{period}_{interval}.csv"
    raw_filepath = os.path.join(raw_dir, raw_filename)
    if not os.path.exists(raw_filepath):
        data_load_tech(ticker, period, interval)
    df = pd.read_csv(raw_filepath, index_col=0, parse_dates=True)
    df_tf = format_for_tensorflow(df, n_steps=n_steps, target_col=target_col)
    if n_cols is not None and n_cols < len(df_tf.columns):
        keep_cols = list(df_tf.columns[:n_cols-1]) + [df_tf.columns[-1]]
        df_tf = df_tf[keep_cols]
    df_tf.to_csv(tf_filepath, index=False)
    print(f"TF formatted file created: {tf_filepath}")
    return tf_filepath

# Exemple d'appel
request_tf_formatted_data("AAPL", "1y", "1h", n_steps=50, target_col='Close', n_cols=None)