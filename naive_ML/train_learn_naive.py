import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def clean_and_prepare_data(file_path):
    """
    Nettoie et prépare les données pour l'apprentissage
    """
    # Charger les données
    df = pd.read_csv(file_path)
    
    print("=== DIAGNOSTIC DES DONNÉES ===")
    print(f"Forme des données : {df.shape}")
    print(f"Types des colonnes :\n{df.dtypes}")
    print(f"Premières lignes :\n{df.head()}")
    print(f"Valeurs manquantes :\n{df.isnull().sum()}")
    
    # Identifier les colonnes non-numériques
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    print(f"Colonnes non-numériques : {list(non_numeric_cols)}")
    
    # Nettoyer les données
    df_clean = df.copy()
    
    # 1. Supprimer les colonnes complètement vides
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # 2. Traiter les colonnes non-numériques
    for col in non_numeric_cols:
        if col in df_clean.columns:
            # Essayer de convertir en numérique
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 3. Sélectionner uniquement les colonnes numériques
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_numeric = df_clean[numeric_columns]
    
    print(f"\n=== APRÈS NETTOYAGE ===")
    print(f"Colonnes numériques conservées : {list(numeric_columns)}")
    print(f"Nouvelle forme : {df_numeric.shape}")
    
    # 4. Supprimer les lignes avec des valeurs manquantes
    df_final = df_numeric.dropna()
    
    print(f"Forme finale après suppression des NaN : {df_final.shape}")
    
    return df_final

def create_sequences(data, sequence_length=50):
    """
    Crée des séquences pour l'apprentissage temporel
    """
    sequences = []
    targets = []
    
    # Supposons que la dernière colonne est le target (prix à prédire)
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    
    for i in range(len(data) - sequence_length):
        seq = features[i:i+sequence_length]
        sequences.append(seq.flatten())  # Aplatir pour format tabulaire
        targets.append(target[i+sequence_length])
    
    return np.array(sequences), np.array(targets)

def create_features(df):
    """
    Crée des features techniques supplémentaires
    """
    df_features = df.copy()
    
    # Si vous avez les colonnes OHLC (Open, High, Low, Close)
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Moyennes mobiles
        df_features['MA_5'] = df['Close'].rolling(window=5).mean()
        df_features['MA_10'] = df['Close'].rolling(window=10).mean()
        df_features['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_features['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatilité
        df_features['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price change
        df_features['Price_Change'] = df['Close'].pct_change()
        
        # Volume ratio (si volume disponible)
        if 'Volume' in df.columns:
            df_features['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df_features['Volume_Ratio'] = df['Volume'] / df_features['Volume_MA']
    
    # Supprimer les NaN créés par les calculs
    df_features = df_features.dropna()
    
    return df_features

def create_improved_model(input_shape):
    """
    Crée un modèle amélioré avec plus de couches et techniques avancées
    """
    model = models.Sequential([
        # Couche d'entrée avec plus de neurones
        layers.Dense(1024, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Couches cachées profondes
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(32, activation='relu'),
        
        # Couche de sortie
        layers.Dense(1, activation='linear')
    ])
    
    return model

def plot_training_history(history):
    """
    Visualise l'historique d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def ensemble_predictions(models, X_test):
    """
    Combine les prédictions de plusieurs modèles
    """
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)
    
    # Moyenne des prédictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

def main():
    # Remplacez par votre chemin de fichier
    file_path = '/Users/jeremy/Desktop/data/tf_ready/AAPL/AAPL_1y_1h_tf_50steps.csv'
    
    try:
        # Nettoyer les données
        df_clean = clean_and_prepare_data(file_path)
        
        # Créer des features techniques supplémentaires
        df_features = create_features(df_clean)
        
        # Créer les séquences
        X, y = create_sequences(df_features, sequence_length=50)
        
        print(f"\n=== SÉQUENCES CRÉÉES ===")
        print(f"Forme des features X : {X.shape}")
        print(f"Forme des targets y : {y.shape}")
        
        # Vérifier qu'il n'y a pas de valeurs non-numériques
        print(f"X contient des NaN : {np.isnan(X).any()}")
        print(f"y contient des NaN : {np.isnan(y).any()}")
        print(f"Type de X : {X.dtype}")
        print(f"Type de y : {y.dtype}")
        
        # Normaliser les données avec RobustScaler (meilleur pour les données financières)
        scaler_X = RobustScaler()  # Plus robuste aux outliers
        scaler_y = RobustScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Diviser en train/test avec stratification temporelle
        # Pour les données temporelles, éviter le mélange aléatoire
        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
        
        print(f"\n=== DIVISION TRAIN/TEST ===")
        print(f"X_train shape : {X_train.shape}")
        print(f"X_test shape : {X_test.shape}")
        print(f"y_train shape : {y_train.shape}")
        print(f"y_test shape : {y_test.shape}")
        
        # Créer plusieurs modèles pour ensemble
        models_list = []
        histories = []
        
        for i in range(3):  # Entraîner 3 modèles différents
            print(f"\n=== ENTRAÎNEMENT MODÈLE {i+1}/3 ===")
            
            # Créer le modèle amélioré
            model = create_improved_model(X_train.shape[1])
            
            # Optimiseur avec learning rate adaptatif
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            if i == 0:
                print(f"\n=== ARCHITECTURE DU MODÈLE ===")
                model.summary()
            
            # Callbacks pour améliorer l'entraînement
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
            
            # Entraîner le modèle
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, lr_scheduler],
                verbose=1 if i == 0 else 0  # Verbose seulement pour le premier modèle
            )
            model.save("AAPLpredict1y1h.keras") 
            models_list.append(model)
            histories.append(history)
        
        # Évaluer chaque modèle
        print(f"\n=== RÉSULTATS INDIVIDUELS ===")
        for i, model in enumerate(models_list):
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"Modèle {i+1} - Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        # Prédictions d'ensemble
        ensemble_pred = ensemble_predictions(models_list, X_test)
        
        # Reconvertir à l'échelle originale
        ensemble_pred_original = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1))
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculer MAE sur l'ensemble
        ensemble_mae = np.mean(np.abs(ensemble_pred_original - y_test_original))
        
        print(f"\n=== RÉSULTATS ENSEMBLE ===")
        print(f"Ensemble MAE : {ensemble_mae:.4f}")
        
        # Exemples de prédictions
        print(f"\n=== EXEMPLES DE PRÉDICTIONS ENSEMBLE ===")
        for i in range(min(10, len(y_test_original))):
            real_val = y_test_original[i][0]
            pred_val = ensemble_pred_original[i][0]
            error = abs(real_val - pred_val)
            print(f"Réel: {real_val:.2f}, Prédit: {pred_val:.2f}, Erreur: {error:.2f}")
        
        # Visualiser l'historique du premier modèle
        if len(histories) > 0:
            plot_training_history(histories[0])
        
    except FileNotFoundError:
        print(f"Fichier non trouvé : {file_path}")
        print("Vérifiez le chemin et le nom du fichier")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()