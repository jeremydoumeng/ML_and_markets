import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from train_learn_naive import clean_and_prepare_data, create_features, create_sequences

# 1. Load the trained model
model = tf.keras.models.load_model('AAPLpredict1y1h.keras', compile=False)

# 2. Load and preprocess the data
file_path = 'tf_ready/AAPL/AAPL_1y_1h_tf_50steps.csv'
df_clean = clean_and_prepare_data(file_path)
df_features = create_features(df_clean)

# 3. Create sequences (windows of 50 steps)
X, y = create_sequences(df_features, sequence_length=50)

# 4. Scale the data (fit on all data for inference)
scaler_X = RobustScaler()
scaler_y = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 5. Predict for all windows
pred_scaled = model.predict(X_scaled)

# 6. Inverse-transform predictions and real values
pred = scaler_y.inverse_transform(pred_scaled)
y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))

# 7. Plot real vs predicted values
plt.figure(figsize=(15, 6))
plt.plot(y_true, label='Real')
plt.plot(pred, label='Predicted')
plt.title('AAPL 1y 1h: Real vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
