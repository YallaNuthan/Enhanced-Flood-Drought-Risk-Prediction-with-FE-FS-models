import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Concatenate

def create_sequences(data, target, time_steps=7):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

def calculate_nse(y_true, y_pred):
    """Calculates the Nash-Sutcliffe Efficiency (NSE)."""
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

def build_hybrid_model(time_steps, num_features):
    input_layer = Input(shape=(time_steps, num_features), name="Hydro_Input")
    
    # LSTM Branch for Slow Dynamics (Drought / 30-day EWMA)
    lstm_branch = LSTM(64, return_sequences=False, name="LSTM_Drought_Memory")(input_layer)
    
    # GRU Branch for Fast Dynamics (Flash Floods / Daily Precip)
    gru_branch = GRU(64, return_sequences=False, name="GRU_Flood_Spikes")(input_layer)
    
    fusion_layer = Concatenate(name="Fusion_Layer")([lstm_branch, gru_branch])
    
    dense_1 = Dense(32, activation='relu')(fusion_layer)
    output_layer = Dense(1, activation='linear', name="Runoff_Prediction")(dense_1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    print("Loading Real-World Causal Data...")
    df = pd.read_csv('model_ready_data.csv', index_col='Date', parse_dates=True)
    
    features = df.drop(columns=['target_runoff']).values
    target = df['target_runoff'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))
    
    time_steps = 7
    X, y = create_sequences(features_scaled, target_scaled, time_steps)
    
    # Split into Training (80%) and Testing (20%) sets to calculate real-world metrics
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} sequences, testing on {len(X_test)} unseen sequences.")
    
    model = build_hybrid_model(time_steps, X.shape[2])
    
    print("\nTraining Hybrid LSTM-GRU Model...")
    model.fit(
        X_train, y_train, 
        epochs=15, 
        batch_size=32, 
        validation_split=0.1,
        verbose=1
    )
    
    print("\n=========================================")
    print("CALCULATING ABSTRACT-PROMISED METRICS (UNSEEN TEST DATA)")
    print("=========================================")
    
    # Run predictions on the unseen test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Denormalize back to actual physical Runoff (millimeters) for accurate metrics
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    nse = calculate_nse(y_test_real, y_pred_real)
    
    print(f"* RMSE (Root Mean Square Error): {rmse:.4f} mm")
    print(f"* MAE (Mean Absolute Error): {mae:.4f} mm")
    print(f"* NSE (Nash-Sutcliffe Efficiency): {nse:.4f}")
    
    # Save the model
    model.save('hydro_hybrid_model.h5')
    print("\nModel saved successfully as hydro_hybrid_model.h5")
    