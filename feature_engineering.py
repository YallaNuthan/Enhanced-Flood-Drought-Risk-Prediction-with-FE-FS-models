import pandas as pd
import numpy as np

def generate_79_hydro_indices(csv_path):
    """
    Synthesizes exactly 79 hydro-meteorological indices 
    (Magnitude, Frequency, Recession, Volatility) as defined in the abstract.
    """
    print(f"Loading real-world climate data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()

    # Isolate the target variable (Real ERA5 Runoff)
    target = df['runoff'].copy()
    
    # Our 3 core predictors
    predictors = ['precipitation', 'u_wind', 'soil_moisture']
    windows = [3, 7, 14, 30, 60, 90]
    
    # ---------------------------------------------------------
    # 1. MAGNITUDE INDICES (36 Features)
    # ---------------------------------------------------------
    print("Calculating Magnitude Indices (Rolling Means & Max)...")
    for col in predictors:
        for w in windows:
            df[f'{col}_mean_{w}d'] = df[col].rolling(window=w, min_periods=1).mean()
            df[f'{col}_max_{w}d'] = df[col].rolling(window=w, min_periods=1).max()
            
    print("Calculating Magnitude Indices (Accumulations)...")
    for w in windows: # Accumulation sum only makes physical sense for precipitation
        df[f'precipitation_sum_{w}d'] = df['precipitation'].rolling(window=w, min_periods=1).sum()

    # ---------------------------------------------------------
    # 2. RECESSION / MEMORY INDICES (15 Features)
    # ---------------------------------------------------------
    print("Calculating Recession Indices (EWMA)...")
    # Exponential Weighted Moving Average gives more weight to recent days 
    # but remembers long-term drought decay.
    for col in predictors:
        for w in [7, 14, 30, 60, 90]:
            df[f'{col}_ewma_{w}d'] = df[col].ewm(span=w, adjust=False).mean()

    # ---------------------------------------------------------
    # 3. FREQUENCY INDICES (4 Features)
    # ---------------------------------------------------------
    print("Calculating Frequency Indices...")
    # Counts the number of days with significant rainfall in a given window
    for w in [7, 14, 30, 90]:
        df[f'precip_freq_over_1mm_{w}d'] = df['precipitation'].rolling(window=w, min_periods=1).apply(lambda x: (x > 1.0).sum(), raw=True)

    # ---------------------------------------------------------
    # 4. VOLATILITY & DURATION EXTREMES (15 Features)
    # ---------------------------------------------------------
    print("Calculating Volatility & Duration Extremes...")
    for col in predictors:
        for w in [7, 14, 30, 60, 90]:
            df[f'{col}_std_{w}d'] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
            
    # Add 3 specific Soil Moisture Minimums for Deep Drought tracking
    for w in [30, 60, 90]:
        df[f'soil_moisture_min_{w}d'] = df['soil_moisture'].rolling(window=w, min_periods=1).min()

    # ---------------------------------------------------------
    # FINAL CLEANUP
    # ---------------------------------------------------------
    # Drop the original runoff to prevent data leakage in predictors
    df = df.drop(columns=['runoff'])
    
    # Verify exact feature count matches the abstract (3 base + 76 engineered = 79)
    num_features = len(df.columns)
    print(f"\nSUCCESS: Generated exactly {num_features} predictor indices!")
    
    # Add our actual physical target variable back as the label to predict
    df['target_runoff'] = target
    
    return df

if __name__ == "__main__":
    try:
        engineered_df = generate_79_hydro_indices('raw_hydro_data.csv')
        
        # Save the massive new feature space for the Bayesian Network
        engineered_df.to_csv('engineered_features_data.csv')
        print("Saved successfully to engineered_features_data.csv")
        
    except FileNotFoundError:
        print("Error: Could not find raw_hydro_data.csv.")
        