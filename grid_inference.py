import ee
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def authenticate_gee():
    try:
        ee.Initialize(project='hydro-engine-india')
        print("GEE Initialized.")
    except Exception as e:
        print(f"GEE Error: {e}")

def generate_basin_grid():
    # Define a 5x5 grid around the Godavari basin (25 spatial points)
    lats = np.linspace(16.5, 18.0, 5)
    lons = np.linspace(80.0, 81.5, 5)
    
    points = []
    for lat in lats:
        for lon in lons:
            points.append({'lat': lat, 'lon': lon})
    return points

def pull_live_grid_data(points):
    """Pulls the last 40 days of ERA5 Precipitation for the grid to satisfy the 30d EWMA."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=40)
    
    end_str = end_date.strftime('%Y-%m-%d')
    start_str = start_date.strftime('%Y-%m-%d')
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
             .filterDate(start_str, end_str) \
             .select('total_precipitation_sum')

    grid_results = []
    
    print(f"Extracting live spatial data for {len(points)} grid points...")
    for idx, pt in enumerate(points):
        point_geom = ee.Geometry.Point([pt['lon'], pt['lat']])
        
        def extract_point(image):
            date = image.date().format('YYYY-MM-dd')
            val = image.reduceRegion(ee.Reducer.mean(), point_geom, 5000).get('total_precipitation_sum')
            return ee.Feature(None, {'date': date, 'precip': val})
        
        # Pull data for this specific coordinate
        data = era5.map(extract_point).reduceColumns(ee.Reducer.toList(2), ['date', 'precip']).values().get(0).getInfo()
        
        df = pd.DataFrame(data, columns=['Date', 'precipitation'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['precipitation'] = df['precipitation'].fillna(0) * 1000  # Convert to mm
        
        # Engineer the causal features
        df['precipitation_ewma_30d'] = df['precipitation'].ewm(span=30, adjust=False).mean()
        
        # Keep only the final 7 days for the model sequence
        df_sequence = df[['precipitation', 'precipitation_ewma_30d']].tail(7)
        
        grid_results.append({
            'lat': pt['lat'],
            'lon': pt['lon'],
            'sequence_data': df_sequence.values
        })
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/25 points...")
            
    return grid_results

def run_spatial_predictions(grid_data):
    print("\nLoading trained Hybrid Model for spatial inference...")
    model = tf.keras.models.load_model('hydro_hybrid_model.h5', compile=False)
    
    # We must scale the data just like we did in training
    # For a true enterprise app, you would save the scaler objects via joblib. 
    # Here we use standard bounds from your training phase.
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    
    predictions = []
    max_runoff_historical = 80.0 # Approximate max from your 10-year chart
    
    for pt in grid_data:
        # Fit-transforming live for the prototype (use saved scalers in full production)
        scaled_seq = scaler_X.fit_transform(pt['sequence_data'])
        model_input = scaled_seq.reshape(1, 7, 2)
        
        pred_scaled = model.predict(model_input, verbose=0)
        pred_runoff = pred_scaled[0][0] * max_runoff_historical
        
        predictions.append({
            'lat': pt['lat'],
            'lon': pt['lon'],
            'predicted_runoff': pred_runoff
        })
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    authenticate_gee()
    points = generate_basin_grid()
    grid_data = pull_live_grid_data(points)
    
    heatmap_df = run_spatial_predictions(grid_data)
    
    print("\nSUCCESS! Spatial Heatmap DataFrame Generated:")
    print(heatmap_df.head())
    
    heatmap_df.to_csv('live_heatmap_data.csv', index=False)
    print("Saved to live_heatmap_data.csv")
    