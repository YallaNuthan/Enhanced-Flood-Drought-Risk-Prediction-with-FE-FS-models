import ee
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def authenticate_gee():
    try:
        ee.Initialize(project='hydro-engine-india')
        print("GEE Initialized for All-India High-Resolution Extraction.")
    except Exception as e:
        print(f"GEE Error: {e}")

def get_target_locations():
    # 1. Macro-grid for the Heatmap (12x12 to save compute)
    lats = np.linspace(8.0, 36.0, 12)
    lons = np.linspace(68.0, 97.0, 12)
    grid_points = [{'name': 'Grid', 'lat': lat, 'lon': lon} for lat in lats for lon in lons]
    
    # 2. Specific City Nodes for "Separate Predictions"
    cities = [
        {"name": "Mumbai (Maharashtra)", "lat": 19.0760, "lon": 72.8777},
        {"name": "Delhi (NCR)", "lat": 28.7041, "lon": 77.1025},
        {"name": "Chennai (Tamil Nadu)", "lat": 13.0827, "lon": 80.2707},
        {"name": "Kolkata (West Bengal)", "lat": 22.5726, "lon": 88.3639},
        {"name": "Hyderabad (Telangana)", "lat": 17.3850, "lon": 78.4867},
        {"name": "Bengaluru (Karnataka)", "lat": 12.9716, "lon": 77.5946},
        {"name": "Ahmedabad (Gujarat)", "lat": 23.0225, "lon": 72.5714},
        {"name": "Guwahati (Assam)", "lat": 26.1445, "lon": 91.7362},
        {"name": "Patna (Bihar)", "lat": 25.5941, "lon": 85.1376},
        {"name": "Srinagar (J&K)", "lat": 34.0837, "lon": 74.7973},
        {"name": "Kochi (Kerala)", "lat": 9.9312, "lon": 76.2673},
        {"name": "Bhubaneswar (Odisha)", "lat": 20.2961, "lon": 85.8245},
        {"name": "Jaipur (Rajasthan)", "lat": 26.9124, "lon": 75.7873},
        {"name": "Lucknow (UP)", "lat": 26.8467, "lon": 80.9462},
        {"name": "Vijayawada (AP)", "lat": 16.5062, "lon": 80.6480}
    ]
    
    return grid_points + cities

def pull_and_predict(points):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=40)
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
             .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
             .select('total_precipitation_sum')

    print("\nLoading Hybrid LSTM-GRU Forecasting Engine...")
    model = tf.keras.models.load_model('hydro_hybrid_model.h5', compile=False)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    
    predictions = []
    print(f"Extracting and executing 7-Day Auto-Regressive Forecast across {len(points)} locations...")
    
    for idx, pt in enumerate(points):
        point_geom = ee.Geometry.Point([pt['lon'], pt['lat']])
        
        def extract_point(image):
            date = image.date().format('YYYY-MM-dd')
            val = image.reduceRegion(ee.Reducer.mean(), point_geom, 25000).get('total_precipitation_sum')
            return ee.Feature(None, {'date': date, 'precip': val})
        
        try:
            data = era5.map(extract_point).reduceColumns(ee.Reducer.toList(2), ['date', 'precip']).values().get(0).getInfo()
            df = pd.DataFrame(data, columns=['Date', 'precipitation'])
            df['precipitation'] = df['precipitation'].fillna(0) * 1000
            df['precipitation_ewma_30d'] = df['precipitation'].ewm(span=30, adjust=False).mean()
            
            # Extract final 7 days as the starting sequence
            current_seq = df[['precipitation', 'precipitation_ewma_30d']].tail(7).values
            current_ewma = current_seq[-1, 1]
            
            future_preds = []
            
            # --- THE AUTO-REGRESSIVE LOOP (7 DAYS) ---
            for day in range(7):
                scaled_seq = scaler_X.fit_transform(current_seq).reshape(1, 7, 2)
                pred_runoff = max(0, model.predict(scaled_seq, verbose=0)[0][0] * 80.0)
                future_preds.append(pred_runoff)
                
                # Shift sequence: Drop oldest day, add tomorrow (Assuming Dry Spell Scenario for recession curve)
                new_precip = 0.0 
                # Recalculate LSTM memory (Alpha for span=30 is 2/31)
                alpha = 2 / (30 + 1)
                new_ewma = (new_precip * alpha) + (current_ewma * (1 - alpha))
                
                current_seq = np.vstack((current_seq[1:], [new_precip, new_ewma]))
                current_ewma = new_ewma
            
            predictions.append({
                'name': pt['name'],
                'lat': pt['lat'],
                'lon': pt['lon'],
                'pred_day_1': future_preds[0],
                'pred_day_2': future_preds[1],
                'pred_day_3': future_preds[2],
                'pred_day_4': future_preds[3],
                'pred_day_5': future_preds[4],
                'pred_day_6': future_preds[5],
                'pred_day_7': future_preds[6]
            })
        except Exception:
            continue
            
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(points)} locations...")
            
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    authenticate_gee()
    points = get_target_locations()
    results_df = pull_and_predict(points)
    
    # Split the results into grid (for map) and cities (for dropdown)
    grid_df = results_df[results_df['name'] == 'Grid']
    city_df = results_df[results_df['name'] != 'Grid']
    
    grid_df.to_csv('national_heatmap_data.csv', index=False)
    city_df.to_csv('city_predictions.csv', index=False)
    
    print("\nSUCCESS! Saved Heatmap and Separate City Predictions.")
    
    