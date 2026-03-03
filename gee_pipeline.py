import ee
import pandas as pd

def authenticate_gee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        # This triggers the browser login to verify your identity
        ee.Authenticate()
        
        # IMPORTANT: You MUST replace 'your-gcp-project-id' with a valid Google Cloud Project ID.
        # Google Earth Engine requires a Cloud Project to track API usage.
        ee.Initialize(project='hydro-engine-india')
        print("Successfully authenticated and initialized Google Earth Engine!")
    except Exception as e:
        print(f"Error initializing GEE: {e}")

def define_data_streams():
    """Defines the ROI and points to the GEE Image Collections."""
    # Defining a Region of Interest (ROI) over India (e.g., Godavari basin area)
    # Format: [longitude_min, latitude_min, longitude_max, latitude_max]
    roi = ee.Geometry.Rectangle([73.0, 16.0, 81.0, 20.0]) 
    
    # 1. High-resolution precipitation (CHIRPS)
    precipitation = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(roi)
                      
    # 2. U-Wind and Temperature (ERA5-Land Daily)
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                  .filterBounds(roi) \
                  .select(['temperature_2m', 'u_component_of_wind_10m'])
                  
    # 3. Surface and Root-zone Soil Moisture (NASA SMAP)
    smap_soil = ee.ImageCollection("NASA/USDA/SMAP/SMAP10KM_SOIL_MOISTURE") \
                  .filterBounds(roi) \
                  .select(['ssm', 'susm'])
                  
    return precipitation, era5_land, smap_soil, roi

import ee
import pandas as pd

def authenticate_gee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        ee.Initialize(project='hydro-engine-india')
        print("Successfully authenticated and initialized Google Earth Engine!")
    except Exception as e:
        print(f"Error initializing GEE: {e}")

def extract_multi_band_series(point, start_date, end_date):
    """Extracts Rain, Soil Moisture, Wind, and Runoff from a unified ERA5 dataset."""
    
    # 1. Load the Unified ERA5-Land Daily Collection
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
             .filterBounds(point) \
             .filterDate(start_date, end_date) \
             .select([
                 'total_precipitation_sum', 
                 'surface_runoff_sum', 
                 'u_component_of_wind_10m', 
                 'volumetric_soil_water_layer_1'
             ])

    # 2. Define extraction function for ALL bands simultaneously
    def extract_data(image):
        date = image.date().format('YYYY-MM-dd')
        # Extract all selected bands at this specific coordinate
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=point, 
            scale=5000
        )
        return ee.Feature(None, stats.set('date', date))

    print("Pulling 10 Years of Unified ERA5 Data (Rain, Runoff, Wind, Soil)...")
    
    # Extract to local machine
    extracted_features = era5.map(extract_data).reduceColumns(
        ee.Reducer.toList(5), 
        ['date', 'total_precipitation_sum', 'surface_runoff_sum', 'u_component_of_wind_10m', 'volumetric_soil_water_layer_1']
    ).values().get(0).getInfo()

    # Convert to Pandas DataFrame
    columns = ['Date', 'precipitation', 'runoff', 'u_wind', 'soil_moisture']
    df = pd.DataFrame(extracted_features, columns=columns)
    
    # Format the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.fillna(0) 
    
    # Convert ERA5 precipitation and runoff from meters to millimeters
    df['precipitation'] = df['precipitation'] * 1000
    df['runoff'] = df['runoff'] * 1000
    
    return df

if __name__ == "__main__":
    authenticate_gee()
    gauge_point = ee.Geometry.Point([80.8, 17.0]) 
    
    print("Extracting multi-variable dataset (2014-2024). Please wait...")
    try:
        master_df = extract_multi_band_series(gauge_point, '2014-01-01', '2024-12-31')
        
        print("\nSuccess! Here is the top of your Real-World Multi-Variable DataFrame:")
        print(master_df.head())
        
        master_df.to_csv('raw_hydro_data.csv')
        print("\nSaved to raw_hydro_data.csv")
        
    except Exception as e:
        print(f"Extraction failed: {e}")


        
        