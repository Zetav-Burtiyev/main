import streamlit as st
import ee
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import folium
import pandas as pd
from folium.plugins import MiniMap
from shapely.geometry import mapping, box, shape
from streamlit_folium import st_folium, folium_static
import tempfile
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import geopandas as gpd
import warnings
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Configuration
# -------------------------------
SENTINEL_BANDS = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
NDVI_THRESHOLD = 0.66  # Threshold for healthy vegetation

# Thresholds for exclusion areas (default values)
ELEVATION_THRESHOLD = 2000  # meters (ignore areas above this elevation)
SNOW_THRESHOLD = 0.3      # NDSI threshold for snow detection
FOREST_COVER_THRESHOLD = 35  # Default value - Minimum tree cover percentage

# Interpolation / rendering params
COARSE_GRID_SIZE = 40        # number of tiles per side for coarse predictions
OUTPUT_RES = 600             # resolution of final interpolated grid
GAUSSIAN_SIGMA = 3.5         # smoothing after interpolation

# Initialize session state
if 'map_created' not in st.session_state:
    st.session_state.map_created = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# -------------------------------
# Azerbaijan Forest Information
# -------------------------------
def display_azerbaijan_forest_info():
    st.sidebar.markdown("## 🌳 Azerbaijan Forest Cover")
    
    forest_stats = {
        "Total Forest Area": "1,021,880 hectares (11.8% of land area)",
        "Primary Forest": "13,840 hectares (1.4% of forest area)",
        "Tree Cover Loss (2001-2022)": "8,610 hectares (-0.84%)",
        "Protected Areas": "10.3% of forest cover",
        "Main Forest Types": {
            "Caspian Hyrcanian": "Northern regions (subtropical rainforest)",
            "Caucasus Mixed": "Mountainous areas (oak, beech, hornbeam)",
            "Juniper Woodlands": "Arid southern regions"
        },
        "Threats": [
            "Illegal logging",
            "Agricultural expansion",
            "Climate change impacts",
            "Forest fires"
        ]
    }
    
    with st.sidebar.expander("📊 Key Statistics"):
        for stat, value in forest_stats.items():
            if isinstance(value, dict):
                st.markdown(f"**{stat}**")
                for subtype, desc in value.items():
                    st.markdown(f"- {subtype}: {desc}")
            elif isinstance(value, list):
                st.markdown(f"**{stat}**")
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**{stat}:** {value}")
    
    st.sidebar.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;">
    <small>Data sources: FAO Global Forest Resources Assessment, World Bank, MODIS Vegetation Continuous Fields</small>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Enhanced Legend System
# -------------------------------
def create_dynamic_legend():
    st.sidebar.markdown("## 🗺️ Forest Analysis Legend")
    
    # Forest Cover Legend
    st.sidebar.markdown("**Forest Cover (%)**")
    cover_colors = {
        "0-20%": "#d9d9d9",
        "20-40%": "#a1d99b",
        "40-60%": "#41ab5d",
        "60-80%": "#238b45",
        "80-100%": "#005a32"
    }
    
    for label, color in cover_colors.items():
        col1, col2 = st.sidebar.columns([1, 10])
        with col1:
            st.markdown(
                f'<div style="width: 20px; height: 20px; background: {color}; border: 1px solid #000;"></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(label)
    
    st.sidebar.markdown("**Forest Loss Probability**")
    loss_colors = {
        "Low (0-30%)": "#ffeda0",
        "Medium (30-60%)": "#feb24c",
        "High (60-90%)": "#f03b20",
        "Very High (90-100%)": "#bd0026"
    }
    
    for label, color in loss_colors.items():
        col1, col2 = st.sidebar.columns([1, 10])
        with col1:
            st.markdown(
                f'<div style="width: 20px; height: 20px; background: {color}; border: 1px solid #000;"></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(label)
    
    with st.sidebar.expander("📖 How to Interpret"):
        st.markdown("""
        - **Forest Cover**: Green colors show baseline tree cover percentage
        - **Loss Probability**: Red colors show likelihood of recent forest loss
        - **Darker Areas**: High forest cover with high loss probability indicate critical areas
        - **Excluded Areas**: Gray areas show non-forest, high elevation, or snow-covered regions
        """)

    with st.sidebar.expander("🧠Use Case Guideline for Each Prediction Method"):
        st.markdown("""
        - **Weighted Average**: General monitoring, Balanced, Use case(General)
        - **Simple Average**: Conservative estimates, Low false positives, Use case(Policy planning) 
        - **Maximum**: Critical areas, High sensitivity, Use case(Endangered forests)
        - **Minimum**: Agricultural borders, High specificity, Use case(Avoiding false alarms)
        - **Product**: Scientific research, Very conservative, Use case(Academic studies)
        """)

# -------------------------------
# Initialize Earth Engine
# -------------------------------
try:
    # Try to initialize Earth Engine
    ee.Initialize()
except Exception as e:
    # If initialization fails, authenticate
    st.warning("Earth Engine needs authentication. Please follow the instructions.")
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as auth_error:
        st.error(f"Earth Engine authentication failed: {str(auth_error)}")
        st.stop()

# -------------------------------
# Model Loading Functions
# -------------------------------
def load_models():
    """Load both ResNet50 and EfficientNet models"""
    try:
        # Load ResNet50 model
        resnet_base = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3)
        )
        resnet_base.trainable = False
        
        resnet_model = tf.keras.Sequential([
            resnet_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Load EfficientNet model
        efficient_base = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3)
        )
        efficient_base.trainable = False
        
        efficient_model = tf.keras.Sequential([
            efficient_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile both models
        for model in [resnet_model, efficient_model]:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return {
            'resnet': resnet_model,
            'efficientnet': efficient_model
        }
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def stacked_prediction(models, image, method='weighted_average'):
    """
    Make predictions using multiple models and combine them
    
    Parameters:
    - models: Dictionary of loaded models
    - image: Preprocessed image for prediction
    - method: How to combine predictions ('weighted_average', 'max', 'min', 'product')
    
    Returns:
    - Combined prediction and individual model predictions
    """
    predictions = {}
    
    # Get predictions from each model
    for name, model in models.items():
        pred = model.predict(np.expand_dims(image, 0), verbose=0)[0][0]
        predictions[name] = float(pred)
    
    # Combine predictions based on selected method
    if method == 'weighted_average':
        # Use custom weights if available
        weights = getattr(st.session_state, 'model_weights', {'resnet': 0.6, 'efficientnet': 0.4})
        combined = (predictions['resnet'] * weights['resnet'] + 
                   predictions['efficientnet'] * weights['efficientnet'])
    elif method == 'max':
        combined = max(predictions.values())
    elif method == 'min':
        combined = min(predictions.values())
    elif method == 'product':
        combined = np.prod(list(predictions.values()))
    else:  # average
        combined = np.mean(list(predictions.values()))
    
    return combined, predictions

@st.cache_data
def load_districts():
    try:
        districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
        azerbaijan_districts = districts.filter(ee.Filter.eq('ADM0_NAME', 'Azerbaijan'))
        district_names = azerbaijan_districts.aggregate_array('ADM2_NAME').getInfo()
        return {
            'ee_object': azerbaijan_districts,
            'names': sorted([name for name in district_names if name is not None])
        }
    except Exception as e:
        st.error(f"Failed to load districts: {str(e)}")
        st.stop()

districts_data = load_districts()
azerbaijan_districts = districts_data['ee_object']
district_names = districts_data['names']

# -------------------------------
# Improved Folium EE layer helper with attribution
# -------------------------------
def add_ee_layer(self, ee_object, vis_params, name, shown=True):
    try:
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True,
                show=shown
            ).add_to(self)
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object = ee_object.style(**{'fillColor': '00000000', 'color': '#3388ff'})
            map_id_dict = ee_object.getMapId()
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True,
                show=shown
            ).add_to(self)
    except Exception as e:
        st.error(f"Could not add EE layer: {str(e)}")

folium.Map.add_ee_layer = add_ee_layer

# -------------------------------
# UI Setup
# -------------------------------
st.set_page_config(layout="wide", page_title="Azerbaijan Forest Analysis", page_icon="🌲")
st.title("🌲 Azerbaijan Forest Cover & Loss Analysis")

# Display forest info and legend in sidebar
display_azerbaijan_forest_info()
create_dynamic_legend()

# Model stacking options
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Stacking Options")

stacking_method = st.sidebar.selectbox(
    "Model Combination Method",
    options=["weighted_average", "average", "max", "min", "product"],
    index=0,
    help="How to combine predictions from multiple models"
)

# Model weights (for weighted average)
if stacking_method == "weighted_average":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        resnet_weight = st.slider("ResNet Weight", 0.0, 1.0, 0.6, 0.1)
    with col2:
        efficient_weight = st.slider("EfficientNet Weight", 0.0, 1.0, 0.4, 0.1)
        
    # Update stacking function with custom weights
    if 'models' in st.session_state:
        st.session_state.stacking_method = stacking_method
        st.session_state.model_weights = {
            'resnet': resnet_weight,
            'efficientnet': efficient_weight
        }

# Load model only when selection changes or when first loading
if (not hasattr(st.session_state, 'models') or 
    not st.session_state.model_loaded):
    
    with st.spinner("Loading models..."):
        models = load_models()
        if models:
            st.session_state.models = models
            st.session_state.model_loaded = True
            st.session_state.stacking_method = stacking_method
            st.sidebar.success("Loaded both ResNet50 and EfficientNet models")
        else:
            st.sidebar.error("Failed to load models")
            st.stop()

# Main content
st.markdown("""
<div style="background-color:#e6f3ff;padding:20px;border-radius:10px;margin-bottom:20px;">
<h3 style="color:#0056b3;">About This Dashboard</h3>
<p>This interactive tool provides comprehensive forest analysis showing both baseline forest cover and recent loss patterns across Azerbaijan.</p>
</div>
""", unsafe_allow_html=True)

# District selection in sidebar
selected_district = st.sidebar.selectbox("Select a district:", district_names)

# Forest cover threshold slider
FOREST_COVER_THRESHOLD = st.sidebar.slider(
    "Forest Cover Threshold (%)",
    min_value=10,
    max_value=50,
    value=FOREST_COVER_THRESHOLD,
    help="Minimum tree cover percentage to consider as forest"
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End date", datetime.now())

# Get the selected district geometry
selected_geom = azerbaijan_districts.filter(ee.Filter.eq('ADM2_NAME', selected_district))

# -------------------------------
# Base Map Display
# -------------------------------
district_center = selected_geom.geometry().centroid().getInfo()['coordinates']
m = folium.Map(
    location=[district_center[1], district_center[0]],
    zoom_start=9,
    tiles=None
)

# Add base layers with proper attribution
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google Satellite Imagery',
    name='Satellite View',
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    name='OpenStreetMap',
    control=True
).add_to(m)

# Add selected district layer with highlighted style
m.add_ee_layer(
    selected_geom.style(**{'fillColor': '00000000', 'color': '#FF0000', 'width': 3}),
    {},
    "Selected District",
    shown=True
)

# Add layer control and display map
folium.LayerControl().add_to(m)
st_folium(m, width=None, height=500)

# -------------------------------
# Helper functions with exclusion logic
# -------------------------------
def get_district_geometry(district_name):
    try:
        district = azerbaijan_districts.filter(ee.Filter.eq('ADM2_NAME', district_name))
        geom = district.first().getInfo()['geometry']
        return geom
    except Exception as e:
        st.error(f"Could not get geometry for {district_name}: {str(e)}")
        return None

def get_tiles(bounds, size=128, grid_size=20):
    """Create a grid of tiles within bounds (coarse grid). Returns list of (tile_geom, (row,col))."""
    minx, miny, maxx, maxy = bounds
    tile_size_deg_x = (maxx - minx) / grid_size
    tile_size_deg_y = (maxy - miny) / grid_size

    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = minx + col * tile_size_deg_x
            y0 = miny + row * tile_size_deg_y
            tile = box(x0, y0, x0 + tile_size_deg_x, y0 + tile_size_deg_y)
            tiles.append((tile, (row, col)))
    return tiles

def should_exclude_tile(tile_geom):
    """Check if tile should be excluded based on elevation, snow cover, or forest cover."""
    try:
        region = ee.Geometry.Polygon(mapping(tile_geom)["coordinates"])
        
        # Get elevation data (SRTM)
        elevation = ee.Image('USGS/SRTMGL1_003').clip(region).select('elevation')
        mean_elev = elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=90,
            maxPixels=1e9
        ).get('elevation').getInfo()
        
        if mean_elev is not None and mean_elev > ELEVATION_THRESHOLD:
            return True, "High elevation"
        
        # Get snow cover (NDSI from MODIS)
        snow_img = ee.ImageCollection('MODIS/061/MOD10A1') \
            .filterBounds(region) \
            .filterDate(ee.Date(start_date.strftime('%Y-%m-%d')), ee.Date(end_date.strftime('%Y-%m-%d'))) \
            .first() \
            .clip(region)
        
        ndsi = snow_img.select('NDSI_Snow_Cover')
        mean_ndsi = ndsi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).get('NDSI_Snow_Cover').getInfo()
        
        if mean_ndsi is not None and mean_ndsi > SNOW_THRESHOLD:
            return True, "Snow cover"
        
        # Get forest cover (Global Forest Change)
        tree_cover = ee.Image('UMD/hansen/global_forest_change_2024_v1_12') \
            .clip(region) \
            .select('treecover2000')
        
        mean_cover = tree_cover.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).get('treecover2000').getInfo()
        
        if mean_cover is not None and mean_cover < FOREST_COVER_THRESHOLD:
            return True, "Low forest cover"
        
        return False, None
        
    except Exception as e:
        st.warning(f"Exclusion check failed for tile: {e}")
        return False, None

def remove_small_patches(prediction_array, min_size=5):
    """Remove small isolated prediction patches"""
    
    # Label connected components
    labeled_array, num_features = ndimage.label(prediction_array > 0.6)
    
    # Remove small components
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size < min_size:
            prediction_array[labeled_array == i] = 0
    
    return prediction_array    

def filter_agricultural_areas(tile_geom, prediction):
    """Reduce predictions in likely agricultural areas"""
    try:
        # Use land cover data to identify agricultural areas
        landcover = ee.Image("ESA/WorldCover/v200/2022") \
            .clip(ee.Geometry.Polygon(mapping(tile_geom)["coordinates"]))
        
        # Class 40: Cropland, Class 50: Urban, Class 60: Bare
        non_forest_classes = landcover.eq(40).Or(landcover.eq(50)).Or(landcover.eq(60))
        non_forest_ratio = non_forest_classes.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry.Polygon(mapping(tile_geom)["coordinates"]),
            scale=10
        ).getInfo().get('classification', 0)
        
        if non_forest_ratio > 0.7:  # Mostly non-forest area
            return prediction * 0.2  # Greatly reduce prediction
        
        return prediction
        
    except Exception:
        return prediction

def download_sentinel_tile(tile_geom, date_range, bands=SENTINEL_BANDS):
    """Download Sentinel-2 tile from Earth Engine. Returns path to a temporary GeoTIFF or None."""
    try:
        region = ee.Geometry.Polygon(mapping(tile_geom)["coordinates"])
        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(date_range[0], date_range[1])
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(bands)
            .median()
            .clip(region)
        )
        temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        temp_file.close()
        import geemap
        geemap.ee_export_image(
            ee_object=image,
            filename=temp_file.name,
            scale=10,
            region=region,
            file_per_band=False,
        )
        for _ in range(60):
            if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                break
            time.sleep(1)
        return temp_file.name
    except Exception as e:
        st.warning(f"Error downloading tile: {e}")
        return None

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for pre-trained model"""
    with rasterio.open(image_path) as src:
        bands = []
        # Use bands 4, 3, 2 for RGB (B4=Red, B3=Green, B2=Blue for Sentinel-2)
        band_indices = [4, 3, 2]  # RGB order for Sentinel-2
        
        for i in band_indices:
            if i <= src.count:  # Check if band exists
                band = src.read(
                    i,
                    out_shape=(target_size[1], target_size[0]),
                    resampling=Resampling.bilinear,
                )
                bands.append(band)
        
        # Ensure we have 3 channels (RGB)
        while len(bands) < 3:
            bands.append(np.zeros_like(bands[0]))
        
        image = np.stack(bands, axis=0)
        image = np.moveaxis(image, 0, -1)
        
        # Normalize for pre-trained models (0-255 range)
        image = image.astype(np.float32) / 255.0
        
        return image

def calculate_ndvi_for_tile(tile_geom):
    """Calculate NDVI for vegetation health assessment using MODIS NDVI (scaled by 10000)."""
    try:
        region = ee.Geometry.Polygon(mapping(tile_geom)["coordinates"])
        ndvi_image = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterBounds(region) \
            .filterDate(
                (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d"),
                datetime.utcnow().strftime("%Y-%m-%d")
            ) \
            .first() \
            .select("NDVI")

        ndvi_dict = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e13
        ).getInfo()

        val = ndvi_dict.get('NDVI', None)
        if val is None:
            return 0.5
        return val / 10000.0
    except Exception as e:
        st.warning(f"NDVI calculation failed: {e}")
        return 0.5

def predict_tile_forest_loss(tile_geom, date_range):
    """Predict forest loss probability for a tile using stacked models. Returns (prob, uncertainty, individual_preds)."""
    # First check if tile should be excluded
    exclude, reason = should_exclude_tile(tile_geom)
    if exclude:
        return -1.0, 0.0, {}  # Special value for excluded tiles
    
    tif_path = download_sentinel_tile(tile_geom, date_range)
    if not tif_path:
        return None, None, {}  # signal missing

    try:
        image = preprocess_image(tif_path)
        
        # Get predictions from each model
        individual_preds = {}
        uncertainties = {}
        
        for name, model in st.session_state.models.items():
            inp = np.expand_dims(image, 0).astype(np.float32)
            
            if any('dropout' in layer.name for layer in model.layers):
                # Use MC Dropout for uncertainty estimation
                preds = [model(inp, training=True).numpy()[0][0] for _ in range(5)]
                prob = float(np.mean(preds))
                uncertainty = float(np.std(preds))
            else:
                prob = float(model.predict(inp, verbose=0)[0][0])
                uncertainty = 0.0
                
            individual_preds[name] = prob
            uncertainties[name] = uncertainty

        # Combine predictions based on selected method
        if st.session_state.stacking_method == 'weighted_average':
            weights = getattr(st.session_state, 'model_weights', {'resnet': 0.6, 'efficientnet': 0.4})
            combined_prob = (individual_preds['resnet'] * weights['resnet'] + 
                           individual_preds['efficientnet'] * weights['efficientnet'])
        elif st.session_state.stacking_method == 'max':
            combined_prob = max(individual_preds.values())
        elif st.session_state.stacking_method == 'min':
            combined_prob = min(individual_preds.values())
        elif st.session_state.stacking_method == 'product':
            combined_prob = np.prod(list(individual_preds.values()))
        else:  # average
            combined_prob = np.mean(list(individual_preds.values()))
        
        # Calculate overall uncertainty
        model_uncertainty = abs(individual_preds['resnet'] - individual_preds['efficientnet'])
        avg_uncertainty = np.mean(list(uncertainties.values()))
        overall_uncertainty = (model_uncertainty + avg_uncertainty) / 2

        os.remove(tif_path)

        # Adjust probability by NDVI (reduce prob for healthy vegetation)
        ndvi = calculate_ndvi_for_tile(tile_geom)
        if ndvi > NDVI_THRESHOLD:
            combined_prob *= 0.66

        combined_prob = max(0.0, min(1.0, combined_prob))
        
        # Add validation steps only if we have a valid probability
        if combined_prob > 0.6:  # Only validate high-confidence predictions
            combined_prob = filter_agricultural_areas(tile_geom, combined_prob)
        
        return combined_prob, overall_uncertainty, individual_preds
        
    except Exception as e:
        st.warning(f"Prediction error for a tile: {e}")
        try:
            os.remove(tif_path)
        except:
            pass
        return None, None, {}

# Interpolation & mask utilities
def interpolate_probabilities(prob_grid, bounds, output_res=OUTPUT_RES, method='cubic'):
    rows, cols = prob_grid.shape
    x = np.linspace(bounds[0], bounds[2], cols)
    y = np.linspace(bounds[1], bounds[3], rows)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = prob_grid.ravel()

    gx = np.linspace(bounds[0], bounds[2], output_res)
    gy = np.linspace(bounds[1], bounds[3], output_res)
    gxx, gyy = np.meshgrid(gx, gy)

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 4:
        grid = np.zeros_like(gxx)
        return grid, gx, gy

    try:
        grid = griddata(points[valid_mask], values[valid_mask], (gxx, gyy), method=method, fill_value=np.nan)
        if np.isnan(grid).any():
            grid_nearest = griddata(points[valid_mask], values[valid_mask], (gxx, gyy), method='nearest')
            grid = np.where(np.isnan(grid), grid_nearest, grid)
    except Exception:
        grid = griddata(points[valid_mask], values[valid_mask], (gxx, gyy), method='nearest')

    grid = gaussian_filter(grid, sigma=GAUSSIAN_SIGMA)
    grid = np.clip(grid, 0.0, 1.0)
    return grid, gx, gy

def create_district_mask(district_geom_geojson, gx, gy):
    poly = shape(district_geom_geojson)
    transform = rasterio.transform.from_bounds(min(gx), min(gy), max(gx), max(gy), len(gx), len(gy))
    mask = rasterize(
        [(poly, 1)],
        out_shape=(len(gy), len(gx)),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

# -------------------------------
# Forest Cover Data Functions
# -------------------------------
def get_forest_cover_data(geometry):
    """Get forest cover data from Google Earth Engine, clipped to district"""
    try:
        # Load Hansen Global Forest Change dataset
        forest_cover = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
        
        # Select the tree cover band for 2000
        tree_cover = forest_cover.select('treecover2000')
        
        # Clip to the district geometry
        clipped_cover = tree_cover.clip(geometry)
        
        return clipped_cover
    except Exception as e:
        st.error(f"Error getting forest cover data: {str(e)}")
        return None

def download_forest_cover_data(geometry, bounds, output_res=OUTPUT_RES):
    """Download forest cover data as numpy array, clipped to district"""
    try:
        forest_data = get_forest_cover_data(geometry)
        if not forest_data:
            return None
        
        # Convert EE geometry to GeoJSON for rasterization
        geometry_info = geometry.getInfo()
        district_shape = shape(geometry_info)
        
        # Create a region for export (use the bounds)
        region = ee.Geometry.Rectangle(bounds)
        
        # Export the data at the target resolution
        temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        temp_file.close()
        
        import geemap
        
        # Calculate the scale based on bounds and output resolution
        width_deg = bounds[2] - bounds[0]
        height_deg = bounds[3] - bounds[1]
        scale_x = (width_deg * 111320) / output_res  # approximate meters per degree
        scale_y = (height_deg * 111320) / output_res
        
        # Use the larger scale to ensure we don't exceed resolution
        target_scale = max(scale_x, scale_y, 30)  # Minimum 30m (Hansen resolution)
        
        geemap.ee_export_image(
            ee_object=forest_data,
            filename=temp_file.name,
            scale=target_scale,
            region=region,
            file_per_band=False,
        )
        
        # Wait for download to complete
        for _ in range(30):
            if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                break
            time.sleep(1)
        
        # Read and resample the data to exact target resolution
        with rasterio.open(temp_file.name) as src:
            # Read and resample to target resolution
            cover_data = src.read(
                1,
                out_shape=(output_res, output_res),
                resampling=Resampling.bilinear
            )
            
        os.remove(temp_file.name)
        
        # Create a mask for the district at target resolution
        transform = rasterio.transform.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            output_res, output_res
        )
        
        district_mask = rasterize(
            [(district_shape, 1)],
            out_shape=(output_res, output_res),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        # Apply the mask - set areas outside district to NaN
        cover_data = cover_data.astype(np.float32)
        cover_data[district_mask == 0] = np.nan
        
        return cover_data
        
    except Exception as e:
        st.error(f"Error downloading forest cover data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# -------------------------------
# Combined Visualization Function
# -------------------------------
def create_combined_forest_map(forest_cover_data, loss_probability_data, district_geom, bounds):
    """Create a single map showing both forest cover and loss probability, clipped to district"""
    
    # Create district mask
    district_shape = shape(district_geom)
    transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        forest_cover_data.shape[1], forest_cover_data.shape[0]
    )
    district_mask = rasterize(
        [(district_shape, 1)],
        out_shape=forest_cover_data.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    # Apply mask to both datasets
    forest_cover_masked = forest_cover_data.copy()
    loss_prob_masked = loss_probability_data.copy()
    
    forest_cover_masked[district_mask == 0] = np.nan
    loss_prob_masked[district_mask == 0] = np.nan
    
    # Normalize data
    forest_cover_norm = np.clip(forest_cover_masked / 100.0, 0, 1)  # 0-100% to 0-1
    loss_prob_norm = np.clip(loss_prob_masked, 0, 1)  # Already 0-1
    
    # Create RGB image where:
    # - Red channel: Loss probability (higher = more red)
    # - Green channel: Forest cover (higher = more green)
    # - Blue channel: Minimal (just for visualization)
    
    red_channel = loss_prob_norm * 255
    green_channel = forest_cover_norm * 255
    blue_channel = np.zeros_like(red_channel)
    
    # Create RGBA image
    rgba_img = np.stack([red_channel, green_channel, blue_channel], axis=-1).astype(np.uint8)
    
    # Add alpha channel - transparent outside district and for very low cover
    alpha_channel = np.ones_like(red_channel) * 255
    alpha_channel[district_mask == 0] = 0  # Outside district
    alpha_channel[forest_cover_masked < 5] = 0  # Very low cover areas
    alpha_channel = alpha_channel.astype(np.uint8)
    
    rgba_img = np.dstack([rgba_img, alpha_channel])
    
    # Save to buffer
    pil_img = Image.fromarray(rgba_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    # Create Folium map
    centroid = district_shape.centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=10,
        tiles=None,
        control_scale=True
    )

    # Add satellite base
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Satellite View',
        overlay=False,
        control=True
    ).add_to(m)

    # Add raster overlay
    img_url = f"data:image/png;base64,{img_base64}"
    
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.8,
        name='Forest Analysis',
        attr='Forest Cover & Loss Analysis',
        zindex=10
    ).add_to(m)

    # District boundary
    folium.GeoJson(
        district_geom,
        name="District Boundary",
        style_function=lambda x: {
            'color': '#FFFF00',
            'weight': 3,
            'fillOpacity': 0.0,
            'dashArray': '5, 5'
        }
    ).add_to(m)

    MiniMap(position="bottomright").add_to(m)

    # Enhanced legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 20px; z-index: 1000;
                background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;
                font-size: 12px; width: 280px; color: white;">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">
            Forest Analysis Legend
        </div>
        <div style="margin-bottom: 5px;">
            <span style="background: linear-gradient(to right, #00ff00, #ff0000); 
                        display: inline-block; width: 100px; height: 15px; 
                        border: 1px solid #fff; margin-right: 10px;"></span>
            <span>Green: Forest Cover | Red: Loss Risk</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
            <div>High Cover + Low Risk</div>
            <div style="color: #00ff00;">🟢 Stable</div>
            <div>High Cover + High Risk</div>
            <div style="color: #ff0000;">🔴 Critical</div>
            <div>Low Cover + High Risk</div>
            <div style="color: #ff4500;">🟠 Degraded</div>
            <div>Low Cover + Low Risk</div>
            <div style="color: #d9d9d9;">⚪ Non-forest</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    
    return m

def create_summary_table(forest_cover_data, loss_interpolated, district_geom, bounds):
    """Create a detailed summary table of forest analysis results"""
    
    # Calculate basic statistics
    total_area_pixels = np.sum(~np.isnan(forest_cover_data))
    pixel_area_sqkm = calculate_pixel_area(bounds, forest_cover_data.shape)
    total_area_sqkm = total_area_pixels * pixel_area_sqkm
    
    # Forest cover categories
    cover_categories = {
        'Dense Forest (80-100%)': (80, 100),
        'High Forest (60-80%)': (60, 80),
        'Medium Forest (40-60%)': (40, 60),
        'Low Forest (20-40%)': (20, 40),
        'Very Low Forest (5-20%)': (5, 20),
        'Non-Forest (0-5%)': (0, 5)
    }
    
    # Loss risk categories
    loss_categories = {
        'Very High Risk (80-100%)': (0.8, 1.0),
        'High Risk (60-80%)': (0.6, 0.8),
        'Medium Risk (40-60%)': (0.4, 0.6),
        'Low Risk (20-40%)': (0.2, 0.4),
        'Very Low Risk (0-20%)': (0.0, 0.2)
    }
    
    # Critical areas (high cover + high risk)
    critical_mask = (forest_cover_data >= 50) & (loss_interpolated >= 0.6)
    
    summary_data = []
    
    # Forest cover summary
    for category, (min_val, max_val) in cover_categories.items():
        if max_val == 100:  # Handle the top range
            mask = (forest_cover_data >= min_val) & (forest_cover_data <= max_val)
        else:
            mask = (forest_cover_data >= min_val) & (forest_cover_data < max_val)
        
        area_pixels = np.sum(mask)
        area_sqkm = area_pixels * pixel_area_sqkm
        percentage = (area_pixels / total_area_pixels * 100) if total_area_pixels > 0 else 0
        
        summary_data.append({
            'Category': category,
            'Area (sq km)': area_sqkm,
            'Percentage': percentage,
            'Type': 'Forest Cover'
        })
    
    # Loss risk summary
    for category, (min_val, max_val) in loss_categories.items():
        if max_val == 1.0:  # Handle the top range
            mask = (loss_interpolated >= min_val) & (loss_interpolated <= max_val)
        else:
            mask = (loss_interpolated >= min_val) & (loss_interpolated < max_val)
        
        area_pixels = np.sum(mask & ~np.isnan(forest_cover_data))
        area_sqkm = area_pixels * pixel_area_sqkm
        percentage = (area_pixels / total_area_pixels * 100) if total_area_pixels > 0 else 0
        
        summary_data.append({
            'Category': category,
            'Area (sq km)': area_sqkm,
            'Percentage': percentage,
            'Type': 'Loss Risk'
        })
    
    # Critical areas summary
    critical_pixels = np.sum(critical_mask)
    critical_area_sqkm = critical_pixels * pixel_area_sqkm
    critical_percentage = (critical_pixels / total_area_pixels * 100) if total_area_pixels > 0 else 0
    
    summary_data.append({
        'Category': 'CRITICAL: High Cover + High Risk',
        'Area (sq km)': critical_area_sqkm,
        'Percentage': critical_percentage,
        'Type': 'Critical Areas'
    })
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    return df_summary, total_area_sqkm, critical_area_sqkm

def calculate_pixel_area(bounds, shape):
    """Calculate area of each pixel in square kilometers"""
    minx, miny, maxx, maxy = bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    
    # Approximate conversion (more accurate would use projection)
    # Average for Azerbaijan: 1° ≈ 111 km
    width_km = width_deg * 111 * np.cos(np.radians((miny + maxy) / 2))
    height_km = height_deg * 111
    
    pixel_width_km = width_km / shape[1]
    pixel_height_km = height_km / shape[0]
    
    return pixel_width_km * pixel_height_km

# Update the main analysis section to include the summary table
if st.button("Analyze Forest Cover & Loss"):
    if not st.session_state.model_loaded:
        st.error("Please wait for the model to load first")
        st.stop()
    with st.spinner("Running comprehensive forest analysis..."):
        # Get district geometry
        district_geom = get_district_geometry(selected_district)
        if not district_geom:
            st.error("Could not load district geometry")
            st.stop()

        try:
            bounds_coords = selected_geom.geometry().bounds().getInfo()['coordinates'][0]
            minx = min(pt[0] for pt in bounds_coords)
            miny = min(pt[1] for pt in bounds_coords)
            maxx = max(pt[0] for pt in bounds_coords)
            maxy = max(pt[1] for pt in bounds_coords)
        except Exception:
            coords = district_geom['coordinates'][0]
            xs = [c[0] for c in coords[0]]
            ys = [c[1] for c in coords[0]]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

        bounds = (minx, miny, maxx, maxy)

        # Get forest cover data
        with st.spinner("Downloading forest cover data..."):
            # Convert district geometry to EE geometry
            district_geom_ee = ee.Geometry(district_geom)
            forest_cover_data = download_forest_cover_data(district_geom_ee, bounds, OUTPUT_RES)

        if forest_cover_data is None:
            st.error("Failed to download forest cover data")
            st.stop()

        # Run loss prediction
        grid_size = COARSE_GRID_SIZE
        xs = np.linspace(minx, maxx, grid_size)
        ys = np.linspace(miny, maxy, grid_size)
        prob_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        
        date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        total_tiles = grid_size * grid_size
        
        with st.spinner("Predicting forest loss patterns..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    idx = i * grid_size + j
                    progress_bar.progress((idx + 1) / total_tiles)
                    
                    dx = (maxx - minx) / grid_size
                    dy = (maxy - miny) / grid_size
                    tile = box(x - dx/2, y - dy/2, x + dx/2, y + dy/2)
                    
                    prob, uncertainty, _ = predict_tile_forest_loss(tile, date_range)
                    
                    if prob == -1.0:  # Excluded tile
                        prob_grid[i, j] = 0.0
                    elif prob is None:
                        prob_grid[i, j] = 0.0
                    else:
                        prob_grid[i, j] = prob

        # Interpolate loss predictions
        loss_interpolated, gx, gy = interpolate_probabilities(prob_grid, bounds, output_res=OUTPUT_RES)

        # Resample forest cover data to match loss data resolution if needed
        if forest_cover_data.shape != loss_interpolated.shape:
            st.warning(f"Resampling forest cover data to match loss data resolution: {forest_cover_data.shape} -> {loss_interpolated.shape}")
            from scipy.ndimage import zoom
            zoom_factors = (loss_interpolated.shape[0] / forest_cover_data.shape[0], 
                           loss_interpolated.shape[1] / forest_cover_data.shape[1])
            forest_cover_data = zoom(forest_cover_data, zoom_factors, order=1)
    
            # Recreate mask for the new resolution
            transform = rasterio.transform.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3],
                loss_interpolated.shape[1], loss_interpolated.shape[0]
            )
            district_shape = shape(district_geom)
            district_mask = rasterize(
                [(district_shape, 1)],
                out_shape=loss_interpolated.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            forest_cover_data[district_mask == 0] = np.nan
        
        # Create combined map
        result_map = create_combined_forest_map(forest_cover_data, loss_interpolated, district_geom, bounds)
        
        # Display results
        st.success("Forest analysis complete!")
        folium_static(result_map, width=1000, height=700)
        
        # Create and display summary table
        with st.spinner("Generating summary report..."):
            df_summary, total_area, critical_area = create_summary_table(
                forest_cover_data, loss_interpolated, district_geom, bounds
            )
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_cover = np.nanmean(forest_cover_data)
            st.metric("Average Forest Cover", f"{avg_cover:.1f}%")
        
        with col2:
            avg_loss_risk = np.nanmean(loss_interpolated) * 100
            st.metric("Average Loss Risk", f"{avg_loss_risk:.1f}%")
        
        with col3:
            valid_forest_mask = forest_cover_data > 0
            if np.any(valid_forest_mask):
                critical_areas = np.sum((forest_cover_data > 50) & (loss_interpolated > 0.6)) / np.sum(valid_forest_mask) * 100
                st.metric("Critical Areas", f"{critical_areas:.1f}%")
            else:
                st.metric("Critical Areas", "0.0%")
        
        with col4:
            st.metric("Total Area", f"{total_area:.1f} km²")
        
        # Detailed summary table
        st.markdown("### 📊 Detailed Analysis Summary")
        
        # Forest Cover Summary
        st.markdown("#### 🌳 Forest Cover Distribution")
        cover_df = df_summary[df_summary['Type'] == 'Forest Cover'].copy()
        cover_df = cover_df.sort_values('Percentage', ascending=False)
        
        st.dataframe(
            cover_df[['Category', 'Area (sq km)', 'Percentage']].style.format({
                'Area (sq km)': '{:.1f}',
                'Percentage': '{:.1f}%'
            }).background_gradient(subset=['Percentage'], cmap='Greens'),
            use_container_width=True
        )
        
        # Loss Risk Summary
        st.markdown("#### ⚠️ Deforestation Risk Distribution")
        risk_df = df_summary[df_summary['Type'] == 'Loss Risk'].copy()
        risk_df = risk_df.sort_values('Percentage', ascending=False)
        
        st.dataframe(
            risk_df[['Category', 'Area (sq km)', 'Percentage']].style.format({
                'Area (sq km)': '{:.1f}',
                'Percentage': '{:.1f}%'
            }).background_gradient(subset=['Percentage'], cmap='Reds'),
            use_container_width=True
        )
        
        # Critical Areas
        st.markdown("#### 🚨 Critical Areas (Immediate Attention Needed)")
        critical_df = df_summary[df_summary['Type'] == 'Critical Areas'].copy()
        
        st.dataframe(
            critical_df[['Category', 'Area (sq km)', 'Percentage']].style.format({
                'Area (sq km)': '{:.1f}',
                'Percentage': '{:.1f}%'
            }).apply(lambda x: ['background-color: #ffcccc' if i == 2 else '' for i in range(len(x))]),
            use_container_width=True
        )
        
        # Export options
        st.markdown("---")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📥 Download Summary CSV"):
                csv = df_summary.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"forest_analysis_{selected_district}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("📊 Generate Detailed Report"):
                with st.expander("📈 Detailed Statistics"):
                    st.write("**Forest Health Indicators:**")
                    
                    # Additional metrics
                    healthy_forest = np.sum((forest_cover_data > 70) & (loss_interpolated < 0.3))
                    declining_forest = np.sum((forest_cover_data > 50) & (loss_interpolated > 0.6))
                    
                    st.metric("Healthy Forest Areas", f"{(healthy_forest/len(forest_cover_data.ravel())*100):.1f}%")
                    st.metric("Declining Forest Areas", f"{(declining_forest/len(forest_cover_data.ravel())*100):.1f}%")
                    
                    # Spatial patterns
                    st.write("**Spatial Patterns:**")
                    if critical_area > total_area * 0.1:  # More than 10% critical
                        st.warning("⚠️ Widespread deforestation pattern detected")
                    elif critical_area > total_area * 0.05:  # More than 5% critical
                        st.info("ℹ️ Moderate deforestation pattern detected")
                    else:
                        st.success("✅ Relatively stable forest pattern")
        
        st.session_state.map_created = True

# Add refresh button if map exists
if st.session_state.map_created and st.button("Refresh Analysis"):
    st.rerun()
