"""
COMPREHENSIVE DEBUGGING SOLUTION
Replace your entire overview.py with this version to identify the exact error location
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
import traceback
import sys

from app.core.data_processor import data_processor
from app.utils.config import config

# Import our new data-driven calculators
from app.core.metrics_calculator import metrics_calculator

# UPDATED IMPORTS - Using cached insights system
try:
    from app.core.groq_insights_generator import add_cache_controls, get_insights_with_cache
except ImportError:
    # Fallback if not available
    def get_insights_with_cache(*args, **kwargs):
        return []

logger = logging.getLogger(__name__)

def debug_print(message, obj=None):
    """Debug printing function"""
    print(f"🔍 DEBUG: {message}")
    if obj is not None:
        if hasattr(obj, 'dtype'):
            print(f"   Type: {type(obj)}, dtype: {obj.dtype}")
            if hasattr(obj, 'head'):
                print(f"   Sample: {obj.head().tolist()}")
        else:
            print(f"   Value: {obj}, Type: {type(obj)}")

def safe_function_wrapper(func_name, func, *args, **kwargs):
    """Wrapper to catch and debug function errors"""
    try:
        debug_print(f"Entering function: {func_name}")
        result = func(*args, **kwargs)
        debug_print(f"Successfully completed: {func_name}")
        return result
    except Exception as e:
        debug_print(f"ERROR in {func_name}: {str(e)}")
        debug_print(f"ERROR type: {type(e)}")
        
        # Print full traceback
        traceback.print_exc()
        
        # Show error in Streamlit
        st.error(f"❌ Error in {func_name}: {str(e)}")
        with st.expander(f"🔍 Full Error Details for {func_name}"):
            st.code(traceback.format_exc())
        
        # Re-raise with more context
        raise Exception(f"Error in {func_name}: {str(e)}") from e

def render_overview_page():
    """DEBUG VERSION: Render the main overview page with extensive debugging"""
    st.title("📊 Dashboard Overview (DEBUG MODE)")
    st.markdown("Real-time insights into cycling safety across your network")
    
    try:
        debug_print("Starting render_overview_page")
        
        # Load all datasets with debugging
        debug_print("Loading datasets...")
        all_data = safe_function_wrapper("load_all_datasets", data_processor.load_all_datasets, force_reload=True)
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        debug_print(f"Available datasets: {available_datasets}")
        
        if not available_datasets:
            debug_print("No datasets available, rendering no data message")
            render_no_data_message()
            return
        
        # Extract dataframes with debugging
        debug_print("Extracting dataframes...")
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # DEBUG: Check data types immediately after loading
        if routes_df is not None:
            debug_print("routes_df loaded", routes_df.shape)
            if 'popularity_rating' in routes_df.columns:
                debug_print("popularity_rating column", routes_df['popularity_rating'])
        
        if braking_df is not None:
            debug_print("braking_df loaded", braking_df.shape)
        
        if swerving_df is not None:
            debug_print("swerving_df loaded", swerving_df.shape)
        
        if time_series_df is not None:
            debug_print("time_series_df loaded", time_series_df.shape)
        
        # Add filters in sidebar with debugging
        debug_print("Rendering filters...")
        filters = safe_function_wrapper("render_overview_filters", render_overview_filters_debug, routes_df, time_series_df)
        debug_print(f"Filters created: {filters}")
        
        # Apply filters with debugging
        debug_print("Applying filters...")
        try:
            filtered_data = safe_function_wrapper("apply_overview_filters", apply_overview_filters_debug, routes_df, braking_df, swerving_df, time_series_df, filters)
            routes_df, braking_df, swerving_df, time_series_df = filtered_data
            debug_print("Filters applied successfully")
        except Exception as e:
            debug_print(f"Filter application failed: {e}")
            st.warning(f"⚠️ Error applying filters: {e}")
            # Continue with unfiltered data
        
        # Render main overview sections with debugging
        debug_print("Rendering key metrics...")
        try:
            safe_function_wrapper("render_key_metrics", render_key_metrics_debug, routes_df, braking_df, swerving_df, time_series_df)
        except Exception as e:
            st.error(f"⚠️ Error rendering key metrics: {e}")
        
        debug_print("Rendering AI insights...")
        try:
            safe_function_wrapper("render_ai_insights_section_cached", render_ai_insights_debug, routes_df, braking_df, swerving_df, time_series_df)
        except Exception as e:
            st.warning(f"⚠️ AI insights unavailable: {e}")
        
        debug_print("Overview page completed successfully")
        
    except Exception as e:
        debug_print(f"MAJOR ERROR in overview page: {e}")
        logger.error(f"Error in overview page: {e}")
        st.error("⚠️ An error occurred while loading the overview page.")
        st.info("Please check your data files and try refreshing the page.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(str(e))
            st.code(traceback.format_exc())
            st.button("🔄 Retry", key="overview_retry")

def render_overview_filters_debug(routes_df, time_series_df):
    """DEBUG VERSION: Render sidebar filters with extensive debugging"""
    debug_print("Starting render_overview_filters_debug")
    st.sidebar.markdown("## 🔍 Filters (DEBUG)")
    
    filters = {}
    
    # Date range filter (skip for debugging focus)
    debug_print("Setting up date filter...")
    filters['date_range'] = None
    
    # Route type filter (skip for debugging focus)
    debug_print("Setting up route type filter...")
    filters['route_type'] = 'All'
    
    # FOCUS ON POPULARITY FILTER - This is likely where the error occurs
    debug_print("Setting up popularity filter...")
    if routes_df is not None and not routes_df.empty:
        try:
            debug_print(f"routes_df shape: {routes_df.shape}")
            debug_print(f"routes_df columns: {routes_df.columns.tolist()}")
            
            if 'popularity_rating' in routes_df.columns:
                debug_print("popularity_rating column found")
                
                # Check original column
                original_column = routes_df['popularity_rating']
                debug_print("Original popularity_rating", original_column)
                
                # Check unique values
                unique_values = original_column.unique()
                debug_print(f"Unique values: {unique_values}")
                debug_print(f"Unique value types: {[type(x) for x in unique_values[:5]]}")
                
                # Try conversion step by step
                debug_print("Attempting pd.to_numeric conversion...")
                try:
                    numeric_popularity = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
                    debug_print("Numeric conversion successful", numeric_popularity)
                    
                    numeric_popularity_clean = numeric_popularity.dropna()
                    debug_print(f"After dropna: {len(numeric_popularity_clean)} values")
                    
                    if not numeric_popularity_clean.empty:
                        debug_print("Computing min/max...")
                        min_val_raw = numeric_popularity_clean.min()
                        max_val_raw = numeric_popularity_clean.max()
                        debug_print(f"Raw min: {min_val_raw} (type: {type(min_val_raw)})")
                        debug_print(f"Raw max: {max_val_raw} (type: {type(max_val_raw)})")
                        
                        # Convert to int
                        min_val = max(1, int(min_val_raw))
                        max_val = min(10, int(max_val_raw))
                        debug_print(f"Final min_val: {min_val}, max_val: {max_val}")
                        
                        # Create slider
                        debug_print("Creating slider...")
                        filters['min_popularity'] = st.sidebar.slider(
                            "📊 Minimum Route Popularity (DEBUG)",
                            min_value=min_val,
                            max_value=max_val,
                            value=min_val,
                            help="Filter routes by minimum popularity rating",
                            key="overview_popularity_filter"
                        )
                        debug_print(f"Slider created with value: {filters['min_popularity']}")
                        
                    else:
                        debug_print("No valid numeric values, using default")
                        filters['min_popularity'] = 1
                except Exception as numeric_error:
                    debug_print(f"Numeric conversion failed: {numeric_error}")
                    filters['min_popularity'] = 1
                    st.sidebar.error(f"Popularity filter error: {numeric_error}")
            else:
                debug_print("No popularity_rating column found")
                filters['min_popularity'] = 1
        except Exception as e:
            debug_print(f"Major error in popularity filter: {e}")
            filters['min_popularity'] = 1
            st.sidebar.error(f"Filter setup error: {e}")
    else:
        debug_print("routes_df is None or empty")
        filters['min_popularity'] = 1
    
    debug_print(f"Final filters: {filters}")
    return filters

def apply_overview_filters_debug(routes_df, braking_df, swerving_df, time_series_df, filters):
    """DEBUG VERSION: Apply filters with extensive debugging"""
    debug_print("Starting apply_overview_filters_debug")
    debug_print(f"Input filters: {filters}")
    
    # Skip date and route type filters for debugging focus
    
    # FOCUS ON POPULARITY FILTER
    if routes_df is not None and filters.get('min_popularity'):
        debug_print(f"Applying popularity filter: {filters['min_popularity']}")
        
        try:
            if 'popularity_rating' in routes_df.columns:
                debug_print("popularity_rating column exists")
                
                # Make a copy
                routes_df = routes_df.copy()
                debug_print("Made copy of routes_df")
                
                # Check before conversion
                debug_print("Before conversion", routes_df['popularity_rating'])
                
                # Force conversion
                debug_print("Converting to numeric...")
                routes_df['popularity_rating'] = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
                debug_print("After conversion", routes_df['popularity_rating'])
                
                # Remove NaN
                before_count = len(routes_df)
                routes_df = routes_df.dropna(subset=['popularity_rating'])
                after_count = len(routes_df)
                debug_print(f"Dropped NaN: {before_count} -> {after_count} rows")
                
                if not routes_df.empty:
                    # THE CRITICAL COMPARISON
                    debug_print("About to perform comparison...")
                    filter_value = filters['min_popularity']
                    debug_print(f"Filter value: {filter_value} (type: {type(filter_value)})")
                    debug_print(f"Column dtype: {routes_df['popularity_rating'].dtype}")
                    
                    # Check for any problematic values
                    sample_values = routes_df['popularity_rating'].head().tolist()
                    debug_print(f"Sample values for comparison: {sample_values}")
                    debug_print(f"Sample value types: {[type(x) for x in sample_values]}")
                    
                    # Perform comparison with explicit debugging
                    debug_print("Executing: routes_df['popularity_rating'] >= filter_value")
                    mask = routes_df['popularity_rating'] >= filter_value
                    debug_print("Comparison successful!")
                    
                    routes_df = routes_df[mask]
                    debug_print(f"Filtering completed: {len(routes_df)} rows remaining")
                else:
                    debug_print("No rows remaining after dropna")
        except Exception as e:
            debug_print(f"ERROR in popularity filtering: {e}")
            traceback.print_exc()
            raise e
    
    debug_print("apply_overview_filters_debug completed")
    return routes_df, braking_df, swerving_df, time_series_df

def render_key_metrics_debug(routes_df, braking_df, swerving_df, time_series_df):
    """DEBUG VERSION: Render key metrics"""
    debug_print("Starting render_key_metrics_debug")
    st.markdown("### 📊 Key Performance Metrics (DEBUG)")
    
    try:
        # Calculate metrics using the metrics calculator
        debug_print("Calling metrics calculator...")
        metrics = metrics_calculator.calculate_all_overview_metrics(
            routes_df, braking_df, swerving_df, time_series_df
        )
        debug_print(f"Metrics calculated: {list(metrics.keys())}")
        
        # Simple metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_routes = metrics.get('total_routes', 0)
            st.metric("🛣️ Total Routes", total_routes)
        
        with col2:
            total_cyclists = metrics.get('total_cyclists', 0)
            st.metric("🚴 Total Cyclists", total_cyclists)
        
        with col3:
            safety_score = metrics.get('safety_score', 0)
            st.metric("🛡️ Safety Score", f"{safety_score:.1f}/10")
        
        with col4:
            total_hotspots = metrics.get('total_hotspots', 0)
            st.metric("🔥 Total Hotspots", total_hotspots)
        
        debug_print("Key metrics rendered successfully")
        
    except Exception as e:
        debug_print(f"ERROR in render_key_metrics: {e}")
        raise e

def render_ai_insights_debug(routes_df, braking_df, swerving_df, time_series_df):
    """DEBUG VERSION: Render AI insights (simplified)"""
    debug_print("Starting render_ai_insights_debug")
    st.markdown("### 🤖 AI Insights (DEBUG - Simplified)")
    st.info("AI insights temporarily simplified for debugging")

def render_no_data_message():
    """Render message when no data is available"""
    st.warning("⚠️ No data available for the overview dashboard.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Upload your data files** to the `data/raw/` directory
    2. **Use the Data Setup page** to validate and load your data
    3. **Return to this overview** to see your cycling safety insights
    """)

# INSTRUCTIONS FOR USE:
"""
TO USE THIS DEBUG VERSION:

1. BACKUP your current overview.py file
2. REPLACE your overview.py with this entire debug version
3. Run your Streamlit app
4. Check the console output (terminal) for detailed DEBUG messages
5. The app will show exactly where the error occurs and what data types are involved

This will pinpoint the EXACT line causing the string vs int comparison error.
"""
