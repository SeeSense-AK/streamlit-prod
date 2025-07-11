"""
Data Setup Page for SeeSense Dashboard
Helps users set up their CSV data files
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

from ..core.data_processor import data_processor
from ..utils.config import config

logger = logging.getLogger(__name__)


def render_data_setup_page():
    """Render the data setup page"""
    st.title("📁 Data Setup")
    st.markdown("Set up your cycling safety data files to power the dashboard.")
    
    # Check current data status
    data_requirements = data_processor.check_data_requirements()
    data_status = data_processor.get_data_status()
    
    # Show setup status
    render_setup_status(data_requirements, data_status)
    
    # Show data requirements
    render_data_requirements()
    
    # Show file upload interface
    render_file_upload()
    
    # Show data validation results
    if data_status['available_datasets'] > 0:
        render_data_validation_results(data_status)
    
    # Show sample data formats
    render_sample_formats()


def render_setup_status(data_requirements: Dict[str, Any], data_status: Dict[str, Any]):
    """Render the current setup status"""
    st.markdown("## 📊 Setup Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Available Files",
            f"{len(data_requirements['available_files'])}/{data_status['total_datasets']}",
            f"{len(data_requirements['available_files']) - len(data_requirements['missing_files'])}"
        )
    
    with col2:
        total_size = sum(f['size_mb'] for f in data_requirements['available_files'])
        st.metric(
            "Total Data Size",
            f"{total_size:.1f} MB"
        )
    
    with col3:
        setup_percentage = (len(data_requirements['available_files']) / data_status['total_datasets']) * 100
        st.metric(
            "Setup Complete",
            f"{setup_percentage:.0f}%"
        )
    
    # Setup status indicator
    if data_requirements['setup_complete']:
        st.success("✅ All required data files are present and ready to use!")
        if st.button("🔄 Reload All Data", type="primary"):
            data_processor.load_all_datasets(force_reload=True)
            st.success("Data reloaded successfully!")
            st.experimental_rerun()
    else:
        st.warning(f"⚠️ {len(data_requirements['missing_files'])} data file(s) missing")
        
        # Show missing files
        if data_requirements['missing_files']:
            st.markdown("### Missing Files:")
            for file_info in data_requirements['missing_files']:
                st.markdown(f"- **{file_info['filename']}** (for {file_info['dataset']} data)")


def render_data_requirements():
    """Render data requirements and setup instructions"""
    st.markdown("## 📋 Data Requirements")
    
    with st.expander("📖 Setup Instructions", expanded=True):
        data_requirements = data_processor.check_data_requirements()
        
        st.markdown("### Step-by-Step Setup:")
        
        for i, instruction in enumerate(data_requirements['setup_instructions'], 1):
            if instruction.startswith('📁') or instruction.startswith('💡'):
                st.markdown(f"**{instruction}**")
            elif instruction.startswith(('1.', '2.', '3.', '4.')):
                st.markdown(f"**{instruction}**")
            elif instruction.strip() == "":
                st.markdown("")
            else:
                st.markdown(instruction)
        
        # Show data directory path
        st.markdown("### 📂 Data Directory Location:")
        st.code(str(data_processor.raw_data_path), language=None)
        
        # Copy button for path
        if st.button("📋 Copy Path to Clipboard"):
            st.write("Path copied! (Note: Actual clipboard copying requires additional setup)")


def render_file_upload():
    """Render file upload interface"""
    st.markdown("## 📤 Upload Data Files")
    
    st.info("""
    **Note**: File upload feature stores files temporarily. For production deployment, 
    place your CSV files directly in the data directory shown above.
    """)
    
    # File upload for each dataset
    datasets = {
        'routes': 'Routes Data',
        'braking_hotspots': 'Braking Hotspots',
        'swerving_hotspots': 'Swerving Hotspots',
        'time_series': 'Time Series Data'
    }
    
    uploaded_files = {}
    
    for dataset_key, dataset_name in datasets.items():
        expected_filename = data_processor.datasets[dataset_key]
        
        uploaded_file = st.file_uploader(
            f"Upload {dataset_name} ({expected_filename})",
            type=['csv'],
            key=f"upload_{dataset_key}",
            help=f"Upload your {dataset_name.lower()} CSV file"
        )
        
        if uploaded_file is not None:
            uploaded_files[dataset_key] = uploaded_file
    
    # Process uploaded files
    if uploaded_files and st.button("💾 Save Uploaded Files", type="primary"):
        success_count = 0
        
        for dataset_key, uploaded_file in uploaded_files.items():
            try:
                # Save file to data directory
                file_path = data_processor.raw_data_path / data_processor.datasets[dataset_key]
                
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Saved {uploaded_file.name}")
                success_count += 1
                
            except Exception as e:
                st.error(f"❌ Failed to save {uploaded_file.name}: {str(e)}")
        
        if success_count > 0:
            st.success(f"Successfully saved {success_count} file(s)! Refresh the page to load the data.")


def render_data_validation_results(data_status: Dict[str, Any]):
    """Render data validation results for available files"""
    st.markdown("## ✅ Data Validation Results")
    
    for dataset_name, status_info in data_status['datasets'].items():
        if not status_info['file_exists']:
            continue
        
        with st.expander(f"📄 {dataset_name.replace('_', ' ').title()}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**File Information:**")
                st.markdown(f"- **Filename**: {status_info['filename']}")
                st.markdown(f"- **Size**: {status_info['file_size_mb']:.2f} MB")
                if status_info['last_modified']:
                    st.markdown(f"- **Last Modified**: {status_info['last_modified'].strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                st.markdown("**Status:**")
                
                if status_info['loaded']:
                    st.success("✅ Loaded successfully")
                    st.markdown(f"- **Rows**: {status_info.get('row_count', 'N/A'):,}")
                    st.markdown(f"- **Columns**: {status_info.get('column_count', 'N/A')}")
                else:
                    st.warning("⚠️ Not loaded yet")
            
            # Try to load and validate the dataset
            if st.button(f"🔍 Validate {dataset_name}", key=f"validate_{dataset_name}"):
                try:
                    df, metadata = data_processor.load_dataset(dataset_name, force_reload=True)
                    
                    if df is not None:
                        st.success("✅ Validation passed!")
                        
                        # Show summary
                        st.markdown("**Data Summary:**")
                        summary = metadata['data_summary']
                        st.markdown(f"- **Rows**: {summary['row_count']:,}")
                        st.markdown(f"- **Columns**: {summary['column_count']}")
                        st.markdown(f"- **Memory Usage**: {summary['memory_usage_mb']:.2f} MB")
                        
                        # Show sample data
                        st.markdown("**Sample Data (first 5 rows):**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                    else:
                        st.error("❌ Validation failed!")
                        
                        if metadata['validation_errors']:
                            st.markdown("**Errors:**")
                            for error in metadata['validation_errors']:
                                st.error(error)
                    
                    if metadata['validation_warnings']:
                        st.markdown("**Warnings:**")
                        for warning in metadata['validation_warnings']:
                            st.warning(warning)
                            
                except Exception as e:
                    st.error(f"Failed to validate {dataset_name}: {str(e)}")


def render_sample_formats():
    """Render sample data formats and schemas"""
    st.markdown("## 📝 Data Format Examples")
    
    st.markdown("""
    Your CSV files should follow these formats. Click on each section to see the expected structure:
    """)
    
    # Routes data format
    with st.expander("🛣️ Routes Data Format (routes.csv)", expanded=False):
        st.markdown("""
        **Required columns:**
        - `route_id`: Unique identifier for each route
        - `start_lat`, `start_lon`: Starting coordinates
        - `end_lat`, `end_lon`: Ending coordinates  
        - `distinct_cyclists`: Number of unique cyclists using this route
        - `days_active`: Number of days the route has been active
        - `popularity_rating`: Rating from 1-10
        - `avg_speed`: Average speed in km/h
        - `avg_duration`: Average trip duration in minutes
        - `route_type`: Category (Commute, Leisure, Exercise, Mixed)
        - `has_bike_lane`: Boolean (True/False)
        
        **Optional columns:**
        - `distance_km`: Route distance (calculated if not provided)
        """)
        
        # Sample data
        sample_routes = pd.DataFrame({
            'route_id': ['R001', 'R002', 'R003'],
            'start_lat': [51.5074, 51.5090, 51.5100],
            'start_lon': [-0.1278, -0.1280, -0.1285],
            'end_lat': [51.5084, 51.5100, 51.5110],
            'end_lon': [-0.1288, -0.1290, -0.1295],
            'distinct_cyclists': [150, 89, 203],
            'days_active': [30, 25, 45],
            'popularity_rating': [8, 6, 9],
            'avg_speed': [18.5, 16.2, 20.1],
            'avg_duration': [25.3, 18.7, 32.1],
            'route_type': ['Commute', 'Leisure', 'Exercise'],
            'has_bike_lane': [True, False, True]
        })
        st.dataframe(sample_routes, use_container_width=True)
    
    # Braking hotspots format
    with st.expander("🛑 Braking Hotspots Format (braking_hotspots.csv)", expanded=False):
        st.markdown("""
        **Required columns:**
        - `hotspot_id`: Unique identifier
        - `lat`, `lon`: Coordinates of the hotspot
        - `intensity`: Intensity score (0-10)
        - `incidents_count`: Number of braking incidents
        - `avg_deceleration`: Average deceleration in m/s²
        - `road_type`: Type of road (Junction, Crossing, Roundabout, Straight)
        - `date_recorded`: Date when data was recorded (YYYY-MM-DD)
        
        **Optional columns:**
        - `surface_quality`: Road surface quality
        - `severity_score`: Calculated severity score
        """)
        
        sample_braking = pd.DataFrame({
            'hotspot_id': ['BRK001', 'BRK002', 'BRK003'],
            'lat': [51.5074, 51.5090, 51.5100],
            'lon': [-0.1278, -0.1280, -0.1285],
            'intensity': [8.5, 6.2, 9.1],
            'incidents_count': [45, 23, 67],
            'avg_deceleration': [5.2, 3.8, 6.1],
            'road_type': ['Junction', 'Crossing', 'Junction'],
            'date_recorded': ['2024-01-15', '2024-01-16', '2024-01-17']
        })
        st.dataframe(sample_braking, use_container_width=True)
    
    # Swerving hotspots format
    with st.expander("↔️ Swerving Hotspots Format (swerving_hotspots.csv)", expanded=False):
        st.markdown("""
        **Required columns:**
        - `hotspot_id`: Unique identifier
        - `lat`, `lon`: Coordinates of the hotspot
        - `intensity`: Intensity score (0-10)
        - `incidents_count`: Number of swerving incidents
        - `avg_lateral_movement`: Average lateral movement in meters
        - `road_type`: Type of road (Junction, Crossing, Roundabout, Straight)
        - `date_recorded`: Date when data was recorded (YYYY-MM-DD)
        
        **Optional columns:**
        - `obstruction_present`: Whether obstruction is present (Yes/No)
        - `cause_category`: Categorized cause of swerving
        """)
        
        sample_swerving = pd.DataFrame({
            'hotspot_id': ['SWV001', 'SWV002', 'SWV003'],
            'lat': [51.5074, 51.5090, 51.5100],
            'lon': [-0.1278, -0.1280, -0.1285],
            'intensity': [7.3, 5.8, 8.9],
            'incidents_count': [32, 18, 54],
            'avg_lateral_movement': [1.2, 0.8, 1.5],
            'road_type': ['Straight', 'Junction', 'Crossing'],
            'obstruction_present': ['Yes', 'No', 'Yes'],
            'date_recorded': ['2024-01-15', '2024-01-16', '2024-01-17']
        })
        st.dataframe(sample_swerving, use_container_width=True)
    
    # Time series format
    with st.expander("📈 Time Series Format (time_series.csv)", expanded=False):
        st.markdown("""
        **Required columns:**
        - `date`: Date (YYYY-MM-DD format)
        - `total_rides`: Total number of rides
        - `incidents`: Number of safety incidents
        - `avg_speed`: Average speed in km/h
        - `avg_braking_events`: Average braking events per ride
        - `avg_swerving_events`: Average swerving events per ride
        
        **Optional columns:**
        - `precipitation_mm`: Daily precipitation in mm
        - `temperature`: Temperature in °C
        - `wind_speed`: Wind speed in km/h
        - `visibility_km`: Visibility in km
        """)
        
        sample_timeseries = pd.DataFrame({
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'total_rides': [1250, 1180, 1340],
            'incidents': [15, 12, 18],
            'avg_speed': [18.5, 17.2, 19.1],
            'avg_braking_events': [2.3, 2.1, 2.5],
            'avg_swerving_events': [1.8, 1.6, 2.0],
            'precipitation_mm': [0.0, 2.5, 0.5],
            'temperature': [15.2, 12.8, 16.5]
        })
        st.dataframe(sample_timeseries, use_container_width=True)
    
    # Download template files
    st.markdown("### 📥 Download Template Files")
    st.markdown("You can download template CSV files with the correct structure:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Routes Template"):
            st.download_button(
                "Download routes.csv template",
                sample_routes.to_csv(index=False),
                "routes_template.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("🛑 Braking Template"):
            st.download_button(
                "Download braking_hotspots.csv template",
                sample_braking.to_csv(index=False),
                "braking_hotspots_template.csv",
                "text/csv"
            )
    
    with col3:
        if st.button("↔️ Swerving Template"):
            st.download_button(
                "Download swerving_hotspots.csv template",
                sample_swerving.to_csv(index=False),
                "swerving_hotspots_template.csv",
                "text/csv"
            )
    
    with col4:
        if st.button("📈 Time Series Template"):
            st.download_button(
                "Download time_series.csv template",
                sample_timeseries.to_csv(index=False),
                "time_series_template.csv",
                "text/csv"
            )
