"""
Data processing pipeline for SeeSense Dashboard - Production Version
Handles only real CSV data, no synthetic data generation
ENHANCED VERSION with centralized data type cleaning to prevent comparison errors
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import streamlit as st

from utils.config import config
from utils.validators import get_data_summary

logger = logging.getLogger(__name__)


class DataProcessor:
    """Production data processing class - real CSV data only with centralized data cleaning"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.raw_data_path = config.get_data_path("raw")
        self.processed_data_path = config.get_data_path("processed")
        self.cache_ttl = config.cache_ttl
        
        # Ensure directories exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'routes': config.get('data.files.routes', 'routes.csv'),
            'braking_hotspots': config.get('data.files.braking_hotspots', 'braking_hotspots.csv'),
            'swerving_hotspots': config.get('data.files.swerving_hotspots', 'swerving_hotspots.csv'),
            'time_series': config.get('data.files.time_series', 'time_series.csv')
        }
        
        self._data_cache = {}
        self._cache_timestamps = {}
    
    def standardize_data_types(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Centralized data type standardization for all datasets.
        This function runs immediately after CSV loading to ensure consistent data types.
        """
        if df is None or df.empty:
            return df
        
        logger.info(f"Standardizing data types for {dataset_name}")
        df_clean = df.copy()
        
        # Define expected data types for each dataset
        type_definitions = {
            'routes': {
                'route_id': 'string',
                'start_lat': 'float',
                'start_lon': 'float', 
                'end_lat': 'float',
                'end_lon': 'float',
                'distinct_cyclists': 'int',
                'days_active': 'int',
                'popularity_rating': 'int',  # THIS IS KEY - ensures it's always int
                'avg_speed': 'float',
                'avg_duration': 'float',
                'route_type': 'string',
                'has_bike_lane': 'boolean',
                'distance_km': 'float'
            },
            'braking_hotspots': {
                'hotspot_id': 'string',
                'lat': 'float',
                'lon': 'float',
                'intensity': 'float',
                'incidents_count': 'int',
                'avg_deceleration': 'float',
                'road_type': 'string',
                'surface_quality': 'string',
                'date_recorded': 'datetime',
                'severity_score': 'float'
            },
            'swerving_hotspots': {
                'hotspot_id': 'string',
                'lat': 'float',
                'lon': 'float',
                'intensity': 'float',
                'incidents_count': 'int',
                'avg_lateral_movement': 'float',
                'road_type': 'string',
                'obstruction_present': 'string',
                'date_recorded': 'datetime',
                'severity_score': 'float'
            },
            'time_series': {
                'date': 'datetime',
                'total_rides': 'int',
                'daily_rides': 'int',
                'incidents': 'int',
                'avg_speed': 'float',
                'safety_score': 'float',
                'incident_rate': 'float',
                'precipitation_mm': 'float',
                'temperature': 'float'
            }
        }
        
        expected_types = type_definitions.get(dataset_name, {})
        
        for column, expected_type in expected_types.items():
            if column not in df_clean.columns:
                continue
                
            try:
                original_dtype = df_clean[column].dtype
                
                if expected_type == 'int':
                    # Clean string data first
                    if df_clean[column].dtype == 'object':
                        df_clean[column] = df_clean[column].astype(str).str.strip()
                        df_clean[column] = df_clean[column].replace(['', 'nan', 'NaN', 'null', 'NULL', 'None'], pd.NA)
                    
                    # Convert to float first to handle decimal strings, then to int
                    numeric_series = pd.to_numeric(df_clean[column], errors='coerce')
                    df_clean[column] = numeric_series.round().astype('Int64')  # Nullable integer
                    
                elif expected_type == 'float':
                    # Clean string data first
                    if df_clean[column].dtype == 'object':
                        df_clean[column] = df_clean[column].astype(str).str.strip()
                        df_clean[column] = df_clean[column].replace(['', 'nan', 'NaN', 'null', 'NULL', 'None'], pd.NA)
                    
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                    
                elif expected_type == 'boolean':
                    # Comprehensive boolean mapping
                    bool_map = {
                        'true': True, 'false': False,
                        'True': True, 'False': False,
                        'TRUE': True, 'FALSE': False,
                        'yes': True, 'no': False,
                        'Yes': True, 'No': False,
                        'YES': True, 'NO': False,
                        'y': True, 'n': False,
                        'Y': True, 'N': False,
                        1: True, 0: False,
                        '1': True, '0': False,
                        1.0: True, 0.0: False
                    }
                    df_clean[column] = df_clean[column].map(bool_map)
                    
                elif expected_type == 'datetime':
                    df_clean[column] = pd.to_datetime(df_clean[column], errors='coerce')
                    
                elif expected_type == 'string':
                    df_clean[column] = df_clean[column].astype(str)
                    df_clean[column] = df_clean[column].replace(['nan', 'NaN', 'None'], pd.NA)
                
                # Log successful conversions
                if original_dtype != df_clean[column].dtype:
                    na_count = df_clean[column].isna().sum()
                    total_count = len(df_clean[column])
                    success_rate = (total_count - na_count) / total_count * 100 if total_count > 0 else 0
                    logger.info(f"  {column}: {original_dtype} → {df_clean[column].dtype} (Success: {success_rate:.1f}%)")
            
            except Exception as e:
                logger.error(f"Failed to convert column '{column}' to {expected_type}: {e}")
                # Continue with original data if conversion fails
        
        # Validate critical numeric columns for comparisons
        critical_numeric_columns = {
            'routes': ['popularity_rating', 'distinct_cyclists', 'days_active'],
            'braking_hotspots': ['intensity', 'incidents_count', 'severity_score'],
            'swerving_hotspots': ['intensity', 'incidents_count', 'severity_score'],
            'time_series': ['total_rides', 'incidents', 'safety_score']
        }
        
        if dataset_name in critical_numeric_columns:
            for col in critical_numeric_columns[dataset_name]:
                if col in df_clean.columns:
                    # Ensure these are definitely numeric
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    na_count = df_clean[col].isna().sum()
                    if na_count > 0:
                        logger.warning(f"  Critical column '{col}' has {na_count} NaN values after conversion")
        
        logger.info(f"Data type standardization completed for {dataset_name}")
        return df_clean

    def apply_data_constraints(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Apply data constraints and validation rules after type conversion
        """
        if df is None or df.empty:
            return df
        
        df_constrained = df.copy()
        
        # Define constraints for each dataset
        constraints = {
            'routes': {
                'start_lat': {'min': -90, 'max': 90},
                'start_lon': {'min': -180, 'max': 180},
                'end_lat': {'min': -90, 'max': 90},
                'end_lon': {'min': -180, 'max': 180},
                'distinct_cyclists': {'min': 1},
                'days_active': {'min': 1},
                'popularity_rating': {'min': 1, 'max': 10},
                'avg_speed': {'min': 0, 'max': 100},
                'route_type': {'values': ["Commute", "Leisure", "Exercise", "Mixed"]}
            },
            'braking_hotspots': {
                'lat': {'min': -90, 'max': 90},
                'lon': {'min': -180, 'max': 180},
                'intensity': {'min': 0, 'max': 10},
                'incidents_count': {'min': 0},
                'avg_deceleration': {'min': 0},
                'road_type': {'values': ["Junction", "Crossing", "Roundabout", "Straight", "Other"]}
            },
            'swerving_hotspots': {
                'lat': {'min': -90, 'max': 90},
                'lon': {'min': -180, 'max': 180},
                'intensity': {'min': 0, 'max': 10},
                'incidents_count': {'min': 0},
                'avg_lateral_movement': {'min': 0}
            },
            'time_series': {
                'total_rides': {'min': 0},
                'daily_rides': {'min': 0},
                'incidents': {'min': 0},
                'safety_score': {'min': 0, 'max': 10},
                'incident_rate': {'min': 0}
            }
        }
        
        dataset_constraints = constraints.get(dataset_name, {})
        original_count = len(df_constrained)
        
        for column, constraint_dict in dataset_constraints.items():
            if column not in df_constrained.columns:
                continue
            
            # Apply min/max constraints (only for numeric columns)
            if 'min' in constraint_dict:
                min_val = constraint_dict['min']
                before_count = len(df_constrained)
                df_constrained = df_constrained[
                    (df_constrained[column].isna()) | (df_constrained[column] >= min_val)
                ]
                removed = before_count - len(df_constrained)
                if removed > 0:
                    logger.info(f"  Removed {removed} rows with {column} < {min_val}")
            
            if 'max' in constraint_dict:
                max_val = constraint_dict['max']
                before_count = len(df_constrained)
                df_constrained = df_constrained[
                    (df_constrained[column].isna()) | (df_constrained[column] <= max_val)
                ]
                removed = before_count - len(df_constrained)
                if removed > 0:
                    logger.info(f"  Removed {removed} rows with {column} > {max_val}")
            
            # Apply categorical constraints
            if 'values' in constraint_dict:
                allowed_values = constraint_dict['values']
                before_count = len(df_constrained)
                df_constrained = df_constrained[
                    (df_constrained[column].isna()) | (df_constrained[column].isin(allowed_values))
                ]
                removed = before_count - len(df_constrained)
                if removed > 0:
                    logger.info(f"  Removed {removed} rows with invalid {column} values")
        
        final_count = len(df_constrained)
        if original_count != final_count:
            logger.info(f"Constraint application: {original_count} → {final_count} rows ({original_count - final_count} removed)")
        
        return df_constrained

    def _is_cache_valid(self, dataset_name: str) -> bool:
        """Check if cached data is still valid"""
        if dataset_name not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[dataset_name]
        return (datetime.now() - cache_time) < timedelta(seconds=self.cache_ttl)

    def _process_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Process dataset after loading and cleaning"""
        if df is None or df.empty:
            return df
        
        df_processed = df.copy()
        
        # Dataset-specific processing
        if dataset_name == 'routes':
            # Calculate distance if missing
            if 'distance_km' not in df_processed.columns or df_processed['distance_km'].isna().all():
                df_processed['distance_km'] = self._calculate_route_distance(df_processed)
        
        elif dataset_name in ['braking_hotspots', 'swerving_hotspots']:
            # Calculate severity score if missing
            if 'severity_score' not in df_processed.columns or df_processed['severity_score'].isna().all():
                df_processed['severity_score'] = self._calculate_severity_score(df_processed)
        
        return df_processed

    def _calculate_route_distance(self, routes_df: pd.DataFrame) -> pd.Series:
        """Calculate route distance using Haversine formula"""
        try:
            # Simple Haversine distance calculation
            lat1, lon1 = np.radians(routes_df['start_lat']), np.radians(routes_df['start_lon'])
            lat2, lon2 = np.radians(routes_df['end_lat']), np.radians(routes_df['end_lon'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # Earth radius in km
            
            return distance
        except Exception as e:
            logger.warning(f"Error calculating route distance: {e}")
            return pd.Series([5.0] * len(routes_df))  # Default 5km

    def _calculate_severity_score(self, hotspots_df: pd.DataFrame) -> pd.Series:
        """Calculate severity score for hotspots"""
        try:
            if 'intensity' in hotspots_df.columns and 'incidents_count' in hotspots_df.columns:
                # Weighted combination of intensity and incident count
                intensity_norm = hotspots_df['intensity'] / 10  # Normalize to 0-1
                incidents_norm = np.log1p(hotspots_df['incidents_count']) / 10  # Log scale
                severity = (intensity_norm * 0.7 + incidents_norm * 0.3) * 10
                return severity.clip(0, 10)
            else:
                return pd.Series([5.0] * len(hotspots_df))  # Default severity
        except Exception as e:
            logger.warning(f"Error calculating severity score: {e}")
            return pd.Series([5.0] * len(hotspots_df))

    def _get_processing_info(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Get processing information for metadata"""
        info = {
            'dataset_type': dataset_name,
            'processing_timestamp': datetime.now(),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_data': df.isnull().sum().sum(),
            'data_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        return info

    @st.cache_data
    def load_dataset(_self, dataset_name: str, force_reload: bool = False) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load and process a dataset from CSV file with comprehensive data cleaning
        BYPASSES the problematic clean_dataframe function from validators.py
        """
        if dataset_name not in _self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
        # Check cache first (if not forcing reload)
        if not force_reload and _self._is_cache_valid(dataset_name):
            logger.info(f"Loading {dataset_name} from cache")
            return _self._data_cache[dataset_name]
    
        # Load from CSV
        csv_file = _self.raw_data_path / _self.datasets[dataset_name]
    
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            metadata = {
                'dataset_name': dataset_name,
                'source_file': str(csv_file),
                'load_timestamp': datetime.now(),
                'validation_errors': [f"File not found: {csv_file}"],
                'validation_warnings': [],
                'data_summary': {},
                'processing_info': {},
                'status': 'file_not_found'
            }
            return None, metadata
    
        try:
            # Load CSV with basic error handling
            logger.info(f"Loading CSV file: {csv_file}")
            df_raw = pd.read_csv(csv_file)
        
            if df_raw.empty:
                logger.warning(f"Empty CSV file: {csv_file}")
                metadata = {
                    'dataset_name': dataset_name,
                    'source_file': str(csv_file),
                    'load_timestamp': datetime.now(),
                    'validation_errors': ["Empty CSV file"],
                    'validation_warnings': [],
                    'data_summary': {},
                    'processing_info': {},
                    'status': 'empty_file'
                }
                return None, metadata
        
            logger.info(f"Raw CSV loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
        
            # STEP 1: Standardize data types (THIS FIXES THE MAIN ISSUE)
            df_typed = _self.standardize_data_types(df_raw, dataset_name)
            logger.info("Data type standardization completed")
        
            # STEP 2: Apply data constraints (our safe version)
            df_constrained = _self.apply_data_constraints(df_typed, dataset_name)
            logger.info("Data constraints applied")
        
            # STEP 3: Remove completely empty rows
            df_clean = df_constrained.dropna(how='all')
        
            # STEP 4: Process dataset-specific features
            df_final = _self._process_dataset(df_clean, dataset_name)
        
            if df_final.empty:
                logger.warning(f"No valid data remaining after cleaning for {dataset_name}")
                metadata = {
                    'dataset_name': dataset_name,
                    'source_file': str(csv_file),
                    'load_timestamp': datetime.now(),
                    'validation_errors': [],
                    'validation_warnings': ["No valid data after cleaning"],
                    'data_summary': {},
                    'processing_info': {},
                    'status': 'empty_after_cleaning'
                }
                return None, metadata
        
            # Basic validation check (just structure, not the problematic cleaning)
            validation_warnings = []
            validation_errors = []
        
            # Simple column checks without the problematic constraint validation
            expected_columns = {
                'routes': ['route_id', 'popularity_rating', 'start_lat', 'start_lon'],
                'braking_hotspots': ['hotspot_id', 'lat', 'lon', 'intensity'],
                'swerving_hotspots': ['hotspot_id', 'lat', 'lon', 'intensity'],
                'time_series': ['date']
            }
        
            required_cols = expected_columns.get(dataset_name, [])
            missing_cols = [col for col in required_cols if col not in df_final.columns]
            if missing_cols:
                validation_warnings.append(f"Missing expected columns: {missing_cols}")
        
            # Generate metadata
            metadata = {
                'dataset_name': dataset_name,
                'source_file': str(csv_file),
                'load_timestamp': datetime.now(),
                'validation_errors': validation_errors,
                'validation_warnings': validation_warnings,
                'data_summary': {
                    'row_count': len(df_final),
                    'column_count': len(df_final.columns),
                    'raw_row_count': len(df_raw),
                    'cleaned_row_count': len(df_final),
                    'data_types': {col: str(dtype) for col, dtype in df_final.dtypes.items()},
                    'columns': list(df_final.columns)
                },
                'processing_info': _self._get_processing_info(df_final, dataset_name),
                'status': 'success'
            }
        
            # Cache the results
            _self._data_cache[dataset_name] = (df_final, metadata)
            _self._cache_timestamps[dataset_name] = datetime.now()
        
            logger.info(f"Successfully loaded and cleaned {dataset_name}: {len(df_final)} rows")
            logger.info(f"Final data types: {df_final.dtypes.to_dict()}")
        
            return df_final, metadata
        
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            metadata = {
                'dataset_name': dataset_name,
                'source_file': str(csv_file),
                'load_timestamp': datetime.now(),
                'validation_errors': [f"Loading error: {str(e)}"],
                'validation_warnings': [],
                'data_summary': {},
                'processing_info': {},
                'status': 'error'
            }
            return None, metadata

    def load_all_datasets(self, force_reload: bool = False) -> Dict[str, Tuple[Optional[pd.DataFrame], Dict[str, Any]]]:
        """
        Load all available datasets
        
        Args:
            force_reload: Force reload all data from CSV
            
        Returns:
            Dictionary mapping dataset names to (DataFrame or None, metadata) tuples
        """
        all_data = {}
        
        for dataset_name in self.datasets.keys():
            try:
                all_data[dataset_name] = self.load_dataset(dataset_name, force_reload)
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {str(e)}")
                metadata = {
                    'dataset_name': dataset_name,
                    'source_file': str(self.raw_data_path / self.datasets[dataset_name]),
                    'load_timestamp': datetime.now(),
                    'validation_errors': [f"Unexpected error: {str(e)}"],
                    'validation_warnings': [],
                    'data_summary': {},
                    'processing_info': {},
                    'status': 'error'
                }
                all_data[dataset_name] = (None, metadata)
        
        return all_data

    def get_data_status(self) -> Dict[str, Any]:
        """
        Get status of all datasets
        
        Returns:
            Dictionary containing status information
        """
        status = {
            'datasets': {},
            'total_datasets': len(self.datasets),
            'available_datasets': 0,
            'loaded_datasets': len(self._data_cache),
            'last_update': max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }
        
        available_count = 0
        
        for dataset_name, filename in self.datasets.items():
            csv_path = self.raw_data_path / filename
            file_exists = csv_path.exists()
            
            if file_exists:
                available_count += 1
            
            dataset_status = {
                'filename': filename,
                'file_exists': file_exists,
                'file_size_mb': csv_path.stat().st_size / (1024 * 1024) if file_exists else 0,
                'last_modified': datetime.fromtimestamp(csv_path.stat().st_mtime) if file_exists else None,
                'loaded': dataset_name in self._data_cache,
                'cache_valid': self._is_cache_valid(dataset_name)
            }
            
            if dataset_name in self._data_cache:
                df, metadata = self._data_cache[dataset_name]
                if df is not None:
                    dataset_status.update({
                        'row_count': metadata['data_summary']['row_count'],
                        'column_count': metadata['data_summary']['column_count'],
                        'load_timestamp': metadata['load_timestamp'],
                        'status': metadata['status']
                    })
                else:
                    dataset_status.update({
                        'row_count': 0,
                        'column_count': 0,
                        'load_timestamp': metadata['load_timestamp'],
                        'status': metadata['status']
                    })
            
            status['datasets'][dataset_name] = dataset_status
        
        status['available_datasets'] = available_count
        return status

    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Data cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        return {
            'cached_datasets': list(self._data_cache.keys()),
            'cache_timestamps': self._cache_timestamps.copy(),
            'cache_size': len(self._data_cache)
        }


# Create singleton instance
data_processor = DataProcessor()
