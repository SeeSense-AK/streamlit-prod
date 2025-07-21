"""
Data processing pipeline for SeeSense Dashboard - Production Version
Handles only real CSV data, no synthetic data generation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import streamlit as st

from utils.config import config
from utils.validators import validate_csv_file, clean_dataframe, get_data_summary

logger = logging.getLogger(__name__)


class DataProcessor:
    """Production data processing class - real CSV data only"""
    
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
    
    @st.cache_data
    def load_dataset(_self, dataset_name: str, force_reload: bool = False) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load and process a dataset from CSV file
        
        Args:
            dataset_name: Name of the dataset to load
            force_reload: Force reload from CSV even if cached
            
        Returns:
            Tuple of (DataFrame or None, metadata)
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
        
        # Validate and load CSV
        is_valid, df, errors, warnings = validate_csv_file(csv_file, dataset_name)
        
        if not is_valid:
            logger.error(f"Validation failed for {dataset_name}: {errors}")
            metadata = {
                'dataset_name': dataset_name,
                'source_file': str(csv_file),
                'load_timestamp': datetime.now(),
                'validation_errors': errors,
                'validation_warnings': warnings,
                'data_summary': {},
                'processing_info': {},
                'status': 'validation_failed'
            }
            return None, metadata
        
        if warnings:
            logger.warning(f"Validation warnings for {dataset_name}: {warnings}")
        
        # Clean and process data
        df_cleaned = clean_dataframe(df, dataset_name)
        
        if df_cleaned.empty:
            logger.warning(f"No valid data remaining after cleaning for {dataset_name}")
            metadata = {
                'dataset_name': dataset_name,
                'source_file': str(csv_file),
                'load_timestamp': datetime.now(),
                'validation_errors': [],
                'validation_warnings': warnings + ["No valid data after cleaning"],
                'data_summary': {},
                'processing_info': {},
                'status': 'empty_after_cleaning'
            }
            return None, metadata
        
        df_processed = _self._process_dataset(df_cleaned, dataset_name)
        
        # Generate metadata
        metadata = {
            'dataset_name': dataset_name,
            'source_file': str(csv_file),
            'load_timestamp': datetime.now(),
            'validation_errors': errors,
            'validation_warnings': warnings,
            'data_summary': get_data_summary(df_processed),
            'processing_info': _self._get_processing_info(df_processed, dataset_name),
            'status': 'success'
        }
        
        # Cache the results
        _self._data_cache[dataset_name] = (df_processed, metadata)
        _self._cache_timestamps[dataset_name] = datetime.now()
        
        logger.info(f"Successfully loaded {dataset_name}: {len(df_processed)} rows")
        return df_processed, metadata
    
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
    
    def get_available_datasets(self) -> List[str]:
        """
        Get list of datasets that have CSV files available
        
        Returns:
            List of dataset names with available CSV files
        """
        available = []
        for dataset_name, filename in self.datasets.items():
            csv_path = self.raw_data_path / filename
            if csv_path.exists():
                available.append(dataset_name)
        return available
    
    def check_data_requirements(self) -> Dict[str, Any]:
        """
        Check which data files are missing and provide guidance
        
        Returns:
            Dictionary with missing files and setup instructions
        """
        missing_files = []
        available_files = []
        
        for dataset_name, filename in self.datasets.items():
            csv_path = self.raw_data_path / filename
            if csv_path.exists():
                available_files.append({
                    'dataset': dataset_name,
                    'filename': filename,
                    'size_mb': csv_path.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(csv_path.stat().st_mtime)
                })
            else:
                missing_files.append({
                    'dataset': dataset_name,
                    'filename': filename,
                    'path': str(csv_path)
                })
        
        return {
            'data_directory': str(self.raw_data_path),
            'missing_files': missing_files,
            'available_files': available_files,
            'setup_complete': len(missing_files) == 0,
            'setup_instructions': self._get_setup_instructions(missing_files)
        }
    
    def _get_setup_instructions(self, missing_files: List[Dict]) -> List[str]:
        """Generate setup instructions for missing files"""
        if not missing_files:
            return ["✅ All required data files are present!"]
        
        instructions = [
            "📁 To set up your dashboard data:",
            f"1. Navigate to the data directory: {self.raw_data_path}",
            "2. Place your CSV files with the following names:"
        ]
        
        for file_info in missing_files:
            instructions.append(f"   • {file_info['filename']} (for {file_info['dataset']} data)")
        
        instructions.extend([
            "",
            "3. Ensure your CSV files match the expected schema (see data_schema.yaml)",
            "4. Refresh the dashboard to load your data",
            "",
            "💡 The dashboard will validate your data and show any issues that need to be resolved."
        ])
        
        return instructions
    
    def _is_cache_valid(self, dataset_name: str) -> bool:
        """Check if cached data is still valid"""
        if dataset_name not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[dataset_name]
        return cache_age.total_seconds() < self.cache_ttl
    
    def _process_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Apply dataset-specific processing
        
        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        if dataset_name == 'routes':
            df_processed = self._process_routes_data(df_processed)
        elif dataset_name == 'braking_hotspots':
            df_processed = self._process_braking_data(df_processed)
        elif dataset_name == 'swerving_hotspots':
            df_processed = self._process_swerving_data(df_processed)
        elif dataset_name == 'time_series':
            df_processed = self._process_time_series_data(df_processed)
        
        return df_processed
    
    def _process_routes_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process routes dataset"""
        # Calculate distance if not present
        if 'distance_km' not in df.columns:
            df['distance_km'] = self._calculate_distance(
                df['start_lat'], df['start_lon'], df['end_lat'], df['end_lon']
            )
        
        # Calculate derived metrics
        df['cyclists_per_day'] = df['distinct_cyclists'] / df['days_active']
        df['popularity_score'] = (df['popularity_rating'] * df['distinct_cyclists'] / 100).round(2)
        
        # Categorize routes by length
        df['route_length_category'] = pd.cut(
            df['distance_km'], 
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Short (<2km)', 'Medium (2-5km)', 'Long (5-10km)', 'Very Long (>10km)']
        )
        
        return df
    
    def _process_braking_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process braking hotspots dataset - FIXED VERSION"""
        # Calculate severity score if not present
        if 'severity_score' not in df.columns:
            # FIXED: Ensure all inputs are numeric before calculation
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df['incidents_count'] = pd.to_numeric(df['incidents_count'], errors='coerce')
            df['avg_deceleration'] = pd.to_numeric(df['avg_deceleration'], errors='coerce')
        
            # Fill NaN values with 0 for calculation
            df['intensity'] = df['intensity'].fillna(0)
            df['incidents_count'] = df['incidents_count'].fillna(0)
            df['avg_deceleration'] = df['avg_deceleration'].fillna(0)
        
            df['severity_score'] = (
                df['intensity'] * 0.4 + 
                (df['incidents_count'] / df['incidents_count'].max() * 10) * 0.3 +
                (df['avg_deceleration'] / df['avg_deceleration'].max() * 10) * 0.3
            ).round(2)
        else:
            # FIXED: Ensure severity_score is numeric
            df['severity_score'] = pd.to_numeric(df['severity_score'], errors='coerce')
    
        # Add time-based features
        if 'date_recorded' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date_recorded']).dt.day_name()
            df['month'] = pd.to_datetime(df['date_recorded']).dt.month
            df['days_since_recorded'] = (datetime.now() - pd.to_datetime(df['date_recorded'])).dt.days
    
        # Risk categorization with safe numeric data
        try:
            df['risk_level'] = pd.cut(
                df['severity_score'],
                bins=[0, 3, 6, 8, 10],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        except Exception as e:
            logger.warning(f"Could not create risk_level categories: {e}")
            df['risk_level'] = 'Unknown'
    
        return df

    def _process_swerving_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process swerving hotspots dataset - FIXED VERSION"""
        # Calculate severity score if not present
        if 'severity_score' not in df.columns:
            # FIXED: Ensure all inputs are numeric before calculation
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df['incidents_count'] = pd.to_numeric(df['incidents_count'], errors='coerce')
            df['avg_lateral_movement'] = pd.to_numeric(df['avg_lateral_movement'], errors='coerce')
        
            # Fill NaN values with 0 for calculation
            df['intensity'] = df['intensity'].fillna(0)
            df['incidents_count'] = df['incidents_count'].fillna(0)
            df['avg_lateral_movement'] = df['avg_lateral_movement'].fillna(0)
        
            df['severity_score'] = (
                df['intensity'] * 0.5 + 
                (df['incidents_count'] / df['incidents_count'].max() * 10) * 0.3 +
                (df['avg_lateral_movement'] / df['avg_lateral_movement'].max() * 10) * 0.2
            ).round(2)
        else:
            # FIXED: Ensure severity_score is numeric
            df['severity_score'] = pd.to_numeric(df['severity_score'], errors='coerce')
    
        # Add time-based features
        if 'date_recorded' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date_recorded']).dt.day_name()
            df['month'] = pd.to_datetime(df['date_recorded']).dt.month
            df['days_since_recorded'] = (datetime.now() - pd.to_datetime(df['date_recorded'])).dt.days
    
        # Risk categorization with safe numeric data
        try:
            df['risk_level'] = pd.cut(
                df['severity_score'],
                bins=[0, 3, 6, 8, 10],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        except Exception as e:
            logger.warning(f"Could not create risk_level categories: {e}")
            df['risk_level'] = 'Unknown'
    
        return df
    
    def _process_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process time series dataset"""
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add time-based features
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['date'].dt.weekday >= 5
        
        # Calculate rolling averages
        df['incidents_7day_avg'] = df['incidents'].rolling(window=7, min_periods=1).mean()
        df['incidents_30day_avg'] = df['incidents'].rolling(window=30, min_periods=1).mean()
        
        # Calculate safety metrics
        df['safety_score'] = 10 - (df['incidents'] / df['total_rides'] * 100).clip(0, 10)
        df['incident_rate'] = (df['incidents'] / df['total_rides'] * 1000).round(2)  # per 1000 rides
        
        # Weather impact indicators
        if 'precipitation_mm' in df.columns:
            df['rainy_day'] = df['precipitation_mm'] > 1
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(
                df['temperature'],
                bins=[-float('inf'), 5, 15, 25, float('inf')],
                labels=['Cold', 'Cool', 'Mild', 'Warm']
            )
        
        return df
    
    def _calculate_distance(self, lat1: pd.Series, lon1: pd.Series, 
                          lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """Calculate haversine distance between coordinates"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _get_processing_info(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Get processing information for the dataset"""
        info = {
            'processed_columns': list(df.columns),
            'derived_features': [],
            'processing_timestamp': datetime.now()
        }
        
        # Track derived features by dataset
        if dataset_name == 'routes':
            derived_features = ['distance_km', 'cyclists_per_day', 'popularity_score', 'route_length_category']
        elif dataset_name in ['braking_hotspots', 'swerving_hotspots']:
            derived_features = ['severity_score', 'day_of_week', 'month', 'days_since_recorded', 'risk_level']
        elif dataset_name == 'time_series':
            derived_features = ['day_of_week', 'month', 'quarter', 'is_weekend', 
                              'incidents_7day_avg', 'incidents_30day_avg', 'safety_score', 'incident_rate']
        else:
            derived_features = []
        
        info['derived_features'] = [col for col in derived_features if col in df.columns]
        return info


# Global data processor instance
data_processor = DataProcessor()
