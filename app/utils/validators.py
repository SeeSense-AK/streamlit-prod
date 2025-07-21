"""
Data validation utilities for SeeSense Dashboard
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
from .config import config

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation class for CSV files"""
    
    def __init__(self, dataset_name: str):
        """
        Initialize validator for a specific dataset
        
        Args:
            dataset_name: Name of the dataset (e.g., 'routes', 'braking_hotspots')
        """
        self.dataset_name = dataset_name
        self.schema = config.get_schema(dataset_name)
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a pandas DataFrame against the schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check if schema exists
        if not self.schema:
            self.validation_errors.append(f"No schema defined for dataset: {self.dataset_name}")
            return False, self.validation_errors, self.validation_warnings
        
        # Validate required columns
        self._validate_required_columns(df)
        
        # Validate data types
        self._validate_data_types(df)
        
        # Validate constraints
        self._validate_constraints(df)
        
        # Check for duplicates
        self._check_duplicates(df)
        
        # Check data quality
        self._check_data_quality(df)
        
        is_valid = len(self.validation_errors) == 0
        
        logger.info(f"Validation completed for {self.dataset_name}: "
                   f"Valid={is_valid}, Errors={len(self.validation_errors)}, "
                   f"Warnings={len(self.validation_warnings)}")
        
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present"""
        required_columns = self.schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            self.validation_errors.append(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Check for extra columns
        expected_columns = set(self.schema.get('column_types', {}).keys())
        extra_columns = set(df.columns) - expected_columns
        
        if extra_columns:
            self.validation_warnings.append(
                f"Unexpected columns found: {', '.join(extra_columns)}"
            )
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types for each column"""
        column_types = self.schema.get('column_types', {})
        
        for column, expected_type in column_types.items():
            if column not in df.columns:
                continue
            
            try:
                if expected_type == 'datetime':
                    pd.to_datetime(df[column], errors='raise')
                elif expected_type == 'float':
                    pd.to_numeric(df[column], errors='raise')
                elif expected_type == 'int':
                    # Check if can be converted to int (allowing for NaN)
                    pd.to_numeric(df[column], errors='raise', downcast='integer')
                elif expected_type == 'boolean':
                    # Check if values are boolean-like
                    unique_vals = df[column].dropna().unique()
                    valid_bool_vals = {True, False, 'true', 'false', 'True', 'False', 
                                     'TRUE', 'FALSE', 1, 0, 'yes', 'no', 'Yes', 'No'}
                    if not all(val in valid_bool_vals for val in unique_vals):
                        raise ValueError(f"Invalid boolean values in {column}")
                
            except (ValueError, TypeError) as e:
                self.validation_errors.append(
                    f"Data type validation failed for column '{column}': {str(e)}"
                )
    
    def _validate_constraints(self, df: pd.DataFrame) -> None:
        """Validate value constraints for each column"""
        constraints = self.schema.get('constraints', {})
        
        for column, constraint_dict in constraints.items():
            if column not in df.columns:
                continue
            
            column_data = df[column].dropna()  # Ignore NaN values
            
            # Min/Max constraints
            if 'min' in constraint_dict:
                min_val = constraint_dict['min']
                if (column_data < min_val).any():
                    violating_count = (column_data < min_val).sum()
                    self.validation_errors.append(
                        f"Column '{column}' has {violating_count} values below minimum ({min_val})"
                    )
            
            if 'max' in constraint_dict:
                max_val = constraint_dict['max']
                if (column_data > max_val).any():
                    violating_count = (column_data > max_val).sum()
                    self.validation_errors.append(
                        f"Column '{column}' has {violating_count} values above maximum ({max_val})"
                    )
            
            # Allowed values constraint
            if 'values' in constraint_dict:
                allowed_values = set(constraint_dict['values'])
                invalid_values = set(column_data.unique()) - allowed_values
                if invalid_values:
                    self.validation_errors.append(
                        f"Column '{column}' has invalid values: {', '.join(map(str, invalid_values))}"
                    )
    
    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate rows and IDs"""
        # Check for completely duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self.validation_warnings.append(
                f"Found {duplicate_rows} duplicate rows"
            )
        
        # Check for duplicate IDs in actual ID columns only
        # Only check columns that are specifically designed to be unique identifiers
        id_columns = []
        
        # Look for columns that are clearly ID columns
        for col in df.columns:
            col_lower = col.lower()
            # Only check columns that end with '_id' or are exactly 'id'
            if (col_lower.endswith('_id') or 
                col_lower == 'id' or 
                col_lower == 'route_id' or 
                col_lower == 'hotspot_id'):
                id_columns.append(col)
        
        # Check for duplicates in actual ID columns
        for id_col in id_columns:
            if id_col in df.columns:
                duplicate_ids = df[id_col].duplicated().sum()
                if duplicate_ids > 0:
                    self.validation_errors.append(
                        f"Found {duplicate_ids} duplicate IDs in column '{id_col}'"
                    )
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check general data quality issues"""
        # Check for high percentage of missing values
        for column in df.columns:
            missing_percentage = (df[column].isnull().sum() / len(df)) * 100
            if missing_percentage > 50:
                self.validation_warnings.append(
                    f"Column '{column}' has {missing_percentage:.1f}% missing values"
                )
            elif missing_percentage > 20:
                self.validation_warnings.append(
                    f"Column '{column}' has {missing_percentage:.1f}% missing values"
                )
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column in df.columns:
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                if outliers > 0:
                    outlier_percentage = (outliers / len(df)) * 100
                    if outlier_percentage > 5:
                        self.validation_warnings.append(
                            f"Column '{column}' has {outlier_percentage:.1f}% outliers"
                        )


def validate_csv_file(file_path: Path, dataset_name: str) -> Tuple[bool, pd.DataFrame, List[str], List[str]]:
    """
    Validate a CSV file and return the DataFrame with validation results
    
    Args:
        file_path: Path to the CSV file
        dataset_name: Name of the dataset for schema validation
        
    Returns:
        Tuple of (is_valid, dataframe, errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        # Check file exists and size
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return False, pd.DataFrame(), errors, warnings
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size = config.max_file_size_mb
        
        if file_size_mb > max_size:
            errors.append(f"File size ({file_size_mb:.1f}MB) exceeds maximum ({max_size}MB)")
            return False, pd.DataFrame(), errors, warnings
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        except Exception as e:
            errors.append(f"Failed to read CSV file: {str(e)}")
            return False, pd.DataFrame(), errors, warnings
        
        # Validate data
        validator = DataValidator(dataset_name)
        is_valid, validation_errors, validation_warnings = validator.validate_dataframe(df)
        
        errors.extend(validation_errors)
        warnings.extend(validation_warnings)
        
        return is_valid, df, errors, warnings
        
    except Exception as e:
        errors.append(f"Unexpected error during validation: {str(e)}")
        return False, pd.DataFrame(), errors, warnings


def clean_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Clean and prepare DataFrame based on schema
    
    Args:
        df: DataFrame to clean
        dataset_name: Name of the dataset for schema reference
        
    Returns:
        Cleaned DataFrame
    """
    schema = config.get_schema(dataset_name)
    if not schema:
        return df
    
    df_cleaned = df.copy()
    column_types = schema.get('column_types', {})
    
    # Convert data types
    for column, expected_type in column_types.items():
        if column not in df_cleaned.columns:
            continue
        
        try:
            if expected_type == 'datetime':
                df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
            elif expected_type == 'float':
                df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            elif expected_type == 'int':
                df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce', downcast='integer')
            elif expected_type == 'boolean':
                # Convert various boolean representations
                bool_map = {
                    'true': True, 'false': False,
                    'True': True, 'False': False,
                    'TRUE': True, 'FALSE': False,
                    'yes': True, 'no': False,
                    'Yes': True, 'No': False,
                    'y': True, 'n': False,
                    'Y': True, 'N': False,
                    1: True, 0: False,
                    '1': True, '0': False
                }
                df_cleaned[column] = df_cleaned[column].map(bool_map).fillna(df_cleaned[column])
        
        except Exception as e:
            logger.warning(f"Failed to convert column '{column}' to {expected_type}: {str(e)}")
    
    # Remove completely empty rows
    df_cleaned = df_cleaned.dropna(how='all')
    
    # Apply constraints and filters
    constraints = schema.get('constraints', {})
    for column, constraint_dict in constraints.items():
        if column not in df_cleaned.columns:
            continue
        
        # Filter out values outside min/max constraints
        if 'min' in constraint_dict:
            min_val = constraint_dict['min']
            # FIXED: Safe numeric conversion before comparison
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            df_cleaned = df_cleaned.dropna(subset=[column])
            df_cleaned = df_cleaned[df_cleaned[column] >= min_val]
        
        if 'max' in constraint_dict:
            max_val = constraint_dict['max']
            # FIXED: Safe numeric conversion before comparison
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            df_cleaned = df_cleaned.dropna(subset=[column])
            df_cleaned = df_cleaned[df_cleaned[column] <= max_val]
        
        # Filter out invalid categorical values
        if 'values' in constraint_dict:
            allowed_values = constraint_dict['values']
            df_cleaned = df_cleaned[df_cleaned[column].isin(allowed_values)]
    
    logger.info(f"Data cleaning completed. Rows: {len(df)} -> {len(df_cleaned)}")
    return df_cleaned


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the DataFrame
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
    }
    
    # Add numeric column statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        summary['numeric_summary'] = df[numeric_columns].describe().to_dict()
    
    # Add categorical column information
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_info = {}
    for col in categorical_columns:
        unique_count = df[col].nunique()
        categorical_info[col] = {
            'unique_values': unique_count,
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    if categorical_info:
        summary['categorical_summary'] = categorical_info
    
    return summary
