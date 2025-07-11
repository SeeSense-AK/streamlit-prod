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
        
        # Check for duplicate IDs in ID columns
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for id_col in id_columns:
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
                            f"Column '{column}
