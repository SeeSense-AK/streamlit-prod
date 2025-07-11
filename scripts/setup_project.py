#!/usr/bin/env python3
"""
Project setup script for SeeSense Dashboard
Creates all necessary directories and __init__.py files
"""
from pathlib import Path

def create_init_files():
    """Create all necessary __init__.py files"""
    
    init_files = {
        "app/__init__.py": '''"""
SeeSense Safety Analytics Platform
Production-ready cycling safety dashboard
"""

__version__ = "1.0.0"
__author__ = "SeeSense"
__description__ = "Production cycling safety analytics dashboard"

# Package-level imports for easier access
from .utils.config import config
from .core.data_processor import data_processor

__all__ = [
    "config",
    "data_processor"
]''',

        "app/utils/__init__.py": '''"""
Utility modules for SeeSense Dashboard
Configuration, validation, caching, and helper functions
"""

from .config import config, setup_logging, get_environment, is_development, is_production
from .validators import DataValidator, validate_csv_file, clean_dataframe, get_data_summary
from .cache import (
    CacheManager, 
    cache_manager, 
    cached_function, 
    streamlit_cache_data,
    clear_streamlit_cache,
    get_cache_info,
    cache_dataframe,
    get_cached_dataframe,
    cache_computation_result,
    get_cached_computation
)

__all__ = [
    # Configuration
    "config",
    "setup_logging", 
    "get_environment",
    "is_development",
    "is_production",
    
    # Validation
    "DataValidator",
    "validate_csv_file",
    "clean_dataframe", 
    "get_data_summary",
    
    # Caching
    "CacheManager",
    "cache_manager",
    "cached_function",
    "streamlit_cache_data",
    "clear_streamlit_cache",
    "get_cache_info",
    "cache_dataframe",
    "get_cached_dataframe", 
    "cache_computation_result",
    "get_cached_computation"
]''',

        "app/core/__init__.py": '''"""
Core business logic modules for SeeSense Dashboard
Data processing, ML models, analytics, and insights generation
"""

from .data_processor import data_processor, DataProcessor

# When we implement these modules, we'll add imports like:
# from .ml_models import RiskPredictor, SafetyClusterer, AnomalyDetector
# from .analytics import TimeSeriesAnalyzer, CorrelationAnalyzer
# from .spatial_analyzer import HotspotAnalyzer, RouteOptimizer
# from .insights_generator import InsightsEngine, RecommendationEngine

__all__ = [
    "data_processor",
    "DataProcessor",
    
    # Placeholder for future ML/Analytics modules
    # "RiskPredictor",
    # "SafetyClusterer", 
    # "AnomalyDetector",
    # "TimeSeriesAnalyzer",
    # "CorrelationAnalyzer",
    # "HotspotAnalyzer",
    # "RouteOptimizer",
    # "InsightsEngine",
    # "RecommendationEngine"
]''',

        "app/pages/__init__.py": '''"""
Dashboard pages for SeeSense Platform
Individual page modules for different dashboard sections
"""

from .data_setup import render_data_setup_page

# When we implement these pages, we'll add imports like:
# from .overview import render_overview_page
# from .ml_insights import render_ml_insights_page
# from .spatial_analysis import render_spatial_analysis_page
# from .advanced_analytics import render_advanced_analytics_page
# from .actionable_insights import render_actionable_insights_page

__all__ = [
    "render_data_setup_page",
    
    # Placeholder for future page modules
    # "render_overview_page",
    # "render_ml_insights_page",
    # "render_spatial_analysis_page", 
    # "render_advanced_analytics_page",
    # "render_actionable_insights_page"
]''',

        "app/components/__init__.py": '''"""
Reusable UI components for SeeSense Dashboard
Charts, maps, metrics cards, filters, and other UI elements
"""

# When we implement these components, we'll add imports like:
# from .metrics_cards import metric_card, kpi_grid, trend_indicator
# from .maps import hotspot_map, route_popularity_map, density_heatmap
# from .charts import time_series_chart, correlation_heatmap, bar_chart, scatter_plot
# from .filters import date_range_filter, spatial_filter, category_filter

__all__ = [
    # Placeholder for future component modules
    # "metric_card",
    # "kpi_grid", 
    # "trend_indicator",
    # "hotspot_map",
    # "route_popularity_map",
    # "density_heatmap",
    # "time_series_chart",
    # "correlation_heatmap",
    # "bar_chart",
    # "scatter_plot",
    # "date_range_filter",
    # "spatial_filter",
    # "category_filter"
]'''
    }
    
    for file_path, content in init_files.items():
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path_obj, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {file_path}")

def create_gitkeep_files():
    """Create .gitkeep files for empty directories"""
    
    gitkeep_files = {
        "data/raw/.gitkeep": """# This file ensures the data/raw directory is preserved in git
# Place your CSV files here:
# - routes.csv
# - braking_hotspots.csv  
# - swerving_hotspots.csv
# - time_series.csv""",

        "data/processed/.gitkeep": """# This file ensures the data/processed directory is preserved in git
# Processed and cached data files will be stored here automatically""",

        "logs/.gitkeep": """# This file ensures the logs directory is preserved in git
# Application logs will be stored here automatically""",

        "assets/styles/.gitkeep": """# This file ensures the assets/styles directory is preserved in git
# Place custom CSS files here (e.g., custom.css)"""
    }
    
    for file_path, content in gitkeep_files.items():
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path_obj, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {file_path}")

def main():
    """Main setup function"""
    print("🚲 Setting up SeeSense Dashboard project structure")
    print("=" * 50)
    
    print("\n📁 Creating __init__.py files...")
    create_init_files()
    
    print("\n📁 Creating .gitkeep files...")
    create_gitkeep_files()
    
    print("\n✅ Project setup complete!")
    print("\n📋 Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Generate sample data: python scripts/generate_sample_data.py")
    print("3. Run dashboard: streamlit run app/main.py")

if __name__ == "__main__":
    main()
