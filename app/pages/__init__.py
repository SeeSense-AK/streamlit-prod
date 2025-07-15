"""
Dashboard pages for SeeSense Platform
Individual page modules for different dashboard sections
"""

# Import with error handling for development
try:
    from .data_setup import render_data_setup_page
    from .overview import render_overview_page
    from .ml_insights import render_ml_insights_page
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.pages.data_setup import render_data_setup_page
    from app.pages.overview import render_overview_page
    from app.pages.ml_insights import render_ml_insights_page

# When we implement these pages, we'll add imports like:
# from .spatial_analysis import render_spatial_analysis_page
# from .advanced_analytics import render_advanced_analytics_page
# from .actionable_insights import render_actionable_insights_page

__all__ = [
    "render_data_setup_page",
    "render_overview_page", 
    "render_ml_insights_page",
    
    # Placeholder for future page modules
    # "render_spatial_analysis_page", 
    # "render_advanced_analytics_page",
    # "render_actionable_insights_page"
]
