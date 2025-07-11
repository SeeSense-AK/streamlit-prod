"""
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
]
