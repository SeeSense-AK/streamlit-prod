"""
Main entry point for SeeSense Production Dashboard
Handles data loading and routing to appropriate pages
"""
import streamlit as st
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable absolute imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import configuration and setup logging
from app.utils.config import config, setup_logging
setup_logging()

# Import core components
from app.core.data_processor import data_processor
from app.pages.data_setup import render_data_setup_page

# Import page modules
from app.pages.overview import render_overview_page
from app.pages.ml_insights import render_ml_insights_page
from app.pages.spatial_analysis import render_spatial_analysis_page
from app.pages.advanced_analytics import render_advanced_analytics_page
from app.pages.actionable_insights import render_actionable_insights_page
from app.pages.data_setup import render_data_setup_page


logger = logging.getLogger(__name__)


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=config.app_title,
        page_icon=config.app_icon,
        layout=config.get('app.layout', 'wide'),
        initial_sidebar_state=config.get('app.initial_sidebar_state', 'expanded')
    )


def load_custom_css():
    """Load custom CSS styles"""
    css_path = config.get_assets_path() / "styles" / "custom.css"
    
    if css_path.exists():
        with open(css_path, 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Use default CSS if file doesn't exist
        st.markdown("""
        <style>
        .main {
            background-color: #f9f9f9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4e89ae;
            color: white;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        .card-title {
            color: #555;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .card-value {
            color: #333;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .card-delta {
            color: #666;
            font-size: 12px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eaeaea;
        }
        </style>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with navigation and status"""
    # Logo section
    logo_path = config.get_assets_path() / "logo.png"
    
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=150)
    else:
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h2>{config.app_icon} SeeSense</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Data status indicator
    data_status = data_processor.get_data_status()
    available_datasets = data_status['available_datasets']
    total_datasets = data_status['total_datasets']
    
    if available_datasets == total_datasets:
        st.sidebar.success(f"✅ All {total_datasets} datasets available")
    elif available_datasets > 0:
        st.sidebar.warning(f"⚠️ {available_datasets}/{total_datasets} datasets available")
    else:
        st.sidebar.error("❌ No datasets available")
    
    # Quick data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        data_processor.load_all_datasets(force_reload=True)
        st.sidebar.success("Data refreshed!")
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation based on data availability
    if available_datasets == 0:
        st.sidebar.warning("⚠️ Set up your data files first")
        return "data_setup"
    else:
        # Navigation menu
        st.sidebar.markdown("### 📊 Dashboard Sections")
        
        pages = {
            "📊 Overview": "overview",
            "🔍 ML Insights": "ml_insights", 
            "🗺️ Spatial Analysis": "spatial_analysis",
            "📈 Advanced Analytics": "advanced_analytics",
            "💡 Actionable Insights": "actionable_insights",
            "📁 Data Setup": "data_setup"
        }
        
        # Create navigation
        selected_page = st.sidebar.radio(
            "Navigate to:",
            list(pages.keys()),
            index=0
        )
        
        return pages[selected_page]


def render_data_not_available_message():
    """Render message when data is not available"""
    st.title("🚲 SeeSense Safety Analytics Platform")
    
    st.markdown("""
    ## Welcome to SeeSense Dashboard!
    
    To get started, you need to set up your cycling safety data files.
    """)
    
    data_requirements = data_processor.check_data_requirements()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Required Data Files:")
        
        for missing_file in data_requirements['missing_files']:
            st.markdown(f"- **{missing_file['filename']}** (for {missing_file['dataset']} data)")
        
        st.markdown("### 📂 Data Directory:")
        st.code(str(data_processor.raw_data_path))
        
        if st.button("📁 Go to Data Setup", type="primary"):
            st.session_state.current_page = "data_setup"
            st.experimental_rerun()
    
    with col2:
        st.info("""
        **Quick Start:**
        
        1. Prepare your CSV files
        2. Place them in the data directory
        3. Use the Data Setup page for validation
        4. Start exploring your dashboard!
        """)


def main():
    """Main application entry point"""
    # Set up page configuration
    setup_page_config()
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None
    
    try:
        # Render sidebar and get selected page
        selected_page = render_sidebar()
        
        # Override with session state if set
        if st.session_state.current_page:
            selected_page = st.session_state.current_page
            st.session_state.current_page = None  # Reset after use
        
        # Route to appropriate page
        if selected_page == "data_setup":
            render_data_setup_page()
        
        elif selected_page == "overview":
            # Check if data is available
            data_status = data_processor.get_data_status()
            if data_status['available_datasets'] == 0:
                render_data_not_available_message()
            else:
                render_overview_page()
        
        elif selected_page == "ml_insights":
            # Check if data is available
            data_status = data_processor.get_data_status()
            if data_status['available_datasets'] == 0:
                render_data_not_available_message()
            else:
                render_ml_insights_page()
        
        elif selected_page == "spatial_analysis":
            # Check if data is available
            data_status = data_processor.get_data_status()
            if data_status['available_datasets'] == 0:
                render_data_not_available_message()
            else:
                render_spatial_analysis_page()
        
        elif selected_page == "advanced_analytics":
            # Check if data is available
            data_status = data_processor.get_data_status()
            if data_status['available_datasets'] == 0:
                render_data_not_available_message()
            else:
                render_advanced_analytics_page()
        
        elif selected_page == "actionable_insights":
            # Check if data is available
            data_status = data_processor.get_data_status()
            if data_status['available_datasets'] == 0:
                render_data_not_available_message()
            else:
                render_actionable_insights_page()
        
        else:
            # Default to data setup if no valid page
            render_data_setup_page()
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        
        # Show data setup page as fallback
        st.markdown("---")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("Try setting up your data files:")
        render_data_setup_page()


if __name__ == "__main__":
    main()
