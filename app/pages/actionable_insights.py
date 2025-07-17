"""
Actionable Insights Page for SeeSense Dashboard
AI-generated recommendations, priority actions, risk assessments, and ROI calculations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import json
from pathlib import Path

# AI and API imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_actionable_insights_page():
    """Render the actionable insights page"""
    st.title("💡 Actionable Insights")
    st.markdown("AI-powered recommendations and priority actions for cycling safety improvements")
    
    # Check if Groq API is available
    if not GROQ_AVAILABLE:
        st.error("⚠️ Groq API library is not installed. Please install it with: `pip install groq`")
        return
    
    # Check for API key
    api_key = get_groq_api_key()
    
    # Check for temporary session key as fallback
    if not api_key and hasattr(st.session_state, 'temp_groq_key'):
        api_key = st.session_state.temp_groq_key
    
    if not api_key:
        render_api_key_setup()
        return
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_insights_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add insights controls in sidebar
        insights_options = render_insights_controls()
        
        # Create tabs for different insight types
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 AI Recommendations",
            "📋 Priority Actions",
            "⚠️ Risk Assessment",
            "💰 ROI Analysis"
        ])
        
        with tab1:
            render_ai_recommendations(routes_df, braking_df, swerving_df, time_series_df, insights_options, api_key)
        
        with tab2:
            render_priority_actions(routes_df, braking_df, swerving_df, time_series_df, insights_options)
        
        with tab3:
            render_risk_assessment(routes_df, braking_df, swerving_df, time_series_df, insights_options)
        
        with tab4:
            render_roi_analysis(routes_df, braking_df, swerving_df, time_series_df, insights_options)
        
    except Exception as e:
        logger.error(f"Error in actionable insights page: {e}")
        st.error("⚠️ An error occurred while loading actionable insights.")
        st.info("Please check your data files and API configuration.")
        
        with st.expander("🔍 Error Details"):
            st.code(str(e))


def get_groq_api_key():
    """Get Groq API key from configuration or environment"""
    try:
        # Method 1: Check environment variable first
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            logger.info("Found Groq API key in environment variable")
            return api_key.strip()
        
        # Method 2: Check config file
        try:
            api_key = config.get('groq.api_key')
            if api_key:
                logger.info("Found Groq API key in config file")
                return api_key.strip()
        except Exception as e:
            logger.debug(f"Could not get API key from config: {e}")
        
        # Method 3: Check secrets file in project root
        project_root = Path(__file__).parent.parent.parent  # Go up from pages/ to project root
        secrets_path = project_root / "secrets" / "groq_api_key.txt"
        
        logger.info(f"Looking for secrets file at: {secrets_path}")
        logger.info(f"Secrets file exists: {secrets_path.exists()}")
        
        if secrets_path.exists():
            try:
                with open(secrets_path, 'r') as f:
                    api_key = f.read().strip()
                if api_key:
                    logger.info("Found Groq API key in secrets file")
                    return api_key
                else:
                    logger.warning("Secrets file exists but is empty")
            except Exception as e:
                logger.error(f"Error reading secrets file: {e}")
        
        # Method 4: Check alternative secrets locations
        alternative_paths = [
            Path.cwd() / "secrets" / "groq_api_key.txt",  # Current working directory
            Path.home() / ".seesense" / "groq_api_key.txt",  # User home directory
        ]
        
        for alt_path in alternative_paths:
            logger.info(f"Checking alternative path: {alt_path}")
            if alt_path.exists():
                try:
                    with open(alt_path, 'r') as f:
                        api_key = f.read().strip()
                    if api_key:
                        logger.info(f"Found Groq API key in alternative location: {alt_path}")
                        return api_key
                except Exception as e:
                    logger.error(f"Error reading alternative secrets file {alt_path}: {e}")
        
        logger.warning("No Groq API key found in any location")
        return None
    
    except Exception as e:
        logger.error(f"Error getting Groq API key: {e}")
        return None


def render_api_key_setup():
    """Render API key setup instructions"""
    st.warning("⚠️ Groq API key not found")
    
    st.markdown("""
    To use AI-powered insights, you need to configure your Groq API key.
    """)
    
    # Debug information
    with st.expander("🔍 Debug Information"):
        project_root = Path(__file__).parent.parent.parent
        secrets_path = project_root / "secrets" / "groq_api_key.txt"
        cwd_path = Path.cwd() / "secrets" / "groq_api_key.txt"
        
        st.markdown("**Checking these locations:**")
        st.code(f"1. Environment variable: GROQ_API_KEY = {'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
        st.code(f"2. Project root path: {secrets_path}")
        st.code(f"   - Exists: {secrets_path.exists()}")
        st.code(f"3. Current working directory: {cwd_path}")
        st.code(f"   - Exists: {cwd_path.exists()}")
        st.code(f"4. Current working directory: {Path.cwd()}")
        
        if secrets_path.exists():
            try:
                with open(secrets_path, 'r') as f:
                    content = f.read()
                st.code(f"   - File size: {len(content)} characters")
                st.code(f"   - Content preview: {content[:20]}..." if len(content) > 20 else f"   - Content: {content}")
            except Exception as e:
                st.code(f"   - Error reading file: {e}")
    
    st.markdown("""
    ### 🔧 Setup Instructions:
    
    #### Option 1: Secrets File (Recommended)
    1. Create the file: `secrets/groq_api_key.txt` in your project root
    2. Add your API key to the file (no extra spaces or newlines)
    3. Make sure the file has proper permissions: `chmod 600 secrets/groq_api_key.txt`
    
    #### Option 2: Environment Variable
    ```bash
    export GROQ_API_KEY="your-api-key-here"
    ```
    
    #### Option 3: Configuration File
    Add to your `config/settings.yaml`:
    ```yaml
    groq:
      api_key: "your-api-key-here"
    ```
    
    ### 📋 Step-by-Step:
    
    1. **Get your API key** from [console.groq.com](https://console.groq.com)
    2. **Create the secrets directory** in your project root:
       ```bash
       mkdir -p secrets
       ```
    3. **Create the API key file**:
       ```bash
       echo "your-api-key-here" > secrets/groq_api_key.txt
       ```
    4. **Set permissions**:
       ```bash
       chmod 600 secrets/groq_api_key.txt
       ```
    5. **Refresh this page**
    """)
    
    # Manual API key input (temporary)
    st.markdown("### 🔑 Temporary API Key (Session Only)")
    st.warning("⚠️ This is for testing only. Your key will be lost when you refresh.")
    
    temp_key = st.text_input("Enter your Groq API key:", type="password", key="temp_groq_key")
    
    if temp_key:
        st.session_state.temp_groq_key = temp_key
        st.success("✅ Temporary API key set! You can now use AI features.")
        if st.button("🔄 Refresh Page"):
            st.experimental_rerun()
    
    # Check if user has set up the key
    if st.button("🔄 Check API Key Again"):
        st.experimental_rerun()


def render_no_insights_data_message():
    """Render message when no data is available for insights"""
    st.warning("⚠️ No data available for actionable insights.")
    st.markdown("""
    To generate actionable insights, you need:
    1. **Safety incident data** for risk analysis
    2. **Route data** for optimization recommendations
    3. **Hotspot data** for targeted interventions
    4. **Time series data** for trend analysis
    
    Please add your data files and refresh the page.
    """)


def render_insights_controls():
    """Render insights controls in sidebar"""
    st.sidebar.markdown("### 💡 Insights Settings")
    
    options = {}
    
    # AI settings
    st.sidebar.markdown("**AI Recommendations**")
    options['ai_model'] = st.sidebar.selectbox(
        "AI Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0,
        help="Select the AI model for generating recommendations"
    )
    
    options['recommendation_focus'] = st.sidebar.selectbox(
        "Focus Area",
        ["Overall Safety", "Infrastructure", "Behavioral", "Environmental", "Policy"],
        help="Primary focus for AI recommendations"
    )
    
    # Priority settings
    st.sidebar.markdown("**Priority Actions**")
    options['priority_threshold'] = st.sidebar.slider(
        "Priority Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        help="Minimum priority score for actions"
    )
    
    options['max_actions'] = st.sidebar.slider(
        "Max Actions to Show",
        min_value=5,
        max_value=50,
        value=20,
        help="Maximum number of priority actions to display"
    )
    
    # Risk assessment settings
    st.sidebar.markdown("**Risk Assessment**")
    options['risk_timeframe'] = st.sidebar.selectbox(
        "Risk Timeframe",
        ["1 month", "3 months", "6 months", "1 year"],
        index=2,
        help="Timeframe for risk assessment"
    )
    
    # ROI settings
    st.sidebar.markdown("**ROI Analysis**")
    options['roi_timeframe'] = st.sidebar.selectbox(
        "ROI Timeframe",
        ["1 year", "2 years", "3 years", "5 years"],
        index=1,
        help="Timeframe for ROI calculations"
    )
    
    options['discount_rate'] = st.sidebar.slider(
        "Discount Rate (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Discount rate for NPV calculations"
    )
    
    return options


def render_ai_recommendations(routes_df, braking_df, swerving_df, time_series_df, insights_options, api_key):
    """Render AI-generated recommendations"""
    st.markdown("### 🤖 AI-Generated Recommendations")
    st.markdown("Intelligent analysis and personalized recommendations powered by advanced AI")
    
    # Initialize Groq client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return
    
    # Data summary for AI context
    data_summary = generate_data_summary(routes_df, braking_df, swerving_df, time_series_df)
    
    # AI recommendation generation
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Generate New Recommendations", type="primary"):
            with st.spinner("Generating AI recommendations..."):
                recommendations = generate_ai_recommendations(
                    client, data_summary, insights_options
                )
                if recommendations:
                    st.session_state.ai_recommendations = recommendations
    
    with col1:
        st.markdown("**AI Analysis Focus:** " + insights_options['recommendation_focus'])
    
    # Display recommendations
    if hasattr(st.session_state, 'ai_recommendations') and st.session_state.ai_recommendations:
        recommendations = st.session_state.ai_recommendations
        
        # Parse and display recommendations
        display_ai_recommendations(recommendations, insights_options)
        
    else:
        st.info("Click 'Generate New Recommendations' to get AI-powered insights")
        
        # Show sample recommendations as placeholder
        render_sample_recommendations()


def render_priority_actions(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Render priority action items"""
    st.markdown("### 📋 Priority Action Items")
    st.markdown("Ranked interventions based on impact, urgency, and feasibility")
    
    # Generate priority actions
    priority_actions = generate_priority_actions(
        routes_df, braking_df, swerving_df, time_series_df, insights_options
    )
    
    if not priority_actions:
        st.warning("No priority actions generated. Please check your data.")
        return
    
    # Priority summary
    st.markdown("#### 📊 Priority Summary")
    
    priority_summary = analyze_priority_distribution(priority_actions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "High Priority",
            priority_summary['high'],
            help="Actions requiring immediate attention"
        )
    
    with col2:
        st.metric(
            "Medium Priority",
            priority_summary['medium'],
            help="Actions for near-term implementation"
        )
    
    with col3:
        st.metric(
            "Low Priority",
            priority_summary['low'],
            help="Actions for future consideration"
        )
    
    with col4:
        st.metric(
            "Total Actions",
            priority_summary['total'],
            help="Total actionable items identified"
        )
    
    # Priority matrix visualization
    st.markdown("#### 🎯 Priority Matrix")
    
    priority_matrix_fig = create_priority_matrix(priority_actions)
    st.plotly_chart(priority_matrix_fig, use_container_width=True)
    
    # Detailed action list
    st.markdown("#### 📝 Detailed Action Items")
    
    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "High", "Medium", "Low"],
            key="priority_filter"
        )
    
    with filter_col2:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + list(set(action['category'] for action in priority_actions)),
            key="category_filter"
        )
    
    # Apply filters
    filtered_actions = filter_priority_actions(
        priority_actions, priority_filter, category_filter
    )
    
    # Display filtered actions
    display_priority_actions(filtered_actions)


def render_risk_assessment(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Render risk assessment summaries"""
    st.markdown("### ⚠️ Risk Assessment")
    st.markdown("Comprehensive analysis of safety risks and potential incidents")
    
    # Generate risk assessment
    risk_assessment = generate_risk_assessment(
        routes_df, braking_df, swerving_df, time_series_df, insights_options
    )
    
    if not risk_assessment:
        st.warning("Unable to generate risk assessment. Please check your data.")
        return
    
    # Risk overview
    st.markdown("#### 📈 Risk Overview")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.metric(
            "Overall Risk Score",
            f"{risk_assessment['overall_score']:.1f}/10",
            delta=f"{risk_assessment['score_change']:+.1f}",
            help="Composite risk score based on multiple factors"
        )
    
    with risk_col2:
        st.metric(
            "High Risk Areas",
            risk_assessment['high_risk_areas'],
            help="Number of locations with high risk scores"
        )
    
    with risk_col3:
        st.metric(
            "Predicted Incidents",
            f"{risk_assessment['predicted_incidents']:.0f}",
            delta=f"{risk_assessment['incident_change']:+.0f}",
            help=f"Predicted incidents in next {insights_options['risk_timeframe']}"
        )
    
    # Risk breakdown
    st.markdown("#### 🔍 Risk Breakdown")
    
    risk_breakdown_fig = create_risk_breakdown_chart(risk_assessment)
    st.plotly_chart(risk_breakdown_fig, use_container_width=True)
    
    # Risk heatmap
    st.markdown("#### 🗺️ Risk Heatmap")
    
    if braking_df is not None and swerving_df is not None:
        risk_heatmap_fig = create_risk_heatmap(braking_df, swerving_df, risk_assessment)
        st.plotly_chart(risk_heatmap_fig, use_container_width=True)
    
    # Risk factors analysis
    st.markdown("#### 📊 Risk Factors Analysis")
    
    risk_factors_fig = create_risk_factors_chart(risk_assessment)
    st.plotly_chart(risk_factors_fig, use_container_width=True)
    
    # Risk mitigation suggestions
    st.markdown("#### 💡 Risk Mitigation Suggestions")
    
    display_risk_mitigation_suggestions(risk_assessment)


def render_roi_analysis(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Render ROI analysis and calculations"""
    st.markdown("### 💰 ROI Analysis")
    st.markdown("Return on investment calculations for safety improvement initiatives")
    
    # Generate ROI analysis
    roi_analysis = generate_roi_analysis(
        routes_df, braking_df, swerving_df, time_series_df, insights_options
    )
    
    if not roi_analysis:
        st.warning("Unable to generate ROI analysis. Please check your data.")
        return
    
    # ROI summary
    st.markdown("#### 💼 Investment Summary")
    
    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
    
    with roi_col1:
        st.metric(
            "Total Investment",
            f"£{roi_analysis['total_investment']:,.0f}",
            help="Total required investment for all recommended actions"
        )
    
    with roi_col2:
        st.metric(
            "Annual Savings",
            f"£{roi_analysis['annual_savings']:,.0f}",
            help="Expected annual savings from safety improvements"
        )
    
    with roi_col3:
        st.metric(
            "Payback Period",
            f"{roi_analysis['payback_period']:.1f} years",
            help="Time to recover initial investment"
        )
    
    with roi_col4:
        st.metric(
            "Net Present Value",
            f"£{roi_analysis['npv']:,.0f}",
            delta=f"ROI: {roi_analysis['roi_percentage']:.1f}%",
            help="NPV and ROI over selected timeframe"
        )
    
    # ROI breakdown by category
    st.markdown("#### 📊 ROI Breakdown by Category")
    
    roi_breakdown_fig = create_roi_breakdown_chart(roi_analysis)
    st.plotly_chart(roi_breakdown_fig, use_container_width=True)
    
    # Investment timeline
    st.markdown("#### 📅 Investment Timeline")
    
    timeline_fig = create_investment_timeline(roi_analysis, insights_options)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Sensitivity analysis
    st.markdown("#### 🎯 Sensitivity Analysis")
    
    sensitivity_fig = create_sensitivity_analysis(roi_analysis, insights_options)
    st.plotly_chart(sensitivity_fig, use_container_width=True)
    
    # Investment recommendations
    st.markdown("#### 💡 Investment Recommendations")
    
    display_investment_recommendations(roi_analysis)


# Helper functions for data processing and AI integration
def generate_data_summary(routes_df, braking_df, swerving_df, time_series_df):
    """Generate a summary of the data for AI context"""
    summary = {
        'total_routes': len(routes_df) if routes_df is not None else 0,
        'total_braking_hotspots': len(braking_df) if braking_df is not None else 0,
        'total_swerving_hotspots': len(swerving_df) if swerving_df is not None else 0,
        'time_series_records': len(time_series_df) if time_series_df is not None else 0,
    }
    
    # Add more detailed analysis
    if time_series_df is not None and len(time_series_df) > 0:
        summary['recent_incidents'] = time_series_df['incidents'].tail(30).sum() if 'incidents' in time_series_df.columns else 0
        summary['incident_trend'] = 'increasing' if time_series_df['incidents'].diff().tail(7).mean() > 0 else 'decreasing'
    
    if braking_df is not None and len(braking_df) > 0:
        summary['top_braking_intensity'] = braking_df['intensity'].max() if 'intensity' in braking_df.columns else 0
    
    if swerving_df is not None and len(swerving_df) > 0:
        summary['top_swerving_intensity'] = swerving_df['intensity'].max() if 'intensity' in swerving_df.columns else 0
    
    return summary


def generate_ai_recommendations(client, data_summary, insights_options):
    """Generate AI recommendations using Groq API"""
    try:
        # Prepare context for AI
        context = f"""
        You are a cycling safety expert analyzing data from a cycling safety dashboard. 
        
        Current data summary:
        - Total routes: {data_summary['total_routes']}
        - Braking hotspots: {data_summary['total_braking_hotspots']}
        - Swerving hotspots: {data_summary['total_swerving_hotspots']}
        - Time series records: {data_summary['time_series_records']}
        - Recent incidents (last 30 days): {data_summary.get('recent_incidents', 'N/A')}
        - Incident trend: {data_summary.get('incident_trend', 'N/A')}
        - Highest braking intensity: {data_summary.get('top_braking_intensity', 'N/A')}
        - Highest swerving intensity: {data_summary.get('top_swerving_intensity', 'N/A')}
        
        Focus area: {insights_options['recommendation_focus']}
        
        Please provide 5 specific, actionable recommendations for improving cycling safety. 
        For each recommendation, include:
        1. Title (brief, action-oriented)
        2. Description (2-3 sentences)
        3. Priority (High/Medium/Low)
        4. Category (Infrastructure/Behavioral/Environmental/Policy)
        5. Implementation timeframe (Short-term/Medium-term/Long-term)
        6. Estimated impact (High/Medium/Low)
        
        Format as JSON with the following structure:
        {{
            "recommendations": [
                {{
                    "title": "...",
                    "description": "...",
                    "priority": "...",
                    "category": "...",
                    "timeframe": "...",
                    "impact": "...",
                    "reasoning": "..."
                }}
            ]
        }}
        """
        
        # Make API call
        response = client.chat.completions.create(
            model=insights_options['ai_model'],
            messages=[
                {"role": "system", "content": "You are a cycling safety expert providing data-driven recommendations."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Try to parse JSON
        try:
            recommendations = json.loads(response_text)
            return recommendations
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured format
            return {
                "recommendations": [
                    {
                        "title": "AI Response Generated",
                        "description": response_text[:200] + "...",
                        "priority": "Medium",
                        "category": "General",
                        "timeframe": "Medium-term",
                        "impact": "Medium",
                        "reasoning": "AI generated recommendation"
                    }
                ]
            }
    
    except Exception as e:
        logger.error(f"Error generating AI recommendations: {e}")
        return None


def display_ai_recommendations(recommendations, insights_options):
    """Display AI recommendations in a structured format"""
    if not recommendations or 'recommendations' not in recommendations:
        st.error("No recommendations received from AI")
        return
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        with st.expander(f"💡 {i}. {rec.get('title', 'Recommendation')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {rec.get('description', 'N/A')}")
                if rec.get('reasoning'):
                    st.markdown(f"**Reasoning:** {rec.get('reasoning', 'N/A')}")
            
            with col2:
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡', 
                    'Low': '🟢'
                }.get(rec.get('priority', 'Medium'), '🟡')
                
                st.markdown(f"**Priority:** {priority_color} {rec.get('priority', 'Medium')}")
                st.markdown(f"**Category:** {rec.get('category', 'N/A')}")
                st.markdown(f"**Timeframe:** {rec.get('timeframe', 'N/A')}")
                st.markdown(f"**Impact:** {rec.get('impact', 'N/A')}")


def render_sample_recommendations():
    """Render sample recommendations as placeholder"""
    st.markdown("#### 📝 Sample Recommendations")
    
    sample_recommendations = [
        {
            "title": "Install Advanced Warning Signs at High-Risk Junctions",
            "description": "Deploy smart warning signs with LED displays at the top 5 braking hotspots to alert both cyclists and drivers.",
            "priority": "High",
            "category": "Infrastructure",
            "timeframe": "Short-term",
            "impact": "High"
        },
        {
            "title": "Implement Protected Bike Lanes on Major Routes",
            "description": "Create physical barriers between cycling lanes and vehicle traffic on routes with highest incident rates.",
            "priority": "High",
            "category": "Infrastructure", 
            "timeframe": "Medium-term",
            "impact": "High"
        },
        {
            "title": "Launch Targeted Safety Education Campaign",
            "description": "Develop educational materials focusing on the most common risk factors identified in the data analysis.",
            "priority": "Medium",
            "category": "Behavioral",
            "timeframe": "Short-term",
            "impact": "Medium"
        }
    ]
    
    for i, rec in enumerate(sample_recommendations, 1):
        with st.expander(f"💡 {i}. {rec['title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {rec['description']}")
            
            with col2:
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(rec['priority'], '🟡')
                
                st.markdown(f"**Priority:** {priority_color} {rec['priority']}")
                st.markdown(f"**Category:** {rec['category']}")
                st.markdown(f"**Timeframe:** {rec['timeframe']}")
                st.markdown(f"**Impact:** {rec['impact']}")


def generate_priority_actions(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Generate priority actions based on data analysis"""
    actions = []
    
    # Analyze braking hotspots
    if braking_df is not None and len(braking_df) > 0:
        high_intensity_braking = braking_df[braking_df.get('intensity', 0) > 7] if 'intensity' in braking_df.columns else braking_df.head(5)
        
        for _, hotspot in high_intensity_braking.iterrows():
            actions.append({
                'title': f"Address Braking Hotspot at ({hotspot.get('lat', 'N/A'):.3f}, {hotspot.get('lon', 'N/A'):.3f})",
                'description': f"High-intensity braking area requiring immediate attention",
                'priority_score': hotspot.get('intensity', 5) / 10,
                'priority_level': 'High' if hotspot.get('intensity', 5) > 7 else 'Medium',
                'category': 'Infrastructure',
                'impact': 'High',
                'effort': 'Medium',
                'timeframe': 'Short-term',
                'cost_estimate': 15000
            })
    
    # Analyze swerving hotspots
    if swerving_df is not None and len(swerving_df) > 0:
        high_intensity_swerving = swerving_df[swerving_df.get('intensity', 0) > 7] if 'intensity' in swerving_df.columns else swerving_df.head(5)
        
        for _, hotspot in high_intensity_swerving.iterrows():
            actions.append({
                'title': f"Address Swerving Hotspot at ({hotspot.get('lat', 'N/A'):.3f}, {hotspot.get('lon', 'N/A'):.3f})",
                'description': f"High-intensity swerving area requiring safety improvements",
                'priority_score': hotspot.get('intensity', 5) / 10,
                'priority_level': 'High' if hotspot.get('intensity', 5) > 7 else 'Medium',
                'category': 'Infrastructure',
                'impact': 'High',
                'effort': 'Medium',
                'timeframe': 'Short-term',
                'cost_estimate': 12000
            })
    
    # Add generic high-impact actions
    generic_actions = [
        {
            'title': 'Implement Real-time Safety Monitoring System',
            'description': 'Deploy IoT sensors and cameras for continuous safety monitoring',
            'priority_score': 0.85,
            'priority_level': 'High',
            'category': 'Technology',
            'impact': 'High',
            'effort': 'High',
            'timeframe': 'Medium-term',
            'cost_estimate': 50000
        },
        {
            'title': 'Enhance Visibility at Night Routes',
            'description': 'Install LED lighting and reflective materials on high-traffic cycling routes',
            'priority_score': 0.75,
            'priority_level': 'High',
            'category': 'Infrastructure',
            'impact': 'Medium',
            'effort': 'Medium',
            'timeframe': 'Short-term',
            'cost_estimate': 25000
        },
        {
            'title': 'Develop Mobile Safety Alert App',
            'description': 'Create smartphone app to alert cyclists of nearby hazards and incidents',
            'priority_score': 0.65,
            'priority_level': 'Medium',
            'category': 'Technology',
            'impact': 'Medium',
            'effort': 'High',
            'timeframe': 'Long-term',
            'cost_estimate': 75000
        }
    ]
    
    actions.extend(generic_actions)
    
    # Filter actions based on priority threshold
    threshold = insights_options['priority_threshold']
    filtered_actions = [a for a in actions if a['priority_score'] >= threshold]
    
    # Sort by priority score and limit results
    filtered_actions.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return filtered_actions[:insights_options['max_actions']]


def analyze_priority_distribution(actions):
    """Analyze the distribution of priority levels"""
    high = len([a for a in actions if a['priority_level'] == 'High'])
    medium = len([a for a in actions if a['priority_level'] == 'Medium'])
    low = len([a for a in actions if a['priority_level'] == 'Low'])
    
    return {
        'high': high,
        'medium': medium,
        'low': low,
        'total': len(actions)
    }


def create_priority_matrix(actions):
    """Create priority matrix visualization"""
    try:
        # Create impact vs effort matrix
        impact_map = {'High': 3, 'Medium': 2, 'Low': 1}
        effort_map = {'High': 3, 'Medium': 2, 'Low': 1}
        
        impact_scores = [impact_map.get(a['impact'], 2) for a in actions]
        effort_scores = [effort_map.get(a['effort'], 2) for a in actions]
        priorities = [a['priority_level'] for a in actions]
        titles = [a['title'][:50] + '...' if len(a['title']) > 50 else a['title'] for a in actions]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color mapping for priorities
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        
        for priority in ['High', 'Medium', 'Low']:
            mask = [p == priority for p in priorities]
            if any(mask):
                fig.add_trace(go.Scatter(
                    x=[effort_scores[i] for i in range(len(effort_scores)) if mask[i]],
                    y=[impact_scores[i] for i in range(len(impact_scores)) if mask[i]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color_map[priority],
                        line=dict(width=2, color='white')
                    ),
                    name=f'{priority} Priority',
                    text=[titles[i] for i in range(len(titles)) if mask[i]],
                    hovertemplate='<b>%{text}</b><br>Impact: %{y}<br>Effort: %{x}<extra></extra>'
                ))
        
        # Add quadrant lines
        fig.add_hline(y=2.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=2.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=1.5, y=3.5, text="Quick Wins", showarrow=False, font=dict(size=14, color="gray"))
        fig.add_annotation(x=3.5, y=3.5, text="Major Projects", showarrow=False, font=dict(size=14, color="gray"))
        fig.add_annotation(x=1.5, y=1.5, text="Fill-ins", showarrow=False, font=dict(size=14, color="gray"))
        fig.add_annotation(x=3.5, y=1.5, text="Questionable", showarrow=False, font=dict(size=14, color="gray"))
        
        fig.update_layout(
            title="Priority Matrix: Impact vs Effort",
            xaxis_title="Effort Required",
            yaxis_title="Expected Impact",
            xaxis=dict(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
            yaxis=dict(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
            height=500
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating priority matrix: {e}")
        return None


def filter_priority_actions(actions, priority_filter, category_filter):
    """Filter actions based on priority and category"""
    filtered = actions.copy()
    
    if priority_filter != "All":
        filtered = [a for a in filtered if a['priority_level'] == priority_filter]
    
    if category_filter != "All":
        filtered = [a for a in filtered if a['category'] == category_filter]
    
    return filtered


def display_priority_actions(actions):
    """Display priority actions in a structured format"""
    if not actions:
        st.info("No actions match the selected filters")
        return
    
    for i, action in enumerate(actions, 1):
        with st.expander(f"📋 {i}. {action['title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {action['description']}")
                st.markdown(f"**Priority Score:** {action['priority_score']:.2f}")
            
            with col2:
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(action['priority_level'], '🟡')
                
                st.markdown(f"**Priority:** {priority_color} {action['priority_level']}")
                st.markdown(f"**Category:** {action['category']}")
                st.markdown(f"**Impact:** {action['impact']}")
                st.markdown(f"**Effort:** {action['effort']}")
                st.markdown(f"**Timeframe:** {action['timeframe']}")
                st.markdown(f"**Cost:** £{action['cost_estimate']:,}")


def generate_risk_assessment(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Generate comprehensive risk assessment"""
    try:
        # Calculate base risk scores
        base_risk = 5.0  # Starting risk score
        
        # Factor in hotspot intensity
        braking_risk = 0
        if braking_df is not None and len(braking_df) > 0:
            braking_risk = braking_df.get('intensity', pd.Series([0])).mean() * 0.3
        
        swerving_risk = 0
        if swerving_df is not None and len(swerving_df) > 0:
            swerving_risk = swerving_df.get('intensity', pd.Series([0])).mean() * 0.3
        
        # Factor in incident trends
        trend_risk = 0
        if time_series_df is not None and len(time_series_df) > 7:
            recent_incidents = time_series_df['incidents'].tail(7).mean() if 'incidents' in time_series_df.columns else 0
            historical_incidents = time_series_df['incidents'].head(7).mean() if 'incidents' in time_series_df.columns else 0
            trend_risk = max(0, (recent_incidents - historical_incidents) * 0.1)
        
        overall_score = min(10, base_risk + braking_risk + swerving_risk + trend_risk)
        
        # Calculate high risk areas
        high_risk_areas = 0
        if braking_df is not None:
            high_risk_areas += len(braking_df[braking_df.get('intensity', 0) > 7])
        if swerving_df is not None:
            high_risk_areas += len(swerving_df[swerving_df.get('intensity', 0) > 7])
        
        # Predict future incidents
        if time_series_df is not None and len(time_series_df) > 0:
            recent_avg = time_series_df['incidents'].tail(30).mean() if 'incidents' in time_series_df.columns else 10
            timeframe_multiplier = {'1 month': 1, '3 months': 3, '6 months': 6, '1 year': 12}
            predicted_incidents = recent_avg * timeframe_multiplier.get(insights_options['risk_timeframe'], 6)
        else:
            predicted_incidents = 50  # Default estimate
        
        return {
            'overall_score': overall_score,
            'score_change': np.random.uniform(-0.5, 0.5),  # Simulated change
            'high_risk_areas': high_risk_areas,
            'predicted_incidents': predicted_incidents,
            'incident_change': np.random.uniform(-5, 10),  # Simulated change
            'risk_factors': {
                'Infrastructure': 7.2,
                'Weather': 5.8,
                'Traffic': 6.5,
                'Lighting': 4.9,
                'Maintenance': 3.8
            },
            'risk_breakdown': {
                'Braking Events': braking_risk,
                'Swerving Events': swerving_risk,
                'Trend Factor': trend_risk,
                'Base Risk': base_risk
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating risk assessment: {e}")
        return None


def create_risk_breakdown_chart(risk_assessment):
    """Create risk breakdown chart"""
    try:
        categories = list(risk_assessment['risk_factors'].keys())
        values = list(risk_assessment['risk_factors'].values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=['red' if v > 7 else 'orange' if v > 5 else 'green' for v in values],
                text=[f"{v:.1f}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Risk Factors Breakdown",
            xaxis_title="Risk Category",
            yaxis_title="Risk Score (0-10)",
            yaxis=dict(range=[0, 10]),
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating risk breakdown chart: {e}")
        return None


def create_risk_heatmap(braking_df, swerving_df, risk_assessment):
    """Create risk heatmap visualization"""
    try:
        # Combine braking and swerving data
        combined_data = pd.DataFrame()
        
        if braking_df is not None and len(braking_df) > 0:
            braking_data = braking_df[['lat', 'lon']].copy()
            braking_data['risk_score'] = braking_df.get('intensity', 5)
            braking_data['type'] = 'Braking'
            combined_data = pd.concat([combined_data, braking_data])
        
        if swerving_df is not None and len(swerving_df) > 0:
            swerving_data = swerving_df[['lat', 'lon']].copy()
            swerving_data['risk_score'] = swerving_df.get('intensity', 5)
            swerving_data['type'] = 'Swerving'
            combined_data = pd.concat([combined_data, swerving_data])
        
        if combined_data.empty:
            return None
        
        # Create scatter map
        fig = px.scatter_mapbox(
            combined_data,
            lat='lat',
            lon='lon',
            color='risk_score',
            size='risk_score',
            color_continuous_scale='Reds',
            size_max=20,
            zoom=12,
            mapbox_style='open-street-map',
            title='Risk Heatmap',
            hover_data=['type']
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating risk heatmap: {e}")
        return None


def create_risk_factors_chart(risk_assessment):
    """Create risk factors radar chart"""
    try:
        categories = list(risk_assessment['risk_factors'].keys())
        values = list(risk_assessment['risk_factors'].values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Score',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            title="Risk Factors Analysis",
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating risk factors chart: {e}")
        return None


def display_risk_mitigation_suggestions(risk_assessment):
    """Display risk mitigation suggestions"""
    suggestions = [
        {
            'risk_factor': 'Infrastructure',
            'suggestion': 'Implement protected bike lanes and improve junction design',
            'priority': 'High',
            'timeline': '6-12 months'
        },
        {
            'risk_factor': 'Traffic',
            'suggestion': 'Install traffic calming measures and optimize signal timing',
            'priority': 'High',
            'timeline': '3-6 months'
        },
        {
            'risk_factor': 'Weather',
            'suggestion': 'Improve drainage and add weather-resistant road markings',
            'priority': 'Medium',
            'timeline': '3-9 months'
        },
        {
            'risk_factor': 'Lighting',
            'suggestion': 'Upgrade street lighting and add reflective materials',
            'priority': 'Medium',
            'timeline': '2-4 months'
        },
        {
            'risk_factor': 'Maintenance',
            'suggestion': 'Establish regular maintenance schedule for cycling infrastructure',
            'priority': 'Low',
            'timeline': '1-3 months'
        }
    ]
    
    for suggestion in suggestions:
        with st.expander(f"💡 {suggestion['risk_factor']} - {suggestion['suggestion']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Mitigation Strategy:** {suggestion['suggestion']}")
            
            with col2:
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(suggestion['priority'], '🟡')
                
                st.markdown(f"**Priority:** {priority_color} {suggestion['priority']}")
                st.markdown(f"**Timeline:** {suggestion['timeline']}")


def generate_roi_analysis(routes_df, braking_df, swerving_df, time_series_df, insights_options):
    """Generate ROI analysis for safety improvements"""
    try:
        # Calculate base investment requirements
        base_investment = 100000  # Base infrastructure investment
        
        # Factor in number of hotspots
        hotspot_investment = 0
        if braking_df is not None:
            hotspot_investment += len(braking_df) * 15000  # £15k per braking hotspot
        if swerving_df is not None:
            hotspot_investment += len(swerving_df) * 12000  # £12k per swerving hotspot
        
        total_investment = base_investment + hotspot_investment
        
        # Calculate expected savings
        if time_series_df is not None and len(time_series_df) > 0:
            avg_incidents = time_series_df['incidents'].mean() if 'incidents' in time_series_df.columns else 20
        else:
            avg_incidents = 20  # Default estimate
        
        # Cost per incident (insurance, medical, infrastructure damage)
        cost_per_incident = 8500  # Conservative estimate
        
        # Expected reduction in incidents (15-30% typical for safety improvements)
        incident_reduction = 0.25
        
        annual_savings = avg_incidents * 12 * cost_per_incident * incident_reduction
        
        # Calculate ROI metrics
        timeframe_years = int(insights_options['roi_timeframe'].split()[0])
        discount_rate = insights_options['discount_rate'] / 100
        
        # Calculate NPV
        npv = -total_investment
        for year in range(1, timeframe_years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)
        
        # Calculate payback period
        payback_period = total_investment / annual_savings if annual_savings > 0 else float('inf')
        
        # Calculate ROI percentage
        roi_percentage = (npv / total_investment) * 100 if total_investment > 0 else 0
        
        return {
            'total_investment': total_investment,
            'annual_savings': annual_savings,
            'npv': npv,
            'payback_period': payback_period,
            'roi_percentage': roi_percentage,
            'investment_breakdown': {
                'Infrastructure': base_investment,
                'Hotspot Remediation': hotspot_investment,
                'Technology': 50000,
                'Training & Education': 25000,
                'Monitoring Systems': 30000
            },
            'savings_breakdown': {
                'Reduced Incidents': annual_savings * 0.6,
                'Lower Insurance': annual_savings * 0.2,
                'Maintenance Savings': annual_savings * 0.1,
                'Productivity Gains': annual_savings * 0.1
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating ROI analysis: {e}")
        return None


def create_roi_breakdown_chart(roi_analysis):
    """Create ROI breakdown chart"""
    try:
        categories = list(roi_analysis['investment_breakdown'].keys())
        values = list(roi_analysis['investment_breakdown'].values())
        
        fig = go.Figure(data=[
            go.Pie(
                labels=categories,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Investment Breakdown",
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating ROI breakdown chart: {e}")
        return None


def create_investment_timeline(roi_analysis, insights_options):
    """Create investment timeline visualization"""
    try:
        timeframe_years = int(insights_options['roi_timeframe'].split()[0])
        
        years = list(range(0, timeframe_years + 1))
        cumulative_investment = [-roi_analysis['total_investment']] + [0] * timeframe_years
        cumulative_savings = [0]
        
        for year in range(1, timeframe_years + 1):
            cumulative_savings.append(cumulative_savings[-1] + roi_analysis['annual_savings'])
        
        net_cashflow = [inv + sav for inv, sav in zip(cumulative_investment, cumulative_savings)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_investment,
            mode='lines+markers',
            name='Cumulative Investment',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_savings,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=net_cashflow,
            mode='lines+markers',
            name='Net Cash Flow',
            line=dict(color='blue', width=3, dash='dash')
        ))
        
        # Add break-even line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title="Investment Timeline and Cash Flow",
            xaxis_title="Years",
            yaxis_title="Amount (£)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating investment timeline: {e}")
        return None


def create_sensitivity_analysis(roi_analysis, insights_options):
    """Create sensitivity analysis chart"""
    try:
        # Vary key parameters
        base_savings = roi_analysis['annual_savings']
        savings_variations = np.arange(0.5, 1.5, 0.1)
        
        base_investment = roi_analysis['total_investment']
        investment_variations = np.arange(0.8, 1.2, 0.05)
        
        # Calculate NPV for different scenarios
        discount_rate = insights_options['discount_rate'] / 100
        timeframe_years = int(insights_options['roi_timeframe'].split()[0])
        
        npv_grid = np.zeros((len(investment_variations), len(savings_variations)))
        
        for i, inv_mult in enumerate(investment_variations):
            for j, sav_mult in enumerate(savings_variations):
                investment = base_investment * inv_mult
                annual_savings = base_savings * sav_mult
                
                npv = -investment
                for year in range(1, timeframe_years + 1):
                    npv += annual_savings / ((1 + discount_rate) ** year)
                
                npv_grid[i, j] = npv
        
        fig = go.Figure(data=go.Heatmap(
            z=npv_grid,
            x=savings_variations,
            y=investment_variations,
            colorscale='RdYlGn',
            colorbar=dict(title="NPV (£)")
        ))
        
        fig.update_layout(
            title="Sensitivity Analysis: NPV vs Investment and Savings",
            xaxis_title="Savings Multiplier",
            yaxis_title="Investment Multiplier",
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating sensitivity analysis: {e}")
        return None


def display_investment_recommendations(roi_analysis):
    """Display investment recommendations"""
    recommendations = [
        {
            'title': 'Prioritize High-Impact Infrastructure',
            'description': f'Focus on the £{roi_analysis["investment_breakdown"]["Infrastructure"]:,.0f} infrastructure investment for maximum safety impact.',
            'justification': 'Infrastructure improvements typically yield the highest ROI in safety projects.'
        },
        {
            'title': 'Phased Implementation Approach',
            'description': f'Implement in phases over {roi_analysis["payback_period"]:.1f} years to manage cash flow.',
            'justification': 'Phased approach reduces financial risk and allows for iterative improvements.'
        },
        {
            'title': 'Monitor and Measure Results',
            'description': 'Establish KPIs to track the actual ROI and adjust strategies accordingly.',
            'justification': 'Regular monitoring ensures the investment delivers expected returns.'
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"💼 {rec['title']}"):
            st.markdown(f"**Recommendation:** {rec['description']}")
            st.markdown(f"**Justification:** {rec['justification']}")
            
            # Add action button
            if st.button(f"Mark as Implemented", key=f"implement_{rec['title']}"):
                st.success(f"✅ {rec['title']} marked as implemented!")
