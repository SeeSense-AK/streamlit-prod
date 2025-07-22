"""
Overview Page for SeeSense Dashboard - Clean Version Without Caching Issues
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging

from app.core.data_processor import data_processor
from app.utils.config import config
from app.core.metrics_calculator import metrics_calculator
from app.core.groq_insights_generator import create_insights_generator

logger = logging.getLogger(__name__)


def render_overview_page():
    """Render the main overview page with real data-driven metrics"""
    st.title("📊 Dashboard Overview")
    st.markdown("Real-time insights into cycling safety across your network")
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add filters in sidebar and get filter values
        filters = render_overview_filters(routes_df, time_series_df)
        
        # Apply filters
        try:
            filtered_data = apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters)
            routes_df, braking_df, swerving_df, time_series_df = filtered_data
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
            # Continue with unfiltered data
        
        # Render main overview sections
        render_key_metrics(routes_df, braking_df, swerving_df, time_series_df)
        render_ai_insights_section(routes_df, braking_df, swerving_df, time_series_df)
        render_safety_maps(braking_df, swerving_df, routes_df)
        render_trends_analysis(time_series_df)
        render_recent_alerts(braking_df, swerving_df)
        
    except Exception as e:
        logger.error(f"Error in overview page: {e}")
        st.error("⚠️ An error occurred while loading the overview page.")
        st.info("Please check your data files and try refreshing the page.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(str(e))
            st.button("🔄 Retry", key="overview_retry")


def render_no_data_message():
    """Render message when no data is available"""
    st.warning("⚠️ No data available for the overview dashboard.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Upload your data files** to the `data/raw/` directory
    2. **Use the Data Setup page** to validate and load your data
    3. **Return to this overview** to see your cycling safety insights
    
    **Required files:**
    - `routes.csv` - Route popularity and safety data
    - `braking_hotspots.csv` - Emergency braking locations
    - `swerving_hotspots.csv` - Swerving incident locations  
    - `time_series.csv` - Daily cycling activity data
    """)
    
    if st.button("📊 Go to Data Setup", type="primary"):
        st.switch_page("data_setup")


def render_overview_filters(routes_df, time_series_df):
    """Render sidebar filters for the overview page with robust data handling"""
    st.sidebar.markdown("## 🔍 Filters")
    
    filters = {}
    
    # Date range filter - Always create UI element first
    if time_series_df is not None and not time_series_df.empty and 'date' in time_series_df.columns:
        try:
            # Convert to datetime if needed
            time_series_df['date'] = pd.to_datetime(time_series_df['date'])
            min_date = time_series_df['date'].min().date()
            max_date = time_series_df['date'].max().date()
            
            # Create date range selector
            date_range = st.sidebar.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="overview_date_filter"
            )
            filters['date_range'] = date_range
        except Exception as e:
            logger.warning(f"Error setting up date filter: {e}")
            filters['date_range'] = None
    else:
        filters['date_range'] = None
    
    # Route type filter - Always create UI element first
    if routes_df is not None and not routes_df.empty and 'route_type' in routes_df.columns:
        try:
            # Get unique route types, handling any NaN values
            unique_types = routes_df['route_type'].dropna().unique().tolist()
            route_types = ['All'] + unique_types
            
            route_type = st.sidebar.selectbox(
                "🛣️ Route Type",
                options=route_types,
                index=0,  # Default to 'All'
                key="overview_route_type_filter"
            )
            filters['route_type'] = route_type
        except Exception as e:
            logger.warning(f"Error setting up route type filter: {e}")
            filters['route_type'] = 'All'
    else:
        filters['route_type'] = 'All'
    
    # Minimum popularity filter - Always create UI element first
    if routes_df is not None and not routes_df.empty and 'popularity_rating' in routes_df.columns:
        try:
            # Convert to numeric for min/max calculation
            numeric_popularity = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
            valid_popularity = numeric_popularity.dropna()
            
            if not valid_popularity.empty:
                min_val = float(valid_popularity.min())
                max_val = float(valid_popularity.max())
                
                min_popularity = st.sidebar.slider(
                    "⭐ Minimum Popularity",
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=0.1,
                    key="overview_popularity_filter"
                )
                filters['min_popularity'] = min_popularity
            else:
                filters['min_popularity'] = None
        except Exception as e:
            logger.warning(f"Error setting up popularity filter: {e}")
            filters['min_popularity'] = None
    else:
        filters['min_popularity'] = None
    
    return filters


def apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply filters to all dataframes with proper data type handling"""
    
    # Apply date range filter to time series
    if time_series_df is not None and filters.get('date_range'):
        try:
            if isinstance(filters['date_range'], (list, tuple)) and len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                time_series_df['date'] = pd.to_datetime(time_series_df['date'])
                # Ensure dates are compared with dates, not strings
                mask = (time_series_df['date'].dt.date >= start_date) & (time_series_df['date'].dt.date <= end_date)
                time_series_df = time_series_df[mask]
        except Exception as e:
            logger.warning(f"Error applying date filter: {e}")
    
    # Apply route type filter
    if routes_df is not None and filters.get('route_type') and filters['route_type'] != 'All':
        try:
            if 'route_type' in routes_df.columns:
                routes_df = routes_df[routes_df['route_type'] == filters['route_type']]
        except Exception as e:
            logger.warning(f"Error applying route type filter: {e}")
    
    # Apply popularity filter with proper data type handling
    if routes_df is not None and filters.get('min_popularity') is not None:
        try:
            if 'popularity_rating' in routes_df.columns:
                # Convert popularity_rating to numeric, handling any non-numeric values
                routes_df['popularity_rating'] = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
                routes_df = routes_df.dropna(subset=['popularity_rating'])
                
                # Ensure the filter value is also numeric
                min_popularity_value = float(filters['min_popularity'])
                routes_df = routes_df[routes_df['popularity_rating'] >= min_popularity_value]
        except Exception as e:
            logger.warning(f"Error applying popularity filter: {e}")
    
    return routes_df, braking_df, swerving_df, time_series_df


def render_key_metrics(routes_df, braking_df, swerving_df, time_series_df):
    """Render key performance metrics"""
    st.markdown("### 📊 Key Metrics")
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safety_score = metrics.get('safety_score', 0)
        delta_safety = metrics.get('safety_score_change', 0)
        st.metric(
            label="🛡️ Safety Score", 
            value=f"{safety_score:.1f}/10",
            delta=f"{delta_safety:+.1f}" if delta_safety != 0 else None
        )
    
    with col2:
        total_routes = metrics.get('total_routes', 0)
        st.metric(
            label="🛣️ Total Routes", 
            value=f"{total_routes:,}"
        )
    
    with col3:
        avg_daily_rides = metrics.get('avg_daily_rides', 0)
        delta_rides = metrics.get('daily_rides_change', 0)
        st.metric(
            label="🚴 Daily Rides", 
            value=f"{avg_daily_rides:,.0f}",
            delta=f"{delta_rides:+,.0f}" if delta_rides != 0 else None
        )
    
    with col4:
        if 'avg_response_time' in metrics:
            avg_response_time = metrics.get('avg_response_time', 0)
            st.metric(
                label="⏱️ Avg Response Time",
                value=f"{avg_response_time:.1f}h"
            )
        else:
            network_efficiency = metrics.get('network_efficiency', 0)
            st.metric(
                label="📊 Network Efficiency",
                value=f"{network_efficiency:.1f}%"
            )


def render_ai_insights_section(routes_df, braking_df, swerving_df, time_series_df):
    """Render AI insights section (simplified without caching)"""
    st.markdown("### 🧠 AI-Powered Insights")
    
    # Calculate metrics for the AI insights
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    try:
        # Generate insights
        generator = create_insights_generator()
        insights = generator.generate_comprehensive_insights(
            metrics=metrics, 
            routes_df=routes_df
        )
        
        # Generate executive summary
        executive_summary = generator.generate_executive_summary(
            insights=insights,
            metrics=metrics
        )
        
        if insights and executive_summary:
            # Display executive summary
            st.markdown("#### 📋 Executive Summary")
            st.info(executive_summary)
            
            # Display insights grouped by priority
            st.markdown("#### 🎯 Key Insights")
            
            # Group insights by impact level
            high_impact = [i for i in insights if i.impact_level == 'High']
            medium_impact = [i for i in insights if i.impact_level == 'Medium']
            low_impact = [i for i in insights if i.impact_level == 'Low']
            
            # Display high impact insights with prominent styling
            if high_impact:
                st.markdown("**🔴 High Priority Issues**")
                for insight in high_impact:
                    with st.expander(f"🚨 {insight.title}", expanded=True):
                        st.markdown(f"""
                        <div style="
                            padding: 16px;
                            border-left: 4px solid #d32f2f;
                            background-color: #fef7f7;
                            border-radius: 0 8px 8px 0;
                            margin-bottom: 16px;
                        ">
                            <p><strong>Impact:</strong> {insight.description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show recommendations
                        st.markdown("**Recommended Actions:**")
                        for rec in insight.recommendations:
                            st.write(f"• {rec}")
            
            # Display medium impact insights
            if medium_impact:
                st.markdown("**🟡 Medium Priority Items**")
                for insight in medium_impact[:3]:  # Show top 3 medium priority
                    with st.expander(f"⚠️ {insight.title}"):
                        st.write(insight.description)
                        st.markdown("**Recommendations:**")
                        for rec in insight.recommendations[:2]:  # Show top 2 recommendations
                            st.write(f"• {rec}")
            
            # Show count of low priority insights
            if low_impact:
                st.markdown(f"**🟢 {len(low_impact)} additional low-priority insights available**")
        
        else:
            st.info("No insights generated at this time. Please check your data.")
            
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        st.warning("Unable to generate AI insights at the moment.")


def render_safety_maps(braking_df, swerving_df, routes_df):
    """Render safety maps and spatial analysis"""
    st.markdown("### 🗺️ Safety Maps")
    
    if braking_df is None and swerving_df is None:
        st.info("No hotspot data available for mapping")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if braking_df is not None and not braking_df.empty:
            st.markdown("#### 🚨 Emergency Braking Hotspots")
            
            # Create braking hotspots map
            if 'lat' in braking_df.columns and 'lon' in braking_df.columns:
                fig = px.scatter_mapbox(
                    braking_df,
                    lat="lat",
                    lon="lon",
                    size="severity_score" if "severity_score" in braking_df.columns else None,
                    color="severity_score" if "severity_score" in braking_df.columns else None,
                    hover_data=["severity_score"] if "severity_score" in braking_df.columns else None,
                    color_continuous_scale="Reds",
                    zoom=12,
                    mapbox_style="carto-positron",
                    height=400
                )
                
                # Center the map on the data
                if len(braking_df) > 0:
                    center_lat = braking_df['lat'].mean()
                    center_lon = braking_df['lon'].mean()
                    fig.update_layout(
                        mapbox=dict(center=dict(lat=center_lat, lon=center_lon))
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Braking data missing latitude/longitude columns")
    
    with col2:
        if swerving_df is not None and not swerving_df.empty:
            st.markdown("#### 🔄 Swerving Hotspots")
            
            # Create swerving hotspots map
            if 'lat' in swerving_df.columns and 'lon' in swerving_df.columns:
                fig = px.scatter_mapbox(
                    swerving_df,
                    lat="lat",
                    lon="lon",
                    size="severity_score" if "severity_score" in swerving_df.columns else None,
                    color="severity_score" if "severity_score" in swerving_df.columns else None,
                    hover_data=["severity_score"] if "severity_score" in swerving_df.columns else None,
                    color_continuous_scale="Oranges",
                    zoom=12,
                    mapbox_style="carto-positron",
                    height=400
                )
                
                # Center the map on the data
                if len(swerving_df) > 0:
                    center_lat = swerving_df['lat'].mean()
                    center_lon = swerving_df['lon'].mean()
                    fig.update_layout(
                        mapbox=dict(center=dict(lat=center_lat, lon=center_lon))
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Swerving data missing latitude/longitude columns")


def render_trends_analysis(time_series_df):
    """Render time series trends analysis"""
    st.markdown("### 📈 Trends Analysis")
    
    if time_series_df is None or time_series_df.empty:
        st.info("No time series data available for trend analysis")
        return
    
    # Ensure we have a date column
    if 'date' not in time_series_df.columns:
        st.warning("Time series data missing 'date' column")
        return
    
    # Convert date column
    time_series_df['date'] = pd.to_datetime(time_series_df['date'])
    
    # Get numeric columns for plotting
    numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.info("No numeric data available for trend analysis")
        return
    
    # Create subplots for trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily rides trend
        if 'daily_rides' in numeric_cols:
            fig = px.line(
                time_series_df, 
                x='date', 
                y='daily_rides',
                title="Daily Rides Trend",
                labels={'daily_rides': 'Number of Rides', 'date': 'Date'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Use the first available numeric column
            if numeric_cols:
                fig = px.line(
                    time_series_df, 
                    x='date', 
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0].title()} Trend"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Safety incidents trend (if available)
        safety_cols = [col for col in numeric_cols if 'incident' in col.lower() or 'safety' in col.lower()]
        
        if safety_cols:
            fig = px.line(
                time_series_df, 
                x='date', 
                y=safety_cols[0],
                title=f"{safety_cols[0].title()} Trend",
                line_shape='spline'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) > 1:
            # Use second available numeric column
            fig = px.line(
                time_series_df, 
                x='date', 
                y=numeric_cols[1],
                title=f"{numeric_cols[1].title()} Trend"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def render_recent_alerts(braking_df, swerving_df):
    """Render recent safety alerts"""
    st.markdown("### 🚨 Recent Safety Alerts")
    
    alerts = []
    
    # Check for high severity braking events
    if braking_df is not None and not braking_df.empty and 'severity_score' in braking_df.columns:
        try:
            # Convert severity to numeric
            braking_df['severity_score'] = pd.to_numeric(braking_df['severity_score'], errors='coerce')
            high_severity_braking = braking_df[braking_df['severity_score'] > 7]
            
            for _, event in high_severity_braking.head(3).iterrows():
                alerts.append({
                    'type': '🚨 High Severity Braking',
                    'location': f"Lat: {event.get('lat', 'N/A'):.4f}, Lon: {event.get('lon', 'N/A'):.4f}",
                    'severity': f"{event.get('severity_score', 'N/A'):.1f}",
                    'timestamp': event.get('timestamp', 'Recent')
                })
        except Exception as e:
            logger.warning(f"Error processing braking alerts: {e}")
    
    # Check for high severity swerving events
    if swerving_df is not None and not swerving_df.empty and 'severity_score' in swerving_df.columns:
        try:
            # Convert severity to numeric
            swerving_df['severity_score'] = pd.to_numeric(swerving_df['severity_score'], errors='coerce')
            high_severity_swerving = swerving_df[swerving_df['severity_score'] > 7]
            
            for _, event in high_severity_swerving.head(3).iterrows():
                alerts.append({
                    'type': '🔄 High Severity Swerving',
                    'location': f"Lat: {event.get('lat', 'N/A'):.4f}, Lon: {event.get('lon', 'N/A'):.4f}",
                    'severity': f"{event.get('severity_score', 'N/A'):.1f}",
                    'timestamp': event.get('timestamp', 'Recent')
                })
        except Exception as e:
            logger.warning(f"Error processing swerving alerts: {e}")
    
    if alerts:
        # Display alerts
        for alert in alerts[:5]:  # Show top 5 alerts
            with st.expander(f"{alert['type']} - Severity: {alert['severity']}"):
                st.write(f"**Location:** {alert['location']}")
                st.write(f"**Time:** {alert['timestamp']}")
                st.write(f"**Severity Score:** {alert['severity']}")
    else:
        st.info("No recent high-severity safety alerts")
