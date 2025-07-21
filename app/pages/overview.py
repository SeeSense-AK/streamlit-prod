"""
Updated Overview Page for SeeSense Dashboard - WITH CACHING
Complete rewrite implementing the cached AI insights system
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

# Import our new data-driven calculators
from app.core.metrics_calculator import metrics_calculator

# UPDATED IMPORTS - Using cached insights system
from app.core.groq_insights_generator import add_cache_controls, get_insights_with_cache

logger = logging.getLogger(__name__)


def render_overview_page():
    """Render the main overview page with real data-driven metrics and cached AI insights"""
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
        
        # UPDATED: Use cached AI insights instead of the old method
        render_ai_insights_section_cached(routes_df, braking_df, swerving_df, time_series_df)
        
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
    
    # Date range filter
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
            filters['date_range'] = None
    
    # Route type filter
    if routes_df is not None and not routes_df.empty and 'route_type' in routes_df.columns:
        try:
            # Get unique route types, handling any NaN values
            unique_types = routes_df['route_type'].dropna().unique()
            route_types = ['All'] + list(unique_types)
            
            filters['route_type'] = st.sidebar.selectbox(
                "🛣️ Route Type",
                options=route_types,
                key="overview_route_type_filter"
            )
        except Exception as e:
            logger.warning(f"Error setting up route type filter: {e}")
            filters['route_type'] = 'All'
    
    # Minimum popularity filter with data validation
    if routes_df is not None and filters.get('min_popularity'):
    try:
        if 'popularity_rating' in routes_df.columns:
            # FIXED: Ensure both sides of comparison are numeric
            routes_df['popularity_rating'] = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
            routes_df = routes_df.dropna(subset=['popularity_rating'])
            
            # CRITICAL FIX: Convert filter value to numeric too
            min_popularity_value = float(filters['min_popularity'])  # <-- Add this line
            
            routes_df = routes_df[routes_df['popularity_rating'] >= min_popularity_value]  # <-- Use converted value
    except Exception as e:
        logger.warning(f"Error applying popularity filter: {e}")
            filters['min_popularity'] = 1
    
    return filters


def apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply filters to all dataframes with proper data type handling"""
    
    # Apply date range filter to time series
    if time_series_df is not None and filters.get('date_range'):
        try:
            if isinstance(filters['date_range'], (list, tuple)) and len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                time_series_df['date'] = pd.to_datetime(time_series_df['date'])
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
    if routes_df is not None and filters.get('min_popularity'):
        try:
            if 'popularity_rating' in routes_df.columns:
                # FIXED: Convert popularity_rating to numeric, handling any non-numeric values
                routes_df['popularity_rating'] = pd.to_numeric(routes_df['popularity_rating'], errors='coerce')
                routes_df = routes_df.dropna(subset=['popularity_rating'])
                routes_df = routes_df[routes_df['popularity_rating'] >= filters['min_popularity']]
        except Exception as e:
            logger.warning(f"Error applying popularity filter: {e}")
    
    
    return routes_df, braking_df, swerving_df, time_series_df


def render_key_metrics(routes_df, braking_df, swerving_df, time_series_df):
    """Render key performance metrics cards"""
    st.markdown("### 📊 Key Performance Metrics")
    
    # Calculate metrics using the metrics calculator
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safety_score = metrics.get('safety_score', 0)
        safety_delta = metrics.get('safety_delta', 0)
        delta_color = "normal" if safety_delta >= 0 else "inverse"
        
        st.metric(
            label="🛡️ Safety Score",
            value=f"{safety_score:.1f}/10",
            delta=f"{safety_delta:+.1f}" if safety_delta != 0 else None,
            delta_color=delta_color
        )
    
    with col2:
        total_routes = metrics.get('total_routes', 0)
        st.metric(
            label="🛣️ Active Routes", 
            value=f"{total_routes:,}"
        )
    
    with col3:
        daily_rides = metrics.get('avg_daily_rides', 0)
        rides_delta = metrics.get('rides_delta', 0)
        
        st.metric(
            label="🚴 Daily Rides",
            value=f"{daily_rides:,}",
            delta=f"{rides_delta:+,}" if rides_delta != 0 else None
        )
    
    with col4:
        infrastructure_coverage = metrics.get('infrastructure_coverage', 0)
        
        st.metric(
            label="🏗️ Infrastructure Coverage",
            value=f"{infrastructure_coverage:.1f}%"
        )
    
    # Additional metrics in a second row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_incidents = metrics.get('total_incidents', 0)
        st.metric(
            label="⚠️ Total Incidents",
            value=f"{total_incidents:,}"
        )
    
    with col6:
        incident_rate = metrics.get('incident_rate', 0)
        st.metric(
            label="📉 Incident Rate",
            value=f"{incident_rate:.2f}/1000 rides"
        )
    
    with col7:
        high_risk_routes = metrics.get('high_risk_routes', 0)
        st.metric(
            label="🚨 High Risk Routes",
            value=f"{high_risk_routes:,}"
        )
    
    with col8:
        avg_response_time = metrics.get('avg_response_time', 0)
        if avg_response_time > 0:
            st.metric(
                label="⏱️ Avg Response Time",
                value=f"{avg_response_time:.1f}h"
            )
        else:
            st.metric(
                label="📊 Network Efficiency",
                value=f"{metrics.get('network_efficiency', 0):.1f}%"
            )


def render_ai_insights_section_cached(routes_df, braking_df, swerving_df, time_series_df):
    """Render AI insights section WITH CACHING - This is the new cached version"""
    st.markdown("### 🧠 AI-Powered Insights")
    
    # ADD CACHE CONTROLS - This adds the refresh and clear cache buttons to the sidebar
    add_cache_controls()
    
    # Calculate metrics for the AI insights
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    try:
        # CACHED INSIGHTS - This replaces the old create_insights_generator() approach
        insights, executive_summary = get_insights_with_cache(
            metrics=metrics, 
            routes_df=routes_df
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
                    # Color coding for impact levels
                    impact_color = "#d32f2f"  # Red for high impact
                    
                    with st.expander(f"🚨 {insight.title}", expanded=True):
                        # Custom styled content
                        st.markdown(f"""
                        <div style="
                            padding: 16px;
                            border-left: 4px solid {impact_color};
                            background-color: #fef7f7;
                            border-radius: 0 8px 8px 0;
                            margin-bottom: 16px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <h4 style="margin: 0; color: #333;">{insight.title}</h4>
                                <span style="
                                    background-color: {impact_color};
                                    color: white;
                                    padding: 4px 12px;
                                    border-radius: 16px;
                                    font-size: 12px;
                                    font-weight: bold;
                                ">{insight.impact_level} Impact</span>
                            </div>
                            <p style="margin: 8px 0; color: #555; line-height: 1.6; font-size: 14px;">{insight.description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show recommendations
                        if insight.recommendations:
                            st.markdown("**🎯 Immediate Actions:**")
                            for rec in insight.recommendations:
                                st.markdown(f"• {rec}")
                        
                        # Show data points
                        if insight.data_points:
                            st.markdown("**📊 Supporting Data:**")
                            for point in insight.data_points:
                                st.markdown(f"• {point}")
                        
                        # Show confidence score
                        confidence = insight.confidence_score
                        st.markdown(f"*Confidence: {confidence:.0%}*")
            
            # Display medium impact insights
            if medium_impact:
                st.markdown("**🟡 Medium Priority Opportunities**")
                for insight in medium_impact:
                    impact_color = "#f57c00"  # Orange for medium impact
                    
                    with st.expander(f"⚡ {insight.title}"):
                        st.markdown(f"""
                        <div style="
                            padding: 16px;
                            border-left: 4px solid {impact_color};
                            background-color: #fff8f0;
                            border-radius: 0 8px 8px 0;
                            margin-bottom: 16px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <h4 style="margin: 0; color: #333;">{insight.title}</h4>
                                <span style="
                                    background-color: {impact_color};
                                    color: white;
                                    padding: 4px 12px;
                                    border-radius: 16px;
                                    font-size: 12px;
                                    font-weight: bold;
                                ">{insight.impact_level} Impact</span>
                            </div>
                            <p style="margin: 8px 0; color: #555; line-height: 1.6; font-size: 14px;">{insight.description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show recommendations and data
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if insight.recommendations:
                                st.markdown("**💡 Recommendations:**")
                                for rec in insight.recommendations:
                                    st.markdown(f"• {rec}")
                        
                        with col2:
                            if insight.data_points:
                                st.markdown("**📊 Key Data:**")
                                for point in insight.data_points:
                                    st.markdown(f"• {point}")
                        
                        st.markdown(f"*Confidence: {insight.confidence_score:.0%}*")
            
            # Display low impact insights (collapsed by default)
            if low_impact:
                with st.expander("🟢 Additional Insights (Low Priority)"):
                    for insight in low_impact:
                        st.markdown(f"**{insight.title}**: {insight.description}")
                        if insight.recommendations:
                            st.markdown(f"*Recommendation: {insight.recommendations[0]}*")
                        st.markdown("---")
        
        else:
            st.info("No AI insights available for current data. Please ensure you have sufficient data and a valid API configuration.")
            
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        st.error("⚠️ Error generating AI insights")
        st.info("AI insights are temporarily unavailable. The dashboard will continue to work with basic metrics.")
        
        # Show error details for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(str(e))


def render_safety_maps(braking_df, swerving_df, routes_df):
    """Render safety hotspot maps"""
    st.markdown("### 🗺️ Safety Hotspot Maps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Braking Hotspots")
        if braking_df is not None and not braking_df.empty:
            # Create braking hotspots map
            try:
                center_lat = braking_df['lat'].mean()
                center_lon = braking_df['lon'].mean()
                
                fig = px.scatter_mapbox(
                    braking_df,
                    lat='lat',
                    lon='lon',
                    size='intensity',
                    color='intensity',
                    color_continuous_scale='Reds',
                    mapbox_style='open-street-map',
                    zoom=12,
                    center={'lat': center_lat, 'lon': center_lon},
                    height=400,
                    title="Emergency Braking Incidents"
                )
                fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error rendering braking hotspots map: {e}")
        else:
            st.info("No braking hotspot data available")
    
    with col2:
        st.markdown("#### 🌪️ Swerving Hotspots")
        if swerving_df is not None and not swerving_df.empty:
            # Create swerving hotspots map
            try:
                center_lat = swerving_df['lat'].mean()
                center_lon = swerving_df['lon'].mean()
                
                fig = px.scatter_mapbox(
                    swerving_df,
                    lat='lat',
                    lon='lon',
                    size='intensity',
                    color='intensity',
                    color_continuous_scale='Blues',
                    mapbox_style='open-street-map',
                    zoom=12,
                    center={'lat': center_lat, 'lon': center_lon},
                    height=400,
                    title="Swerving Incidents"
                )
                fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error rendering swerving hotspots map: {e}")
        else:
            st.info("No swerving hotspot data available")


def render_trends_analysis(time_series_df):
    """Render time series trends analysis"""
    st.markdown("### 📈 Trends Analysis")
    
    if time_series_df is not None and not time_series_df.empty:
        try:
            # Ensure date column is datetime
            time_series_df['date'] = pd.to_datetime(time_series_df['date'])
            
            # Create trends charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily rides trend
                if 'daily_rides' in time_series_df.columns:
                    fig = px.line(
                        time_series_df,
                        x='date',
                        y='daily_rides',
                        title='Daily Cycling Activity',
                        labels={'daily_rides': 'Number of Rides', 'date': 'Date'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Daily rides data not available")
            
            with col2:
                # Safety incidents trend
                if 'incident_count' in time_series_df.columns:
                    fig = px.line(
                        time_series_df,
                        x='date',
                        y='incident_count',
                        title='Safety Incidents Over Time',
                        labels={'incident_count': 'Number of Incidents', 'date': 'Date'},
                        line_shape='spline'
                    )
                    fig.update_traces(line_color='red')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Incident count data not available")
            
            # Weekly patterns
            st.markdown("#### 📅 Weekly Patterns")
            try:
                # Add day of week
                time_series_df['day_of_week'] = time_series_df['date'].dt.day_name()
                
                # Group by day of week
                if 'daily_rides' in time_series_df.columns:
                    weekly_pattern = time_series_df.groupby('day_of_week')['daily_rides'].mean().reset_index()
                    
                    # Reorder days
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekly_pattern['day_of_week'] = pd.Categorical(weekly_pattern['day_of_week'], categories=day_order, ordered=True)
                    weekly_pattern = weekly_pattern.sort_values('day_of_week')
                    
                    fig = px.bar(
                        weekly_pattern,
                        x='day_of_week',
                        y='daily_rides',
                        title='Average Daily Rides by Day of Week',
                        labels={'day_of_week': 'Day of Week', 'daily_rides': 'Average Rides'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                logger.warning(f"Error creating weekly patterns: {e}")
                
        except Exception as e:
            logger.error(f"Error rendering trends analysis: {e}")
            st.error("Error loading trends analysis")
    else:
        st.info("No time series data available for trends analysis")


def render_recent_alerts(braking_df, swerving_df):
    """Render recent safety alerts with robust data handling"""
    st.markdown("### 🚨 Recent Safety Alerts")
    
    alerts = []
    
    # Get recent high-intensity braking incidents
    if braking_df is not None and not braking_df.empty:
        try:
            if 'date_recorded' in braking_df.columns and 'intensity' in braking_df.columns:
                # Convert intensity to numeric
                braking_df['intensity'] = pd.to_numeric(braking_df['intensity'], errors='coerce')
                braking_df = braking_df.dropna(subset=['intensity'])
                
                if not braking_df.empty:
                    recent_braking = braking_df[braking_df['intensity'] >= 8.0].copy()
                    
                    if not recent_braking.empty:
                        recent_braking['date_recorded'] = pd.to_datetime(recent_braking['date_recorded'], errors='coerce')
                        recent_braking = recent_braking.dropna(subset=['date_recorded'])
                        recent_braking = recent_braking.sort_values('date_recorded', ascending=False).head(5)
                        
                        for _, row in recent_braking.iterrows():
                            alerts.append({
                                'type': '🚨 High-Intensity Braking',
                                'location': f"({row['lat']:.4f}, {row['lon']:.4f})" if 'lat' in row and 'lon' in row else "Location unavailable",
                                'intensity': row['intensity'],
                                'date': row['date_recorded'].strftime('%Y-%m-%d') if pd.notnull(row['date_recorded']) else "Date unavailable",
                                'severity': 'High' if row['intensity'] >= 9.0 else 'Medium'
                            })
        except Exception as e:
            logger.warning(f"Error processing braking alerts: {e}")
    
    # Get recent high-intensity swerving incidents
    if swerving_df is not None and not swerving_df.empty:
        try:
            if 'date_recorded' in swerving_df.columns and 'intensity' in swerving_df.columns:
                # Convert intensity to numeric
                swerving_df['intensity'] = pd.to_numeric(swerving_df['intensity'], errors='coerce')
                swerving_df = swerving_df.dropna(subset=['intensity'])
                
                if not swerving_df.empty:
                    recent_swerving = swerving_df[swerving_df['intensity'] >= 8.0].copy()
                    
                    if not recent_swerving.empty:
                        recent_swerving['date_recorded'] = pd.to_datetime(recent_swerving['date_recorded'], errors='coerce')
                        recent_swerving = recent_swerving.dropna(subset=['date_recorded'])
                        recent_swerving = recent_swerving.sort_values('date_recorded', ascending=False).head(5)
                        
                        for _, row in recent_swerving.iterrows():
                            alerts.append({
                                'type': '🌪️ High-Intensity Swerving',
                                'location': f"({row['lat']:.4f}, {row['lon']:.4f})" if 'lat' in row and 'lon' in row else "Location unavailable",
                                'intensity': row['intensity'],
                                'date': row['date_recorded'].strftime('%Y-%m-%d') if pd.notnull(row['date_recorded']) else "Date unavailable",
                                'severity': 'High' if row['intensity'] >= 9.0 else 'Medium'
                            })
        except Exception as e:
            logger.warning(f"Error processing swerving alerts: {e}")
    
    # Display alerts
    if alerts:
        # Sort by date and intensity
        alerts_df = pd.DataFrame(alerts)
        
        # Handle date sorting safely
        try:
            alerts_df['date_for_sort'] = pd.to_datetime(alerts_df['date'], errors='coerce')
            alerts_df = alerts_df.sort_values(['date_for_sort', 'intensity'], ascending=[False, False], na_position='last')
        except:
            # Fallback to intensity-only sorting
            alerts_df = alerts_df.sort_values('intensity', ascending=False)
        
        for _, alert in alerts_df.head(10).iterrows():
            severity_color = "🔴" if alert['severity'] == 'High' else "🟡"
            
            st.markdown(f"""
            **{severity_color} {alert['type']}**  
            📍 Location: {alert['location']}  
            📊 Intensity: {alert['intensity']:.1f}/10  
            📅 Date: {alert['date']}
            """)
            st.markdown("---")
    else:
        st.info("No recent high-intensity safety alerts to display")
