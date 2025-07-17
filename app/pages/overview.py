"""
Updated Overview Page for SeeSense Dashboard
Now uses real data-driven calculations instead of hardcoded placeholders
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

# Import our new calculators
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
        
        # Calculate real metrics using our new calculator
        real_metrics = metrics_calculator.calculate_all_overview_metrics(
            routes_df, braking_df, swerving_df, time_series_df
        )
        
        # Calculate priority scores
        priority_data = metrics_calculator.calculate_priority_scores(
            braking_df, swerving_df, routes_df
        )
        
        # Initialize Groq insights generator
        insights_generator = create_insights_generator()
        
        # Generate AI-powered insights
        insights = insights_generator.generate_comprehensive_insights(
            real_metrics, routes_df, priority_data, time_series_df
        )
        
        # Generate executive summary
        executive_summary = insights_generator.generate_executive_summary(
            insights, real_metrics
        )
        
        # Render main overview sections
        render_executive_summary(executive_summary)
        render_key_metrics_enhanced(real_metrics)
        render_ai_insights(insights)
        render_priority_hotspots(priority_data)
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


def render_executive_summary(summary: str):
    """Render AI-generated executive summary"""
    st.markdown("### 🎯 Executive Summary")
    
    # Create an attractive summary card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="margin: 0 0 12px 0; color: white;">🚴‍♂️ Network Performance Overview</h4>
        <div style="font-size: 16px; line-height: 1.6; white-space: pre-line;">
            {summary}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_key_metrics_enhanced(metrics: Dict[str, Any]):
    """Render key performance metrics with real data-driven deltas"""
    st.markdown("### 📈 Key Safety Metrics")
    
    # Display metrics in columns with real deltas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Routes Analyzed",
            value=f"{metrics['total_routes']:,}",
            delta=metrics.get('routes_delta', 'N/A')
        )
    
    with col2:
        st.metric(
            label="Active Hotspots",
            value=f"{metrics['total_hotspots']:,}",
            delta=metrics.get('hotspots_delta', 'N/A')
        )
    
    with col3:
        st.metric(
            label="Safety Score",
            value=f"{metrics['safety_score']:.1f}/10",
            delta=metrics.get('safety_delta', 'N/A')
        )
    
    with col4:
        st.metric(
            label="Incident Rate",
            value=f"{metrics['incident_rate']:.1f}/1000",
            delta=metrics.get('incident_delta', 'N/A'),
            help="Incidents per 1000 rides"
        )
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="Total Cyclists",
            value=f"{metrics['total_cyclists']:,}",
            delta=metrics.get('cyclists_delta', 'N/A')
        )
    
    with col6:
        st.metric(
            label="Avg Daily Rides",
            value=f"{metrics['avg_daily_rides']:,}",
            delta=metrics.get('rides_delta', 'N/A')
        )
    
    with col7:
        st.metric(
            label="High-Risk Areas",
            value=f"{metrics['high_risk_areas']:,}",
            delta=metrics.get('risk_delta', 'N/A')
        )
    
    with col8:
        st.metric(
            label="Infrastructure Coverage",
            value=f"{metrics['infrastructure_coverage']:.1f}%",
            delta=metrics.get('infrastructure_delta', 'N/A')
        )


def render_ai_insights(insights: list):
    """Render AI-generated insights with enhanced formatting"""
    st.markdown("### 🧠 AI-Powered Insights")
    
    if not insights:
        st.info("No insights available. Please check your data.")
        return
    
    # Create tabs for different insight categories
    insight_categories = {}
    for insight in insights:
        category = insight.category
        if category not in insight_categories:
            insight_categories[category] = []
        insight_categories[category].append(insight)
    
    # Create tabs
    tab_names = list(insight_categories.keys())
    tabs = st.tabs([f"🔍 {name}" for name in tab_names])
    
    for tab, category in zip(tabs, tab_names):
        with tab:
            category_insights = insight_categories[category]
            
            for insight in category_insights:
                # Impact level color coding
                impact_colors = {
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
                
                impact_color = impact_colors.get(insight.impact_level, '#6c757d')
                
                # Create insight card
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {impact_color};
                    padding: 16px;
                    margin: 16px 0;
                    background-color: #f8f9fa;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #333;">{insight.title}</h4>
                        <span style="
                            background-color: {impact_color};
                            color: white;
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 12px;
                            font-weight: bold;
                        ">{insight.impact_level} Impact</span>
                    </div>
                    <p style="margin: 8px 0; color: #555; line-height: 1.5;">{insight.description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show data points and recommendations
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if insight.data_points:
                        st.markdown("**📊 Key Data Points:**")
                        for point in insight.data_points:
                            st.markdown(f"• {point}")
                
                with col2:
                    if insight.recommendations:
                        st.markdown("**💡 Recommendations:**")
                        for rec in insight.recommendations:
                            st.markdown(f"• {rec}")
                
                # Confidence score
                st.markdown(f"*Confidence Score: {insight.confidence_score:.0%}*")
                st.markdown("---")


def render_priority_hotspots(priority_data: Dict[str, Any]):
    """Render priority hotspots with data-driven scoring"""
    st.markdown("### 🎯 Priority Hotspots & Routes")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Priority", priority_data['high_priority_count'])
    
    with col2:
        st.metric("Medium Priority", priority_data['medium_priority_count'])
    
    with col3:
        st.metric("Low Priority", priority_data['low_priority_count'])
    
    # Create tabs for different priority types
    hotspot_tab, route_tab = st.tabs(["🚨 Hotspot Priorities", "🛣️ Route Priorities"])
    
    with hotspot_tab:
        if priority_data['hotspot_priorities']:
            # Sort by priority score
            sorted_hotspots = sorted(
                priority_data['hotspot_priorities'],
                key=lambda x: x['priority_score'],
                reverse=True
            )
            
            # Display top 10 hotspots
            st.markdown("**Top Priority Hotspots**")
            
            for i, hotspot in enumerate(sorted_hotspots[:10]):
                priority_colors = {
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
                
                color = priority_colors.get(hotspot['priority_level'], '#6c757d')
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px;
                    margin: 8px 0;
                    background-color: white;
                    border-radius: 8px;
                    border-left: 4px solid {color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div>
                        <strong>#{i+1}: {hotspot['type'].title()} Hotspot</strong><br>
                        <small>{hotspot['description']}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: bold; color: {color};">
                            {hotspot['priority_score']:.1f}
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            {hotspot['priority_level']} Priority
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No hotspot priority data available")
    
    with route_tab:
        if priority_data['route_priorities']:
            # Sort by priority score
            sorted_routes = sorted(
                priority_data['route_priorities'],
                key=lambda x: x['priority_score'],
                reverse=True
            )
            
            # Display top 10 routes
            st.markdown("**Top Priority Routes**")
            
            for i, route in enumerate(sorted_routes[:10]):
                priority_colors = {
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
                
                color = priority_colors.get(route['priority_level'], '#6c757d')
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px;
                    margin: 8px 0;
                    background-color: white;
                    border-radius: 8px;
                    border-left: 4px solid {color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div>
                        <strong>#{i+1}: Route {route['id']}</strong><br>
                        <small>{route['description']}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: bold; color: {color};">
                            {route['priority_score']:.1f}
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            {route['priority_level']} Priority
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No route priority data available")


def render_overview_filters(routes_df, time_series_df):
    """Render sidebar filters for overview page"""
    st.sidebar.markdown("### 🔍 Filters")
    
    filters = {}
    
    # Date range filter
    if time_series_df is not None and 'date' in time_series_df.columns:
        time_series_df['date'] = pd.to_datetime(time_series_df['date'])
        min_date = time_series_df['date'].min().date()
        max_date = time_series_df['date'].max().date()
        
        filters['date_range'] = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="overview_date_range"
        )
    
    # Route type filter
    if routes_df is not None and 'route_type' in routes_df.columns:
        route_types = ['All'] + sorted(routes_df['route_type'].unique().tolist())
        filters['route_type'] = st.sidebar.selectbox(
            "Route Type",
            route_types,
            key="overview_route_type"
        )
    
    # Infrastructure filter
    if routes_df is not None and 'has_bike_lane' in routes_df.columns:
        filters['infrastructure'] = st.sidebar.selectbox(
            "Infrastructure",
            ['All', 'With Bike Lane', 'Without Bike Lane'],
            key="overview_infrastructure"
        )
    
    # Severity threshold for hotspots
    filters['severity_threshold'] = st.sidebar.slider(
        "Min Severity Score",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        key="overview_severity"
    )
    
    return filters


def apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply filters to all dataframes"""
    
    # Apply date range filter to time series
    if time_series_df is not None and 'date_range' in filters:
        date_range = filters['date_range']
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            time_series_df = time_series_df[
                (time_series_df['date'].dt.date >= start_date) & 
                (time_series_df['date'].dt.date <= end_date)
            ]
    
    # Apply route type filter
    if routes_df is not None and 'route_type' in filters:
        route_type = filters['route_type']
        if route_type != 'All':
            routes_df = routes_df[routes_df['route_type'] == route_type]
    
    # Apply infrastructure filter
    if routes_df is not None and 'infrastructure' in filters:
        infrastructure = filters['infrastructure']
        if infrastructure == 'With Bike Lane':
            routes_df = routes_df[routes_df['has_bike_lane'] == True]
        elif infrastructure == 'Without Bike Lane':
            routes_df = routes_df[routes_df['has_bike_lane'] == False]
    
    # Apply severity threshold
    severity_threshold = filters.get('severity_threshold', 0)
    
    if braking_df is not None and 'severity_score' in braking_df.columns:
        braking_df = braking_df[braking_df['severity_score'] >= severity_threshold]
    
    if swerving_df is not None and 'severity_score' in swerving_df.columns:
        swerving_df = swerving_df[swerving_df['severity_score'] >= severity_threshold]
    
    return routes_df, braking_df, swerving_df, time_series_df


def render_no_data_message():
    """Render message when no data is available"""
    st.warning("⚠️ No data available for the overview dashboard.")
    st.markdown("""
    To use the overview dashboard, you need to:
    1. **Add your data files** to the `data/raw/` directory
    2. **Go to the Data Setup page** to validate your files
    3. **Refresh this page** after adding your data
    
    Required files:
    - `routes.csv` - Route data with popularity metrics
    - `braking_hotspots.csv` - Sudden braking incident locations
    - `swerving_hotspots.csv` - Swerving incident locations
    - `time_series.csv` - Daily aggregated cycling data
    """)


def render_safety_maps(braking_df, swerving_df, routes_df):
    """Render safety hotspot maps"""
    st.markdown("### 🗺️ Safety Hotspot Maps")
    
    # Create map tabs
    map_tab1, map_tab2, map_tab3 = st.tabs(["🛑 Braking Hotspots", "↔️ Swerving Hotspots", "🛣️ Route Popularity"])
    
    with map_tab1:
        if braking_df is not None and len(braking_df) > 0:
            render_braking_map(braking_df)
        else:
            st.info("No braking hotspot data available")
    
    with map_tab2:
        if swerving_df is not None and len(swerving_df) > 0:
            render_swerving_map(swerving_df)
        else:
            st.info("No swerving hotspot data available")
    
    with map_tab3:
        if routes_df is not None and len(routes_df) > 0:
            render_route_popularity_map(routes_df)
        else:
            st.info("No route data available")


def render_braking_map(braking_df):
    """Render braking hotspots map"""
    st.markdown("**Sudden braking incident locations**")
    
    # Map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        map_style = st.selectbox(
            "Map Style",
            ["carto-positron", "open-street-map", "carto-darkmatter"],
            key="braking_map_style"
        )
        
        show_severity = st.checkbox("Color by Severity", value=True, key="braking_severity")
    
    with col1:
        # Create the map
        if show_severity and 'severity_score' in braking_df.columns:
            fig = px.scatter_mapbox(
                braking_df,
                lat="lat",
                lon="lon",
                size="incidents_count",
                color="severity_score",
                color_continuous_scale="Reds",
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "incidents_count": True,
                    "severity_score": ":.1f",
                    "road_type": True
                },
                title="Braking Hotspots by Severity"
            )
        else:
            fig = px.scatter_mapbox(
                braking_df,
                lat="lat",
                lon="lon",
                size="incidents_count",
                color_discrete_sequence=["red"],
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "incidents_count": True,
                    "road_type": True
                },
                title="Braking Hotspots"
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_swerving_map(swerving_df):
    """Render swerving hotspots map"""
    st.markdown("**Swerving incident locations**")
    
    # Map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        map_style = st.selectbox(
            "Map Style",
            ["carto-positron", "open-street-map", "carto-darkmatter"],
            key="swerving_map_style"
        )
        
        show_severity = st.checkbox("Color by Severity", value=True, key="swerving_severity")
    
    with col1:
        # Create the map
        if show_severity and 'severity_score' in swerving_df.columns:
            fig = px.scatter_mapbox(
                swerving_df,
                lat="lat",
                lon="lon",
                size="incidents_count",
                color="severity_score",
                color_continuous_scale="Oranges",
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "incidents_count": True,
                    "severity_score": ":.1f",
                    "road_type": True
                },
                title="Swerving Hotspots by Severity"
            )
        else:
            fig = px.scatter_mapbox(
                swerving_df,
                lat="lat",
                lon="lon",
                size="incidents_count",
                color_discrete_sequence=["orange"],
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "incidents_count": True,
                    "road_type": True
                },
                title="Swerving Hotspots"
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_route_popularity_map(routes_df):
    """Render route popularity map"""
    st.markdown("**Route popularity and usage patterns**")
    
    # Map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        map_style = st.selectbox(
            "Map Style",
            ["carto-positron", "open-street-map", "carto-darkmatter"],
            key="routes_map_style"
        )
        
        color_by = st.selectbox(
            "Color By",
            ["popularity_rating", "distinct_cyclists", "route_type"],
            key="routes_color_by"
        )
    
    with col1:
        # Create route lines or points
        if 'start_lat' in routes_df.columns and 'start_lon' in routes_df.columns:
            # Use starting points for routes
            fig = px.scatter_mapbox(
                routes_df,
                lat="start_lat",
                lon="start_lon",
                size="distinct_cyclists",
                color=color_by,
                size_max=15,
                zoom=11,
                mapbox_style=map_style,
                hover_name="route_id",
                hover_data={
                    "distinct_cyclists": True,
                    "popularity_rating": True,
                    "route_type": True,
                    "has_bike_lane": True
                },
                title="Route Popularity"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Route location data not available for mapping")


def render_trends_analysis(time_series_df):
    """Render trends analysis with real data"""
    st.markdown("### 📈 Trends Analysis")
    
    if time_series_df is None or len(time_series_df) == 0:
        st.info("No time series data available for trends analysis")
        return
    
    # Create trend tabs
    trend_tab1, trend_tab2, trend_tab3 = st.tabs(["📊 Daily Trends", "📅 Weekly Patterns", "🌤️ Weather Impact"])
    
    with trend_tab1:
        render_daily_trends(time_series_df)
    
    with trend_tab2:
        render_weekly_patterns(time_series_df)
    
    with trend_tab3:
        render_weather_impact(time_series_df)


def render_daily_trends(time_series_df):
    """Render daily trends analysis"""
    st.markdown("**Daily cycling safety trends**")
    
    # Ensure date column is datetime
    time_series_df['date'] = pd.to_datetime(time_series_df['date'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Incidents', 'Safety Score Trend', 'Total Rides', 'Incident Rate'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Daily incidents
    if 'incidents' in time_series_df.columns:
        fig.add_trace(
            go.Scatter(x=time_series_df['date'], y=time_series_df['incidents'], 
                      name='Incidents', line=dict(color='red')),
            row=1, col=1
        )
    
    # Safety score trend
    if 'safety_score' in time_series_df.columns:
        fig.add_trace(
            go.Scatter(x=time_series_df['date'], y=time_series_df['safety_score'], 
                      name='Safety Score', line=dict(color='green')),
            row=1, col=2
        )
    
    # Total rides
    if 'total_rides' in time_series_df.columns:
        fig.add_trace(
            go.Scatter(x=time_series_df['date'], y=time_series_df['total_rides'], 
                      name='Total Rides', line=dict(color='blue')),
            row=2, col=1
        )
    
    # Incident rate
    if 'incident_rate' in time_series_df.columns:
        fig.add_trace(
            go.Scatter(x=time_series_df['date'], y=time_series_df['incident_rate'], 
                      name='Incident Rate', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(
        height=500,
        title_text="Daily Safety Trends",
        showlegend=False
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def render_weekly_patterns(time_series_df):
    """Render weekly patterns analysis"""
    st.markdown("**Weekly patterns in cycling safety**")
    
    # Calculate weekly averages
    if 'day_of_week' not in time_series_df.columns:
        time_series_df['day_of_week'] = pd.to_datetime(time_series_df['date']).dt.day_name()
    
    weekly_data = time_series_df.groupby('day_of_week').agg({
        'incidents': 'mean',
        'total_rides': 'mean',
        'safety_score': 'mean',
        'incident_rate': 'mean'
    }).round(2)
    
    # Ensure proper day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data = weekly_data.reindex([day for day in day_order if day in weekly_data.index])
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Daily Incidents', 'Average Daily Rides', 'Safety Score by Day', 'Incident Rate by Day'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Incidents by day
    if 'incidents' in weekly_data.columns:
        fig.add_trace(
            go.Bar(x=weekly_data.index, y=weekly_data['incidents'], name='Incidents', marker_color='red'),
            row=1, col=1
        )
    
    # Rides by day
    if 'total_rides' in weekly_data.columns:
        fig.add_trace(
            go.Bar(x=weekly_data.index, y=weekly_data['total_rides'], name='Total Rides', marker_color='blue'),
            row=1, col=2
        )
    
    # Safety score by day
    if 'safety_score' in weekly_data.columns:
        fig.add_trace(
            go.Bar(x=weekly_data.index, y=weekly_data['safety_score'], name='Safety Score', marker_color='green'),
            row=2, col=1
        )
    
    # Incident rate by day
    if 'incident_rate' in weekly_data.columns:
        fig.add_trace(
            go.Bar(x=weekly_data.index, y=weekly_data['incident_rate'], name='Incident Rate', marker_color='orange'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=500,
        title_text="Weekly Safety Patterns",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_weather_impact(time_series_df):
    """Render weather impact analysis"""
    st.markdown("**Impact of weather conditions on cycling safety**")
    
    if 'precipitation_mm' not in time_series_df.columns and 'temperature' not in time_series_df.columns:
        st.info("No weather data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'precipitation_mm' in time_series_df.columns:
            # Precipitation vs incidents
            fig = px.scatter(
                time_series_df,
                x='precipitation_mm',
                y='incidents',
                title="Precipitation vs Incidents",
                labels={'precipitation_mm': 'Precipitation (mm)', 'incidents': 'Incidents'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'temperature' in time_series_df.columns:
            # Temperature vs safety score
            fig = px.scatter(
                time_series_df,
                x='temperature',
                y='safety_score',
                title="Temperature vs Safety Score",
                labels={'temperature': 'Temperature (°C)', 'safety_score': 'Safety Score'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)


def render_recent_alerts(braking_df, swerving_df):
    """Render recent safety alerts"""
    st.markdown("### 🚨 Recent Safety Alerts")
    
    alerts = []
    
    # Get recent braking hotspots
    if braking_df is not None and len(braking_df) > 0:
        if 'date_recorded' in braking_df.columns:
            braking_df['date_recorded'] = pd.to_datetime(braking_df['date_recorded'])
            recent_braking = braking_df[
                braking_df['date_recorded'] >= (braking_df['date_recorded'].max() - timedelta(days=7))
            ]
            
            for _, alert in recent_braking.iterrows():
                alerts.append({
                    'type': 'Braking',
                    'location': f"Lat: {alert['lat']:.4f}, Lon: {alert['lon']:.4f}",
                    'severity': alert.get('severity_score', 'N/A'),
                    'date': alert['date_recorded'],
                    'description': f"New braking hotspot with {alert.get('incidents_count', 0)} incidents"
                })
    
    # Get recent swerving hotspots
    if swerving_df is not None and len(swerving_df) > 0:
        if 'date_recorded' in swerving_df.columns:
            swerving_df['date_recorded'] = pd.to_datetime(swerving_df['date_recorded'])
            recent_swerving = swerving_df[
                swerving_df['date_recorded'] >= (swerving_df['date_recorded'].max() - timedelta(days=7))
            ]
            
            for _, alert in recent_swerving.iterrows():
                alerts.append({
                    'type': 'Swerving',
                    'location': f"Lat: {alert['lat']:.4f}, Lon: {alert['lon']:.4f}",
                    'severity': alert.get('severity_score', 'N/A'),
                    'date': alert['date_recorded'],
                    'description': f"New swerving hotspot with {alert.get('incidents_count', 0)} incidents"
                })
    
    if alerts:
        # Sort by date and severity
        alerts.sort(key=lambda x: (x['date'], x['severity']), reverse=True)
        
        # Display recent alerts
        for alert in alerts[:5]:  # Show top 5 recent alerts
            severity_color = '#dc3545' if alert['severity'] >= 7 else '#ffc107' if alert['severity'] >= 4 else '#28a745'
            
            st.markdown(f"""
            <div style="
                padding: 12px;
                margin: 8px 0;
                background-color: white;
                border-radius: 8px;
                border-left: 4px solid {severity_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>🚨 {alert['type']} Alert</strong><br>
                        <small>{alert['description']}</small><br>
                        <small>📍 {alert['location']}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {severity_color}; font-weight: bold;">
                            Severity: {alert['severity']}
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            {alert['date'].strftime('%Y-%m-%d')}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent safety alerts")
