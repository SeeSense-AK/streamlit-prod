"""
Overview Page for SeeSense Dashboard
Main dashboard with key metrics, maps, and trends
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

logger = logging.getLogger(__name__)


def render_overview_page():
    """Render the main overview page"""
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
    To use the overview dashboard, you need to:
    1. **Add your data files** to the `data/raw/` directory
    2. **Go to the Data Setup page** to validate your files
    3. **Refresh this page** after your data is loaded
    """)
    
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()


def render_overview_filters(routes_df: Optional[pd.DataFrame], time_series_df: Optional[pd.DataFrame]):
    """Render filters in the sidebar and return filter values"""
    st.sidebar.markdown("### 🎛️ Dashboard Filters")
    
    filters = {}
    
    # Date range filter
    if time_series_df is not None and len(time_series_df) > 0:
        min_date = time_series_df['date'].min().date()
        max_date = time_series_df['date'].max().date()
        
        # Default to last 30 days
        default_start = max(min_date, max_date - timedelta(days=30))
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="overview_date_range"
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filters['start_date'] = date_range[0]
            filters['end_date'] = date_range[1]
        else:
            filters['start_date'] = default_start
            filters['end_date'] = max_date
    
    # Route type filter
    if routes_df is not None and 'route_type' in routes_df.columns:
        route_types = ['All'] + list(routes_df['route_type'].unique())
        selected_route_type = st.sidebar.selectbox(
            "Route Type",
            options=route_types,
            index=0,
            key="overview_route_type"
        )
        filters['route_type'] = selected_route_type
    
    # Minimum severity filter
    severity_threshold = st.sidebar.slider(
        "Minimum Severity Level",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Filter hotspots by minimum severity score",
        key="overview_severity"
    )
    filters['severity'] = severity_threshold
    
    return filters


def apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply filters to all datasets"""
    
    # Apply date filter to time series
    if time_series_df is not None and 'start_date' in filters:
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        time_series_df = time_series_df[
            (time_series_df['date'] >= start_date) & 
            (time_series_df['date'] <= end_date)
        ]
    
    # Apply route type filter
    if routes_df is not None and filters.get('route_type') != 'All':
        route_type = filters.get('route_type')
        if route_type and route_type != 'All':
            routes_df = routes_df[routes_df['route_type'] == route_type]
    
    # Apply severity filter to hotspots
    if 'severity' in filters:
        severity_threshold = filters['severity']
        
        if braking_df is not None and 'severity_score' in braking_df.columns:
            braking_df = braking_df[braking_df['severity_score'] >= severity_threshold]
        
        if swerving_df is not None and 'severity_score' in swerving_df.columns:
            swerving_df = swerving_df[swerving_df['severity_score'] >= severity_threshold]
    
    return routes_df, braking_df, swerving_df, time_series_df


def render_key_metrics(routes_df, braking_df, swerving_df, time_series_df):
    """Render key performance metrics"""
    st.markdown("### 📈 Key Safety Metrics")
    
    # Calculate metrics
    metrics = calculate_key_metrics(routes_df, braking_df, swerving_df, time_series_df)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Routes Analyzed",
            value=f"{metrics['total_routes']:,}",
            delta=metrics['routes_delta']
        )
    
    with col2:
        st.metric(
            label="Active Hotspots",
            value=f"{metrics['total_hotspots']:,}",
            delta=metrics['hotspots_delta']
        )
    
    with col3:
        st.metric(
            label="Safety Score",
            value=f"{metrics['safety_score']:.1f}/10",
            delta=metrics['safety_delta']
        )
    
    with col4:
        st.metric(
            label="Incident Rate",
            value=f"{metrics['incident_rate']:.1f}/1000",
            delta=metrics['incident_delta'],
            help="Incidents per 1000 rides"
        )
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="Total Cyclists",
            value=f"{metrics['total_cyclists']:,}",
            delta=metrics['cyclists_delta']
        )
    
    with col6:
        st.metric(
            label="Avg. Daily Rides",
            value=f"{metrics['avg_daily_rides']:,}",
            delta=metrics['rides_delta']
        )
    
    with col7:
        st.metric(
            label="High-Risk Areas",
            value=f"{metrics['high_risk_areas']:,}",
            delta=metrics['risk_delta']
        )
    
    with col8:
        st.metric(
            label="Infrastructure Coverage",
            value=f"{metrics['infrastructure_coverage']:.1f}%",
            delta=metrics['infrastructure_delta']
        )


def calculate_key_metrics(routes_df, braking_df, swerving_df, time_series_df):
    """Calculate all key metrics for the overview"""
    metrics = {}
    
    # Routes metrics
    if routes_df is not None and len(routes_df) > 0:
        metrics['total_routes'] = len(routes_df)
        metrics['total_cyclists'] = routes_df['distinct_cyclists'].sum()
        metrics['infrastructure_coverage'] = (routes_df['has_bike_lane'].sum() / len(routes_df)) * 100
        metrics['routes_delta'] = "+5% vs prev month"  # Placeholder
        metrics['cyclists_delta'] = "+12% vs prev month"
        metrics['infrastructure_delta'] = "+2.3% vs prev month"
    else:
        metrics['total_routes'] = 0
        metrics['total_cyclists'] = 0
        metrics['infrastructure_coverage'] = 0
        metrics['routes_delta'] = None
        metrics['cyclists_delta'] = None
        metrics['infrastructure_delta'] = None
    
    # Hotspots metrics
    hotspots_count = 0
    high_risk_count = 0
    
    if braking_df is not None and len(braking_df) > 0:
        hotspots_count += len(braking_df)
        if 'severity_score' in braking_df.columns:
            high_risk_count += len(braking_df[braking_df['severity_score'] >= 7])
    
    if swerving_df is not None and len(swerving_df) > 0:
        hotspots_count += len(swerving_df)
        if 'severity_score' in swerving_df.columns:
            high_risk_count += len(swerving_df[swerving_df['severity_score'] >= 7])
    
    metrics['total_hotspots'] = hotspots_count
    metrics['high_risk_areas'] = high_risk_count
    metrics['hotspots_delta'] = "-8% vs prev month"
    metrics['risk_delta'] = "-15% vs prev month"
    
    # Time series metrics
    if time_series_df is not None and len(time_series_df) > 0:
        metrics['avg_daily_rides'] = int(time_series_df['total_rides'].mean())
        metrics['safety_score'] = time_series_df['safety_score'].mean()
        metrics['incident_rate'] = time_series_df['incident_rate'].mean()
        metrics['rides_delta'] = "+8% vs prev month"
        metrics['safety_delta'] = "+0.6 vs prev month"
        metrics['incident_delta'] = "-12% vs prev month"
    else:
        metrics['avg_daily_rides'] = 0
        metrics['safety_score'] = 0
        metrics['incident_rate'] = 0
        metrics['rides_delta'] = None
        metrics['safety_delta'] = None
        metrics['incident_delta'] = None
    
    return metrics


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
                    "lat": False,
                    "lon": False,
                    "severity_score": ":.1f",
                    "incidents_count": True,
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
                color="road_type",
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "lat": False,
                    "lon": False,
                    "incidents_count": True,
                    "road_type": True
                },
                title="Braking Hotspots by Road Type"
            )
        
        # Center the map on the data
        center_lat = braking_df['lat'].mean()
        center_lon = braking_df['lon'].mean()
        
        fig.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon)),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
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
                color_continuous_scale="Purples",
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "lat": False,
                    "lon": False,
                    "severity_score": ":.1f",
                    "incidents_count": True,
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
                color="road_type",
                size_max=20,
                zoom=12,
                mapbox_style=map_style,
                hover_name="hotspot_id",
                hover_data={
                    "lat": False,
                    "lon": False,
                    "incidents_count": True,
                    "road_type": True
                },
                title="Swerving Hotspots by Road Type"
            )
        
        # Center the map on the data
        center_lat = swerving_df['lat'].mean()
        center_lon = swerving_df['lon'].mean()
        
        fig.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon)),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_route_popularity_map(routes_df):
    """Render route popularity map using PyDeck"""
    st.markdown("**Route popularity and usage patterns**")
    
    # Map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        max_routes = st.slider(
            "Max Routes to Show",
            min_value=100,
            max_value=min(2000, len(routes_df)),
            value=min(500, len(routes_df)),
            key="route_map_limit"
        )
        
        color_by = st.selectbox(
            "Color Routes By",
            ["popularity_rating", "distinct_cyclists", "route_type"],
            key="route_color_by"
        )
    
    with col1:
        # Sample routes to avoid performance issues
        sampled_routes = routes_df.sample(min(max_routes, len(routes_df)))
        
        # Create route data for PyDeck
        route_data = []
        for _, row in sampled_routes.iterrows():
            # Color based on selection
            if color_by == "popularity_rating":
                color_intensity = row['popularity_rating'] / 10
                color = [int(255 * color_intensity), int(255 * (1 - color_intensity)), 50, 160]
            elif color_by == "distinct_cyclists":
                max_cyclists = sampled_routes['distinct_cyclists'].max()
                color_intensity = row['distinct_cyclists'] / max_cyclists
                color = [50, int(255 * color_intensity), 100, 160]
            else:  # route_type
                type_colors = {
                    'Commute': [255, 100, 100, 160],
                    'Leisure': [100, 255, 100, 160],
                    'Exercise': [100, 100, 255, 160],
                    'Mixed': [255, 255, 100, 160]
                }
                color = type_colors.get(row['route_type'], [128, 128, 128, 160])
            
            route_data.append({
                'path': [[row['start_lon'], row['start_lat']], [row['end_lon'], row['end_lat']]],
                'color': color,
                'width': max(1, int(row['popularity_rating'] * 2)),
                'route_id': row['route_id'],
                'popularity': row['popularity_rating'],
                'cyclists': row['distinct_cyclists'],
                'type': row['route_type']
            })
        
        # Create PyDeck chart
        view_state = pdk.ViewState(
            latitude=sampled_routes['start_lat'].mean(),
            longitude=sampled_routes['start_lon'].mean(),
            zoom=12,
            pitch=0
        )
        
        layer = pdk.Layer(
            "PathLayer",
            data=route_data,
            get_path="path",
            get_color="color",
            get_width="width",
            width_scale=1,
            width_min_pixels=1,
            pickable=True,
            auto_highlight=True
        )
        
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Route:</b> {route_id}<br>"
                       "<b>Popularity:</b> {popularity}/10<br>"
                       "<b>Cyclists:</b> {cyclists}<br>"
                       "<b>Type:</b> {type}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        )
        
        st.pydeck_chart(deck, use_container_width=True)


def render_trends_analysis(time_series_df):
    """Render trends and time series analysis"""
    st.markdown("### 📈 Safety Trends Analysis")
    
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
    """Render daily trends chart"""
    st.markdown("**Daily safety metrics over time**")
    
    # Create multi-line chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Safety Score & Incident Rate', 'Total Rides & Incidents'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Top chart: Safety score and incident rate
    fig.add_trace(
        go.Scatter(
            x=time_series_df['date'],
            y=time_series_df['safety_score'],
            name='Safety Score',
            line=dict(color='green', width=2),
            yaxis='y'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_series_df['date'],
            y=time_series_df['incident_rate'],
            name='Incident Rate',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # Bottom chart: Total rides and incidents
    fig.add_trace(
        go.Scatter(
            x=time_series_df['date'],
            y=time_series_df['total_rides'],
            name='Total Rides',
            line=dict(color='blue', width=2),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_series_df['date'],
            y=time_series_df['incidents'],
            name='Incidents',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Safety Trends Over Time",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Safety Score", row=1, col=1)
    fig.update_yaxes(title_text="Incident Rate", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_weekly_patterns(time_series_df):
    """Render weekly patterns analysis"""
    st.markdown("**Weekly patterns in cycling safety**")
    
    # Calculate weekly averages
    weekly_data = time_series_df.groupby('day_of_week').agg({
        'incidents': 'mean',
        'total_rides': 'mean',
        'safety_score': 'mean',
        'incident_rate': 'mean'
    }).round(2)
    
    # Ensure proper day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data = weekly_data.reindex(day_order)
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Daily Incidents', 'Average Daily Rides', 'Safety Score by Day', 'Incident Rate by Day'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Incidents by day
    fig.add_trace(
        go.Bar(x=weekly_data.index, y=weekly_data['incidents'], name='Incidents', marker_color='red'),
        row=1, col=1
    )
    
    # Rides by day
    fig.add_trace(
        go.Bar(x=weekly_data.index, y=weekly_data['total_rides'], name='Total Rides', marker_color='blue'),
        row=1, col=2
    )
    
    # Safety score by day
    fig.add_trace(
        go.Bar(x=weekly_data.index, y=weekly_data['safety_score'], name='Safety Score', marker_color='green'),
        row=2, col=1
    )
    
    # Incident rate by day
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
                color='safety_score',
                size='total_rides',
                title="Precipitation vs Safety Incidents",
                labels={
                    'precipitation_mm': 'Precipitation (mm)',
                    'incidents': 'Number of Incidents',
                    'safety_score': 'Safety Score'
                },
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'temperature' in time_series_df.columns:
            # Temperature vs incidents
            fig = px.scatter(
                time_series_df,
                x='temperature',
                y='incidents',
                color='safety_score',
                size='total_rides',
                title="Temperature vs Safety Incidents",
                labels={
                    'temperature': 'Temperature (°C)',
                    'incidents': 'Number of Incidents',
                    'safety_score': 'Safety Score'
                },
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_recent_alerts(braking_df, swerving_df):
    """Render recent safety alerts and notifications"""
    st.markdown("### 🚨 Recent Safety Alerts")
    
    # Generate alerts from recent high-severity incidents
    alerts = generate_safety_alerts(braking_df, swerving_df)
    
    if not alerts:
        st.success("✅ No recent safety alerts")
        return
    
    # Display alerts
    for alert in alerts[:5]:  # Show top 5 alerts
        severity_color = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(alert['severity'], '⚪')
        
        with st.expander(f"{severity_color} {alert['title']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Location:** {alert['location']}")
                st.markdown(f"**Issue:** {alert['description']}")
                st.markdown(f"**Recommendation:** {alert['recommendation']}")
            
            with col2:
                st.metric("Severity", f"{alert['severity_score']:.1f}/10")
                st.metric("Incidents", alert['incident_count'])
                st.markdown(f"**Type:** {alert['type']}")


def generate_safety_alerts(braking_df, swerving_df):
    """Generate safety alerts from hotspot data"""
    alerts = []
    
    # Generate alerts from braking hotspots
    if braking_df is not None and len(braking_df) > 0:
        high_severity_braking = braking_df[
            braking_df.get('severity_score', 0) >= 7
        ].head(3)
        
        for _, hotspot in high_severity_braking.iterrows():
            alerts.append({
                'title': f"High-Severity Braking Hotspot Detected",
                'location': f"{hotspot['road_type']} at ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'description': f"Sudden braking incidents increased by {hotspot['incidents_count']} events",
                'recommendation': f"Review junction design and consider traffic calming measures",
                'severity': 'High',
                'severity_score': hotspot.get('severity_score', 8.0),
                'incident_count': hotspot['incidents_count'],
                'type': 'Braking',
                'date': hotspot.get('date_recorded', datetime.now())
            })
    
    # Generate alerts from swerving hotspots
    if swerving_df is not None and len(swerving_df) > 0:
        high_severity_swerving = swerving_df[
            swerving_df.get('severity_score', 0) >= 7
        ].head(3)
        
        for _, hotspot in high_severity_swerving.iterrows():
            alerts.append({
                'title': f"High-Severity Swerving Hotspot Detected",
                'location': f"{hotspot['road_type']} at ({hotspot['lat']:.4f}, {hotspot['lon']:.4f})",
                'description': f"Swerving incidents increased by {hotspot['incidents_count']} events",
                'recommendation': f"Investigate road surface and potential obstructions",
                'severity': 'High',
                'severity_score': hotspot.get('severity_score', 8.0),
                'incident_count': hotspot['incidents_count'],
                'type': 'Swerving',
                'date': hotspot.get('date_recorded', datetime.now())
            })
    
    # Sort alerts by severity score
    alerts.sort(key=lambda x: x['severity_score'], reverse=True)
    
    return alerts
