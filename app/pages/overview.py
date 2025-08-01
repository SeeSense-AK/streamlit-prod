"""
Simple & Clean Enhanced Overview Page for SeeSense Dashboard
Builds on existing working components with modern styling
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


def load_clean_css():
    """Load clean, modern CSS styling"""
    st.markdown("""
    <style>
    /* Clean background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Modern metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Clean section headers */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Card containers */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-good {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Insights container */
    .insights-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px 0 rgba(102, 126, 234, 0.25);
    }
    
    /* Clean alerts */
    .alert-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .alert-critical {
        border-left-color: #ef4444;
    }
    
    .alert-warning {
        border-left-color: #f59e0b;
    }
    
    .alert-info {
        border-left-color: #3b82f6;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


def render_overview_page():
    """Main overview page renderer - clean and functional"""
    
    # Load styling
    load_clean_css()
    
    # Page header
    st.title("🚲 SeeSense Dashboard Overview")
    st.markdown("**Real-time cycling safety analytics and insights**")
    
    try:
        # Load data using existing data processor
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_state()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Show data status
        render_data_status(available_datasets)
        
        # Add simple filters
        filters = render_simple_filters(routes_df, time_series_df)
        
        # Apply filters if any
        if filters:
            try:
                routes_df, braking_df, swerving_df, time_series_df = apply_simple_filters(
                    routes_df, braking_df, swerving_df, time_series_df, filters
                )
            except Exception as e:
                st.warning(f"Filter error: {e}")
        
        # Render main sections
        render_key_metrics_section(routes_df, braking_df, swerving_df, time_series_df)
        render_ai_insights_section(routes_df, braking_df, swerving_df, time_series_df)
        render_data_visualizations(routes_df, braking_df, swerving_df, time_series_df)
        render_summary_alerts(braking_df, swerving_df, time_series_df)
        
    except Exception as e:
        logger.error(f"Overview page error: {e}")
        st.error("❌ Error loading dashboard")
        st.info("Please check your data files in the Data Setup page.")
        
        with st.expander("Error Details"):
            st.code(str(e))


def render_no_data_state():
    """Clean no data message"""
    st.markdown("""
    <div class="info-card" style="text-align: center; padding: 3rem;">
        <h2>📊 No Data Available</h2>
        <p>Upload your cycling safety data to get started with powerful analytics.</p>
        <p><strong>Next Steps:</strong></p>
        <ol style="text-align: left; display: inline-block;">
            <li>Go to the <strong>Data Setup</strong> page</li>
            <li>Upload your CSV files</li>
            <li>Validate your data</li>
            <li>Return here to see your dashboard</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def render_data_status(available_datasets):
    """Show current data status"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🟢 Active" if len(available_datasets) > 0 else "🔴 No Data"
        st.markdown(f"**Data Status:** {status}")
    
    with col2:
        st.markdown(f"**Datasets:** {len(available_datasets)}/4")
    
    with col3:
        last_updated = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Updated:** {last_updated}")


def render_simple_filters(routes_df, time_series_df):
    """Simple sidebar filters"""
    filters = {}
    
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        
        # Date filter
        if time_series_df is not None and ('date' in time_series_df.columns or 'timestamp' in time_series_df.columns):
            st.markdown("**📅 Time Period**")
            
            period = st.selectbox(
                "Select Period",
                ["All Time", "Last 30 Days", "Last 7 Days", "Custom"],
                index=0
            )
            
            if period == "Custom":
                start_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
                end_date = st.date_input("To", value=datetime.now())
                filters['date_range'] = (start_date, end_date)
            elif period != "All Time":
                days = 30 if period == "Last 30 Days" else 7
                filters['date_range'] = (datetime.now() - timedelta(days=days), datetime.now())
        
        # Route filter
        if routes_df is not None and 'route_name' in routes_df.columns:
            st.markdown("**🛣️ Routes**")
            
            route_options = ["All Routes"] + sorted(routes_df['route_name'].unique().tolist())
            selected_routes = st.multiselect(
                "Select Routes",
                route_options,
                default=["All Routes"]
            )
            
            if "All Routes" not in selected_routes:
                filters['selected_routes'] = selected_routes
        
        # Simple toggle filters
        st.markdown("**📊 Display Options**")
        filters['show_trends'] = st.checkbox("Show Trends", value=True)
        filters['show_maps'] = st.checkbox("Show Maps", value=True)
        filters['show_details'] = st.checkbox("Show Details", value=False)
    
    return filters


def apply_simple_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply simple filters to data"""
    
    # Date filtering
    if filters.get('date_range') and time_series_df is not None:
        start_date, end_date = filters['date_range']
        date_col = 'date' if 'date' in time_series_df.columns else 'timestamp'
        
        if date_col in time_series_df.columns:
            time_series_df[date_col] = pd.to_datetime(time_series_df[date_col])
            mask = (time_series_df[date_col].dt.date >= start_date) & (time_series_df[date_col].dt.date <= end_date)
            time_series_df = time_series_df[mask]
    
    # Route filtering
    if filters.get('selected_routes') and routes_df is not None:
        if 'route_name' in routes_df.columns:
            routes_df = routes_df[routes_df['route_name'].isin(filters['selected_routes'])]
    
    return routes_df, braking_df, swerving_df, time_series_df


def render_key_metrics_section(routes_df, braking_df, swerving_df, time_series_df):
    """Render key metrics using Streamlit's built-in metrics"""
    
    st.markdown('<div class="section-header">📊 Key Performance Metrics</div>', unsafe_allow_html=True)
    
    # Calculate metrics using existing calculator
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    # Display in clean columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safety_score = metrics.get('safety_score', 0)
        safety_delta = metrics.get('safety_score_change', 0)
        st.metric(
            label="🛡️ Safety Score",
            value=f"{safety_score:.1f}/10",
            delta=f"{safety_delta:+.1f}" if safety_delta != 0 else None
        )
    
    with col2:
        total_incidents = metrics.get('total_incidents', 0)
        incidents_change = metrics.get('incidents_change', 0)
        st.metric(
            label="🚨 Total Incidents",
            value=f"{total_incidents:,}",
            delta=f"{incidents_change:+.0f}" if incidents_change != 0 else None,
            delta_color="inverse"  # Lower is better for incidents
        )
    
    with col3:
        total_routes = metrics.get('total_routes', 0)
        routes_change = metrics.get('routes_change', 0)
        st.metric(
            label="🛣️ Active Routes",
            value=f"{total_routes:,}",
            delta=f"{routes_change:+.0f}" if routes_change != 0 else None
        )
    
    with col4:
        daily_rides = metrics.get('avg_daily_rides', 0)
        rides_change = metrics.get('daily_rides_change', 0)
        st.metric(
            label="🚴 Daily Rides",
            value=f"{daily_rides:,.0f}",
            delta=f"{rides_change:+.0f}" if rides_change != 0 else None
        )
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        response_time = metrics.get('avg_response_time', 0)
        response_delta = metrics.get('response_time_change', 0)
        st.metric(
            label="⏱️ Response Time",
            value=f"{response_time:.1f}h",
            delta=f"{response_delta:+.1f}h" if response_delta != 0 else None,
            delta_color="inverse"  # Lower is better
        )
    
    with col2:
        coverage = metrics.get('network_coverage', 0)
        coverage_delta = metrics.get('coverage_change', 0)
        st.metric(
            label="📡 Coverage",
            value=f"{coverage:.1f}%",
            delta=f"{coverage_delta:+.1f}%" if coverage_delta != 0 else None
        )
    
    with col3:
        efficiency = metrics.get('network_efficiency', 0)
        efficiency_delta = metrics.get('efficiency_change', 0)
        st.metric(
            label="⚡ Efficiency",
            value=f"{efficiency:.1f}%",
            delta=f"{efficiency_delta:+.1f}%" if efficiency_delta != 0 else None
        )
    
    with col4:
        satisfaction = metrics.get('user_satisfaction', 0)
        satisfaction_delta = metrics.get('satisfaction_change', 0)
        st.metric(
            label="⭐ Satisfaction",
            value=f"{satisfaction:.1f}/5",
            delta=f"{satisfaction_delta:+.1f}" if satisfaction_delta != 0 else None
        )


def render_ai_insights_section(routes_df, braking_df, swerving_df, time_series_df):
    """Clean AI insights section"""
    
    st.markdown('<div class="section-header">🧠 AI-Generated Insights</div>', unsafe_allow_html=True)
    
    # Calculate metrics for AI
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    try:
        # Generate insights using existing AI system
        generator = create_insights_generator()
        insights = generator.generate_comprehensive_insights(
            metrics=metrics, 
            routes_df=routes_df
        )
        
        executive_summary = generator.generate_executive_summary(
            insights=insights,
            metrics=metrics
        )
        
        # Display in clean container
        st.markdown("""
        <div class="insights-container">
            <h3 style="margin-top: 0; color: white;">📋 Executive Summary</h3>
        """, unsafe_allow_html=True)
        
        if executive_summary:
            st.markdown(f'<p style="color: white; line-height: 1.6;">{executive_summary}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: white;">Generating insights from your data...</p>', unsafe_allow_html=True)
        
        # Show key insights if available
        if insights:
            st.markdown('<h4 style="color: white; margin-top: 1.5rem;">🎯 Key Recommendations</h4>', unsafe_allow_html=True)
            
            for i, insight in enumerate(insights[:3]):  # Show top 3
                priority = getattr(insight, 'impact_level', 'Medium')
                title = getattr(insight, 'title', f'Insight {i+1}')
                description = getattr(insight, 'description', 'Analysis in progress...')
                
                priority_color = {
                    'High': '#ef4444',
                    'Medium': '#f59e0b', 
                    'Low': '#10b981'
                }.get(priority, '#6b7280')
                
                st.markdown(f"""
                <div class="alert-item" style="border-left-color: {priority_color};">
                    <strong>{title}</strong><br>
                    <small style="opacity: 0.9;">{description}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        logger.warning(f"AI insights error: {e}")
        
        # Fallback insights based on data
        st.markdown("""
        <div class="insights-container">
            <h3 style="margin-top: 0; color: white;">📊 Data Summary</h3>
        """, unsafe_allow_html=True)
        
        # Simple data-driven insights
        safety_score = metrics.get('safety_score', 0)
        total_incidents = metrics.get('total_incidents', 0)
        
        if safety_score > 7:
            insight = "✅ Your network is performing well with a strong safety score. Continue monitoring key metrics."
        elif safety_score > 5:
            insight = "⚠️ Safety performance is moderate. Consider focusing on high-incident areas for improvement."
        else:
            insight = "🚨 Safety score indicates areas needing immediate attention. Review incident hotspots urgently."
        
        st.markdown(f'<p style="color: white; line-height: 1.6;">{insight}</p>', unsafe_allow_html=True)
        
        if total_incidents > 0:
            st.markdown(f'<p style="color: white;">📈 Currently tracking {total_incidents:,} incidents across your network.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_data_visualizations(routes_df, braking_df, swerving_df, time_series_df):
    """Clean data visualizations"""
    
    st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Trends", "🗺️ Maps", "📋 Summary"])
    
    with tab1:
        render_trends_section(time_series_df)
    
    with tab2:
        render_maps_section(braking_df, swerving_df, routes_df)
    
    with tab3:
        render_data_summary(routes_df, braking_df, swerving_df, time_series_df)


def render_trends_section(time_series_df):
    """Simple trends visualization"""
    
    if time_series_df is not None and len(time_series_df) > 0:
        st.markdown("### 📈 Time Series Trends")
        
        # Find numeric columns for plotting
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns.tolist()
        date_col = None
        
        # Find date column
        for col in ['date', 'timestamp', 'created_at']:
            if col in time_series_df.columns:
                date_col = col
                break
        
        if date_col and numeric_cols:
            # Convert date column
            time_series_df[date_col] = pd.to_datetime(time_series_df[date_col])
            
            # Select metric to plot
            metric_to_plot = st.selectbox(
                "Select Metric",
                numeric_cols,
                index=0
            )
            
            # Create simple line chart
            fig = px.line(
                time_series_df,
                x=date_col,
                y=metric_to_plot,
                title=f"{metric_to_plot.replace('_', ' ').title()} Over Time",
                template="plotly_white"
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No time series data available for trend analysis")
    else:
        st.info("📈 Upload time series data to see trends")


def render_maps_section(braking_df, swerving_df, routes_df):
    """Simple maps visualization"""
    
    st.markdown("### 🗺️ Geographic Overview")
    
    # Check for location data
    has_location_data = False
    map_data = []
    
    if braking_df is not None and 'latitude' in braking_df.columns and 'longitude' in braking_df.columns:
        braking_sample = braking_df.head(100)  # Limit for performance
        braking_sample['type'] = 'Braking Event'
        braking_sample['color'] = '#ef4444'  # Red
        map_data.append(braking_sample[['latitude', 'longitude', 'type', 'color']])
        has_location_data = True
    
    if swerving_df is not None and 'latitude' in swerving_df.columns and 'longitude' in swerving_df.columns:
        swerving_sample = swerving_df.head(100)  # Limit for performance
        swerving_sample['type'] = 'Swerving Event'
        swerving_sample['color'] = '#f59e0b'  # Orange
        map_data.append(swerving_sample[['latitude', 'longitude', 'type', 'color']])
        has_location_data = True
    
    if has_location_data and map_data:
        # Combine data
        combined_data = pd.concat(map_data, ignore_index=True)
        
        # Create simple scatter map
        fig = px.scatter_mapbox(
            combined_data,
            lat="latitude",
            lon="longitude",
            color="type",
            color_discrete_map={
                'Braking Event': '#ef4444',
                'Swerving Event': '#f59e0b'
            },
            zoom=10,
            height=500,
            title="Safety Events Locations"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary
        col1, col2 = st.columns(2)
        with col1:
            if braking_df is not None:
                st.metric("🛑 Braking Events", f"{len(braking_df):,}")
        with col2:
            if swerving_df is not None:
                st.metric("🔄 Swerving Events", f"{len(swerving_df):,}")
    else:
        st.info("🗺️ No location data available for mapping")


def render_data_summary(routes_df, braking_df, swerving_df, time_series_df):
    """Simple data summary"""
    
    st.markdown("### 📋 Data Summary")
    
    # Create summary table
    summary_data = []
    
    datasets = [
        ("Routes", routes_df),
        ("Braking Events", braking_df),
        ("Swerving Events", swerving_df),
        ("Time Series", time_series_df)
    ]
    
    for name, df in datasets:
        if df is not None:
            summary_data.append({
                "Dataset": name,
                "Records": f"{len(df):,}",
                "Columns": len(df.columns),
                "Status": "✅ Loaded"
            })
        else:
            summary_data.append({
                "Dataset": name,
                "Records": "0",
                "Columns": "0",
                "Status": "❌ Not Available"
            })
    
    # Display as table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Show column information for available datasets
    for name, df in datasets:
        if df is not None:
            with st.expander(f"📊 {name} Columns"):
                cols_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    cols_info.append({
                        "Column": col,
                        "Type": dtype,
                        "Non-Null": f"{non_null:,}"
                    })
                
                cols_df = pd.DataFrame(cols_info)
                st.dataframe(cols_df, use_container_width=True, hide_index=True)


def render_summary_alerts(braking_df, swerving_df, time_series_df):
    """Simple summary alerts"""
    
    st.markdown('<div class="section-header">🚨 Summary & Alerts</div>', unsafe_allow_html=True)
    
    alerts = []
    
    # Generate simple alerts based on data
    if braking_df is not None and len(braking_df) > 0:
        if len(braking_df) > 100:
            alerts.append({
                'type': 'warning',
                'title': 'High Braking Activity',
                'message': f'{len(braking_df)} braking events detected - consider safety review'
            })
        else:
            alerts.append({
                'type': 'info',
                'title': 'Braking Events',
                'message': f'{len(braking_df)} braking events recorded'
            })
    
    if swerving_df is not None and len(swerving_df) > 0:
        if len(swerving_df) > 50:
            alerts.append({
                'type': 'warning',
                'title': 'Swerving Pattern',
                'message': f'{len(swerving_df)} swerving events may indicate road hazards'
            })
        else:
            alerts.append({
                'type': 'info',
                'title': 'Swerving Events',
                'message': f'{len(swerving_df)} swerving events recorded'
            })
    
    if not alerts:
        alerts.append({
            'type': 'info', 
            'title': 'System Status',
            'message': 'Dashboard is operational - upload data to see detailed alerts'
        })
    
    # Display alerts
    for alert in alerts:
        alert_class = f"status-{alert['type']}" if alert['type'] in ['good', 'warning', 'danger'] else 'status-info'
        icon = {'warning': '⚠️', 'danger': '🚨', 'info': 'ℹ️', 'good': '✅'}.get(alert['type'], 'ℹ️')
        
        st.markdown(f"""
        <div class="info-card">
            <span class="status-badge {alert_class}">
                {icon} {alert['title']}
            </span>
            <p style="margin: 0.5rem 0 0 0; color: #64748b;">{alert['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_events = 0
        if braking_df is not None:
            total_events += len(braking_df)
        if swerving_df is not None:
            total_events += len(swerving_df)
        
        st.markdown(f"""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #1e293b; margin: 0;">📊 {total_events:,}</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        datasets_loaded = sum([
            1 for df in [routes_df, braking_df, swerving_df, time_series_df] 
            if df is not None
        ])
        
        st.markdown(f"""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #1e293b; margin: 0;">📁 {datasets_loaded}/4</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Datasets Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate data freshness
        data_freshness = "Current"
        if time_series_df is not None:
            date_col = None
            for col in ['date', 'timestamp', 'created_at']:
                if col in time_series_df.columns:
                    date_col = col
                    break
            
            if date_col:
                try:
                    latest_date = pd.to_datetime(time_series_df[date_col]).max()
                    days_old = (datetime.now() - latest_date).days
                    if days_old > 7:
                        data_freshness = f"{days_old}d old"
                    else:
                        data_freshness = "Recent"
                except:
                    data_freshness = "Unknown"
        
        st.markdown(f"""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #1e293b; margin: 0;">🕒 {data_freshness}</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Data Freshness</p>
        </div>
        """, unsafe_allow_html=True)


# Additional helper functions for data processing

def get_data_health_score(routes_df, braking_df, swerving_df, time_series_df):
    """Calculate overall data health score"""
    score = 0
    max_score = 100
    
    # Dataset availability (40 points)
    datasets = [routes_df, braking_df, swerving_df, time_series_df]
    available_count = sum(1 for df in datasets if df is not None and len(df) > 0)
    score += (available_count / 4) * 40
    
    # Data quality (30 points)
    quality_score = 0
    quality_checks = 0
    
    for df in datasets:
        if df is not None and len(df) > 0:
            quality_checks += 1
            # Check for missing values
            if df.isnull().sum().sum() < len(df) * len(df.columns) * 0.1:  # Less than 10% missing
                quality_score += 1
    
    if quality_checks > 0:
        score += (quality_score / quality_checks) * 30
    
    # Data recency (30 points)
    if time_series_df is not None:
        date_col = None
        for col in ['date', 'timestamp', 'created_at']:
            if col in time_series_df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                latest_date = pd.to_datetime(time_series_df[date_col]).max()
                days_old = (datetime.now() - latest_date).days
                if days_old <= 1:
                    score += 30
                elif days_old <= 7:
                    score += 20
                elif days_old <= 30:
                    score += 10
            except:
                pass
    
    return min(score, max_score)


def create_summary_chart(routes_df, braking_df, swerving_df):
    """Create a simple summary chart"""
    
    # Prepare data for chart
    chart_data = []
    
    if routes_df is not None:
        chart_data.append({'Category': 'Routes', 'Count': len(routes_df), 'Type': 'Infrastructure'})
    
    if braking_df is not None:
        chart_data.append({'Category': 'Braking Events', 'Count': len(braking_df), 'Type': 'Safety Events'})
    
    if swerving_df is not None:
        chart_data.append({'Category': 'Swerving Events', 'Count': len(swerving_df), 'Type': 'Safety Events'})
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        
        fig = px.bar(
            chart_df,
            x='Category',
            y='Count',
            color='Type',
            title='Data Overview',
            template='plotly_white',
            color_discrete_map={
                'Infrastructure': '#3b82f6',
                'Safety Events': '#ef4444'
            }
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True
        )
        
        return fig
    
    return None


def render_data_health_indicator():
    """Render data health indicator"""
    
    # This would be called from the main function with actual data
    # For now, showing the structure
    
    st.markdown("### 🏥 Data Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #10b981; margin: 0;">✅ Good</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Data Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #f59e0b; margin: 0;">⚠️ Check</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Data Completeness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #3b82f6; margin: 0;">📊 85%</h3>
            <p style="color: #64748b; margin: 0.25rem 0 0 0;">Health Score</p>
        </div>
        """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons"""
    
    st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Export functionality would be implemented here")
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.info("Settings panel would open here")
    
    with col4:
        if st.button("❓ Help", use_container_width=True):
            st.info("Help documentation would open here")


# Update the main render function to include additional sections
def render_enhanced_overview_page():
    """Enhanced version with additional sections"""
    
    # Load styling
    load_clean_css()
    
    # Page header
    st.title("🚲 SeeSense Dashboard Overview")
    st.markdown("**Real-time cycling safety analytics and insights**")
    
    try:
        # Load data using existing data processor
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_state()
            render_quick_actions()  # Show actions even with no data
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Show data status with health score
        render_data_status_with_health(available_datasets, routes_df, braking_df, swerving_df, time_series_df)
        
        # Add simple filters
        filters = render_simple_filters(routes_df, time_series_df)
        
        # Apply filters if any
        if filters:
            try:
                routes_df, braking_df, swerving_df, time_series_df = apply_simple_filters(
                    routes_df, braking_df, swerving_df, time_series_df, filters
                )
            except Exception as e:
                st.warning(f"Filter error: {e}")
        
        # Render main sections
        render_key_metrics_section(routes_df, braking_df, swerving_df, time_series_df)
        render_ai_insights_section(routes_df, braking_df, swerving_df, time_series_df)
        render_data_visualizations(routes_df, braking_df, swerving_df, time_series_df)
        render_summary_alerts(routes_df, braking_df, swerving_df, time_series_df)
        render_quick_actions()
        
    except Exception as e:
        logger.error(f"Overview page error: {e}")
        st.error("❌ Error loading dashboard")
        st.info("Please check your data files in the Data Setup page.")
        
        with st.expander("Error Details"):
            st.code(str(e))
        
        render_quick_actions()  # Show actions even on error


def render_data_status_with_health(available_datasets, routes_df, braking_df, swerving_df, time_series_df):
    """Enhanced data status with health indicators"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Active" if len(available_datasets) > 0 else "🔴 No Data"
        st.markdown(f"**Status:** {status}")
    
    with col2:
        st.markdown(f"**Datasets:** {len(available_datasets)}/4")
    
    with col3:
        # Calculate health score
        health_score = get_data_health_score(routes_df, braking_df, swerving_df, time_series_df)
        health_icon = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        st.markdown(f"**Health:** {health_icon} {health_score:.0f}%")
    
    with col4:
        last_updated = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Updated:** {last_updated}")
    
    # Show data summary chart if we have data
    if available_datasets:
        summary_fig = create_summary_chart(routes_df, braking_df, swerving_df)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
