"""
User-Friendly Advanced Analytics Page for SeeSense Dashboard
Simplified interface with AI-generated insights for non-technical audiences
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
import random

# Advanced analytics imports (kept minimal)
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_advanced_analytics_page():
    """Render the user-friendly advanced analytics page"""
    
    # Modern header with gradient styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        animation: slideInUp 0.6s ease-out;
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .analysis-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .ai-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🧠 Smart Analytics</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            AI-powered insights that reveal what your data is really telling you
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Quick overview metrics
        render_quick_insights_overview(routes_df, braking_df, swerving_df, time_series_df)
        
        # Create user-friendly analysis sections
        st.markdown("## 📊 Choose Your Analysis")
        
        analysis_type = st.selectbox(
            "What would you like to discover?",
            [
                "🔍 Safety Patterns - Find when and where cycling is safest",
                "⚠️ Risk Detection - Spot unusual danger zones", 
                "📈 Trend Prediction - See what's coming next",
                "🔗 Hidden Connections - Discover surprising relationships"
            ],
            help="Choose the type of insights you want to explore"
        )
        
        if "Safety Patterns" in analysis_type:
            render_safety_patterns_analysis(time_series_df, routes_df)
        
        elif "Risk Detection" in analysis_type:
            render_risk_detection_analysis(braking_df, swerving_df, time_series_df)
        
        elif "Trend Prediction" in analysis_type:
            render_trend_prediction_analysis(time_series_df, routes_df)
        
        elif "Hidden Connections" in analysis_type:
            render_connections_analysis(routes_df, braking_df, swerving_df, time_series_df)
            
    except Exception as e:
        logger.error(f"Error in advanced analytics page: {e}")
        st.error("⚠️ Something went wrong while analyzing your data.")
        st.info("Try refreshing the page or check that your data files are properly formatted.")


def render_no_data_message():
    """Render friendly message when no data is available"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: #8B4513;">📊 Ready to Discover Insights?</h2>
        <p style="font-size: 1.1rem; color: #8B4513; margin: 1rem 0;">
            Upload your cycling data to unlock powerful AI-driven safety insights!
        </p>
        <p style="color: #8B4513;">
            Go to <strong>Data Setup</strong> to get started with your analysis journey.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_insights_overview(routes_df, braking_df, swerving_df, time_series_df):
    """Render quick overview metrics with AI insights"""
    st.markdown("## ⚡ Quick Insights")
    
    # Calculate basic metrics
    total_routes = len(routes_df) if routes_df is not None else 0
    total_braking_events = len(braking_df) if braking_df is not None else 0
    total_swerving_events = len(swerving_df) if swerving_df is not None else 0
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{total_routes:,}</h3>
            <p style="margin: 0; opacity: 0.9;">Routes Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{total_braking_events:,}</h3>
            <p style="margin: 0; opacity: 0.9;">Braking Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{total_swerving_events:,}</h3>
            <p style="margin: 0; opacity: 0.9;">Swerving Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_score = calculate_overall_risk_score(braking_df, swerving_df, total_routes)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{risk_score}/10</h3>
            <p style="margin: 0; opacity: 0.9;">Safety Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI-generated overview insight
    overview_insight = generate_overview_insight(total_routes, total_braking_events, total_swerving_events, risk_score)
    
    st.markdown(f"""
    <div class="insight-card">
        <h4 style="margin: 0 0 1rem 0;">🤖 AI Insight <span class="ai-badge">DYNAMIC</span></h4>
        <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{overview_insight}</p>
    </div>
    """, unsafe_allow_html=True)


def render_safety_patterns_analysis(time_series_df, routes_df):
    """Render safety patterns analysis in user-friendly format"""
    st.markdown("""
    <div class="analysis-section">
        <h2 style="color: #667eea;">🔍 Safety Patterns Discovery</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Let's find out when and where cycling is safest for you
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if time_series_df is None or len(time_series_df) < 7:
        st.warning("📅 Need at least a week of data to find meaningful patterns")
        return
    
    # Prepare time series data
    ts_data = prepare_user_friendly_time_series(time_series_df)
    
    if ts_data is None:
        st.error("Unable to analyze time patterns in your data")
        return
    
    # Day of week analysis
    st.markdown("### 📅 Best Days to Cycle")
    
    daily_safety = analyze_daily_safety_patterns(ts_data)
    
    if daily_safety is not None:
        # Create appealing visualization
        fig = create_daily_safety_chart(daily_safety)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # AI insight for daily patterns
        daily_insight = generate_daily_pattern_insight(daily_safety)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 What This Means <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{daily_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Time of day analysis
    st.markdown("### 🕐 Safest Times to Ride")
    
    hourly_safety = analyze_hourly_safety_patterns(ts_data)
    
    if hourly_safety is not None:
        fig = create_hourly_safety_chart(hourly_safety)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # AI insight for hourly patterns
        hourly_insight = generate_hourly_pattern_insight(hourly_safety)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Time-Based Insights <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{hourly_insight}</p>
        </div>
        """, unsafe_allow_html=True)


def render_risk_detection_analysis(braking_df, swerving_df, time_series_df):
    """Render risk detection analysis in user-friendly format"""
    st.markdown("""
    <div class="analysis-section">
        <h2 style="color: #f5576c;">⚠️ Risk Detection</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            AI helps identify unusual danger zones and risky situations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if braking_df is None and swerving_df is None:
        st.warning("🚫 Need braking or swerving data to detect risks")
        return
    
    # Detect high-risk zones
    st.markdown("### 🗺️ High-Risk Zone Detection")
    
    risk_zones = detect_user_friendly_risk_zones(braking_df, swerving_df)
    
    if risk_zones is not None and not risk_zones.empty:
        # Create risk zone visualization
        fig = create_risk_zone_map(risk_zones)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show top risk zones in a friendly table
        st.markdown("#### 🚨 Top Areas to Watch Out For")
        
        risk_display = risk_zones.head(5).copy()
        risk_display['risk_level'] = risk_display['risk_score'].apply(get_risk_level_label)
        risk_display['location_description'] = risk_display.apply(generate_location_description, axis=1)
        
        display_cols = ['location_description', 'risk_level', 'incident_count']
        display_names = ['Location', 'Risk Level', 'Total Incidents']
        
        risk_table = risk_display[display_cols].copy()
        risk_table.columns = display_names
        
        st.dataframe(risk_table, use_container_width=True, hide_index=True)
        
        # AI insight for risk zones
        risk_insight = generate_risk_detection_insight(risk_zones)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Risk Analysis <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{risk_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Unusual incident detection
    if time_series_df is not None and len(time_series_df) > 14:
        st.markdown("### 📊 Unusual Activity Detection")
        
        anomalies = detect_user_friendly_anomalies(time_series_df)
        
        if anomalies is not None:
            fig = create_anomaly_chart(anomalies)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # AI insight for anomalies
            anomaly_insight = generate_anomaly_insight(anomalies)
            st.markdown(f"""
            <div class="insight-card">
                <h4 style="margin: 0 0 1rem 0;">🤖 Unusual Patterns <span class="ai-badge">AI GENERATED</span></h4>
                <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{anomaly_insight}</p>
            </div>
            """, unsafe_allow_html=True)


def render_trend_prediction_analysis(time_series_df, routes_df):
    """Render trend prediction analysis in user-friendly format"""
    st.markdown("""
    <div class="analysis-section">
        <h2 style="color: #4ECDC4;">📈 What's Coming Next</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            AI predicts future safety trends based on your cycling patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if time_series_df is None or len(time_series_df) < 14:
        st.warning("📈 Need at least 2 weeks of data to make reliable predictions")
        return
    
    # Simple trend analysis
    st.markdown("### 🔮 Safety Trend Forecast")
    
    predictions = generate_user_friendly_predictions(time_series_df)
    
    if predictions is not None:
        fig = create_prediction_chart(predictions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # AI insight for predictions
        prediction_insight = generate_prediction_insight(predictions)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Future Outlook <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{prediction_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route popularity predictions
    if routes_df is not None and len(routes_df) > 10:
        st.markdown("### 🛣️ Route Recommendations")
        
        route_recommendations = generate_route_recommendations(routes_df)
        
        if route_recommendations:
            st.markdown("#### 🌟 AI-Recommended Safe Routes")
            
            for i, rec in enumerate(route_recommendations[:3], 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 1rem; border-radius: 10px; margin: 1rem 0; color: #2c3e50;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #2c3e50;">Route #{i}</h5>
                    <p style="margin: 0; font-weight: 500;">{rec}</p>
                </div>
                """, unsafe_allow_html=True)


def render_connections_analysis(routes_df, braking_df, swerving_df, time_series_df):
    """Render connections analysis in user-friendly format"""
    st.markdown("""
    <div class="analysis-section">
        <h2 style="color: #FF6B6B;">🔗 Hidden Connections</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Discover surprising relationships in your cycling data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Find meaningful correlations
    st.markdown("### 💡 Surprising Relationships")
    
    connections = find_user_friendly_correlations(routes_df, braking_df, swerving_df, time_series_df)
    
    if connections and len(connections) > 0:
        for i, connection in enumerate(connections[:3], 1):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                       padding: 1.5rem; border-radius: 12px; margin: 1rem 0; color: #2c3e50;">
                <h5 style="margin: 0 0 1rem 0; color: #2c3e50;">Discovery #{i}</h5>
                <p style="margin: 0; font-size: 1.1rem; line-height: 1.6; font-weight: 500;">{connection['description']}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
                    Strength: {connection['strength']} | Confidence: {connection['confidence']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # AI insight for connections
        connections_insight = generate_connections_insight(connections)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Why This Matters <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{connections_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🔍 Need more data to discover meaningful connections")


# Helper functions for user-friendly analysis

def calculate_overall_risk_score(braking_df, swerving_df, total_routes):
    """Calculate a simple 1-10 risk score"""
    try:
        if total_routes == 0:
            return 5  # neutral score
        
        total_incidents = 0
        if braking_df is not None:
            total_incidents += len(braking_df)
        if swerving_df is not None:
            total_incidents += len(swerving_df)
        
        # Simple risk calculation
        incident_rate = total_incidents / max(total_routes, 1)
        
        # Convert to 1-10 scale (inverse - lower incidents = higher safety score)
        risk_score = max(1, min(10, 10 - (incident_rate * 2)))
        
        return round(risk_score, 1)
    
    except Exception:
        return 5.0  # neutral score


def prepare_user_friendly_time_series(time_series_df):
    """Prepare time series data for user-friendly analysis"""
    try:
        df = time_series_df.copy()
        
        if 'date' not in df.columns:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add time-based features
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['month'] = df['date'].dt.month_name()
        
        # Calculate incident rate if we have incidents
        if 'incidents' not in df.columns:
            # Create a synthetic incidents column from available data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['incidents'] = df[numeric_cols].sum(axis=1)
            else:
                df['incidents'] = 1  # default
        
        return df
    
    except Exception as e:
        logger.error(f"Error preparing time series data: {e}")
        return None


def analyze_daily_safety_patterns(ts_data):
    """Analyze safety patterns by day of week"""
    try:
        daily_stats = ts_data.groupby('day_of_week')['incidents'].agg([
            'mean', 'sum', 'count'
        ]).round(2)
        
        # Calculate safety score (inverse of incidents)
        max_incidents = daily_stats['mean'].max()
        daily_stats['safety_score'] = round(10 - (daily_stats['mean'] / max_incidents * 9), 1)
        
        # Reorder by actual day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = daily_stats.reindex([day for day in day_order if day in daily_stats.index])
        
        return daily_stats
    
    except Exception as e:
        logger.error(f"Error analyzing daily patterns: {e}")
        return None


def analyze_hourly_safety_patterns(ts_data):
    """Analyze safety patterns by hour of day"""
    try:
        hourly_stats = ts_data.groupby('hour')['incidents'].agg([
            'mean', 'sum', 'count'
        ]).round(2)
        
        # Calculate safety score
        max_incidents = hourly_stats['mean'].max()
        hourly_stats['safety_score'] = round(10 - (hourly_stats['mean'] / max_incidents * 9), 1)
        
        return hourly_stats
    
    except Exception as e:
        logger.error(f"Error analyzing hourly patterns: {e}")
        return None


def create_daily_safety_chart(daily_stats):
    """Create appealing daily safety chart"""
    try:
        fig = go.Figure()
        
        # Add safety score bars
        fig.add_trace(go.Bar(
            x=daily_stats.index,
            y=daily_stats['safety_score'],
            name='Safety Score',
            marker=dict(
                color=daily_stats['safety_score'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=10,
                colorbar=dict(title="Safety Score")
            ),
            text=daily_stats['safety_score'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(
                text="🗓️ Safety Score by Day of Week",
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis_title="Day of Week",
            yaxis_title="Safety Score (1-10)",
            yaxis=dict(range=[0, 10]),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating daily safety chart: {e}")
        return None


def create_hourly_safety_chart(hourly_stats):
    """Create appealing hourly safety chart"""
    try:
        fig = go.Figure()
        
        # Add safety score line
        fig.add_trace(go.Scatter(
            x=hourly_stats.index,
            y=hourly_stats['safety_score'],
            mode='lines+markers',
            name='Safety Score',
            line=dict(width=3, color='#4ECDC4'),
            marker=dict(size=8, color='#FF6B6B'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=dict(
                text="🕐 Safety Score Throughout the Day",
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis_title="Hour of Day",
            yaxis_title="Safety Score (1-10)",
            yaxis=dict(range=[0, 10]),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating hourly safety chart: {e}")
        return None


def detect_user_friendly_risk_zones(braking_df, swerving_df):
    """Detect risk zones in user-friendly format"""
    try:
        risk_zones = []
        
        # Process braking hotspots
        if braking_df is not None and len(braking_df) > 0:
            for _, row in braking_df.iterrows():
                if 'lat' in row and 'lon' in row:
                    risk_zones.append({
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'type': 'Heavy Braking',
                        'incident_count': row.get('count', 1),
                        'risk_score': min(10, row.get('count', 1) * 2)
                    })
        
        # Process swerving hotspots
        if swerving_df is not None and len(swerving_df) > 0:
            for _, row in swerving_df.iterrows():
                if 'lat' in row and 'lon' in row:
                    risk_zones.append({
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'type': 'Sudden Swerving',
                        'incident_count': row.get('count', 1),
                        'risk_score': min(10, row.get('count', 1) * 2.5)
                    })
        
        if risk_zones:
            risk_df = pd.DataFrame(risk_zones)
            risk_df = risk_df.sort_values('risk_score', ascending=False)
            return risk_df
        
        return None
    
    except Exception as e:
        logger.error(f"Error detecting risk zones: {e}")
        return None


def create_risk_zone_map(risk_zones):
    """Create risk zone map visualization"""
    try:
        fig = px.scatter_mapbox(
            risk_zones,
            lat='lat',
            lon='lon',
            size='risk_score',
            color='risk_score',
            color_continuous_scale='Reds',
            hover_data=['type', 'incident_count'],
            zoom=12,
            height=500,
            title="🗺️ Risk Zones Detected by AI"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating risk zone map: {
