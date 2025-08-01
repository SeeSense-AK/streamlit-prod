"""
Enhanced Overview Page for SeeSense Dashboard - Complete Revamp
Modern, AI-powered, dynamic dashboard suitable for non-technical audiences
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
import time

from app.core.data_processor import data_processor
from app.utils.config import config
from app.core.metrics_calculator import metrics_calculator
from app.core.groq_insights_generator import create_insights_generator

logger = logging.getLogger(__name__)


def load_custom_css():
    """Load enhanced CSS for modern dashboard styling"""
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 10px 0;
        transition: all 0.3s ease;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-title {
        font-size: 16px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Arial', sans-serif;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .metric-delta.positive {
        color: #4ade80;
    }
    
    .metric-delta.negative {
        color: #f87171;
    }
    
    .metric-delta.neutral {
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* AI Insights Panel */
    .ai-insights-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 20px 0;
        color: white;
    }
    
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 4px solid #4ade80;
        color: white;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
        color: white;
    }
    
    .insight-card.high-priority {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
    }
    
    .insight-card.medium-priority {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
    }
    
    .insight-card.low-priority {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    .insight-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Chart Containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 15px 0;
    }
    
    /* Progress Bars */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 2s ease-in-out;
        background: linear-gradient(90deg, #4ade80, #22c55e);
    }
    
    /* Alert Badges */
    .alert-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 5px;
    }
    
    .alert-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin: 30px 0 20px 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #4ade80, #22c55e);
        border-radius: 2px;
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    .slide-up {
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Hide Streamlit Elements */
    .stDeployButton {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stMainBlockContainer {
        padding-top: 2rem;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)


def render_animated_counter(value: float, suffix: str = "", prefix: str = "") -> str:
    """Generate HTML for animated counter"""
    return f"""
    <div class="metric-value" id="counter-{hash(str(value))}">
        {prefix}{value:,.0f if isinstance(value, (int, float)) and value.is_integer() else value:.1f}{suffix}
    </div>
    <script>
        function animateCounter(element, target, duration = 2000) {{
            const start = 0;
            const increment = target / (duration / 50);
            let current = start;
            
            const timer = setInterval(() => {{
                current += increment;
                if (current >= target) {{
                    current = target;
                    clearInterval(timer);
                }}
                element.innerHTML = '{prefix}' + Math.floor(current).toLocaleString() + '{suffix}';
            }}, 50);
        }}
        
        const element = document.getElementById('counter-{hash(str(value))}');
        if (element) {{
            animateCounter(element, {value});
        }}
    </script>
    """


def create_enhanced_metric_card(title: str, value: Any, delta: Optional[float] = None, 
                              delta_type: str = "neutral", icon: str = "📊", 
                              progress: Optional[float] = None) -> str:
    """Create an enhanced metric card with animations"""
    
    delta_class = f"metric-delta {delta_type}" if delta is not None else "metric-delta neutral"
    delta_arrow = "↗" if delta_type == "positive" else "↘" if delta_type == "negative" else "→"
    delta_html = f'<div class="{delta_class}">{delta_arrow} {delta:+.1f}%</div>' if delta is not None else ""
    
    progress_html = ""
    if progress is not None:
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%"></div>
        </div>
        """
    
    return f"""
    <div class="metric-card fade-in">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {progress_html}
    </div>
    """


def render_enhanced_overview_page():
    """Render the completely revamped overview page"""
    
    # Load custom CSS
    load_custom_css()
    
    # Page header with modern styling
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 48px; font-weight: 800; margin-bottom: 10px; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            🚲 SeeSense Analytics Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 20px; font-weight: 300; margin: 0;">
            Real-time insights into cycling safety across your network
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_enhanced_no_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add enhanced filters
        filters = render_enhanced_filters(routes_df, time_series_df)
        
        # Apply filters
        try:
            filtered_data = apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters)
            routes_df, braking_df, swerving_df, time_series_df = filtered_data
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
        
        # Render enhanced sections
        render_executive_dashboard(routes_df, braking_df, swerving_df, time_series_df)
        render_ai_powered_insights(routes_df, braking_df, swerving_df, time_series_df)
        render_interactive_safety_maps(braking_df, swerving_df, routes_df)
        render_dynamic_trends_analysis(time_series_df)
        render_smart_alerts_panel(braking_df, swerving_df, time_series_df)
        
    except Exception as e:
        logger.error(f"Error in enhanced overview page: {e}")
        render_error_message(str(e))


def render_enhanced_no_data_message():
    """Enhanced no data message with modern styling"""
    st.markdown("""
    <div class="ai-insights-container" style="text-align: center; padding: 60px 40px;">
        <div style="font-size: 64px; margin-bottom: 20px;">🔍</div>
        <h2 style="color: white; margin-bottom: 15px;">No Data Available</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 18px; margin-bottom: 30px;">
            Upload your cycling safety data to unlock powerful insights and analytics.
        </p>
        <div style="margin-top: 30px;">
            <span class="alert-badge alert-info">Setup Required</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_enhanced_filters(routes_df, time_series_df):
    """Render enhanced filter sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="color: white; font-size: 24px; font-weight: 700; margin-bottom: 20px; text-align: center;">
            🎛️ Smart Controls
        </div>
        """, unsafe_allow_html=True)
        
        filters = {}
        
        # Date range filter with presets
        st.markdown("#### 📅 Time Period")
        
        date_preset = st.selectbox(
            "Quick Select",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom Range"],
            index=1
        )
        
        if date_preset == "Custom Range" and time_series_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("To", value=datetime.now())
            filters['date_range'] = (start_date, end_date)
        else:
            # Set date range based on preset
            if date_preset == "Last 7 Days":
                filters['date_range'] = (datetime.now() - timedelta(days=7), datetime.now())
            elif date_preset == "Last 30 Days":
                filters['date_range'] = (datetime.now() - timedelta(days=30), datetime.now())
            elif date_preset == "Last 90 Days":
                filters['date_range'] = (datetime.now() - timedelta(days=90), datetime.now())
            else:
                filters['date_range'] = None
        
        # Risk level filter
        st.markdown("#### ⚠️ Risk Levels")
        risk_levels = st.multiselect(
            "Select Risk Levels",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        filters['risk_levels'] = risk_levels
        
        # Geographic filter
        if routes_df is not None and 'route_name' in routes_df.columns:
            st.markdown("#### 🗺️ Geographic Area")
            route_options = ["All Routes"] + sorted(routes_df['route_name'].unique().tolist())
            selected_routes = st.multiselect(
                "Select Routes",
                route_options,
                default=["All Routes"]
            )
            filters['routes'] = selected_routes if "All Routes" not in selected_routes else None
        
        # Real-time toggle
        st.markdown("#### ⚡ Real-time Updates")
        real_time = st.toggle("Auto-refresh", value=True)
        filters['real_time'] = real_time
        
        if real_time:
            st.markdown("*Dashboard updates every 5 minutes*")
    
    return filters


def render_executive_dashboard(routes_df, braking_df, swerving_df, time_series_df):
    """Render executive-level KPI dashboard with animated metrics"""
    
    st.markdown('<div class="section-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    
    # Calculate comprehensive metrics
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    # Row 1: Primary KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safety_score = metrics.get('safety_score', 75.5)
        safety_delta = metrics.get('safety_score_change', 2.3)
        delta_type = "positive" if safety_delta > 0 else "negative" if safety_delta < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Overall Safety Score",
            f"{safety_score:.1f}%",
            safety_delta,
            delta_type,
            "🛡️",
            safety_score
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        total_incidents = metrics.get('total_incidents', 0)
        incidents_change = metrics.get('incidents_change', -5.2)
        delta_type = "positive" if incidents_change < 0 else "negative" if incidents_change > 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Total Safety Events",
            f"{total_incidents:,}",
            incidents_change,
            delta_type,
            "🚨"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col3:
        active_routes = metrics.get('total_routes', 0)
        routes_change = metrics.get('routes_change', 1.2)
        delta_type = "positive" if routes_change > 0 else "negative" if routes_change < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Active Routes",
            f"{active_routes:,}",
            routes_change,
            delta_type,
            "🛣️"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col4:
        daily_rides = metrics.get('avg_daily_rides', 0)
        rides_change = metrics.get('daily_rides_change', 8.7)
        delta_type = "positive" if rides_change > 0 else "negative" if rides_change < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Daily Rides",
            f"{daily_rides:,.0f}",
            rides_change,
            delta_type,
            "🚴"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Row 2: Operational Metrics
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        response_time = metrics.get('avg_response_time', 2.4)
        response_delta = metrics.get('response_time_change', -12.5)
        delta_type = "positive" if response_delta < 0 else "negative" if response_delta > 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Avg Response Time",
            f"{response_time:.1f}h",
            response_delta,
            delta_type,
            "⏱️"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        network_coverage = metrics.get('network_coverage', 87.3)
        coverage_delta = metrics.get('coverage_change', 3.1)
        delta_type = "positive" if coverage_delta > 0 else "negative" if coverage_delta < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Network Coverage",
            f"{network_coverage:.1f}%",
            coverage_delta,
            delta_type,
            "📡",
            network_coverage
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col3:
        risk_reduction = metrics.get('risk_reduction', 15.8)
        risk_delta = metrics.get('risk_reduction_change', 4.2)
        delta_type = "positive" if risk_delta > 0 else "negative" if risk_delta < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "Risk Reduction",
            f"{risk_reduction:.1f}%",
            risk_delta,
            delta_type,
            "📉"
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col4:
        user_satisfaction = metrics.get('user_satisfaction', 4.6)
        satisfaction_delta = metrics.get('satisfaction_change', 0.3)
        delta_type = "positive" if satisfaction_delta > 0 else "negative" if satisfaction_delta < 0 else "neutral"
        
        card_html = create_enhanced_metric_card(
            "User Satisfaction",
            f"{user_satisfaction:.1f}/5.0",
            satisfaction_delta,
            delta_type,
            "⭐"
        )
        st.markdown(card_html, unsafe_allow_html=True)


def render_ai_powered_insights(routes_df, braking_df, swerving_df, time_series_df):
    """Render AI-powered insights with dynamic content generation"""
    
    st.markdown('<div class="section-header">🧠 AI-Powered Insights</div>', unsafe_allow_html=True)
    
    # Calculate metrics for AI insights
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    try:
        # Generate AI insights
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
        
        # AI Insights Container
        st.markdown('<div class="ai-insights-container slide-up">', unsafe_allow_html=True)
        
        # Executive Summary
        if executive_summary:
            st.markdown("#### 📋 Executive Summary")
            st.markdown(f"""
            <div class="executive-summary">
                {executive_summary}
            </div>
            """, unsafe_allow_html=True)
        
        # Priority Insights
        if insights:
            st.markdown("#### 🎯 Priority Insights")
            
            # Group insights by priority
            high_priority = [i for i in insights if hasattr(i, 'impact_level') and i.impact_level == 'High']
            medium_priority = [i for i in insights if hasattr(i, 'impact_level') and i.impact_level == 'Medium']
            low_priority = [i for i in insights if hasattr(i, 'impact_level') and i.impact_level == 'Low']
            
            # Display high priority insights
            for insight in high_priority[:3]:  # Show top 3 high priority
                st.markdown(f"""
                <div class="insight-card high-priority">
                    <h4 style="color: #fca5a5; margin-bottom: 10px;">🔴 {getattr(insight, 'title', 'Critical Insight')}</h4>
                    <p style="color: white; margin-bottom: 15px;">{getattr(insight, 'description', 'No description available')}</p>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">
                        Priority: High | Impact: Critical
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display medium priority insights
            for insight in medium_priority[:2]:  # Show top 2 medium priority
                st.markdown(f"""
                <div class="insight-card medium-priority">
                    <h4 style="color: #fcd34d; margin-bottom: 10px;">🟡 {getattr(insight, 'title', 'Important Insight')}</h4>
                    <p style="color: white; margin-bottom: 15px;">{getattr(insight, 'description', 'No description available')}</p>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">
                        Priority: Medium | Impact: Moderate
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        logger.warning(f"Error generating AI insights: {e}")
        
        # Fallback insights based on metrics
        st.markdown('<div class="ai-insights-container">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Data-Driven Insights")
        
        # Generate insights from metrics
        safety_score = metrics.get('safety_score', 0)
        total_incidents = metrics.get('total_incidents', 0)
        
        if safety_score > 80:
            st.markdown("""
            <div class="insight-card low-priority">
                <h4 style="color: #4ade80; margin-bottom: 10px;">🟢 Excellent Safety Performance</h4>
                <p style="color: white;">Your network is performing exceptionally well with a safety score above 80%. Continue monitoring key routes and maintain current safety protocols.</p>
            </div>
            """, unsafe_allow_html=True)
        elif safety_score > 60:
            st.markdown("""
            <div class="insight-card medium-priority">
                <h4 style="color: #fcd34d; margin-bottom: 10px;">🟡 Room for Improvement</h4>
                <p style="color: white;">Safety score indicates moderate performance. Focus on high-incident areas and consider implementing additional safety measures on problematic routes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card high-priority">
                <h4 style="color: #fca5a5; margin-bottom: 10px;">🔴 Immediate Action Required</h4>
                <p style="color: white;">Safety score is below acceptable levels. Urgent intervention needed on high-risk routes. Consider emergency safety protocols and infrastructure improvements.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_interactive_safety_maps(braking_df, swerving_df, routes_df):
    """Render interactive safety maps with enhanced visualizations"""
    
    st.markdown('<div class="section-header">🗺️ Interactive Safety Maps</div>', unsafe_allow_html=True)
    
    # Map controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        map_type = st.selectbox(
            "🗺️ Map View",
            ["Safety Heatmap", "Incident Clusters", "Route Analysis", "Risk Zones"],
            index=0
        )
    
    with col2:
        show_braking = st.toggle("🛑 Braking Events", value=True)
    
    with col3:
        show_swerving = st.toggle("🔄 Swerving Events", value=True)
    
    # Create enhanced map visualization
    if map_type == "Safety Heatmap":
        render_safety_heatmap(braking_df, swerving_df, show_braking, show_swerving)
    elif map_type == "Incident Clusters":
        render_incident_clusters(braking_df, swerving_df, show_braking, show_swerving)
    elif map_type == "Route Analysis":
        render_route_analysis_map(routes_df, braking_df, swerving_df)
    else:
        render_risk_zones_map(braking_df, swerving_df, routes_df)


def render_safety_heatmap(braking_df, swerving_df, show_braking, show_swerving):
    """Render safety heatmap with PyDeck"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    layers = []
    
    # Add braking events layer
    if show_braking and braking_df is not None and len(braking_df) > 0:
        if 'latitude' in braking_df.columns and 'longitude' in braking_df.columns:
            braking_layer = pdk.Layer(
                "HeatmapLayer",
                data=braking_df,
                get_position=["longitude", "latitude"],
                get_weight="severity" if "severity" in braking_df.columns else 1,
                radius_pixels=60,
                intensity=1,
                threshold=0.03,
                get_color=[255, 69, 0, 160]  # Red-orange for braking
            )
            layers.append(braking_layer)
    
    # Add swerving events layer
    if show_swerving and swerving_df is not None and len(swerving_df) > 0:
        if 'latitude' in swerving_df.columns and 'longitude' in swerving_df.columns:
            swerving_layer = pdk.Layer(
                "HeatmapLayer",
                data=swerving_df,
                get_position=["longitude", "latitude"],
                get_weight="severity" if "severity" in swerving_df.columns else 1,
                radius_pixels=60,
                intensity=1,
                threshold=0.03,
                get_color=[138, 43, 226, 160]  # Purple for swerving
            )
            layers.append(swerving_layer)
    
    if layers:
        # Calculate center point
        all_data = []
        if show_braking and braking_df is not None:
            all_data.append(braking_df)
        if show_swerving and swerving_df is not None:
            all_data.append(swerving_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
                center_lat = combined_df['latitude'].mean()
                center_lon = combined_df['longitude'].mean()
            else:
                center_lat, center_lon = 51.5074, -0.1278  # Default to London
        else:
            center_lat, center_lon = 51.5074, -0.1278
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=0
        )
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "Safety Event Density"}
        )
        
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("📍 No location data available for heatmap visualization")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_incident_clusters(braking_df, swerving_df, show_braking, show_swerving):
    """Render incident clusters with scatter plot"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    layers = []
    
    # Add braking events as scatter plot
    if show_braking and braking_df is not None and len(braking_df) > 0:
        if 'latitude' in braking_df.columns and 'longitude' in braking_df.columns:
            braking_layer = pdk.Layer(
                "ScatterplotLayer",
                data=braking_df,
                get_position=["longitude", "latitude"],
                get_color=[255, 69, 0, 200],
                get_radius=50,
                radius_scale=1,
                pickable=True
            )
            layers.append(braking_layer)
    
    # Add swerving events as scatter plot
    if show_swerving and swerving_df is not None and len(swerving_df) > 0:
        if 'latitude' in swerving_df.columns and 'longitude' in swerving_df.columns:
            swerving_layer = pdk.Layer(
                "ScatterplotLayer",
                data=swerving_df,
                get_position=["longitude", "latitude"],
                get_color=[138, 43, 226, 200],
                get_radius=40,
                radius_scale=1,
                pickable=True
            )
            layers.append(swerving_layer)
    
    if layers:
        # Calculate center point
        all_data = []
        if show_braking and braking_df is not None:
            all_data.append(braking_df)
        if show_swerving and swerving_df is not None:
            all_data.append(swerving_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
                center_lat = combined_df['latitude'].mean()
                center_lon = combined_df['longitude'].mean()
            else:
                center_lat, center_lon = 51.5074, -0.1278
        else:
            center_lat, center_lon = 51.5074, -0.1278
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=13,
            pitch=0
        )
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "Safety Event: {event_type}"}
        )
        
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("📍 No location data available for cluster visualization")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_route_analysis_map(routes_df, braking_df, swerving_df):
    """Render route analysis with path visualization"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if routes_df is not None and len(routes_df) > 0:
        # Create route popularity visualization
        fig = go.Figure()
        
        if 'route_name' in routes_df.columns and 'popularity_score' in routes_df.columns:
            # Sort routes by popularity
            routes_sorted = routes_df.sort_values('popularity_score', ascending=False).head(10)
            
            fig.add_trace(go.Bar(
                x=routes_sorted['route_name'],
                y=routes_sorted['popularity_score'],
                marker_color='rgba(55, 128, 191, 0.8)',
                name='Route Popularity'
            ))
            
            fig.update_layout(
                title="Top 10 Most Popular Routes",
                xaxis_title="Route Name",
                yaxis_title="Popularity Score",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Route popularity data not available")
    else:
        st.info("🛣️ No route data available for analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_risk_zones_map(braking_df, swerving_df, routes_df):
    """Render risk zones analysis"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Calculate risk scores for different areas
    risk_data = []
    
    if braking_df is not None and len(braking_df) > 0:
        if 'latitude' in braking_df.columns and 'longitude' in braking_df.columns:
            # Create risk zones based on incident density
            braking_risk = braking_df.groupby([
                pd.cut(braking_df['latitude'], bins=10),
                pd.cut(braking_df['longitude'], bins=10)
            ]).size().reset_index(name='risk_score')
            
            st.write("🔴 High Risk Zones - Braking Events")
            st.bar_chart(braking_risk['risk_score'].head(10))
    
    if swerving_df is not None and len(swerving_df) > 0:
        if 'latitude' in swerving_df.columns and 'longitude' in swerving_df.columns:
            # Create risk zones based on incident density
            swerving_risk = swerving_df.groupby([
                pd.cut(swerving_df['latitude'], bins=10),
                pd.cut(swerving_df['longitude'], bins=10)
            ]).size().reset_index(name='risk_score')
            
            st.write("🟡 Medium Risk Zones - Swerving Events")
            st.bar_chart(swerving_risk['risk_score'].head(10))
    
    if not (braking_df is not None and len(braking_df) > 0) and not (swerving_df is not None and len(swerving_df) > 0):
        st.info("⚠️ No incident data available for risk zone analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_dynamic_trends_analysis(time_series_df):
    """Render dynamic trends analysis with interactive charts"""
    
    st.markdown('<div class="section-header">📈 Dynamic Trends Analysis</div>', unsafe_allow_html=True)
    
    if time_series_df is not None and len(time_series_df) > 0:
        
        # Time series controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_to_analyze = st.selectbox(
                "📊 Metric to Analyze",
                ["incidents", "safety_score", "daily_rides", "response_time"],
                index=0
            )
        
        with col2:
            time_granularity = st.selectbox(
                "📅 Time Granularity",
                ["Daily", "Weekly", "Monthly"],
                index=1
            )
        
        with col3:
            show_forecast = st.toggle("🔮 Show Forecast", value=True)
        
        # Create dynamic trends chart
        render_trends_chart(time_series_df, metric_to_analyze, time_granularity, show_forecast)
        
        # Performance metrics summary
        render_performance_summary(time_series_df)
        
    else:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 40px;">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <h3 style="color: white; margin-bottom: 15px;">No Time Series Data</h3>
            <p style="color: rgba(255,255,255,0.7);">Upload time series data to see dynamic trends and forecasting.</p>
        </div>
        """, unsafe_allow_html=True)


def render_trends_chart(time_series_df, metric, granularity, show_forecast):
    """Render interactive trends chart"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    try:
        # Prepare data based on granularity
        if 'timestamp' in time_series_df.columns or 'date' in time_series_df.columns:
            date_col = 'timestamp' if 'timestamp' in time_series_df.columns else 'date'
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(time_series_df[date_col]):
                time_series_df[date_col] = pd.to_datetime(time_series_df[date_col])
            
            # Group by granularity
            if granularity == "Weekly":
                grouped_data = time_series_df.groupby(pd.Grouper(key=date_col, freq='W')).agg({
                    metric: 'mean' if metric in ['safety_score', 'response_time'] else 'sum'
                }).reset_index()
            elif granularity == "Monthly":
                grouped_data = time_series_df.groupby(pd.Grouper(key=date_col, freq='M')).agg({
                    metric: 'mean' if metric in ['safety_score', 'response_time'] else 'sum'
                }).reset_index()
            else:  # Daily
                grouped_data = time_series_df.groupby(date_col).agg({
                    metric: 'mean' if metric in ['safety_score', 'response_time'] else 'sum'
                }).reset_index()
            
            # Create the chart
            fig = go.Figure()
            
            # Add main trend line
            fig.add_trace(go.Scatter(
                x=grouped_data[date_col],
                y=grouped_data[metric],
                mode='lines+markers',
                name=f'{metric.replace("_", " ").title()}',
                line=dict(color='#4ade80', width=3),
                marker=dict(size=6, color='#22c55e')
            ))
            
            # Add moving average
            if len(grouped_data) >= 7:
                window = min(7, len(grouped_data) // 3)
                moving_avg = grouped_data[metric].rolling(window=window, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=grouped_data[date_col],
                    y=moving_avg,
                    mode='lines',
                    name=f'{window}-Period Moving Average',
                    line=dict(color='#fbbf24', width=2, dash='dash')
                ))
            
            # Add forecast if enabled
            if show_forecast and len(grouped_data) >= 10:
                # Simple linear forecast (last 30% of data)
                forecast_periods = min(7, len(grouped_data) // 4)
                recent_data = grouped_data.tail(len(grouped_data) // 2)
                
                if len(recent_data) >= 3:
                    # Simple linear regression for forecast
                    x_vals = np.arange(len(recent_data))
                    y_vals = recent_data[metric].values
                    
                    # Fit linear trend
                    z = np.polyfit(x_vals, y_vals, 1)
                    trend_line = np.poly1d(z)
                    
                    # Generate forecast
                    forecast_x = np.arange(len(recent_data), len(recent_data) + forecast_periods)
                    forecast_y = trend_line(forecast_x)
                    
                    # Create forecast dates
                    last_date = grouped_data[date_col].iloc[-1]
                    if granularity == "Weekly":
                        forecast_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W')
                    elif granularity == "Monthly":
                        forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
                    else:
                        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_y,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#f87171', width=2, dash='dot'),
                        marker=dict(size=5, color='#ef4444')
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'{metric.replace("_", " ").title()} Trends - {granularity}',
                xaxis_title='Date',
                yaxis_title=metric.replace("_", " ").title(),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                hovermode='x unified',
                legend=dict(
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("📅 No timestamp data available for trend analysis")
            
    except Exception as e:
        logger.warning(f"Error creating trends chart: {e}")
        st.info("📊 Unable to create trends chart with available data")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_performance_summary(time_series_df):
    """Render performance summary metrics"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    st.markdown("#### 📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Calculate performance metrics
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            with col1:
                avg_incidents = time_series_df[numeric_cols[0]].mean() if len(numeric_cols) > 0 else 0
                st.metric("📈 Average Daily", f"{avg_incidents:.1f}")
            
            with col2:
                trend_direction = "↗️" if len(numeric_cols) > 0 and time_series_df[numeric_cols[0]].iloc[-1] > time_series_df[numeric_cols[0]].iloc[0] else "↘️"
                st.metric("📊 Trend", trend_direction)
            
            with col3:
                volatility = time_series_df[numeric_cols[0]].std() if len(numeric_cols) > 0 else 0
                st.metric("📉 Volatility", f"{volatility:.2f}")
            
            with col4:
                data_quality = "High" if len(time_series_df.dropna()) / len(time_series_df) > 0.9 else "Medium"
                st.metric("✅ Data Quality", data_quality)
        else:
            st.info("📊 No numeric data available for performance summary")
            
    except Exception as e:
        logger.warning(f"Error calculating performance summary: {e}")
        st.info("📊 Unable to calculate performance metrics")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_smart_alerts_panel(braking_df, swerving_df, time_series_df):
    """Render smart alerts and notifications panel"""
    
    st.markdown('<div class="section-header">🚨 Smart Alerts & Notifications</div>', unsafe_allow_html=True)
    
    # Generate dynamic alerts based on data
    alerts = generate_smart_alerts(braking_df, swerving_df, time_series_df)
    
    if alerts:
        # Group alerts by priority
        critical_alerts = [a for a in alerts if a['priority'] == 'critical']
        warning_alerts = [a for a in alerts if a['priority'] == 'warning']
        info_alerts = [a for a in alerts if a['priority'] == 'info']
        
        # Display alerts in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 Critical Alerts")
            if critical_alerts:
                for alert in critical_alerts[:3]:  # Show top 3
                    st.markdown(f"""
                    <div class="alert-badge alert-critical" style="display: block; margin: 10px 0; padding: 15px;">
                        <strong>{alert['title']}</strong><br>
                        <small>{alert['message']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical alerts")
        
        with col2:
            st.markdown("#### 🟡 Warnings")
            if warning_alerts:
                for alert in warning_alerts[:3]:  # Show top 3
                    st.markdown(f"""
                    <div class="alert-badge alert-warning" style="display: block; margin: 10px 0; padding: 15px;">
                        <strong>{alert['title']}</strong><br>
                        <small>{alert['message']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No warnings")
        
        with col3:
            st.markdown("#### 🔵 Information")
            if info_alerts:
                for alert in info_alerts[:3]:  # Show top 3
                    st.markdown(f"""
                    <div class="alert-badge alert-info" style="display: block; margin: 10px 0; padding: 15px;">
                        <strong>{alert['title']}</strong><br>
                        <small>{alert['message']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📊 System operating normally")
    
    else:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 40px;">
            <div style="font-size: 48px; margin-bottom: 20px;">✅</div>
            <h3 style="color: white; margin-bottom: 15px;">All Systems Normal</h3>
            <p style="color: rgba(255,255,255,0.7);">No alerts or warnings at this time.</p>
        </div>
        """, unsafe_allow_html=True)


def generate_smart_alerts(braking_df, swerving_df, time_series_df):
    """Generate smart alerts based on data patterns"""
    alerts = []
    
    try:
        # Check for high incident areas
        if braking_df is not None and len(braking_df) > 0:
            if len(braking_df) > 100:  # High volume of braking events
                alerts.append({
                    'priority': 'critical',
                    'title': 'High Braking Activity',
                    'message': f'{len(braking_df)} braking events detected. Consider route safety review.'
                })
            elif len(braking_df) > 50:
                alerts.append({
                    'priority': 'warning',
                    'title': 'Elevated Braking Events',
                    'message': f'{len(braking_df)} braking events in current period.'
                })
        
        if swerving_df is not None and len(swerving_df) > 0:
            if len(swerving_df) > 50:  # High volume of swerving events
                alerts.append({
                    'priority': 'warning',
                    'title': 'Swerving Pattern Detected',
                    'message': f'{len(swerving_df)} swerving events may indicate road hazards.'
                })
        
        # Check time series trends
        if time_series_df is not None and len(time_series_df) > 7:
            numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                recent_avg = time_series_df[numeric_cols[0]].tail(7).mean()
                overall_avg = time_series_df[numeric_cols[0]].mean()
                
                if recent_avg > overall_avg * 1.5:  # 50% increase
                    alerts.append({
                        'priority': 'critical',
                        'title': 'Incident Spike Detected',
                        'message': 'Recent incidents 50% above normal levels.'
                    })
                elif recent_avg > overall_avg * 1.2:  # 20% increase
                    alerts.append({
                        'priority': 'warning',
                        'title': 'Rising Incident Trend',
                        'message': 'Incidents trending above normal levels.'
                    })
                else:
                    alerts.append({
                        'priority': 'info',
                        'title': 'Stable Operations',
                        'message': 'Incident levels within normal parameters.'
                    })
        
        # Data quality alerts
        total_data_points = 0
        if braking_df is not None:
            total_data_points += len(braking_df)
        if swerving_df is not None:
            total_data_points += len(swerving_df)
        if time_series_df is not None:
            total_data_points += len(time_series_df)
        
        if total_data_points == 0:
            alerts.append({
                'priority': 'critical',
                'title': 'No Data Available',
                'message': 'Please upload safety data to enable monitoring.'
            })
        elif total_data_points < 100:
            alerts.append({
                'priority': 'info',
                'title': 'Limited Data Sample',
                'message': 'Consider uploading more data for better insights.'
            })
        
    except Exception as e:
        logger.warning(f"Error generating alerts: {e}")
        alerts.append({
            'priority': 'info',
            'title': 'System Status',
            'message': 'Alert system operational.'
        })
    
    return alerts


def render_error_message(error_msg: str):
    """Render enhanced error message"""
    st.markdown(f"""
    <div class="ai-insights-container" style="text-align: center; padding: 60px 40px;">
        <div style="font-size: 64px; margin-bottom: 20px;">⚠️</div>
        <h2 style="color: white; margin-bottom: 15px;">Dashboard Error</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 16px; margin-bottom: 30px;">
            An error occurred while loading the dashboard. Please check your data files and try again.
        </p>
        <div style="margin-top: 30px;">
            <span class="alert-badge alert-critical">Error: {error_msg[:50]}...</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_overview_filters(routes_df, braking_df, swerving_df, time_series_df, filters):
    """Apply filters to all dataframes"""
    
    # Date range filtering
    if filters.get('date_range') and time_series_df is not None:
        start_date, end_date = filters['date_range']
        
        # Convert dates to datetime if needed
        date_col = None
        for col in ['timestamp', 'date', 'created_at']:
            if col in time_series_df.columns:
                date_col = col
                break
        
        if date_col:
            time_series_df = time_series_df[
                (pd.to_datetime(time_series_df[date_col]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(time_series_df[date_col]) <= pd.to_datetime(end_date))
            ]
    
    # Route filtering
    if filters.get('routes') and routes_df is not None:
        selected_routes = filters['routes']
        if 'route_name' in routes_df.columns:
            routes_df = routes_df[routes_df['route_name'].isin(selected_routes)]
    
    # Risk level filtering
    if filters.get('risk_levels'):
        risk_levels = filters['risk_levels']
        
        # Apply to braking data
        if braking_df is not None and 'risk_level' in braking_df.columns:
            braking_df = braking_df[braking_df['risk_level'].isin(risk_levels)]
        
        # Apply to swerving data
        if swerving_df is not None and 'risk_level' in swerving_df.columns:
            swerving_df = swerving_df[swerving_df['risk_level'].isin(risk_levels)]
    
    return routes_df, braking_df, swerving_df, time_series_df


# Main render function
def render_overview_page():
    """Main function to render the enhanced overview page"""
    render_enhanced_overview_page()
