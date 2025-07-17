"""
Actionable Insights Page for SeeSense Dashboard
Now uses centralized insights generator - no direct groq import needed
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

from app.core.data_processor import data_processor
from app.core.metrics_calculator import metrics_calculator
from app.core.groq_insights_generator import create_insights_generator

logger = logging.getLogger(__name__)

def render_actionable_insights_page():
    """Render the actionable insights page"""
    st.title("💡 Actionable Insights")
    st.markdown("AI-powered recommendations for improving cycling safety")
    
    try:
        # Load data
        all_data = data_processor.load_all_datasets()
        
        # Check if we have data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            st.warning("⚠️ No data available for insights generation.")
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Generate insights
        render_insights_dashboard(routes_df, braking_df, swerving_df, time_series_df)
        
    except Exception as e:
        logger.error(f"Error in actionable insights page: {e}")
        st.error("⚠️ An error occurred while generating insights.")

def render_insights_dashboard(routes_df, braking_df, swerving_df, time_series_df):
    """Render the main insights dashboard"""
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    # Generate insights
    insights_generator = create_insights_generator()
    insights = insights_generator.generate_comprehensive_insights(
        metrics, routes_df, None, time_series_df
    )
    
    # Render insights sections
    render_priority_actions(insights)
    render_implementation_roadmap(insights)
    render_impact_analysis(insights, metrics)
    render_resource_allocation(insights)

def render_priority_actions(insights):
    """Render priority actions section"""
    st.markdown("### 🎯 Priority Actions")
    
    # Filter high-impact insights
    high_impact = [i for i in insights if i.impact_level == 'High']
    medium_impact = [i for i in insights if i.impact_level == 'Medium']
    
    if high_impact:
        st.markdown("#### 🔴 Immediate Actions Required")
        for i, insight in enumerate(high_impact, 1):
            with st.expander(f"Priority {i}: {insight.title}"):
                st.write(insight.description)
                st.markdown("**🎯 Action Items:**")
                for rec in insight.recommendations:
                    st.write(f"• {rec}")
    
    if medium_impact:
        st.markdown("#### 🟡 Medium Priority Actions")
        for i, insight in enumerate(medium_impact, 1):
            with st.expander(f"Action {i}: {insight.title}"):
                st.write(insight.description)
                st.markdown("**💡 Recommendations:**")
                for rec in insight.recommendations:
                    st.write(f"• {rec}")

def render_implementation_roadmap(insights):
    """Render implementation roadmap"""
    st.markdown("### 🗺️ Implementation Roadmap")
    
    # Create timeline
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📅 Next 30 Days")
        st.markdown("**Quick Wins:**")
        for insight in insights[:2]:
            if insight.recommendations:
                st.write(f"• {insight.recommendations[0]}")
    
    with col2:
        st.markdown("#### 📅 Next 90 Days")
        st.markdown("**Medium-term Goals:**")
        for insight in insights[2:4]:
            if insight.recommendations:
                st.write(f"• {insight.recommendations[0]}")
    
    with col3:
        st.markdown("#### 📅 Next 6 Months")
        st.markdown("**Long-term Strategy:**")
        for insight in insights[4:]:
            if insight.recommendations:
                st.write(f"• {insight.recommendations[0]}")

def render_impact_analysis(insights, metrics):
    """Render impact analysis"""
    st.markdown("### 📊 Impact Analysis")
    
    # Show potential improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Potential Improvements")
        
        # Estimate impact based on insights
        if metrics.get('safety_score', 0) < 7:
            st.metric("Safety Score Improvement", "+1.5 points", "Implementing top recommendations")
        
        if metrics.get('incident_rate', 0) > 10:
            st.metric("Incident Rate Reduction", "-25%", "Focus on high-risk areas")
    
    with col2:
        st.markdown("#### 💰 Cost-Benefit Analysis")
        
        # Simple cost-benefit estimates
        st.markdown("**High ROI Actions:**")
        for insight in insights:
            if insight.impact_level == 'High' and insight.category == 'Safety':
                st.write(f"• {insight.title}: High impact, moderate cost")

def render_resource_allocation(insights):
    """Render resource allocation recommendations"""
    st.markdown("### 💼 Resource Allocation")
    
    # Group insights by category
    categories = {}
    for insight in insights:
        if insight.category not in categories:
            categories[insight.category] = []
        categories[insight.category].append(insight)
    
    # Show resource needs by category
    for category, category_insights in categories.items():
        with st.expander(f"📋 {category} Resources"):
            st.markdown(f"**{category} Focus Areas:**")
            for insight in category_insights:
                st.write(f"• {insight.title}")
                if insight.recommendations:
                    st.write(f"  → {insight.recommendations[0]}")

# You can also add this function if you need it elsewhere
def get_actionable_recommendations(routes_df, braking_df, swerving_df, time_series_df):
    """Get actionable recommendations programmatically"""
    try:
        # Calculate metrics
        metrics = metrics_calculator.calculate_all_overview_metrics(
            routes_df, braking_df, swerving_df, time_series_df
        )
        
        # Generate insights
        insights_generator = create_insights_generator()
        insights = insights_generator.generate_comprehensive_insights(
            metrics, routes_df, None, time_series_df
        )
        
        # Extract recommendations
        recommendations = []
        for insight in insights:
            for rec in insight.recommendations:
                recommendations.append({
                    'recommendation': rec,
                    'category': insight.category,
                    'impact_level': insight.impact_level,
                    'confidence': insight.confidence_score
                })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return []
