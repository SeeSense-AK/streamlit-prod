"""
User-Friendly Advanced Analytics Page for SeeSense Dashboard
Complete, modular, and optimized version with integrated AI insights and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import random
import json

# Advanced analytics imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# Time series analysis
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from app.core.data_processor import data_processor
from app.utils.config import config

# Insights generator
try:
    from app.core.groq_insights_generator import create_insights_generator
    INSIGHTS_GENERATOR_AVAILABLE = True
except ImportError:
    INSIGHTS_GENERATOR_AVAILABLE = False
    logging.warning("Insights generator not available")

logger = logging.getLogger(__name__)

# Custom CSS for UI styling
def add_custom_css() -> None:
    """Add custom CSS for enhanced UI appearance"""
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

# Data Preparation and Processing
def prepare_user_friendly_time_series(time_series_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Prepare time series data for user-friendly analysis"""
    try:
        if time_series_df is None or 'date' not in time_series_df.columns:
            return None

        df = time_series_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add time-based features
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['month'] = df['date'].dt.month_name()
        
        # Calculate incident rate if we have incidents
        if 'incidents' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['incidents'] = df[numeric_cols].sum(axis=1)
            else:
                df['incidents'] = 1  # default
        
        return df
    except Exception as e:
        logger.error(f"Error preparing time series data: {e}")
        return None

def detect_user_friendly_risk_zones(braking_df: Optional[pd.DataFrame], swerving_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Detect risk zones in user-friendly format"""
    try:
        risk_zones = []
        
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
            return risk_df.sort_values('risk_score', ascending=False)
        
        return None
    except Exception as e:
        logger.error(f"Error detecting risk zones: {e}")
        return None

def detect_user_friendly_anomalies(time_series_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Detect anomalies in user-friendly format"""
    try:
        df = time_series_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if 'incidents' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['incidents'] = df[numeric_cols].sum(axis=1)
            else:
                return None
        
        mean_incidents = df['incidents'].mean()
        std_incidents = df['incidents'].std()
        
        df['is_unusual'] = df['incidents'] > (mean_incidents + 2 * std_incidents)
        df['severity'] = df['incidents'].apply(lambda x: 
            'High' if x > mean_incidents + 2 * std_incidents else
            'Medium' if x > mean_incidents + std_incidents else 'Normal'
        )
        
        return df
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return None

def generate_user_friendly_predictions(time_series_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Generate simple predictions for user-friendly display"""
    try:
        df = time_series_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if 'incidents' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['incidents'] = df[numeric_cols].sum(axis=1)
            else:
                return None
        
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        X = df['days_since_start'].values.reshape(-1, 1)
        y = df['incidents'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = df['days_since_start'].max()
        future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
        future_predictions = np.maximum(model.predict(future_days), 0)
        
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        return pd.DataFrame({
            'date': list(df['date']) + future_dates,
            'incidents': list(df['incidents']) + list(future_predictions),
            'is_prediction': [False] * len(df) + [True] * len(future_predictions)
        })
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None

# Visualization Functions
def create_daily_safety_chart(daily_stats: pd.DataFrame) -> Optional[go.Figure]:
    """Create appealing daily safety chart"""
    try:
        fig = go.Figure()
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
            title=dict(text="🗓️ Safety Score by Day of Week", font=dict(size=20, color='#2c3e50')),
            xaxis_title="Day of Week",
            yaxis_title="Safety Score (1-10)",
            yaxis=dict(range=[0, 10]),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating daily safety chart: {e}")
        return None

def create_hourly_safety_chart(hourly_stats: pd.DataFrame) -> Optional[go.Figure]:
    """Create appealing hourly safety chart"""
    try:
        fig = go.Figure()
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
            title=dict(text="🕐 Safety Score Throughout the Day", font=dict(size=20, color='#2c3e50')),
            xaxis_title="Hour of Day",
            yaxis_title="Safety Score (1-10)",
            yaxis=dict(range=[0, 10]),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating hourly safety chart: {e}")
        return None

def create_risk_zone_map(risk_zones: pd.DataFrame) -> Optional[go.Figure]:
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
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Error creating risk zone map: {e}")
        return None

def create_anomaly_chart(anomalies: pd.DataFrame) -> Optional[go.Figure]:
    """Create user-friendly anomaly chart"""
    try:
        fig = go.Figure()
        
        normal_data = anomalies[~anomalies['is_unusual']]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['incidents'],
            mode='markers',
            name='Normal Days',
            marker=dict(color='#4ECDC4', size=8),
            hovertemplate='<b>%{x}</b><br>Incidents: %{y}<extra></extra>'
        ))
        
        unusual_data = anomalies[anomalies['is_unusual']]
        if not unusual_data.empty:
            fig.add_trace(go.Scatter(
                x=unusual_data['date'],
                y=unusual_data['incidents'],
                mode='markers',
                name='Unusual Days',
                marker=dict(color='#FF6B6B', size=12, symbol='diamond'),
                hovertemplate='<b>%{x}</b><br>Incidents: %{y}<br>⚠️ Unusual Activity<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text="📊 Unusual Activity Detection", font=dict(size=20, color='#2c3e50')),
            xaxis_title="Date",
            yaxis_title="Incident Count",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating anomaly chart: {e}")
        return None

def create_prediction_chart(predictions: pd.DataFrame) -> Optional[go.Figure]:
    """Create prediction chart"""
    try:
        fig = go.Figure()
        
        historical = predictions[~predictions['is_prediction']]
        fig.add_trace(go.Scatter(
            x=historical['date'],
            y=historical['incidents'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=6)
        ))
        
        predicted = predictions[predictions['is_prediction']]
        fig.add_trace(go.Scatter(
            x=predicted['date'],
            y=predicted['incidents'],
            mode='lines+markers',
            name='AI Predictions',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=dict(text="🔮 AI Safety Trend Predictions", font=dict(size=20, color='#2c3e50')),
            xaxis_title="Date",
            yaxis_title="Expected Incidents",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating prediction chart: {e}")
        return None

# Analysis Functions
def analyze_daily_safety_patterns(ts_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Analyze safety patterns by day of week"""
    try:
        daily_stats = ts_data.groupby('day_of_week')['incidents'].agg(['mean', 'sum', 'count']).round(2)
        
        max_incidents = daily_stats['mean'].max()
        daily_stats['safety_score'] = 10 if max_incidents == 0 else round(10 - (daily_stats['mean'] / max_incidents * 9), 1)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return daily_stats.reindex([day for day in day_order if day in daily_stats.index])
    except Exception as e:
        logger.error(f"Error analyzing daily patterns: {e}")
        return None

def analyze_hourly_safety_patterns(ts_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Analyze safety patterns by hour of day"""
    try:
        hourly_stats = ts_data.groupby('hour')['incidents'].agg(['mean', 'sum', 'count']).round(2)
        
        max_incidents = hourly_stats['mean'].max()
        hourly_stats['safety_score'] = 10 if max_incidents == 0 else round(10 - (hourly_stats['mean'] / max_incidents * 9), 1)
        
        return hourly_stats
    except Exception as e:
        logger.error(f"Error analyzing hourly patterns: {e}")
        return None

def find_user_friendly_correlations(routes_df: Optional[pd.DataFrame], braking_df: Optional[pd.DataFrame], 
                                  swerving_df: Optional[pd.DataFrame], time_series_df: Optional[pd.DataFrame]) -> List[Dict]:
    """Find meaningful correlations in user-friendly language"""
    try:
        connections = []
        
        if routes_df is not None and time_series_df is not None:
            if len(routes_df) > 10 and len(time_series_df) > 10 and 'distance' in routes_df.columns and 'incidents' in time_series_df.columns:
                avg_distance = routes_df['distance'].mean()
                if avg_distance > 5:
                    connections.append({
                        'description': f"Longer cycling routes (avg {avg_distance:.1f}km) tend to have more safety events. Consider shorter routes for safer rides.",
                        'strength': 'Moderate',
                        'confidence': '75%'
                    })
        
        if braking_df is not None and swerving_df is not None:
            if len(braking_df) > 5 and len(swerving_df) > 5:
                braking_count = len(braking_df)
                swerving_count = len(swerving_df)
                
                if braking_count > swerving_count * 1.5:
                    connections.append({
                        'description': f"You brake heavily {braking_count} times vs swerving {swerving_count} times. This suggests good anticipation.",
                        'strength': 'Strong',
                        'confidence': '85%'
                    })
                elif swerving_count > braking_count * 1.5:
                    connections.append({
                        'description': f"More swerving ({swerving_count}) than heavy braking ({braking_count}) suggests unexpected obstacles.",
                        'strength': 'Strong',
                        'confidence': '80%'
                    })
        
        if time_series_df is not None and len(time_series_df) > 14:
            df = time_series_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            
            if 'incidents' in df.columns:
                weekend_incidents = df[df['day_of_week'].isin([5, 6])]['incidents'].mean()
                weekday_incidents = df[~df['day_of_week'].isin([5, 6])]['incidents'].mean()
                
                if weekend_incidents > weekday_incidents * 1.3:
                    connections.append({
                        'description': f"Weekend cycling shows {weekend_incidents:.1f} vs {weekday_incidents:.1f} weekday incidents.",
                        'strength': 'Moderate',
                        'confidence': '70%'
                    })
                elif weekday_incidents > weekend_incidents * 1.3:
                    connections.append({
                        'description': f"Weekday cycling is riskier ({weekday_incidents:.1f} vs {weekend_incidents:.1f} weekend incidents).",
                        'strength': 'Moderate',
                        'confidence': '70%'
                    })
        
        return connections
    except Exception as e:
        logger.error(f"Error finding correlations: {e}")
        return []

# Insight Generation
def calculate_overall_risk_score(braking_df: Optional[pd.DataFrame], swerving_df: Optional[pd.DataFrame], total_routes: int) -> float:
    """Calculate a simple 1-10 risk score"""
    try:
        if total_routes == 0:
            return 5.0
        
        total_incidents = (len(braking_df) if braking_df is not None else 0) + (len(swerving_df) if swerving_df is not None else 0)
        incident_rate = total_incidents / max(total_routes, 1)
        return round(max(1, min(10, 10 - (incident_rate * 2))), 1)
    except Exception:
        return 5.0

def generate_overview_insight(total_routes: int, total_braking: int, total_swerving: int, risk_score: float) -> str:
    """Generate dynamic overview insight based on data"""
    insights = []
    
    if risk_score >= 8:
        insights.append("🎉 Excellent news! Your cycling safety score is very high.")
    elif risk_score >= 6:
        insights.append("👍 Your cycling safety is above average, but there's room for improvement.")
    else:
        insights.append("⚠️ Your data shows some safety concerns.")
    
    if total_routes > 50:
        insights.append(f"With {total_routes} routes analyzed, we have excellent data.")
    elif total_routes > 20:
        insights.append(f"Your {total_routes} routes provide good data.")
    
    incident_rate = (total_braking + total_swerving) / max(total_routes, 1)
    if incident_rate < 0.5:
        insights.append("Your incident rate is low.")
    elif incident_rate > 2:
        insights.append("Higher incident rates detected.")
    
    return " ".join(insights)

def generate_daily_pattern_insight(daily_stats: pd.DataFrame) -> str:
    """Generate insight for daily patterns"""
    try:
        safest_day = daily_stats['safety_score'].idxmax()
        riskiest_day = daily_stats['safety_score'].idxmin()
        safest_score = daily_stats['safety_score'].max()
        riskiest_score = daily_stats['safety_score'].min()
        
        insights = [
            f"🌟 {safest_day} is your safest cycling day with a score of {safest_score}/10.",
            f"⚠️ {riskiest_day} shows the highest risk with a score of {riskiest_score}/10."
        ]
        
        if safest_day in ['Saturday', 'Sunday']:
            insights.append("Weekend cycling appears safer.")
        elif riskiest_day in ['Monday', 'Friday']:
            insights.append("Weekday rush hours may increase risks.")
        
        return " ".join(insights)
    except Exception as e:
        logger.error(f"Error generating daily insight: {e}")
        return "Daily cycling patterns show interesting safety variations."

def generate_hourly_pattern_insight(hourly_stats: pd.DataFrame) -> str:
    """Generate insight for hourly patterns"""
    try:
        safest_hour = hourly_stats['safety_score'].idxmax()
        riskiest_hour = hourly_stats['safety_score'].idxmin()
        
        insights = []
        if 10 <= safest_hour <= 14:
            insights.append(f"🌞 Midday cycling (around {safest_hour}:00) is safest.")
        elif 6 <= safest_hour <= 9:
            insights.append(f"🌅 Early morning cycling (around {safest_hour}:00) shows high safety.")
        
        if 17 <= riskiest_hour <= 19:
            insights.append(f"🚦 Evening rush hour (around {riskiest_hour}:00) shows increased risks.")
        elif 20 <= riskiest_hour <= 23:
            insights.append(f"🌙 Late evening cycling (around {riskiest_hour}:00) carries higher risks.")
        
        return " ".join(insights) or "Hourly cycling patterns reveal safety trends."
    except Exception as e:
        logger.error(f"Error generating hourly insight: {e}")
        return "Hourly cycling patterns reveal interesting safety trends."

def generate_risk_detection_insight(risk_zones: pd.DataFrame) -> str:
    """Generate insight for risk detection"""
    try:
        high_risk_count = len(risk_zones[risk_zones['risk_score'] >= 7])
        total_zones = len(risk_zones)
        
        insights = []
        if high_risk_count == 0:
            insights.append("🎉 No extremely high-risk zones detected.")
        elif high_risk_count < total_zones * 0.2:
            insights.append(f"⚠️ {high_risk_count} high-risk zones identified.")
        else:
            insights.append(f"🚨 {high_risk_count} high-risk zones need attention.")
        
        if not risk_zones.empty:
            most_common_type = risk_zones['type'].mode().iloc[0]
            if most_common_type == 'Heavy Braking':
                insights.append("Heavy braking incidents dominate.")
            elif most_common_type == 'Sudden Swerving':
                insights.append("Swerving incidents are more common.")
        
        return " ".join(insights)
    except Exception as e:
        logger.error(f"Error generating risk insight: {e}")
        return "Risk analysis complete."

def generate_anomaly_insight(anomalies: pd.DataFrame) -> str:
    """Generate insight for anomaly detection"""
    try:
        unusual_days = anomalies[anomalies['is_unusual']]
        total_days = len(anomalies)
        unusual_count = len(unusual_days)
        
        insights = []
        if unusual_count == 0:
            insights.append("✅ No unusual activity patterns detected.")
        elif unusual_count < total_days * 0.1:
            insights.append(f"📊 {unusual_count} unusual days detected.")
        else:
            insights.append(f"⚠️ {unusual_count} days with unusual activity patterns.")
        
        if not unusual_days.empty:
            avg_unusual_incidents = unusual_days['incidents'].mean()
            avg_normal_incidents = anomalies[~anomalies['is_unusual']]['incidents'].mean()
            ratio = avg_unusual_incidents / max(avg_normal_incidents, 1)
            if ratio > 3:
                insights.append(f"Unusual days show {ratio:.1f}x more incidents.")
        
        return " ".join(insights)
    except Exception as e:
        logger.error(f"Error generating anomaly insight: {e}")
        return "Anomaly analysis reveals interesting patterns."

def generate_prediction_insight(predictions: pd.DataFrame) -> str:
    """Generate insight for predictions"""
    try:
        historical = predictions[~predictions['is_prediction']]['incidents']
        predicted = predictions[predictions['is_prediction']]['incidents']
        
        insights = []
        trend_direction = "increasing" if predicted.iloc[-1] > historical.iloc[-1] else "decreasing"
        trend_magnitude = abs(predicted.iloc[-1] - historical.iloc[-1]) / max(historical.iloc[-1], 1)
        
        if trend_direction == "increasing" and trend_magnitude > 0.2:
            insights.append("📈 AI predicts increasing safety incidents.")
        elif trend_direction == "decreasing" and trend_magnitude > 0.2:
            insights.append("📉 AI predicts improving safety conditions.")
        else:
            insights.append("📊 AI predicts stable safety conditions.")
        
        predicted_avg = predicted.mean()
        historical_avg = historical.mean()
        
        if predicted_avg > historical_avg * 1.2:
            insights.append("Future weeks may see 20% more incidents.")
        elif predicted_avg < historical_avg * 0.8:
            insights.append("Future weeks look 20% safer.")
        
        return " ".join(insights)
    except Exception as e:
        logger.error(f"Error generating prediction insight: {e}")
        return "Trend predictions show interesting patterns."

def generate_connections_insight(connections: List[Dict]) -> str:
    """Generate insight for connections analysis"""
    try:
        if not connections:
            return "🔍 No significant patterns found yet."
        
        insights = []
        strong_connections = [c for c in connections if c.get('strength') == 'Strong']
        
        if len(strong_connections) >= 2:
            insights.append("🔗 Multiple strong patterns discovered.")
        elif len(strong_connections) == 1:
            insights.append("💡 One strong pattern identified.")
        
        high_confidence = [c for c in connections if '8' in str(c.get('confidence', '')) or '9' in str(c.get('confidence', ''))]
        if len(high_confidence) >= 2:
            insights.append("🎯 High-confidence patterns detected.")
        
        insights.append("Use these discoveries to make informed decisions.")
        return " ".join(insights)
    except Exception as e:
        logger.error(f"Error generating connections insight: {e}")
        return "Connection analysis reveals interesting relationships."

def generate_route_recommendations(routes_df: Optional[pd.DataFrame]) -> List[str]:
    """Generate AI route recommendations"""
    try:
        recommendations = []
        
        if routes_df is not None and len(routes_df) > 5:
            if 'distance' in routes_df.columns:
                safe_distance = routes_df['distance'].quantile(0.3)
                recommendations.append(f"🛣️ Optimal route length: {safe_distance:.1f}km.")
            
            if 'elevation_gain' in routes_df.columns:
                low_elevation = routes_df['elevation_gain'].quantile(0.25)
                recommendations.append(f"⛰️ Flatter routes under {low_elevation:.0f}m elevation gain.")
            
            recommendations.append("🕐 Cycling between 10 AM - 2 PM shows lowest incident rates.")
        
        return recommendations[:3]
    except Exception as e:
        logger.error(f"Error generating route recommendations: {e}")
        return []

# UI Rendering Functions
def render_no_data_message() -> None:
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

def render_quick_insights_overview(routes_df: Optional[pd.DataFrame], braking_df: Optional[pd.DataFrame], 
                                 swerving_df: Optional[pd.DataFrame], time_series_df: Optional[pd.DataFrame]) -> None:
    """Render quick overview metrics with AI insights"""
    st.markdown("## ⚡ Quick Insights")
    
    total_routes = len(routes_df) if routes_df is not None else 0
    total_braking_events = len(braking_df) if braking_df is not None else 0
    total_swerving_events = len(swerving_df) if swerving_df is not None else 0
    risk_score = calculate_overall_risk_score(braking_df, swerving_df, total_routes)
    
    col1, col2, col3, col4 = st.columns(4)
    
    for col, metric, value in [
        (col1, "Routes Analyzed", f"{total_routes:,}"),
        (col2, "Braking Events", f"{total_braking_events:,}"),
        (col3, "Swerving Events", f"{total_swerving_events:,}"),
        (col4, "Safety Score", f"{risk_score}/10")
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem;">{value}</h3>
                <p style="margin: 0; opacity: 0.9;">{metric}</p>
            </div>
            """, unsafe_allow_html=True)
    
    try:
        if INSIGHTS_GENERATOR_AVAILABLE:
            metrics = {
                'safety_score': risk_score,
                'total_routes': total_routes,
                'total_hotspots': total_braking_events + total_swerving_events,
                'avg_daily_rides': total_routes // 30 if total_routes > 0 else 0,
                'infrastructure_coverage': min(100, (total_routes / 50) * 100) if total_routes > 0 else 0
            }
            insights_generator = create_insights_generator()
            insights = insights_generator.generate_comprehensive_insights(metrics=metrics, routes_df=routes_df)
            if insights and len(insights) > 0:
                first_insight = insights[0]
                st.markdown(f"""
                <div class="insight-card">
                    <h4 style="margin: 0 0 1rem 0;">🤖 AI Insight <span class="ai-badge">DYNAMIC</span></h4>
                    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;"><strong>{first_insight.title}:</strong> {first_insight.description}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                raise Exception("No insights generated")
        else:
            raise Exception("Insights generator not available")
    except Exception as e:
        logger.warning(f"AI insights failed, using fallback: {e}")
        overview_insight = generate_overview_insight(total_routes, total_braking_events, total_swerving_events, risk_score)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">📊 Data Insight <span class="ai-badge">DYNAMIC</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{overview_insight}</p>
        </div>
        """, unsafe_allow_html=True)

def render_safety_patterns_analysis(time_series_df: Optional[pd.DataFrame], routes_df: Optional[pd.DataFrame]) -> None:
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
    
    ts_data = prepare_user_friendly_time_series(time_series_df)
    if ts_data is None:
        st.error("Unable to analyze time patterns in your data")
        return
    
    st.markdown("### 📅 Best Days to Cycle")
    daily_safety = analyze_daily_safety_patterns(ts_data)
    if daily_safety is not None:
        fig = create_daily_safety_chart(daily_safety)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        daily_insight = generate_daily_pattern_insight(daily_safety)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 What This Means <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{daily_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🕐 Safest Times to Ride")
    hourly_safety = analyze_hourly_safety_patterns(ts_data)
    if hourly_safety is not None:
        fig = create_hourly_safety_chart(hourly_safety)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        hourly_insight = generate_hourly_pattern_insight(hourly_safety)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Time-Based Insights <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{hourly_insight}</p>
        </div>
        """, unsafe_allow_html=True)

def render_risk_detection_analysis(braking_df: Optional[pd.DataFrame], swerving_df: Optional[pd.DataFrame], 
                                 time_series_df: Optional[pd.DataFrame]) -> None:
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
    
    st.markdown("### 🗺️ High-Risk Zone Detection")
    risk_zones = detect_user_friendly_risk_zones(braking_df, swerving_df)
    if risk_zones is not None and not risk_zones.empty:
        fig = create_risk_zone_map(risk_zones)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 🚨 Top Areas to Watch Out For")
        risk_display = risk_zones.head(5).copy()
        risk_display['risk_level'] = risk_display['risk_score'].apply(lambda x: 
            "🔴 High Risk" if x >= 8 else "🟡 Medium Risk" if x >= 5 else "🟢 Low Risk")
        risk_display['location_description'] = risk_display.apply(
            lambda row: f"Area near {row['lat']:.3f}, {row['lon']:.3f} - {row['type']} incidents", axis=1)
        
        st.dataframe(risk_display[['location_description', 'risk_level', 'incident_count']].rename(
            columns={'location_description': 'Location', 'risk_level': 'Risk Level', 'incident_count': 'Total Incidents'}),
            use_container_width=True, hide_index=True)
        
        risk_insight = generate_risk_detection_insight(risk_zones)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Risk Analysis <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{risk_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if time_series_df is not None and len(time_series_df) > 14:
        st.markdown("### 📊 Unusual Activity Detection")
        anomalies = detect_user_friendly_anomalies(time_series_df)
        if anomalies is not None:
            fig = create_anomaly_chart(anomalies)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            anomaly_insight = generate_anomaly_insight(anomalies)
            st.markdown(f"""
            <div class="insight-card">
                <h4 style="margin: 0 0 1rem 0;">🤖 Unusual Patterns <span class="ai-badge">AI GENERATED</span></h4>
                <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{anomaly_insight}</p>
            </div>
            """, unsafe_allow_html=True)

def render_trend_prediction_analysis(time_series_df: Optional[pd.DataFrame], routes_df: Optional[pd.DataFrame]) -> None:
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
    
    st.markdown("### 🔮 Safety Trend Forecast")
    predictions = generate_user_friendly_predictions(time_series_df)
    if predictions is not None:
        fig = create_prediction_chart(predictions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        prediction_insight = generate_prediction_insight(predictions)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Future Outlook <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{prediction_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
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

def render_connections_analysis(routes_df: Optional[pd.DataFrame], braking_df: Optional[pd.DataFrame], 
                              swerving_df: Optional[pd.DataFrame], time_series_df: Optional[pd.DataFrame]) -> None:
    """Render connections analysis in user-friendly format"""
    st.markdown("""
    <div class="analysis-section">
        <h2 style="color: #FF6B6B;">🔗 Hidden Connections</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Discover surprising relationships in your cycling data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        connections_insight = generate_connections_insight(connections)
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 1rem 0;">🤖 Why This Matters <span class="ai-badge">AI GENERATED</span></h4>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{connections_insight}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🔍 Need more data to discover meaningful connections")

# Main Render Function
def render_advanced_analytics_page() -> None:
    """Render the user-friendly advanced analytics page"""
    add_custom_css()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🧠 Smart Analytics</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            AI-powered insights that reveal what your data is really telling you
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_message()
            return
        
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        render_quick_insights_overview(routes_df, braking_df, swerving_df, time_series_df)
        
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
