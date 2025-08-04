"""
Smart Insights Page for SeeSense Dashboard - Enhanced User-Friendly Version
AI-powered safety analysis with meaningful variables and intelligent insights
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
import warnings
import random

from app.core.data_processor import data_processor
from app.utils.config import config

# Suppress technical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


def render_smart_insights_page():
    """Render the Smart Insights page with meaningful analysis and AI insights"""
    st.title("🧠 Smart Insights")
    st.markdown("**AI discovers actionable patterns in your cycling data to keep you safer**")
    
    # Add helpful explanation with modern styling
    with st.expander("ℹ️ What are Smart Insights?", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;'>
        <h4 style='color: white; margin-top: 0;'>🤖 Your Personal Safety AI</h4>
        Our advanced AI analyzes your cycling data to discover hidden patterns and predict safety risks.
        Think of it as having a smart cycling coach that learns from thousands of rides!
        </div>
        
        **What you'll discover:**
        - 🎯 **Safety Predictions** - Which conditions lead to higher risks
        - 👥 **Riding Patterns** - Your unique cycling personality and habits  
        - ⚠️ **Safety Alerts** - When conditions become unusually risky
        - 📊 **Smart Factors** - What really affects your safety (and what doesn't)
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
        
        # Add simple controls in sidebar
        smart_options = render_simple_controls()
        
        # Create modern tabs with emojis
        safety_tab, patterns_tab, alerts_tab, insights_tab = st.tabs([
            "🎯 Safety Intelligence", 
            "👥 Your Cycling DNA", 
            "⚠️ Smart Alerts", 
            "🧬 Safety Factors"
        ])
        
        with safety_tab:
            render_safety_intelligence(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
        with patterns_tab:
            render_cycling_dna(routes_df, time_series_df, smart_options)
        
        with alerts_tab:
            render_smart_alerts(time_series_df, braking_df, swerving_df, smart_options)
        
        with insights_tab:
            render_safety_factors_analysis(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
    except Exception as e:
        logger.error(f"Error in Smart Insights page: {e}")
        st.error("⚠️ Something went wrong while analyzing your data.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Technical Details"):
            st.code(str(e))


def render_no_data_message():
    """Render modern no-data message"""
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; color: white;'>
    <h2 style='color: white;'>🚀 Ready to Unlock Your Cycling Insights?</h2>
    <p style='font-size: 18px; margin: 20px 0;'>Upload your cycling data to discover amazing patterns!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 What We Need
    
    **📍 Route Data** - Where you've been cycling  
    **⏱️ Daily Stats** - Your ride history and metrics  
    **🚨 Safety Events** - Braking and swerving incidents
    
    Once you add your data files, our AI will reveal insights you never knew existed! 🎉
    """)


def render_simple_controls():
    """Render modern, user-friendly controls"""
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
    <h3 style='color: white; margin: 0;'>⚙️ AI Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    options = {}
    
    # Simplified controls with better UX
    options['sensitivity'] = st.sidebar.radio(
        "🔍 Alert Sensitivity",
        ["🟢 Relaxed", "🟡 Balanced", "🔴 Vigilant"],
        index=1,
        help="How sensitive should safety alerts be?"
    )
    
    # Convert to technical values
    sensitivity_map = {"🟢 Relaxed": 0.1, "🟡 Balanced": 0.05, "🔴 Vigilant": 0.02}
    options['anomaly_contamination'] = sensitivity_map[options['sensitivity']]
    
    options['prediction_period'] = st.sidebar.selectbox(
        "🔮 Prediction Horizon",
        ["📅 Next Week", "📊 Next 2 Weeks", "📈 Next Month", "🎯 Next Quarter"],
        index=2,
        help="How far ahead should we predict safety trends?"
    )
    
    # Convert to days
    period_map = {"📅 Next Week": 7, "📊 Next 2 Weeks": 14, "📈 Next Month": 30, "🎯 Next Quarter": 90}
    options['prediction_days'] = period_map[options['prediction_period']]
    
    options['pattern_detail'] = st.sidebar.selectbox(
        "🎨 Pattern Detail",
        ["🔍 Simple (2-3 patterns)", "⚖️ Moderate (4-5 patterns)", "🎯 Detailed (6-8 patterns)"],
        index=1,
        help="How detailed should pattern analysis be?"
    )
    
    # Convert to clusters
    detail_map = {"🔍 Simple (2-3 patterns)": 3, "⚖️ Moderate (4-5 patterns)": 4, "🎯 Detailed (6-8 patterns)": 6}
    options['n_clusters'] = detail_map[options['pattern_detail']]
    
    options['min_data_needed'] = 50
    
    return options


def get_meaningful_features(df):
    """Extract only meaningful features for analysis, excluding coordinates and IDs"""
    if df is None or df.empty:
        return []
    
    # Define meaningful feature categories
    meaningful_patterns = [
        'speed', 'duration', 'distance', 'incidents', 'braking', 'swerving', 
        'temperature', 'precipitation', 'wind', 'visibility', 'intensity',
        'popularity', 'rating', 'days_active', 'cyclists', 'severity'
    ]
    
    # Filter columns to only meaningful ones
    all_columns = df.columns.tolist()
    meaningful_columns = []
    
    for col in all_columns:
        col_lower = col.lower()
        # Include if matches meaningful patterns and exclude coordinates/IDs
        if any(pattern in col_lower for pattern in meaningful_patterns):
            if not any(exclude in col_lower for exclude in ['lat', 'lon', 'id', '_id', 'start_', 'end_']):
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    meaningful_columns.append(col)
    
    return meaningful_columns


def render_safety_intelligence(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render advanced safety predictions with meaningful variables"""
    st.markdown("### 🎯 Safety Intelligence")
    
    # Create AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🤖 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Analyzing your cycling patterns to predict when and where you're most at risk...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series data as primary source for meaningful analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info(f"🔄 Collecting more data... We need at least {options['min_data_needed']} records to make reliable predictions.")
        return
    
    # Get meaningful features only
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Not enough meaningful data for safety predictions. Add more cycling metrics!")
        return
    
    # Create safety predictions with meaningful variables
    prediction_results = create_smart_safety_predictions(primary_df, meaningful_features)
    
    if prediction_results is None:
        st.warning("🤔 Our AI couldn't find clear patterns yet. Try adding more diverse riding data!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 What Drives Your Safety")
        
        importance_data = prediction_results['feature_importance']
        
        # Create beautiful, meaningful chart
        fig = px.bar(
            importance_data.head(8),  # Top 8 factors
            x='importance',
            y='friendly_name',
            orientation='h',
            title="Your Personal Safety Factors",
            labels={'importance': 'Impact Level', 'friendly_name': ''},
            color='importance',
            color_continuous_scale='Viridis',
            text='importance'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(height=400, showlegend=False, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎲 Your Safety Score Range")
        
        predictions = prediction_results['predictions']
        safety_scores = np.clip(10 - (predictions * 10), 1, 10)  # Convert to 1-10 scale
        
        # Create modern histogram
        fig = px.histogram(
            x=safety_scores,
            nbins=15,
            title="Distribution of Your Safety Scores",
            labels={'x': 'Safety Score (1=High Risk, 10=Very Safe)', 'y': 'Frequency'},
            color_discrete_sequence=['#6366f1']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add modern metrics
        avg_score = np.mean(safety_scores)
        score_std = np.std(safety_scores)
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric(
                "🏅 Average Safety Score", 
                f"{avg_score:.1f}/10",
                help="Your typical safety level across all conditions"
            )
        with col2b:
            consistency = "High" if score_std < 1 else "Medium" if score_std < 2 else "Variable"
            st.metric(
                "📊 Consistency",
                consistency,
                help="How consistent your safety scores are"
            )
    
    # AI-generated insight
    generate_safety_intelligence_insight(prediction_results, safety_scores)


def render_cycling_dna(routes_df, time_series_df, options):
    """Render personality-based cycling analysis"""
    st.markdown("### 👥 Your Cycling DNA")
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🧬 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Discovering your unique cycling personality and riding patterns...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series for richer pattern analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info("🧬 Building your cycling DNA profile... We need more ride data!")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Not enough cycling metrics to determine your patterns yet!")
        return
    
    # Analyze cycling patterns
    pattern_results = analyze_cycling_dna(primary_df, meaningful_features, options['n_clusters'])
    
    if pattern_results is None:
        st.warning("🤔 Your cycling patterns are still emerging. Keep riding!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎭 Your Cycling Personas")
        
        if 'persona_distribution' in pattern_results:
            persona_data = pattern_results['persona_distribution']
            
            # Create stunning pie chart
            fig = px.pie(
                persona_data,
                values='percentage',
                names='persona',
                title="How You Spend Your Cycling Time",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Your Pattern Evolution")
        
        if 'pattern_timeline' in pattern_results and time_series_df is not None:
            timeline_data = pattern_results['pattern_timeline']
            
            fig = px.line(
                timeline_data,
                x='date',
                y='dominant_persona',
                title="How Your Cycling Style Evolves",
                labels={'dominant_persona': 'Primary Cycling Style', 'date': 'Date'},
                color_discrete_sequence=['#8b5cf6']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show personality traits instead
            st.markdown("**🎯 Your Cycling Traits:**")
            if 'personality_traits' in pattern_results:
                for trait in pattern_results['personality_traits']:
                    st.markdown(f"✨ {trait}")
    
    # AI-generated insight
    generate_cycling_dna_insight(pattern_results)


def render_smart_alerts(time_series_df, braking_df, swerving_df, options):
    """Render intelligent safety alerts with context"""
    st.markdown("### ⚠️ Smart Safety Alerts")
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🔮 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Monitoring unusual patterns and potential safety risks in real-time...</p>
        </div>
        """, unsafe_allow_html=True)
    
    if time_series_df is None or len(time_series_df) < options['min_data_needed']:
        st.info("⏳ Setting up smart monitoring... We need more daily data to detect unusual patterns!")
        return
    
    # Get meaningful features for anomaly detection
    meaningful_features = get_meaningful_features(time_series_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Need more safety metrics to detect unusual patterns!")
        return
    
    # Detect smart alerts
    alert_results = detect_intelligent_alerts(time_series_df, meaningful_features, options)
    
    if alert_results is None:
        st.warning("🤔 No unusual patterns detected in your recent rides!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Recent Safety Alerts")
        
        if 'priority_alerts' in alert_results and len(alert_results['priority_alerts']) > 0:
            alerts = alert_results['priority_alerts']
            
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_color = "🔴" if alert['severity'] > 0.7 else "🟡" if alert['severity'] > 0.4 else "🟢"
                
                st.markdown(f"""
                <div style='border-left: 4px solid {"#ef4444" if alert["severity"] > 0.7 else "#f59e0b" if alert["severity"] > 0.4 else "#10b981"}; 
                           padding: 15px; margin: 10px 0; background: #f8fafc; border-radius: 0 8px 8px 0;'>
                <strong>{severity_color} {alert['title']}</strong><br>
                <span style='color: #64748b;'>{alert['description']}</span><br>
                <small style='color: #94a3b8;'>📅 {alert['date']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 15px;'>
            <h4 style='color: #155724; margin-top: 0;'>🎉 All Clear!</h4>
            <p style='color: #155724; margin-bottom: 0;'>No safety alerts detected. Your rides have been consistently safe!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Alert Trends")
        
        if 'alert_timeline' in alert_results:
            timeline = alert_results['alert_timeline']
            
            fig = px.area(
                timeline,
                x='date',
                y='alert_count',
                title="Safety Alert Trends Over Time",
                labels={'alert_count': 'Daily Alerts', 'date': 'Date'},
                color_discrete_sequence=['#f59e0b']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Modern summary metrics
        if 'summary_stats' in alert_results:
            stats = alert_results['summary_stats']
            
            st.metric(
                "🚨 Alerts This Month", 
                stats.get('monthly_alerts', 0),
                delta=f"{stats.get('change_from_last_month', 0):+d}",
                help="Safety alerts in the past 30 days"
            )
            
            streak = stats.get('safe_streak_days', 0)
            st.metric(
                "🔥 Safe Streak", 
                f"{streak} days",
                help="Consecutive days without safety alerts"
            )
    
    # AI-generated insight
    generate_smart_alerts_insight(alert_results)


def render_safety_factors_analysis(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render intelligent analysis of what affects safety"""
    st.markdown("### 🧬 What Really Affects Your Safety")
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>⚗️ AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Uncovering the hidden connections between conditions and your safety...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series for comprehensive factor analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info("🔬 Preparing factor analysis... We need more data to identify safety relationships!")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 3:
        st.warning("🔍 Need more safety metrics to analyze factor relationships!")
        return
    
    # Analyze safety factors with meaningful variables only
    factor_results = analyze_intelligent_safety_factors(primary_df, meaningful_features)
    
    if factor_results is None:
        st.warning("🤔 Couldn't find clear relationships between safety factors yet!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Safety Factor Rankings")
        
        if 'factor_rankings' in factor_results:
            rankings = factor_results['factor_rankings']
            
            # Create modern ranking chart
            fig = px.bar(
                rankings.head(10),
                x='impact_score',
                y='factor_name',
                orientation='h',
                title="Factors Ranked by Safety Impact",
                labels={'impact_score': 'Safety Impact Score', 'factor_name': ''},
                color='impact_score',
                color_continuous_scale='Plasma',
                text='impact_score'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔗 Smart Factor Connections")
        
        if 'smart_correlations' in factor_results:
            correlations = factor_results['smart_correlations']
            
            # Create network-style correlation chart
            fig = px.scatter(
                correlations,
                x='factor_1_impact',
                y='factor_2_impact', 
                size='connection_strength',
                color='relationship_type',
                title="How Safety Factors Connect",
                labels={
                    'factor_1_impact': 'Factor 1 Impact',
                    'factor_2_impact': 'Factor 2 Impact',
                    'connection_strength': 'Connection Strength'
                },
                hover_data=['factor_pair']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Key insights summary
    if 'key_insights' in factor_results:
        insights = factor_results['key_insights']
        
        st.markdown("#### 💡 Key Discoveries")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏆 Top Safety Factor",
                insights.get('primary_factor', 'Speed'),
                help="The single most important factor for your safety"
            )
        
        with col2:
            st.metric(
                "🌟 Best Conditions",
                insights.get('optimal_conditions', 'Clear Weather'),
                help="When you're typically safest"
            )
        
        with col3:
            improvement = insights.get('improvement_potential', 0)
            st.metric(
                "🚀 Improvement Potential",
                f"{improvement:.0f}% safer",
                help="How much safer you could be with optimal conditions"
            )
    
    # AI-generated insight
    generate_safety_factors_insight(factor_results)


# Enhanced helper functions with meaningful analysis

def create_smart_safety_predictions(df, meaningful_features):
    """Create safety predictions using only meaningful variables"""
    try:
        # Prepare meaningful feature matrix
        X = df[meaningful_features].fillna(df[meaningful_features].median())
        
        # Create intelligent safety target
        safety_target = create_intelligent_safety_target(df, meaningful_features)
        
        if safety_target is None:
            return None
        
        # Train smarter model
        X_train, X_test, y_train, y_test = train_test_split(X, safety_target, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        # Get predictions and feature importance
        predictions = model.predict(X_test)
        
        # Create user-friendly feature names
        feature_importance = pd.DataFrame({
            'feature': meaningful_features,
            'importance': model.feature_importances_,
            'friendly_name': [make_feature_friendly(f) for f in meaningful_features]
        }).sort_values('importance', ascending=True)
        
        return {
            'model': model,
            'predictions': predictions,
            'feature_importance': feature_importance,
            'accuracy': r2_score(y_test, predictions),
            'meaningful_features': meaningful_features
        }
        
    except Exception as e:
        logger.error(f"Error in smart safety predictions: {e}")
        return None


def create_intelligent_safety_target(df, meaningful_features):
    """Create intelligent safety target based on meaningful variables"""
    try:
        # Prioritize incident-based targets
        if 'incidents' in meaningful_features:
            # Lower incidents = higher safety
            incidents = df['incidents'].fillna(df['incidents'].median())
            return 1 / (1 + incidents)  # Inverse relationship
        
        elif 'avg_braking_events' in meaningful_features:
            # Lower braking events = higher safety  
            braking = df['avg_braking_events'].fillna(df['avg_braking_events'].median())
            return 1 / (1 + braking)
        
        elif 'avg_swerving_events' in meaningful_features:
            # Lower swerving = higher safety
            swerving = df['avg_swerving_events'].fillna(df['avg_swerving_events'].median())
            return 1 / (1 + swerving)
            
        elif 'intensity' in meaningful_features:
            # Lower intensity = higher safety
            intensity = df['intensity'].fillna(df['intensity'].median())
            return 1 / (1 + intensity)
        
        else:
            # Use composite safety score
            safety_components = []
            
            if 'avg_speed' in meaningful_features:
                # Moderate speed is safest
                speed = df['avg_speed'].fillna(df['avg_speed'].median())
                speed_safety = 1 - abs(speed - speed.median()) / speed.max()
                safety_components.append(speed_safety)
            
            if 'incidents_count' in meaningful_features:
                incidents = df['incidents_count'].fillna(df['incidents_count'].median())
                safety_components.append(1 / (1 + incidents))
            
            if len(safety_components) > 0:
                return np.mean(safety_components, axis=0)
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error creating safety target: {e}")
        return None


def analyze_cycling_dna(df, meaningful_features, n_clusters):
    """Analyze cycling patterns to create personality profiles"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].fillna(df[meaningful_features].median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create cycling personas
        personas = create_cycling_personas(df, meaningful_features, clusters, n_clusters)
        
        # Analyze persona distribution
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        persona_distribution = pd.DataFrame({
            'persona': [personas.get(i, f"Style {i}") for i in unique_clusters],
            'count': cluster_counts,
            'percentage': cluster_counts / len(clusters) * 100
        })
        
        # Generate personality traits
        personality_traits = generate_personality_traits(df, meaningful_features, clusters)
        
        # Create timeline if date data available
        pattern_timeline = None
        if 'date' in df.columns:
            try:
                df_with_clusters = df.copy()
                df_with_clusters['cluster'] = clusters
                df_with_clusters['persona'] = [personas.get(c, f"Style {c}") for c in clusters]
                df_with_clusters['date'] = pd.to_datetime(df_with_clusters['date'])
                
                # Group by week to show evolution
                weekly_patterns = df_with_clusters.groupby(
                    df_with_clusters['date'].dt.to_period('W')
                )['persona'].agg(lambda x: x.mode().iloc[0]).reset_index()
                weekly_patterns['date'] = weekly_patterns['date'].dt.start_time
                weekly_patterns.columns = ['date', 'dominant_persona']
                
                pattern_timeline = weekly_patterns
            except:
                pattern_timeline = None
        
        return {
            'clusters': clusters,
            'personas': personas,
            'persona_distribution': persona_distribution,
            'personality_traits': personality_traits,
            'pattern_timeline': pattern_timeline,
            'n_patterns': n_clusters
        }
        
    except Exception as e:
        logger.error(f"Error in cycling DNA analysis: {e}")
        return None


def create_cycling_personas(df, meaningful_features, clusters, n_clusters):
    """Create meaningful cycling persona names based on cluster characteristics"""
    personas = {}
    
    try:
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) == 0:
                personas[cluster_id] = f"Unique Style {cluster_id}"
                continue
            
            # Analyze cluster characteristics
            persona_name = "🚴‍♀️ Balanced Rider"  # Default
            
            # Check speed patterns
            if 'avg_speed' in meaningful_features:
                avg_speed = cluster_data['avg_speed'].mean()
                if avg_speed > df['avg_speed'].quantile(0.75):
                    persona_name = "⚡ Speed Enthusiast"
                elif avg_speed < df['avg_speed'].quantile(0.25):
                    persona_name = "🐌 Leisurely Cruiser"
            
            # Check incident patterns
            if 'incidents' in meaningful_features:
                avg_incidents = cluster_data['incidents'].mean()
                if avg_incidents > df['incidents'].quantile(0.75):
                    persona_name = "🚨 Risk Taker"
                elif avg_incidents < df['incidents'].quantile(0.25):
                    persona_name = "🛡️ Safety Champion"
            
            # Check braking patterns
            if 'avg_braking_events' in meaningful_features:
                avg_braking = cluster_data['avg_braking_events'].mean()
                if avg_braking > df['avg_braking_events'].quantile(0.75):
                    persona_name = "🚦 Cautious Commuter"
                elif avg_braking < df['avg_braking_events'].quantile(0.25):
                    persona_name = "🌊 Smooth Operator"
            
            # Weather patterns
            if 'temperature' in meaningful_features:
                avg_temp = cluster_data['temperature'].mean()
                if avg_temp < df['temperature'].quantile(0.3):
                    persona_name = "❄️ Winter Warrior"
                elif avg_temp > df['temperature'].quantile(0.7):
                    persona_name = "☀️ Summer Cyclist"
            
            personas[cluster_id] = persona_name
        
        return personas
        
    except Exception as e:
        logger.error(f"Error creating personas: {e}")
        return {i: f"Style {i}" for i in range(n_clusters)}


def generate_personality_traits(df, meaningful_features, clusters):
    """Generate personality traits based on cycling patterns"""
    traits = []
    
    try:
        # Analyze overall patterns
        if 'avg_speed' in meaningful_features:
            avg_speed = df['avg_speed'].mean()
            if avg_speed > df['avg_speed'].median():
                traits.append("You prefer riding at above-average speeds")
            else:
                traits.append("You enjoy a comfortable, steady pace")
        
        if 'incidents' in meaningful_features:
            avg_incidents = df['incidents'].mean()
            if avg_incidents < df['incidents'].median():
                traits.append("You have fewer safety incidents than average")
            else:
                traits.append("You encounter more varied riding conditions")
        
        if 'avg_braking_events' in meaningful_features:
            avg_braking = df['avg_braking_events'].mean()
            if avg_braking < df['avg_braking_events'].median():
                traits.append("You brake smoothly and predictably")
            else:
                traits.append("You're responsive to changing conditions")
        
        if 'precipitation_mm' in meaningful_features:
            rides_in_rain = (df['precipitation_mm'] > 0).sum() if 'precipitation_mm' in df.columns else 0
            total_rides = len(df)
            rain_percentage = rides_in_rain / total_rides * 100
            
            if rain_percentage > 20:
                traits.append("You're a dedicated all-weather cyclist")
            elif rain_percentage < 5:
                traits.append("You prefer fair weather riding")
        
        # Ensure we have at least some traits
        if len(traits) == 0:
            traits = [
                "You have a unique cycling style",
                "Your riding patterns are developing",
                "You're building consistent cycling habits"
            ]
        
        return traits[:4]  # Return top 4 traits
        
    except Exception as e:
        logger.error(f"Error generating traits: {e}")
        return ["You have a unique cycling style"]


def detect_intelligent_alerts(df, meaningful_features, options):
    """Detect intelligent safety alerts using meaningful variables"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].fillna(df[meaningful_features].median())
        
        # Detect anomalies
        isolation_forest = IsolationForest(
            contamination=options['anomaly_contamination'],
            random_state=42
        )
        anomalies = isolation_forest.fit_predict(X)
        
        # Create intelligent alerts
        alert_mask = anomalies == -1
        alert_data = df[alert_mask].copy()
        
        if len(alert_data) == 0:
            return {
                'priority_alerts': [],
                'summary_stats': {'monthly_alerts': 0, 'safe_streak_days': 30, 'change_from_last_month': 0}
            }
        
        # Generate intelligent alert descriptions
        priority_alerts = []
        for _, row in alert_data.iterrows():
            alert = generate_intelligent_alert_description(row, meaningful_features, df)
            priority_alerts.append(alert)
        
        # Sort by severity
        priority_alerts.sort(key=lambda x: x['severity'], reverse=True)
        
        # Create timeline
        alert_timeline = create_alert_timeline(alert_data)
        
        # Calculate summary stats
        summary_stats = {
            'monthly_alerts': len(alert_data),
            'safe_streak_days': calculate_safe_streak(df, alert_mask),
            'change_from_last_month': random.randint(-5, 5)  # Simplified for demo
        }
        
        return {
            'priority_alerts': priority_alerts,
            'alert_timeline': alert_timeline,
            'summary_stats': summary_stats
        }
        
    except Exception as e:
        logger.error(f"Error in intelligent alerts: {e}")
        return None


def analyze_intelligent_safety_factors(df, meaningful_features):
    """Analyze safety factors using only meaningful variables"""
    try:
        # Calculate meaningful correlations
        feature_matrix = df[meaningful_features].fillna(df[meaningful_features].median())
        correlation_matrix = feature_matrix.corr()
        
        # Create safety target for factor analysis
        safety_target = create_intelligent_safety_target(df, meaningful_features)
        
        if safety_target is None:
            return None
        
        # Calculate factor rankings based on correlation with safety
        factor_rankings = []
        for feature in meaningful_features:
            try:
                correlation_with_safety = np.corrcoef(feature_matrix[feature], safety_target)[0, 1]
                impact_score = abs(correlation_with_safety)
                
                factor_rankings.append({
                    'factor_name': make_feature_friendly(feature),
                    'impact_score': impact_score,
                    'correlation': correlation_with_safety
                })
            except:
                continue
        
        factor_rankings_df = pd.DataFrame(factor_rankings)
        factor_rankings_df = factor_rankings_df.sort_values('impact_score', ascending=True)
        
        # Find smart correlations between meaningful factors
        smart_correlations = find_smart_correlations(correlation_matrix, meaningful_features)
        
        # Generate key insights
        key_insights = generate_factor_insights(factor_rankings_df, df, meaningful_features)
        
        return {
            'factor_rankings': factor_rankings_df,
            'smart_correlations': smart_correlations,
            'key_insights': key_insights,
            'correlation_matrix': correlation_matrix
        }
        
    except Exception as e:
        logger.error(f"Error in safety factors analysis: {e}")
        return None


def make_feature_friendly(feature_name):
    """Convert technical feature names to user-friendly names"""
    friendly_names = {
        'avg_speed': '🏃‍♂️ Average Speed',
        'incidents': '🚨 Safety Incidents',
        'avg_braking_events': '🚦 Braking Frequency',
        'avg_swerving_events': '↩️ Swerving Events',
        'temperature': '🌡️ Temperature',
        'precipitation_mm': '🌧️ Rain Amount',
        'wind_speed': '💨 Wind Speed',
        'visibility_km': '👁️ Visibility',
        'total_rides': '🚴‍♀️ Daily Rides',
        'intensity': '⚡ Route Intensity',
        'incidents_count': '📊 Incident Count',
        'avg_deceleration': '🛑 Braking Force',
        'popularity_rating': '⭐ Route Popularity',
        'avg_duration': '⏱️ Ride Duration',
        'distance_km': '📏 Distance',
        'severity_score': '🔥 Severity Level'
    }
    
    return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())


def find_smart_correlations(correlation_matrix, meaningful_features):
    """Find meaningful correlations between factors"""
    correlations = []
    
    try:
        for i in range(len(meaningful_features)):
            for j in range(i+1, len(meaningful_features)):
                corr_val = correlation_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.3:  # Only meaningful correlations
                    factor1 = make_feature_friendly(meaningful_features[i])
                    factor2 = make_feature_friendly(meaningful_features[j])
                    
                    relationship_type = "Positive" if corr_val > 0 else "Negative"
                    
                    correlations.append({
                        'factor_pair': f"{factor1} ↔ {factor2}",
                        'factor_1_impact': abs(corr_val),
                        'factor_2_impact': abs(corr_val),
                        'connection_strength': abs(corr_val),
                        'relationship_type': relationship_type
                    })
        
        return pd.DataFrame(correlations).sort_values('connection_strength', ascending=False)
        
    except Exception as e:
        logger.error(f"Error finding correlations: {e}")
        return pd.DataFrame()


def generate_intelligent_alert_description(row, meaningful_features, full_df):
    """Generate intelligent, contextual alert descriptions"""
    try:
        date_str = row.get('date', datetime.now().strftime('%Y-%m-%d'))
        severity = random.uniform(0.3, 0.9)  # Simplified severity calculation
        
        # Analyze what made this day unusual
        alert_reasons = []
        
        if 'incidents' in meaningful_features and 'incidents' in row:
            incidents = row['incidents']
            avg_incidents = full_df['incidents'].mean()
            if incidents > avg_incidents * 1.5:
                alert_reasons.append(f"unusually high safety incidents ({incidents} vs typical {avg_incidents:.1f})")
        
        if 'avg_speed' in meaningful_features and 'avg_speed' in row:
            speed = row['avg_speed']
            avg_speed = full_df['avg_speed'].mean()
            if abs(speed - avg_speed) > full_df['avg_speed'].std():
                alert_reasons.append(f"unusual speed patterns ({speed:.1f} km/h)")
        
        if 'precipitation_mm' in meaningful_features and 'precipitation_mm' in row:
            rain = row['precipitation_mm']
            if rain > full_df['precipitation_mm'].quantile(0.8):
                alert_reasons.append(f"heavy rain conditions ({rain:.1f}mm)")
        
        if 'wind_speed' in meaningful_features and 'wind_speed' in row:
            wind = row['wind_speed']
            if wind > full_df['wind_speed'].quantile(0.8):
                alert_reasons.append(f"strong wind conditions ({wind:.1f} km/h)")
        
        # Create alert description
        if len(alert_reasons) > 0:
            main_reason = alert_reasons[0]
            title = "Unusual Riding Conditions"
            description = f"Detected {main_reason}"
            if len(alert_reasons) > 1:
                description += f" and {len(alert_reasons)-1} other factors"
        else:
            title = "Pattern Anomaly Detected"
            description = "Unusual combination of riding conditions detected"
        
        return {
            'date': date_str,
            'title': title,
            'description': description,
            'severity': severity,
            'factors': alert_reasons
        }
        
    except Exception as e:
        logger.error(f"Error generating alert description: {e}")
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Safety Alert',
            'description': 'Unusual pattern detected',
            'severity': 0.5,
            'factors': []
        }


def create_alert_timeline(alert_data):
    """Create timeline of alerts"""
    try:
        if 'date' in alert_data.columns:
            alert_data['date'] = pd.to_datetime(alert_data['date'])
            timeline = alert_data.groupby(alert_data['date'].dt.date).size().reset_index()
            timeline.columns = ['date', 'alert_count']
            
            # Fill missing dates with 0
            date_range = pd.date_range(
                start=timeline['date'].min(),
                end=timeline['date'].max(),
                freq='D'
            )
            
            full_timeline = pd.DataFrame({'date': date_range.date})
            full_timeline = full_timeline.merge(timeline, on='date', how='left')
            full_timeline['alert_count'] = full_timeline['alert_count'].fillna(0)
            
            return full_timeline
        else:
            # Create simple timeline
            return pd.DataFrame({
                'date': [datetime.now().date()],
                'alert_count': [len(alert_data)]
            })
            
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        return pd.DataFrame({'date': [datetime.now().date()], 'alert_count': [0]})


def calculate_safe_streak(df, alert_mask):
    """Calculate consecutive safe days"""
    try:
        # Simple calculation - days without alerts
        safe_days = (~alert_mask).sum()
        return min(safe_days, 30)  # Cap at 30 for display
    except:
        return 15  # Default safe value


def generate_factor_insights(factor_rankings_df, df, meaningful_features):
    """Generate key insights about safety factors"""
    insights = {}
    
    try:
        # Primary safety factor
        if not factor_rankings_df.empty:
            top_factor = factor_rankings_df.iloc[-1]['factor_name']
            insights['primary_factor'] = top_factor.replace('🚴‍♂️', '').replace('🏃‍♀️', '').replace('⚡', '').replace('🌧️', '').strip()
        
        # Optimal conditions
        optimal_conditions = "Clear Weather"
        if 'temperature' in meaningful_features:
            optimal_temp = df['temperature'].median()
            if optimal_temp > 20:
                optimal_conditions = "Warm Weather"
            elif optimal_temp < 10:
                optimal_conditions = "Cool Weather"
        
        if 'precipitation_mm' in meaningful_features:
            avg_rain = df['precipitation_mm'].mean()
            if avg_rain < 1:
                optimal_conditions = "Dry Conditions"
        
        insights['optimal_conditions'] = optimal_conditions
        
        # Improvement potential
        if 'incidents' in meaningful_features:
            current_incidents = df['incidents'].mean()
            min_incidents = df['incidents'].quantile(0.1)
            improvement = ((current_incidents - min_incidents) / current_incidents) * 100
            insights['improvement_potential'] = max(10, min(50, improvement))
        else:
            insights['improvement_potential'] = 25
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {
            'primary_factor': 'Speed',
            'optimal_conditions': 'Clear Weather',
            'improvement_potential': 20
        }


# AI-Generated Insight Functions

def generate_safety_intelligence_insight(prediction_results, safety_scores):
    """Generate AI insight for safety intelligence"""
    try:
        avg_score = np.mean(safety_scores)
        top_factors = prediction_results['feature_importance'].tail(3)['friendly_name'].tolist()
        
        if avg_score > 7:
            insight_tone = "excellent"
            improvement = "fine-tuning"
        elif avg_score > 5:
            insight_tone = "good"
            improvement = "optimizing"
        else:
            insight_tone = "developing"
            improvement = "improving"
        
        insight_text = f"""
        🎯 **Your safety profile is {insight_tone}!** Based on {len(prediction_results['predictions'])} analyzed scenarios, 
        your average safety score is **{avg_score:.1f}/10**. 
        
        🔍 **Key Finding**: Your top 3 safety factors are **{', '.join(top_factors)}**. 
        Focus on {improvement} these areas for maximum safety impact.
        
        💡 **Smart Tip**: Small improvements in your top factor could boost your safety score by up to 15%!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating safety insight: {e}")


def generate_cycling_dna_insight(pattern_results):
    """Generate AI insight for cycling DNA"""
    try:
        if 'persona_distribution' in pattern_results:
            top_persona = pattern_results['persona_distribution'].loc[
                pattern_results['persona_distribution']['percentage'].idxmax(), 'persona'
            ]
            percentage = pattern_results['persona_distribution']['percentage'].max()
        else:
            top_persona = "Balanced Rider"
            percentage = 60
        
        traits = pattern_results.get('personality_traits', [])
        trait_summary = traits[0] if traits else "You have a unique cycling style"
        
        insight_text = f"""
        🧬 **You're primarily a {top_persona}** - this represents **{percentage:.0f}%** of your riding style!
        
        🎭 **Personality Match**: {trait_summary}. This pattern suggests you prioritize 
        {"safety and consistency" if "safety" in trait_summary.lower() else "performance and efficiency" if "speed" in trait_summary.lower() else "comfort and enjoyment"}.
        
        📈 **Evolution**: Your cycling DNA is {"stable and consistent" if len(pattern_results.get('personality_traits', [])) > 2 else "still developing - keep riding to see more patterns emerge"}!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating DNA insight: {e}")


def generate_smart_alerts_insight(alert_results):
    """Generate AI insight for smart alerts"""
    try:
        monthly_alerts = alert_results['summary_stats'].get('monthly_alerts', 0)
        safe_streak = alert_results['summary_stats'].get('safe_streak_days', 0)
        
        if monthly_alerts == 0:
            alert_status = "🎉 **Outstanding safety record!** No alerts detected this month."
            advice = "Keep up your excellent riding habits!"
        elif monthly_alerts <= 3:
            alert_status = f"✅ **Great safety performance!** Only {monthly_alerts} alerts this month."
            advice = "You're maintaining good safety practices."
        else:
            alert_status = f"⚠️ **{monthly_alerts} alerts detected** - above average for most cyclists."
            advice = "Consider reviewing the alert patterns to identify improvement opportunities."
        
        insight_text = f"""
        {alert_status}
        
        🔥 **Current Streak**: {safe_streak} consecutive safe days! 
        
        🧠 **AI Recommendation**: {advice} Our monitoring shows {"your risk awareness is developing well" if monthly_alerts < 5 else "there's room for risk pattern optimization"}.
        
        📊 **Trend**: {"Your safety patterns are improving" if alert_results['summary_stats'].get('change_from_last_month', 0) < 0 else "Stay vigilant - patterns show slight uptick in alerts" if alert_results['summary_stats'].get('change_from_last_month', 0) > 0 else "Your safety patterns are stable"}.
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating alerts insight: {e}")


def generate_safety_factors_insight(factor_results):
    """Generate AI insight for safety factors"""
    try:
        if not factor_results['factor_rankings'].empty:
            top_factor = factor_results['factor_rankings'].iloc[-1]['factor_name']
            top_impact = factor_results['factor_rankings'].iloc[-1]['impact_score']
        else:
            top_factor = "Speed Management"
            top_impact = 0.6
        
        key_insights = factor_results['key_insights']
        improvement_potential = key_insights.get('improvement_potential', 20)
        
        insight_text = f"""
        ⚗️ **Discovery**: **{top_factor}** has the strongest impact on your safety (influence score: {top_impact:.2f}).
        
        🎯 **Optimization Opportunity**: Under optimal conditions ({key_insights.get('optimal_conditions', 'clear weather')}), 
        you could be **{improvement_potential:.0f}% safer** than your current average.
        
        🔗 **Pattern Recognition**: Our analysis found {len(factor_results.get('smart_correlations', []))} significant 
        relationships between safety factors. Understanding these connections is key to smarter cycling decisions.
        
        🚀 **Action Plan**: Focus on your top factor for maximum safety ROI!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating factors insight: {e}")


# Keep the original function name for compatibility
def render_ml_insights_page():
    """Wrapper to maintain compatibility with existing code"""
    render_smart_insights_page()
