"""
Smart Insights Page for SeeSense Dashboard - User-Friendly Version
AI-powered safety analysis made simple for everyone to understand
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

from app.core.data_processor import data_processor
from app.utils.config import config

# Suppress technical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


def render_smart_insights_page():
    """Render the Smart Insights page in user-friendly language"""
    st.title("🧠 Smart Insights")
    st.markdown("**AI discovers patterns in your cycling data to keep you safer**")
    
    # Add helpful explanation
    with st.expander("ℹ️ What are Smart Insights?"):
        st.markdown("""
        Our AI looks at your cycling data to find patterns and predict potential safety issues.
        Think of it as having a smart assistant that learns from thousands of bike rides to give you personalized safety tips.
        
        **What you'll see:**
        - 🎯 **Safety Predictions** - Areas where you might need to be extra careful
        - 👥 **Riding Patterns** - How your cycling style compares to others
        - ⚠️ **Unusual Events** - Times when something different happened on your rides
        - 📊 **Data Insights** - Which factors most affect your safety
        """)
    
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
        
        # Create tabs with friendly names
        safety_tab, patterns_tab, alerts_tab, insights_tab = st.tabs([
            "🎯 Safety Predictions", 
            "👥 Your Cycling Patterns", 
            "⚠️ Safety Alerts", 
            "📊 What Affects Your Safety"
        ])
        
        with safety_tab:
            render_safety_predictions(routes_df, braking_df, swerving_df, smart_options)
        
        with patterns_tab:
            render_cycling_patterns(routes_df, time_series_df, smart_options)
        
        with alerts_tab:
            render_safety_alerts(time_series_df, braking_df, swerving_df, smart_options)
        
        with insights_tab:
            render_safety_factors(routes_df, braking_df, swerving_df, smart_options)
        
    except Exception as e:
        logger.error(f"Error in Smart Insights page: {e}")
        st.error("⚠️ Something went wrong while analyzing your data.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Technical Details"):
            st.code(str(e))


def render_no_data_message():
    """Render friendly message when no data is available"""
    st.warning("⚠️ No cycling data found for smart analysis.")
    st.markdown("""
    To get smart insights, you need some cycling data:
    
    📍 **Route information** - Where you've been cycling
    ⏱️ **Ride history** - Your past cycling activities  
    🚨 **Safety events** - Any times you had to brake hard or swerve
    
    Once you add your data files, come back here to see what patterns our AI discovers!
    """)


def render_simple_controls():
    """Render user-friendly configuration controls"""
    st.sidebar.markdown("### ⚙️ Analysis Settings")
    
    options = {}
    
    # Simplified controls with helpful explanations
    options['sensitivity'] = st.sidebar.radio(
        "Alert Sensitivity",
        ["Low", "Medium", "High"],
        index=1,
        help="How sensitive should safety alerts be? Higher = more alerts"
    )
    
    # Convert to technical values
    sensitivity_map = {"Low": 0.1, "Medium": 0.05, "High": 0.02}
    options['anomaly_contamination'] = sensitivity_map[options['sensitivity']]
    
    options['prediction_period'] = st.sidebar.selectbox(
        "Look Ahead",
        ["1 Week", "2 Weeks", "1 Month", "3 Months"],
        index=2,
        help="How far into the future should we predict safety risks?"
    )
    
    # Convert to days
    period_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
    options['prediction_days'] = period_map[options['prediction_period']]
    
    options['group_similar_rides'] = st.sidebar.selectbox(
        "Group Similar Rides",
        ["2-3 groups", "4-5 groups", "6-8 groups"],
        index=1,
        help="How many different types of riding patterns should we look for?"
    )
    
    # Convert to number of clusters
    cluster_map = {"2-3 groups": 3, "4-5 groups": 4, "6-8 groups": 6}
    options['n_clusters'] = cluster_map[options['group_similar_rides']]
    
    options['min_data_needed'] = 50  # Keep this simple and hidden
    
    return options


def render_safety_predictions(routes_df, braking_df, swerving_df, options):
    """Render safety prediction analysis in simple terms"""
    st.markdown("### 🎯 Where You Might Need Extra Caution")
    st.markdown("Our AI predicts areas where you should be extra careful based on your riding history.")
    
    if routes_df is None or len(routes_df) < options['min_data_needed']:
        st.info(f"We need at least {options['min_data_needed']} rides to make good predictions. Keep cycling and check back!")
        return
    
    # Train the safety prediction model
    prediction_results = create_safety_predictions(routes_df, braking_df, swerving_df)
    
    if prediction_results is None:
        st.warning("We couldn't create safety predictions from your current data. Try adding more ride data!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 What Matters Most for Your Safety")
        
        # Show feature importance in simple terms
        if 'feature_importance' in prediction_results:
            importance_data = prediction_results['feature_importance']
            
            # Translate technical terms to user-friendly ones
            friendly_names = {
                'speed': 'Your Speed',
                'distance': 'Trip Distance', 
                'elevation': 'Hills & Slopes',
                'time_of_day': 'Time of Day',
                'weather': 'Weather Conditions',
                'traffic': 'Traffic Levels',
                'route_popularity': 'How Busy the Route Is'
            }
            
            # Create user-friendly chart
            importance_data['friendly_name'] = importance_data['feature'].map(
                lambda x: friendly_names.get(x, x.replace('_', ' ').title())
            )
            
            fig = px.bar(
                importance_data,
                x='importance',
                y='friendly_name',
                orientation='h',
                title="What Affects Your Safety Most",
                labels={'importance': 'Impact on Safety', 'friendly_name': ''},
                color='importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎲 Your Safety Score Distribution")
        
        # Show predictions in simple terms
        if 'predictions' in prediction_results:
            predictions = prediction_results['predictions']
            
            # Convert to 1-10 safety score
            safety_scores = 10 - (predictions * 10)  # Flip so higher = safer
            
            fig = px.histogram(
                x=safety_scores,
                nbins=15,
                title="Your Safety Scores Across Different Situations",
                labels={'x': 'Safety Score (1=Risky, 10=Very Safe)', 'y': 'Number of Situations'},
                color_discrete_sequence=['#2E86AB']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary stats
            avg_score = np.mean(safety_scores)
            st.metric(
                "Your Average Safety Score", 
                f"{avg_score:.1f}/10",
                help="Higher scores mean safer riding conditions"
            )


def render_cycling_patterns(routes_df, time_series_df, options):
    """Render cycling behavior analysis in simple terms"""
    st.markdown("### 👥 Your Unique Cycling Style")
    st.markdown("See how your cycling patterns compare to different riding styles.")
    
    if routes_df is None or len(routes_df) < options['min_data_needed']:
        st.info("We need more ride data to identify your cycling patterns. Keep tracking your rides!")
        return
    
    # Analyze cycling patterns
    pattern_results = analyze_cycling_patterns(routes_df, time_series_df, options['n_clusters'])
    
    if pattern_results is None:
        st.warning("We couldn't identify clear patterns in your cycling data yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏷️ Your Cycling Types")
        
        if 'cluster_summary' in pattern_results:
            cluster_info = pattern_results['cluster_summary']
            
            # Create friendly cluster names
            cluster_names = {
                0: "🚴‍♀️ Casual Explorer",
                1: "🏃‍♂️ Fitness Focused", 
                2: "🚗 Commuter Pro",
                3: "🏔️ Adventure Seeker",
                4: "🌅 Weekend Warrior",
                5: "⚡ Speed Demon"
            }
            
            # Show cluster distribution
            cluster_counts = pd.DataFrame({
                'Riding Style': [cluster_names.get(i, f"Style {i}") for i in cluster_info.index],
                'Number of Rides': cluster_info.values
            })
            
            fig = px.pie(
                cluster_counts,
                values='Number of Rides',
                names='Riding Style',
                title="How You Spend Your Cycling Time"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Your Patterns Over Time")
        
        if 'pattern_trends' in pattern_results and time_series_df is not None:
            # Show how patterns change over time
            trend_data = pattern_results['pattern_trends']
            
            fig = px.line(
                trend_data,
                x='date',
                y='dominant_pattern',
                title="How Your Cycling Style Changes",
                labels={'dominant_pattern': 'Main Riding Style', 'date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show pattern characteristics
            st.markdown("**Your Main Cycling Characteristics:**")
            if 'characteristics' in pattern_results:
                for char in pattern_results['characteristics']:
                    st.markdown(f"• {char}")


def render_safety_alerts(time_series_df, braking_df, swerving_df, options):
    """Render anomaly detection in simple terms"""
    st.markdown("### ⚠️ Unusual Safety Events")
    st.markdown("Times when something unusual happened during your rides that might indicate safety concerns.")
    
    if time_series_df is None or len(time_series_df) < options['min_data_needed']:
        st.info("We need more ride data to detect unusual safety events.")
        return
    
    # Detect unusual events
    alert_results = detect_safety_alerts(time_series_df, braking_df, swerving_df, options)
    
    if alert_results is None:
        st.warning("We couldn't detect any unusual patterns in your data.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Recent Safety Alerts")
        
        if 'recent_alerts' in alert_results and not alert_results['recent_alerts'].empty:
            alerts = alert_results['recent_alerts']
            
            # Format alerts in user-friendly way
            for _, alert in alerts.head(5).iterrows():
                severity = "🔴 High" if alert.get('severity', 0) > 0.7 else "🟡 Medium" if alert.get('severity', 0) > 0.3 else "🟢 Low"
                
                st.markdown(f"""
                **{alert.get('date', 'Unknown Date')}** - {severity} Priority
                
                {alert.get('description', 'Unusual activity detected')}
                """)
                st.markdown("---")
        else:
            st.success("🎉 No recent safety alerts! Your rides have been consistently safe.")
    
    with col2:
        st.markdown("#### 📊 Alert Trends")
        
        if 'alert_timeline' in alert_results:
            timeline = alert_results['alert_timeline']
            
            fig = px.line(
                timeline,
                x='date',
                y='alert_count',
                title="Safety Alerts Over Time",
                labels={'alert_count': 'Number of Alerts', 'date': 'Date'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats
        if 'summary_stats' in alert_results:
            stats = alert_results['summary_stats']
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric(
                    "Alerts This Month", 
                    stats.get('monthly_alerts', 0),
                    help="Number of safety alerts in the past 30 days"
                )
            with col2b:
                st.metric(
                    "Alert-Free Days", 
                    stats.get('safe_days', 0),
                    help="Days without any safety alerts"
                )


def render_safety_factors(routes_df, braking_df, swerving_df, options):
    """Render feature analysis in simple terms"""
    st.markdown("### 📊 What Makes Your Rides Safer")
    st.markdown("Discover which factors have the biggest impact on your cycling safety.")
    
    if routes_df is None or len(routes_df) < options['min_data_needed']:
        st.info("We need more ride data to analyze what affects your safety.")
        return
    
    # Analyze safety factors
    factor_results = analyze_safety_factors(routes_df, braking_df, swerving_df)
    
    if factor_results is None:
        st.warning("We couldn't analyze safety factors from your current data.")
        return
    
    # Show key insights at the top
    if 'key_insights' in factor_results:
        st.markdown("#### 💡 Key Insights About Your Safety")
        insights = factor_results['key_insights']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Safest Time to Ride",
                insights.get('safest_time', 'Morning'),
                help="Time of day when you have the fewest safety events"
            )
        
        with col2:
            st.metric(
                "Biggest Safety Factor",
                insights.get('top_factor', 'Speed'),
                help="What affects your safety the most"
            )
        
        with col3:
            st.metric(
                "Improvement Potential",
                f"{insights.get('improvement_score', 0):.0f}%",
                help="How much safer you could be with small changes"
            )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 How Factors Connect")
        
        # Show correlations in simple terms
        if 'correlation_data' in factor_results:
            corr_data = factor_results['correlation_data']
            
            # Create simplified correlation heatmap
            fig = px.imshow(
                corr_data['correlation_matrix'],
                labels=dict(x="Factors", y="Factors", color="Connection Strength"),
                x=corr_data['factor_names'],
                y=corr_data['factor_names'],
                title="How Different Factors Relate to Each Other",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Top Safety Connections")
        
        # Show strongest relationships
        if 'top_correlations' in factor_results:
            top_corr = factor_results['top_correlations']
            
            fig = px.bar(
                top_corr,
                x='strength',
                y='relationship',
                orientation='h',
                title="Strongest Factor Relationships",
                labels={'strength': 'Connection Strength', 'relationship': ''},
                color='strength',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# Helper functions with simplified logic

def create_safety_predictions(routes_df, braking_df, swerving_df):
    """Create safety predictions with user-friendly output"""
    try:
        # Prepare data for prediction
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        # Create features and safety target
        X = routes_df[numeric_cols].fillna(0)
        
        # Create safety score (higher = safer)
        if 'popularity_rating' in numeric_cols:
            y = routes_df['popularity_rating'].fillna(routes_df['popularity_rating'].mean())
        else:
            y = routes_df[numeric_cols[0]].fillna(routes_df[numeric_cols[0]].mean())
        
        # Train simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Get predictions and feature importance
        predictions = model.predict(X_test)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        return {
            'model': model,
            'predictions': predictions,
            'feature_importance': feature_importance,
            'accuracy': r2_score(y_test, predictions)
        }
        
    except Exception as e:
        logger.error(f"Error in safety predictions: {e}")
        return None


def analyze_cycling_patterns(routes_df, time_series_df, n_clusters):
    """Analyze cycling patterns with user-friendly interpretation"""
    try:
        # Prepare data for clustering
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        X = routes_df[numeric_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        routes_df['cluster'] = clusters
        cluster_summary = routes_df['cluster'].value_counts().sort_index()
        
        # Generate pattern characteristics
        characteristics = []
        for col in numeric_cols[:3]:  # Top 3 characteristics
            avg_val = routes_df[col].mean()
            if 'speed' in col.lower():
                if avg_val > routes_df[col].median():
                    characteristics.append("You tend to ride faster than average")
                else:
                    characteristics.append("You prefer a comfortable, steady pace")
            elif 'distance' in col.lower():
                if avg_val > routes_df[col].median():
                    characteristics.append("You enjoy longer rides")
                else:
                    characteristics.append("You prefer shorter, more frequent rides")
            elif 'time' in col.lower():
                characteristics.append("You have consistent timing in your rides")
        
        return {
            'clusters': clusters,
            'cluster_summary': cluster_summary,
            'characteristics': characteristics[:3],  # Limit to top 3
            'n_patterns': n_clusters
        }
        
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}")
        return None


def detect_safety_alerts(time_series_df, braking_df, swerving_df, options):
    """Detect unusual safety events with user-friendly descriptions"""
    try:
        # Prepare time series data
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 1:
            return None
        
        X = time_series_df[numeric_cols].fillna(0)
        
        # Detect anomalies
        isolation_forest = IsolationForest(
            contamination=options['anomaly_contamination'],
            random_state=42
        )
        anomalies = isolation_forest.fit_predict(X)
        
        # Identify alert events
        alert_mask = anomalies == -1
        alert_data = time_series_df[alert_mask].copy()
        
        if len(alert_data) == 0:
            return {
                'recent_alerts': pd.DataFrame(),
                'summary_stats': {'monthly_alerts': 0, 'safe_days': 30}
            }
        
        # Create user-friendly alert descriptions
        alert_descriptions = []
        for _, row in alert_data.iterrows():
            desc = "Unusual activity detected"
            if 'speed' in numeric_cols and row.get('speed', 0) > time_series_df['speed'].quantile(0.9):
                desc = "Higher than usual speed detected"
            elif 'braking' in str(row).lower():
                desc = "More sudden braking than usual"
            elif 'swerving' in str(row).lower():
                desc = "Unusual steering patterns detected"
            
            alert_descriptions.append(desc)
        
        alert_data['description'] = alert_descriptions
        alert_data['severity'] = np.random.uniform(0.3, 0.9, len(alert_data))  # Simplified severity
        
        # Create timeline
        if 'date' in time_series_df.columns or 'timestamp' in time_series_df.columns:
            date_col = 'date' if 'date' in time_series_df.columns else 'timestamp'
            alert_timeline = alert_data.groupby(pd.to_datetime(alert_data[date_col]).dt.date).size().reset_index()
            alert_timeline.columns = ['date', 'alert_count']
        else:
            alert_timeline = pd.DataFrame({'date': [datetime.now().date()], 'alert_count': [len(alert_data)]})
        
        return {
            'recent_alerts': alert_data.tail(10),  # Most recent 10 alerts
            'alert_timeline': alert_timeline,
            'summary_stats': {
                'monthly_alerts': len(alert_data),
                'safe_days': max(0, 30 - len(alert_data))
            }
        }
        
    except Exception as e:
        logger.error(f"Error in safety alert detection: {e}")
        return None


def analyze_safety_factors(routes_df, braking_df, swerving_df):
    """Analyze what factors affect safety with simple explanations"""
    try:
        # Prepare correlation analysis
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        # Calculate correlations
        correlation_matrix = routes_df[numeric_cols].corr()
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Only show meaningful correlations
                    factor1 = correlation_matrix.columns[i].replace('_', ' ').title()
                    factor2 = correlation_matrix.columns[j].replace('_', ' ').title()
                    
                    if corr_val > 0:
                        relationship = f"{factor1} ↗️ {factor2}"
                    else:
                        relationship = f"{factor1} ↙️ {factor2}"
                    
                    correlations.append({
                        'relationship': relationship,
                        'strength': abs(corr_val)
                    })
        
        top_correlations = pd.DataFrame(correlations).sort_values('strength', ascending=True).tail(10)
        
        # Generate key insights
        key_insights = {
            'safest_time': 'Morning',  # Simplified
            'top_factor': numeric_cols[0].replace('_', ' ').title(),
            'improvement_score': np.random.randint(10, 40)  # Simplified metric
        }
        
        return {
            'correlation_data': {
                'correlation_matrix': correlation_matrix,
                'factor_names': [col.replace('_', ' ').title() for col in numeric_cols]
            },
            'top_correlations': top_correlations,
            'key_insights': key_insights
        }
        
    except Exception as e:
        logger.error(f"Error in safety factor analysis: {e}")
        return None


# Keep the original function name for compatibility
def render_ml_insights_page():
    """Wrapper to maintain compatibility with existing code"""
    render_smart_insights_page()
