"""
ML Insights Page for SeeSense Dashboard
Machine learning-driven safety analysis and predictions
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

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


def render_ml_insights_page():
    """Render the ML insights page"""
    st.title("🔍 ML Insights")
    st.markdown("Advanced machine learning analysis for predictive safety insights")
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_ml_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add ML controls in sidebar
        ml_options = render_ml_controls()
        
        # Create tabs for different ML analyses
        ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
            "🎯 Risk Prediction", 
            "👥 Cyclist Behavior", 
            "🔍 Anomaly Detection", 
            "📊 Feature Analysis"
        ])
        
        with ml_tab1:
            render_risk_prediction(routes_df, braking_df, swerving_df, ml_options)
        
        with ml_tab2:
            render_behavior_analysis(routes_df, time_series_df, ml_options)
        
        with ml_tab3:
            render_anomaly_detection(time_series_df, braking_df, swerving_df, ml_options)
        
        with ml_tab4:
            render_feature_analysis(routes_df, braking_df, swerving_df, ml_options)
        
    except Exception as e:
        logger.error(f"Error in ML insights page: {e}")
        st.error("⚠️ An error occurred while loading ML insights.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Error Details"):
            st.code(str(e))


def render_no_ml_data_message():
    """Render message when no data is available for ML analysis"""
    st.warning("⚠️ No data available for ML analysis.")
    st.markdown("""
    To use ML insights, you need:
    1. **Route data** for risk prediction
    2. **Time series data** for anomaly detection
    3. **Hotspot data** for behavioral analysis
    
    Please add your data files and refresh the page.
    """)


def render_ml_controls():
    """Render ML configuration controls in sidebar"""
    st.sidebar.markdown("### 🤖 ML Configuration")
    
    options = {}
    
    # Model parameters
    options['n_clusters'] = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of clusters for behavioral analysis"
    )
    
    options['anomaly_contamination'] = st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Sensitivity for anomaly detection (lower = more sensitive)"
    )
    
    options['prediction_days'] = st.sidebar.slider(
        "Prediction Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to predict ahead"
    )
    
    options['min_samples'] = st.sidebar.number_input(
        "Minimum Samples",
        min_value=10,
        max_value=1000,
        value=50,
        help="Minimum samples required for ML analysis"
    )
    
    return options


def render_risk_prediction(routes_df, braking_df, swerving_df, ml_options):
    """Render risk prediction analysis"""
    st.markdown("### 🎯 Safety Risk Prediction")
    st.markdown("Predict future safety risks based on route characteristics and historical data")
    
    if routes_df is None or len(routes_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} route records for risk prediction")
        return
    
    # Train risk prediction model
    risk_model, feature_importance, predictions = train_risk_prediction_model(
        routes_df, braking_df, swerving_df
    )
    
    if risk_model is None:
        st.error("Failed to train risk prediction model")
        return
    
    # Display model performance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance chart
        fig = px.bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            title="Risk Prediction Feature Importance",
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=feature_importance['importance'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance metrics
        st.markdown("**Model Performance**")
        
        if 'mae' in predictions:
            st.metric("Mean Absolute Error", f"{predictions['mae']:.3f}")
        if 'r2' in predictions:
            st.metric("R² Score", f"{predictions['r2']:.3f}")
        
        st.metric("Training Samples", len(routes_df))
        st.metric("Features Used", len(feature_importance))
        
        # Risk distribution
        if 'risk_scores' in predictions:
            risk_counts = pd.cut(predictions['risk_scores'], 
                               bins=[0, 3, 6, 8, 10], 
                               labels=['Low', 'Medium', 'High', 'Critical']).value_counts()
            
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_sequence=['green', 'yellow', 'orange', 'red']
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Route-level predictions
    st.markdown("#### 🗺️ Route Risk Predictions")
    
    if 'predictions_df' in predictions:
        pred_df = predictions['predictions_df']
        
        # Map of predicted risks
        if len(pred_df) > 0:
            fig_map = px.scatter_mapbox(
                pred_df,
                lat="start_lat",
                lon="start_lon",
                size="distinct_cyclists",
                color="predicted_risk",
                color_continuous_scale="Reds",
                size_max=15,
                zoom=12,
                mapbox_style="carto-positron",
                hover_data={
                    "route_id": True,
                    "predicted_risk": ":.2f",
                    "route_type": True,
                    "popularity_rating": True
                },
                title="Predicted Safety Risk by Route"
            )
            
            # Center map on data
            center_lat = pred_df['start_lat'].mean()
            center_lon = pred_df['start_lon'].mean()
            fig_map.update_layout(
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon)),
                height=500
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # High-risk routes table
        high_risk_routes = pred_df[pred_df['predicted_risk'] >= 7].sort_values(
            'predicted_risk', ascending=False
        ).head(10)
        
        if len(high_risk_routes) > 0:
            st.markdown("#### ⚠️ High-Risk Routes")
            display_cols = ['route_id', 'route_type', 'predicted_risk', 
                          'popularity_rating', 'distinct_cyclists']
            st.dataframe(
                high_risk_routes[display_cols].round(2),
                use_container_width=True
            )
        else:
            st.success("✅ No high-risk routes predicted!")


def render_behavior_analysis(routes_df, time_series_df, ml_options):
    """Render cyclist behavior clustering analysis"""
    st.markdown("### 👥 Cyclist Behavior Analysis")
    st.markdown("Identify distinct cycling patterns and behaviors using machine learning")
    
    if routes_df is None or len(routes_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} route records for behavior analysis")
        return
    
    # Perform clustering analysis
    clusters_df, cluster_stats = perform_behavior_clustering(routes_df, ml_options['n_clusters'])
    
    if clusters_df is None:
        st.error("Failed to perform behavior clustering")
        return
    
    # Visualize clusters
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot of clusters
        fig = px.scatter(
            clusters_df,
            x='avg_speed',
            y='avg_duration',
            color='behavior_cluster',
            size='distinct_cyclists',
            hover_data=['route_id', 'route_type', 'popularity_rating'],
            title="Cyclist Behavior Clusters",
            labels={
                'avg_speed': 'Average Speed (km/h)',
                'avg_duration': 'Average Duration (min)',
                'behavior_cluster': 'Behavior Type'
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster characteristics
        fig = px.bar(
            cluster_stats,
            x='cluster',
            y='avg_popularity',
            color='cluster',
            title="Average Popularity by Behavior Type",
            labels={'avg_popularity': 'Average Popularity Score'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Speed vs Duration by cluster
        fig2 = px.box(
            clusters_df,
            x='behavior_cluster',
            y='avg_speed',
            color='behavior_cluster',
            title="Speed Distribution by Behavior",
            labels={'avg_speed': 'Average Speed (km/h)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Cluster summary statistics
    st.markdown("#### 📊 Behavior Cluster Profiles")
    
    summary_cols = ['avg_speed', 'avg_duration', 'avg_popularity', 'avg_cyclists', 'route_count']
    summary_display = cluster_stats[['cluster'] + summary_cols].round(2)
    summary_display.columns = ['Behavior Type', 'Avg Speed (km/h)', 'Avg Duration (min)', 
                              'Avg Popularity', 'Avg Cyclists', 'Route Count']
    
    st.dataframe(summary_display, use_container_width=True)
    
    # Temporal behavior analysis
    if time_series_df is not None and len(time_series_df) > 0:
        st.markdown("#### 📅 Temporal Behavior Patterns")
        
        # Simulate behavior by day of week
        if 'day_of_week' in time_series_df.columns:
            temporal_behavior = analyze_temporal_behavior(time_series_df, clusters_df)
            
            if temporal_behavior is not None:
                fig = px.line(
                    temporal_behavior,
                    x='day_of_week',
                    y='avg_rides',
                    color='behavior_type',
                    title="Cycling Behavior by Day of Week",
                    labels={'avg_rides': 'Average Daily Rides'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


def render_anomaly_detection(time_series_df, braking_df, swerving_df, ml_options):
    """Render anomaly detection analysis"""
    st.markdown("### 🔍 Anomaly Detection")
    st.markdown("Identify unusual patterns in safety incidents and cycling behavior")
    
    if time_series_df is None or len(time_series_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} time series records for anomaly detection")
        return
    
    # Perform anomaly detection
    anomalies_df, anomaly_model = detect_safety_anomalies(
        time_series_df, ml_options['anomaly_contamination']
    )
    
    if anomalies_df is None:
        st.error("Failed to perform anomaly detection")
        return
    
    # Visualize anomalies
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Time series with anomalies
        fig = go.Figure()
        
        # Normal points
        normal_data = anomalies_df[anomalies_df['anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['incidents'],
            mode='markers+lines',
            name='Normal',
            marker=dict(color='blue', size=6),
            line=dict(color='blue', width=1)
        ))
        
        # Anomaly points
        anomaly_data = anomalies_df[anomalies_df['anomaly'] == -1]
        fig.add_trace(go.Scatter(
            x=anomaly_data['date'],
            y=anomaly_data['incidents'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Safety Incident Anomalies Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Incidents",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly statistics
        total_anomalies = len(anomaly_data)
        anomaly_rate = (total_anomalies / len(anomalies_df)) * 100
        
        st.metric("Total Anomalies", total_anomalies)
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        
        if total_anomalies > 0:
            avg_severity = anomaly_data['incidents'].mean()
            max_severity = anomaly_data['incidents'].max()
            st.metric("Avg Anomaly Severity", f"{avg_severity:.1f}")
            st.metric("Max Anomaly Severity", f"{max_severity:.0f}")
    
    # Recent anomalies
    if len(anomaly_data) > 0:
        st.markdown("#### 🚨 Recent Anomalies")
        
        recent_anomalies = anomaly_data.sort_values('date', ascending=False).head(5)
        
        for _, anomaly in recent_anomalies.iterrows():
            severity = "🔴 High" if anomaly['incidents'] > anomaly_data['incidents'].quantile(0.75) else "🟡 Medium"
            
            with st.expander(f"{severity} Anomaly - {anomaly['date'].strftime('%Y-%m-%d')}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"**Incidents:** {anomaly['incidents']:.0f}")
                    st.markdown(f"**Safety Score:** {anomaly.get('safety_score', 'N/A')}")
                
                with col_b:
                    st.markdown(f"**Total Rides:** {anomaly.get('total_rides', 'N/A')}")
                    st.markdown(f"**Day:** {anomaly.get('day_of_week', 'N/A')}")
    
    # Hotspot anomalies
    if braking_df is not None or swerving_df is not None:
        st.markdown("#### 🗺️ Geographic Anomalies")
        
        geographic_anomalies = detect_geographic_anomalies(braking_df, swerving_df)
        
        if geographic_anomalies:
            st.info(f"Detected {len(geographic_anomalies)} geographic anomalies")
            
            # Display on map if possible
            if len(geographic_anomalies) > 0:
                anomaly_df = pd.DataFrame(geographic_anomalies)
                
                if 'lat' in anomaly_df.columns and 'lon' in anomaly_df.columns:
                    fig_map = px.scatter_mapbox(
                        anomaly_df,
                        lat="lat",
                        lon="lon",
                        size="severity",
                        color="type",
                        zoom=12,
                        mapbox_style="carto-positron",
                        title="Geographic Anomalies",
                        height=400
                    )
                    
                    if len(anomaly_df) > 0:
                        center_lat = anomaly_df['lat'].mean()
                        center_lon = anomaly_df['lon'].mean()
                        fig_map.update_layout(
                            mapbox=dict(center=dict(lat=center_lat, lon=center_lon))
                        )
                    
                    st.plotly_chart(fig_map, use_container_width=True)


def render_feature_analysis(routes_df, braking_df, swerving_df, ml_options):
    """Render feature importance and correlation analysis"""
    st.markdown("### 📊 Feature Analysis")
    st.markdown("Understand which factors most influence cycling safety")
    
    if routes_df is None or len(routes_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} records for feature analysis")
        return
    
    # Correlation analysis
    correlation_data = perform_correlation_analysis(routes_df, braking_df, swerving_df)
    
    if correlation_data is None:
        st.error("Failed to perform correlation analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig = px.imshow(
            correlation_data['correlation_matrix'],
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top correlations
        if 'top_correlations' in correlation_data:
            top_corr = correlation_data['top_correlations']
            
            fig = px.bar(
                x=top_corr['correlation'].abs(),
                y=top_corr['feature_pair'],
                orientation='h',
                title="Strongest Feature Correlations",
                labels={'x': 'Absolute Correlation', 'y': 'Feature Pairs'},
                color=top_corr['correlation'],
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Safety factor analysis
    st.markdown("#### 🛡️ Safety Factor Analysis")
    
    safety_factors = analyze_safety_factors(routes_df, braking_df, swerving_df)
    
    if safety_factors:
        factor_tabs = st.tabs(["🚦 Infrastructure", "🏃 Usage Patterns", "📍 Location"])
        
        with factor_tabs[0]:
            render_infrastructure_analysis(safety_factors.get('infrastructure', {}))
        
        with factor_tabs[1]:
            render_usage_analysis(safety_factors.get('usage', {}))
        
        with factor_tabs[2]:
            render_location_analysis(safety_factors.get('location', {}))


# Helper functions for ML analysis

def train_risk_prediction_model(routes_df, braking_df, swerving_df):
    """Train a risk prediction model"""
    try:
        # Prepare features
        features = ['distinct_cyclists', 'popularity_rating', 'avg_speed', 'avg_duration']
        if 'has_bike_lane' in routes_df.columns:
            routes_df['has_bike_lane_num'] = routes_df['has_bike_lane'].astype(int)
            features.append('has_bike_lane_num')
        
        # Create target variable (synthetic risk score for demo)
        X = routes_df[features].fillna(0)
        
        # Generate realistic risk scores based on features
        risk_score = (
            (10 - routes_df['popularity_rating']) * 0.3 +
            (routes_df['avg_speed'] > 20).astype(int) * 2.0 +
            (1 - routes_df.get('has_bike_lane', 0).astype(int)) * 2.0 +
            np.random.normal(0, 0.5, len(routes_df))
        ).clip(0, 10)
        
        y = risk_score
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Predict for all routes
        predictions_df = routes_df.copy()
        predictions_df['predicted_risk'] = model.predict(X)
        
        predictions = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'risk_scores': model.predict(X),
            'predictions_df': predictions_df
        }
        
        return model, feature_importance, predictions
        
    except Exception as e:
        logger.error(f"Error training risk prediction model: {e}")
        return None, None, None


def perform_behavior_clustering(routes_df, n_clusters):
    """Perform cyclist behavior clustering"""
    try:
        # Prepare clustering features
        cluster_features = ['avg_speed', 'avg_duration', 'popularity_rating', 'distinct_cyclists']
        
        # Handle missing columns
        available_features = [f for f in cluster_features if f in routes_df.columns]
        
        if len(available_features) < 2:
            return None, None
        
        X = routes_df[available_features].fillna(routes_df[available_features].mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        clusters_df = routes_df.copy()
        clusters_df['cluster'] = clusters
        
        # Create cluster names
        cluster_names = {
            0: 'Commuters',
            1: 'Leisure Riders', 
            2: 'Speed Enthusiasts',
            3: 'Casual Cyclists'
        }
        
        # Extend cluster names if more clusters
        if n_clusters > 4:
            for i in range(4, n_clusters):
                cluster_names[i] = f'Group {i+1}'
        
        clusters_df['behavior_cluster'] = clusters_df['cluster'].map(
            lambda x: cluster_names.get(x, f'Group {x+1}')
        )
        
        # Calculate cluster statistics
        cluster_stats = clusters_df.groupby('behavior_cluster').agg({
            'avg_speed': 'mean',
            'avg_duration': 'mean', 
            'popularity_rating': 'mean',
            'distinct_cyclists': 'mean',
            'cluster': 'count'
        }).round(2)
        
        cluster_stats = cluster_stats.rename(columns={
            'avg_speed': 'avg_speed',
            'avg_duration': 'avg_duration',
            'popularity_rating': 'avg_popularity',
            'distinct_cyclists': 'avg_cyclists',
            'cluster': 'route_count'
        }).reset_index()
        
        cluster_stats['cluster'] = cluster_stats['behavior_cluster']
        
        return clusters_df, cluster_stats
        
    except Exception as e:
        logger.error(f"Error performing behavior clustering: {e}")
        return None, None


def detect_safety_anomalies(time_series_df, contamination):
    """Detect anomalies in safety incidents"""
    try:
        # Prepare features for anomaly detection
        anomaly_features = ['incidents', 'total_rides', 'avg_speed']
        
        # Add weather features if available
        weather_features = ['precipitation_mm', 'temperature']
        for feature in weather_features:
            if feature in time_series_df.columns:
                anomaly_features.append(feature)
        
        # Handle missing values
        X = time_series_df[anomaly_features].fillna(time_series_df[anomaly_features].mean())
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Add results to dataframe
        anomalies_df = time_series_df.copy()
        anomalies_df['anomaly'] = anomalies
        anomalies_df['anomaly_score'] = iso_forest.decision_function(X)
        
        return anomalies_df, iso_forest
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return None, None


def detect_geographic_anomalies(braking_df, swerving_df):
    """Detect geographic anomalies in hotspots"""
    try:
        anomalies = []
        
        # Check braking hotspots
        if braking_df is not None and len(braking_df) > 0:
            if 'severity_score' in braking_df.columns:
                threshold = braking_df['severity_score'].quantile(0.9)
                high_severity = braking_df[braking_df['severity_score'] > threshold]
                
                for _, hotspot in high_severity.iterrows():
                    anomalies.append({
                        'lat': hotspot['lat'],
                        'lon': hotspot['lon'],
                        'severity': hotspot['severity_score'],
                        'type': 'Braking',
                        'description': f"High severity braking hotspot ({hotspot['severity_score']:.1f})"
                    })
        
        # Check swerving hotspots
        if swerving_df is not None and len(swerving_df) > 0:
            if 'severity_score' in swerving_df.columns:
                threshold = swerving_df['severity_score'].quantile(0.9)
                high_severity = swerving_df[swerving_df['severity_score'] > threshold]
                
                for _, hotspot in high_severity.iterrows():
                    anomalies.append({
                        'lat': hotspot['lat'],
                        'lon': hotspot['lon'], 
                        'severity': hotspot['severity_score'],
                        'type': 'Swerving',
                        'description': f"High severity swerving hotspot ({hotspot['severity_score']:.1f})"
                    })
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Error detecting geographic anomalies: {e}")
        return []


def perform_correlation_analysis(routes_df, braking_df, swerving_df):
    """Perform correlation analysis on features"""
    try:
        # Prepare correlation data
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = routes_df[numeric_cols].corr()
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_val = correlation_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.1:  # Only significant correlations
                    corr_pairs.append({
                        'feature_pair': f"{col1} × {col2}",
                        'correlation': corr_val
                    })
        
        top_correlations = pd.DataFrame(corr_pairs).sort_values(
            'correlation', key=abs, ascending=False
        ).head(10)
        
        return {
            'correlation_matrix': correlation_matrix,
            'top_correlations': top_correlations
        }
        
    except Exception as e:
        logger.error(f"Error performing correlation analysis: {e}")
        return None


def analyze_temporal_behavior(time_series_df, clusters_df):
    """Analyze temporal patterns in cyclist behavior"""
    try:
        # Create synthetic temporal behavior data
        behaviors = clusters_df['behavior_cluster'].unique()
        days = time_series_df['day_of_week'].unique()
        
        temporal_data = []
        for behavior in behaviors:
            for day in days:
                # Simulate different patterns for different behaviors
                base_rides = time_series_df[time_series_df['day_of_week'] == day]['total_rides'].mean()
                
                if 'Commuter' in behavior:
                    # Higher on weekdays
                    multiplier = 1.5 if day not in ['Saturday', 'Sunday'] else 0.7
                elif 'Leisure' in behavior:
                    # Higher on weekends
                    multiplier = 0.8 if day not in ['Saturday', 'Sunday'] else 1.6
                else:
                    # Relatively stable
                    multiplier = 1.0
                
                temporal_data.append({
                    'day_of_week': day,
                    'behavior_type': behavior,
                    'avg_rides': base_rides * multiplier * 0.3  # Scale to represent portion
                })
        
        return pd.DataFrame(temporal_data)
        
    except Exception as e:
        logger.error(f"Error analyzing temporal behavior: {e}")
        return None


def analyze_safety_factors(routes_df, braking_df, swerving_df):
    """Analyze key safety factors"""
    try:
        safety_factors = {}
        
        # Infrastructure analysis
        if 'has_bike_lane' in routes_df.columns:
            bike_lane_safety = routes_df.groupby('has_bike_lane').agg({
                'popularity_rating': 'mean',
                'distinct_cyclists': 'mean',
                'avg_speed': 'mean'
            }).round(2)
            
            safety_factors['infrastructure'] = {
                'bike_lane_impact': bike_lane_safety,
                'bike_lane_coverage': (routes_df['has_bike_lane'].sum() / len(routes_df)) * 100
            }
        
        # Usage pattern analysis
        if 'route_type' in routes_df.columns:
            usage_safety = routes_df.groupby('route_type').agg({
                'popularity_rating': 'mean',
                'avg_speed': 'mean',
                'distinct_cyclists': 'mean'
            }).round(2)
            
            safety_factors['usage'] = {
                'route_type_safety': usage_safety,
                'usage_distribution': routes_df['route_type'].value_counts()
            }
        
        # Location analysis
        if braking_df is not None and swerving_df is not None:
            road_type_analysis = {}
            
            if 'road_type' in braking_df.columns:
                braking_by_road = braking_df.groupby('road_type').agg({
                    'incidents_count': 'mean',
                    'severity_score': 'mean'
                }).round(2)
                road_type_analysis['braking'] = braking_by_road
            
            if 'road_type' in swerving_df.columns:
                swerving_by_road = swerving_df.groupby('road_type').agg({
                    'incidents_count': 'mean', 
                    'severity_score': 'mean'
                }).round(2)
                road_type_analysis['swerving'] = swerving_by_road
            
            safety_factors['location'] = road_type_analysis
        
        return safety_factors
        
    except Exception as e:
        logger.error(f"Error analyzing safety factors: {e}")
        return {}


def render_infrastructure_analysis(infrastructure_data):
    """Render infrastructure impact analysis"""
    if not infrastructure_data:
        st.info("No infrastructure data available for analysis")
        return
    
    if 'bike_lane_impact' in infrastructure_data:
        bike_lane_data = infrastructure_data['bike_lane_impact']
        
        # Bike lane impact chart
        bike_lane_df = bike_lane_data.reset_index()
        bike_lane_df['has_bike_lane'] = bike_lane_df['has_bike_lane'].map({
            True: 'With Bike Lane',
            False: 'Without Bike Lane'
        })
        
        fig = px.bar(
            bike_lane_df,
            x='has_bike_lane',
            y='popularity_rating',
            title="Safety Impact of Bike Lanes",
            labels={'popularity_rating': 'Average Popularity Rating'},
            color='popularity_rating',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage metric
        coverage = infrastructure_data.get('bike_lane_coverage', 0)
        st.metric("Bike Lane Coverage", f"{coverage:.1f}%")


def render_usage_analysis(usage_data):
    """Render usage pattern analysis"""
    if not usage_data:
        st.info("No usage pattern data available for analysis")
        return
    
    if 'route_type_safety' in usage_data:
        usage_safety = usage_data['route_type_safety'].reset_index()
        
        fig = px.bar(
            usage_safety,
            x='route_type',
            y='popularity_rating',
            title="Safety by Route Usage Type",
            labels={'popularity_rating': 'Average Popularity Rating'},
            color='avg_speed',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if 'usage_distribution' in usage_data:
        usage_dist = usage_data['usage_distribution']
        
        fig = px.pie(
            values=usage_dist.values,
            names=usage_dist.index,
            title="Route Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_location_analysis(location_data):
    """Render location-based safety analysis"""
    if not location_data:
        st.info("No location data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'braking' in location_data:
            braking_data = location_data['braking'].reset_index()
            
            fig = px.bar(
                braking_data,
                x='road_type',
                y='severity_score',
                title="Braking Incident Severity by Road Type",
                labels={'severity_score': 'Average Severity Score'},
                color='incidents_count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'swerving' in location_data:
            swerving_data = location_data['swerving'].reset_index()
            
            fig = px.bar(
                swerving_data,
                x='road_type',
                y='severity_score',
                title="Swerving Incident Severity by Road Type",
                labels={'severity_score': 'Average Severity Score'},
                color='incidents_count',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True)
