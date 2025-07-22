"""
ML Insights Page for SeeSense Dashboard - Clean Version
Machine learning-driven safety analysis and predictions without caching issues
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance chart
        if feature_importance is not None:
            fig = px.bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h',
                title="Feature Importance for Risk Prediction",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution
        if predictions is not None:
            fig = px.histogram(
                x=predictions,
                nbins=20,
                title="Predicted Risk Distribution",
                labels={'x': 'Risk Score', 'y': 'Count'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # High-risk routes
    if predictions is not None:
        st.markdown("#### 🚨 High-Risk Routes")
        high_risk_threshold = np.percentile(predictions, 90)
        high_risk_indices = np.where(predictions >= high_risk_threshold)[0]
        
        if len(high_risk_indices) > 0:
            high_risk_routes = routes_df.iloc[high_risk_indices].copy()
            high_risk_routes['predicted_risk'] = predictions[high_risk_indices]
            
            # Display top high-risk routes
            display_cols = ['route_id', 'predicted_risk'] if 'route_id' in high_risk_routes.columns else ['predicted_risk']
            if 'popularity_rating' in high_risk_routes.columns:
                display_cols.append('popularity_rating')
            
            st.dataframe(
                high_risk_routes[display_cols].head(10),
                use_container_width=True
            )
        else:
            st.info("No high-risk routes identified")


def render_behavior_analysis(routes_df, time_series_df, ml_options):
    """Render cyclist behavior clustering analysis"""
    st.markdown("### 👥 Cyclist Behavior Analysis")
    st.markdown("Identify patterns and clusters in cyclist behavior")
    
    if time_series_df is None or len(time_series_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} time series records for behavior analysis")
        return
    
    # Perform clustering analysis
    clusters, cluster_centers, cluster_labels = perform_behavior_clustering(
        time_series_df, ml_options['n_clusters']
    )
    
    if clusters is None:
        st.error("Failed to perform behavior clustering")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster visualization
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                x=time_series_df[numeric_cols[0]],
                y=time_series_df[numeric_cols[1]],
                color=cluster_labels.astype(str),
                title="Behavior Clusters",
                labels={
                    'x': numeric_cols[0],
                    'y': numeric_cols[1],
                    'color': 'Cluster'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster sizes
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Cluster Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.markdown("#### 📈 Cluster Characteristics")
    for i in range(ml_options['n_clusters']):
        cluster_data = time_series_df[cluster_labels == i]
        if len(cluster_data) > 0:
            with st.expander(f"Cluster {i} ({len(cluster_data)} records)"):
                numeric_summary = cluster_data.select_dtypes(include=[np.number]).describe()
                st.dataframe(numeric_summary.transpose())


def render_anomaly_detection(time_series_df, braking_df, swerving_df, ml_options):
    """Render anomaly detection analysis"""
    st.markdown("### 🔍 Anomaly Detection")
    st.markdown("Identify unusual patterns and outliers in cycling data")
    
    if time_series_df is None or len(time_series_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} records for anomaly detection")
        return
    
    # Temporal anomalies
    temporal_anomalies = detect_temporal_anomalies(time_series_df, ml_options['anomaly_contamination'])
    
    # Geographic anomalies
    geographic_anomalies = detect_geographic_anomalies(braking_df, swerving_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏰ Temporal Anomalies")
        if temporal_anomalies is not None and len(temporal_anomalies) > 0:
            # Plot temporal anomalies
            if 'date' in time_series_df.columns:
                time_series_df['date'] = pd.to_datetime(time_series_df['date'])
                time_series_df['is_anomaly'] = temporal_anomalies
                
                numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col_to_plot = numeric_cols[0]
                    
                    fig = px.scatter(
                        time_series_df,
                        x='date',
                        y=col_to_plot,
                        color='is_anomaly',
                        title=f"Temporal Anomalies in {col_to_plot}",
                        color_discrete_map={True: 'red', False: 'blue'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show anomaly count
                    anomaly_count = temporal_anomalies.sum()
                    st.metric("Anomalies Detected", f"{anomaly_count}/{len(temporal_anomalies)}")
            else:
                st.info("No date column found for temporal analysis")
        else:
            st.info("No temporal anomalies detected")
    
    with col2:
        st.markdown("#### 🗺️ Geographic Anomalies")
        if geographic_anomalies and len(geographic_anomalies) > 0:
            # Create DataFrame from geographic anomalies
            anomaly_df = pd.DataFrame(geographic_anomalies)
            
            if 'lat' in anomaly_df.columns and 'lon' in anomaly_df.columns:
                fig = px.scatter_mapbox(
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
                    fig.update_layout(
                        mapbox=dict(center=dict(lat=center_lat, lon=center_lon))
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Geographic Anomalies", len(geographic_anomalies))
        else:
            st.info("No geographic anomalies detected")


def render_feature_analysis(routes_df, braking_df, swerving_df, ml_options):
    """Render feature importance and correlation analysis - FIXED VERSION"""
    st.markdown("### 📊 Feature Analysis")
    st.markdown("Understand which factors most influence cycling safety")
    
    if routes_df is None or len(routes_df) < ml_options['min_samples']:
        st.warning(f"Need at least {ml_options['min_samples']} records for feature analysis")
        return
    
    # Correlation analysis with proper error handling
    correlation_data = perform_correlation_analysis(routes_df, braking_df, swerving_df)
    
    if correlation_data is None or not correlation_data.get('success', False):
        st.error("Unable to perform correlation analysis - insufficient numeric data")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        try:
            correlation_matrix = correlation_data['correlation_matrix']
            fig = px.imshow(
                correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
    
    with col2:
        # Top correlations
        try:
            if 'top_correlations' in correlation_data and not correlation_data['top_correlations'].empty:
                top_corr = correlation_data['top_correlations']
                
                fig = px.bar(
                    top_corr,
                    x='correlation',
                    y='feature_pair',
                    orientation='h',
                    title="Strongest Feature Correlations",
                    labels={'correlation': 'Correlation Coefficient', 'feature_pair': 'Feature Pairs'},
                    color='correlation',
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant correlations found")
        except Exception as e:
            st.error(f"Error creating correlation chart: {e}")


# Helper functions for ML analysis

def train_risk_prediction_model(routes_df, braking_df, swerving_df):
    """Train a risk prediction model"""
    try:
        # Prepare features
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric features for risk prediction")
            return None, None, None
        
        # Create features and target
        X = routes_df[numeric_cols].fillna(0)
        
        # Create synthetic risk target based on available data
        if 'popularity_rating' in numeric_cols:
            # Higher popularity with lower safety features = higher risk
            y = routes_df['popularity_rating'].fillna(routes_df['popularity_rating'].mean())
            y = 1 / (1 + y)  # Inverse relationship
        else:
            # Use first numeric column as proxy target
            y = routes_df[numeric_cols[0]].fillna(routes_df[numeric_cols[0]].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        return model, feature_importance, predictions
        
    except Exception as e:
        logger.error(f"Error training risk prediction model: {e}")
        return None, None, None


def perform_behavior_clustering(time_series_df, n_clusters):
    """Perform K-means clustering on time series data"""
    try:
        # Get numeric columns
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns for clustering")
            return None, None, None
        
        # Prepare data
        X = time_series_df[numeric_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        return X_scaled, kmeans.cluster_centers_, cluster_labels
        
    except Exception as e:
        logger.error(f"Error performing behavior clustering: {e}")
        return None, None, None


def detect_temporal_anomalies(time_series_df, contamination):
    """Detect temporal anomalies using Isolation Forest"""
    try:
        # Get numeric columns
        numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns for anomaly detection")
            return None
        
        # Prepare data
        X = time_series_df[numeric_cols].fillna(0)
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Convert to boolean (True = anomaly)
        return anomalies == -1
        
    except Exception as e:
        logger.error(f"Error detecting temporal anomalies: {e}")
        return None


def detect_geographic_anomalies(braking_df, swerving_df):
    """Detect geographic anomalies in hotspot data"""
    try:
        anomalies = []
        
        # Check braking hotspots with safe comparison
        if braking_df is not None and len(braking_df) > 0:
            if 'severity_score' in braking_df.columns:
                # Safe numeric conversion before quantile and comparison
                severity_numeric = pd.to_numeric(braking_df['severity_score'], errors='coerce')
                severity_clean = severity_numeric.dropna()
                
                if not severity_clean.empty:
                    threshold = severity_clean.quantile(0.9)
                    
                    # Create a temporary dataframe with numeric severity_score
                    temp_braking = braking_df.copy()
                    temp_braking['severity_score'] = severity_numeric
                    temp_braking = temp_braking.dropna(subset=['severity_score'])
                    
                    high_severity = temp_braking[temp_braking['severity_score'] > threshold]
                    
                    for _, hotspot in high_severity.iterrows():
                        anomalies.append({
                            'lat': hotspot['lat'],
                            'lon': hotspot['lon'],
                            'severity': hotspot['severity_score'],
                            'type': 'Braking',
                            'description': f"High severity braking hotspot ({hotspot['severity_score']:.1f})"
                        })
        
        # Check swerving hotspots with safe comparison
        if swerving_df is not None and len(swerving_df) > 0:
            if 'severity_score' in swerving_df.columns:
                # Safe numeric conversion before quantile and comparison
                severity_numeric = pd.to_numeric(swerving_df['severity_score'], errors='coerce')
                severity_clean = severity_numeric.dropna()
                
                if not severity_clean.empty:
                    threshold = severity_clean.quantile(0.9)
                    
                    # Create a temporary dataframe with numeric severity_score
                    temp_swerving = swerving_df.copy()
                    temp_swerving['severity_score'] = severity_numeric
                    temp_swerving = temp_swerving.dropna(subset=['severity_score'])
                    
                    high_severity = temp_swerving[temp_swerving['severity_score'] > threshold]
                    
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
    """Perform correlation analysis on features - FIXED VERSION"""
    try:
        if routes_df is None or len(routes_df) == 0:
            logger.warning("No routes data available for correlation analysis")
            return None
            
        # Prepare correlation data with proper error handling
        numeric_cols = routes_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for correlation analysis")
            return None
            
        # Calculate correlation matrix with proper error handling
        correlation_df = routes_df[numeric_cols].copy()
        correlation_df = correlation_df.dropna()  # Remove rows with NaN values
        
        if len(correlation_df) == 0:
            logger.warning("No valid data remaining after removing NaN values")
            return None
            
        correlation_matrix = correlation_df.corr()
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_val = correlation_matrix.iloc[i, j]
                
                # Skip NaN correlations
                if pd.isna(corr_val):
                    continue
                    
                if abs(corr_val) > 0.1:  # Only include meaningful correlations
                    corr_pairs.append({
                        'feature_pair': f"{col1} vs {col2}",
                        'correlation': corr_val
                    })
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_correlations = pd.DataFrame(corr_pairs[:10])  # Top 10 correlations
        
        # Return the result dictionary with proper structure
        result = {
            'correlation_matrix': correlation_matrix,
            'top_correlations': top_correlations,
            'success': True
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error performing correlation analysis: {e}")
        return None
