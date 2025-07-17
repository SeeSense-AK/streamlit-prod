"""
Advanced Analytics Page for SeeSense Dashboard
Time series analysis, correlation analysis, anomaly detection, and predictive modeling
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

# Advanced analytics imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_advanced_analytics_page():
    """Render the advanced analytics page"""
    st.title("📈 Advanced Analytics")
    st.markdown("Deep statistical analysis and predictive modeling for cycling safety")
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_advanced_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add analytics controls in sidebar
        analytics_options = render_analytics_controls()
        
        # Create tabs for different advanced analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Time Series Analysis",
            "🔗 Correlation Analysis", 
            "🔍 Anomaly Detection",
            "📈 Predictive Modeling",
            "🎯 Statistical Testing"
        ])
        
        with tab1:
            render_time_series_analysis(time_series_df, analytics_options)
        
        with tab2:
            render_correlation_analysis(routes_df, braking_df, swerving_df, time_series_df, analytics_options)
        
        with tab3:
            render_anomaly_detection(time_series_df, braking_df, swerving_df, analytics_options)
        
        with tab4:
            render_predictive_modeling(time_series_df, routes_df, analytics_options)
        
        with tab5:
            render_statistical_testing(time_series_df, braking_df, swerving_df, analytics_options)
        
    except Exception as e:
        logger.error(f"Error in advanced analytics page: {e}")
        st.error("⚠️ An error occurred while loading advanced analytics.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Error Details"):
            st.code(str(e))


def render_no_advanced_data_message():
    """Render message when no data is available for advanced analytics"""
    st.warning("⚠️ No data available for advanced analytics.")
    st.markdown("""
    To use advanced analytics, you need:
    1. **Time series data** for temporal analysis
    2. **Sufficient data points** for statistical significance
    3. **Numeric features** for correlation analysis
    
    Please add your data files and refresh the page.
    """)


def render_analytics_controls():
    """Render advanced analytics controls in sidebar"""
    st.sidebar.markdown("### 📈 Advanced Analytics Settings")
    
    options = {}
    
    # Time series settings
    st.sidebar.markdown("**Time Series Analysis**")
    options['decomposition_period'] = st.sidebar.slider(
        "Seasonal Period (days)",
        min_value=7,
        max_value=365,
        value=7,
        help="Period for seasonal decomposition"
    )
    
    options['forecast_periods'] = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast"
    )
    
    # Anomaly detection settings
    st.sidebar.markdown("**Anomaly Detection**")
    options['anomaly_contamination'] = st.sidebar.slider(
        "Anomaly Contamination",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalies"
    )
    
    # Correlation settings
    st.sidebar.markdown("**Correlation Analysis**")
    options['correlation_method'] = st.sidebar.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"],
        help="Method for correlation calculation"
    )
    
    options['min_correlation'] = st.sidebar.slider(
        "Minimum Correlation Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        help="Minimum correlation to highlight"
    )
    
    # Statistical testing settings
    st.sidebar.markdown("**Statistical Testing**")
    options['confidence_level'] = st.sidebar.slider(
        "Confidence Level",
        min_value=0.9,
        max_value=0.99,
        value=0.95,
        help="Confidence level for tests"
    )
    
    return options


def render_time_series_analysis(time_series_df, analytics_options):
    """Render time series analysis with decomposition and forecasting"""
    st.markdown("### 📊 Time Series Analysis")
    st.markdown("Analyze temporal patterns, seasonality, and trends in safety data")
    
    if time_series_df is None or len(time_series_df) < 14:
        st.warning("Need at least 14 time series records for meaningful analysis")
        return
    
    if not STATSMODELS_AVAILABLE:
        st.error("statsmodels library is required for time series analysis")
        return
    
    # Prepare time series data
    ts_data = prepare_time_series_data(time_series_df)
    
    if ts_data is None:
        st.error("Failed to prepare time series data")
        return
    
    # Metric selection
    numeric_columns = ts_data.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_metric = st.selectbox(
            "Select Metric to Analyze",
            options=numeric_columns,
            index=0 if 'incidents' in numeric_columns else 0
        )
    
    with col1:
        st.markdown(f"**Analyzing: {selected_metric.replace('_', ' ').title()}**")
    
    # Time series decomposition
    if len(ts_data) >= analytics_options['decomposition_period'] * 2:
        st.markdown("#### 🔄 Seasonal Decomposition")
        
        decomposition_fig = create_time_series_decomposition(
            ts_data, selected_metric, analytics_options['decomposition_period']
        )
        st.plotly_chart(decomposition_fig, use_container_width=True)
    
    # Trend analysis
    st.markdown("#### 📈 Trend Analysis")
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Moving averages
        ma_fig = create_moving_averages_chart(ts_data, selected_metric)
        st.plotly_chart(ma_fig, use_container_width=True)
    
    with trend_col2:
        # Daily/weekly patterns
        pattern_fig = create_temporal_patterns_chart(ts_data, selected_metric)
        st.plotly_chart(pattern_fig, use_container_width=True)
    
    # Forecasting
    st.markdown("#### 🔮 Forecasting")
    forecast_fig, forecast_metrics = create_forecast_chart(
        ts_data, selected_metric, analytics_options['forecast_periods']
    )
    
    if forecast_fig is not None:
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Show forecast metrics
        if forecast_metrics:
            metric_cols = st.columns(len(forecast_metrics))
            for i, (metric_name, metric_value) in enumerate(forecast_metrics.items()):
                with metric_cols[i]:
                    st.metric(metric_name, f"{metric_value:.3f}")


def render_correlation_analysis(routes_df, braking_df, swerving_df, time_series_df, analytics_options):
    """Render correlation analysis and feature relationships"""
    st.markdown("### 🔗 Correlation Analysis")
    st.markdown("Explore relationships between different safety metrics and features")
    
    # Combine datasets for correlation analysis
    combined_data = prepare_correlation_data(routes_df, braking_df, swerving_df, time_series_df)
    
    if combined_data is None or len(combined_data) < 10:
        st.warning("Insufficient data for correlation analysis")
        return
    
    # Feature correlation heatmap
    st.markdown("#### 🌡️ Feature Correlation Matrix")
    
    correlation_fig = create_correlation_heatmap(
        combined_data, analytics_options['correlation_method']
    )
    st.plotly_chart(correlation_fig, use_container_width=True)
    
    # Significant correlations
    st.markdown("#### 📊 Significant Correlations")
    
    significant_correlations = find_significant_correlations(
        combined_data, analytics_options['min_correlation'], analytics_options['correlation_method']
    )
    
    if not significant_correlations.empty:
        st.dataframe(significant_correlations, use_container_width=True)
    else:
        st.info("No significant correlations found above the threshold")
    
    # Scatter plot matrix for top correlations
    if not significant_correlations.empty:
        st.markdown("#### 🔍 Relationship Visualization")
        
        # Select top correlations for visualization
        top_correlations = significant_correlations.head(6)
        
        scatter_cols = st.columns(2)
        
        for i, (_, row) in enumerate(top_correlations.iterrows()):
            col_idx = i % 2
            with scatter_cols[col_idx]:
                scatter_fig = create_correlation_scatter(
                    combined_data, row['Feature 1'], row['Feature 2'], row['Correlation']
                )
                st.plotly_chart(scatter_fig, use_container_width=True)


def render_anomaly_detection(time_series_df, braking_df, swerving_df, analytics_options):
    """Render anomaly detection analysis"""
    st.markdown("### 🔍 Anomaly Detection")
    st.markdown("Identify unusual patterns in safety incidents and cycling behavior")
    
    # Time series anomaly detection
    if time_series_df is not None and len(time_series_df) > 20:
        st.markdown("#### 📅 Time Series Anomalies")
        
        ts_anomalies = detect_time_series_anomalies(
            time_series_df, analytics_options['anomaly_contamination']
        )
        
        if ts_anomalies is not None:
            anomaly_fig = create_anomaly_visualization(ts_anomalies)
            st.plotly_chart(anomaly_fig, use_container_width=True)
            
            # Show anomaly details
            anomaly_details = ts_anomalies[ts_anomalies['is_anomaly'] == True]
            
            if not anomaly_details.empty:
                st.markdown("#### 🚨 Detected Anomalies")
                
                anomaly_display = anomaly_details[['date', 'incidents', 'anomaly_score']].copy()
                anomaly_display['anomaly_score'] = anomaly_display['anomaly_score'].round(3)
                anomaly_display.columns = ['Date', 'Incident Count', 'Anomaly Score']
                
                st.dataframe(anomaly_display, use_container_width=True)
    
    # Spatial anomaly detection
    st.markdown("#### 🗺️ Spatial Anomalies")
    
    if braking_df is not None and swerving_df is not None:
        spatial_anomalies = detect_spatial_anomalies(
            braking_df, swerving_df, analytics_options['anomaly_contamination']
        )
        
        if spatial_anomalies is not None:
            spatial_fig = create_spatial_anomaly_map(spatial_anomalies)
            st.plotly_chart(spatial_fig, use_container_width=True)


def render_predictive_modeling(time_series_df, routes_df, analytics_options):
    """Render predictive modeling analysis"""
    st.markdown("### 📈 Predictive Modeling")
    st.markdown("Machine learning models for safety prediction and risk assessment")
    
    if time_series_df is None or len(time_series_df) < 30:
        st.warning("Need at least 30 time series records for predictive modeling")
        return
    
    # Prepare features for modeling
    modeling_data = prepare_modeling_data(time_series_df, routes_df)
    
    if modeling_data is None:
        st.error("Failed to prepare modeling data")
        return
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Linear Regression", "Random Forest", "Time Series Forecast"],
        help="Choose the type of predictive model"
    )
    
    if model_type == "Linear Regression":
        render_linear_regression_analysis(modeling_data)
    
    elif model_type == "Random Forest":
        render_random_forest_analysis(modeling_data)
    
    elif model_type == "Time Series Forecast":
        render_time_series_forecast(time_series_df, analytics_options)


def render_statistical_testing(time_series_df, braking_df, swerving_df, analytics_options):
    """Render statistical testing analysis"""
    st.markdown("### 🎯 Statistical Testing")
    st.markdown("Hypothesis testing and statistical significance analysis")
    
    if time_series_df is None or len(time_series_df) < 20:
        st.warning("Need at least 20 records for statistical testing")
        return
    
    # Prepare data for testing
    test_data = prepare_statistical_test_data(time_series_df, braking_df, swerving_df)
    
    if test_data is None:
        st.error("Failed to prepare data for statistical testing")
        return
    
    # Test selection
    test_type = st.selectbox(
        "Select Statistical Test",
        ["T-Test", "ANOVA", "Chi-Square", "Kolmogorov-Smirnov"],
        help="Choose the statistical test to perform"
    )
    
    if test_type == "T-Test":
        render_t_test_analysis(test_data, analytics_options)
    
    elif test_type == "ANOVA":
        render_anova_analysis(test_data, analytics_options)
    
    elif test_type == "Chi-Square":
        render_chi_square_analysis(test_data, analytics_options)
    
    elif test_type == "Kolmogorov-Smirnov":
        render_ks_test_analysis(test_data, analytics_options)


# Helper functions for data preparation and analysis
def prepare_time_series_data(time_series_df):
    """Prepare time series data for analysis"""
    try:
        df = time_series_df.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
        else:
            return None
        
        # Fill missing values
        df = df.ffill().bfill()
        
        return df
    
    except Exception as e:
        logger.error(f"Error preparing time series data: {e}")
        return None


def create_time_series_decomposition(data, column, period):
    """Create time series decomposition visualization"""
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data[column], model='additive', period=period)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Add traces
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Observed', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, mode='lines', name='Residual', line=dict(color='orange')), row=4, col=1)
        
        fig.update_layout(
            height=800,
            title_text=f"Time Series Decomposition: {column.replace('_', ' ').title()}",
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating time series decomposition: {e}")
        return None


def create_moving_averages_chart(data, column):
    """Create moving averages chart"""
    try:
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name='Original',
            line=dict(color='lightblue', width=1)
        ))
        
        # 7-day moving average (only if we have enough data)
        if len(data) >= 7:
            ma_7 = data[column].rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma_7,
                mode='lines',
                name='7-day MA',
                line=dict(color='blue', width=2)
            ))
        
        # 30-day moving average (only if we have enough data)
        if len(data) >= 30:
            ma_30 = data[column].rolling(window=30, center=True).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma_30,
                mode='lines',
                name='30-day MA',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title=f"Moving Averages: {column.replace('_', ' ').title()}",
            xaxis_title="Date",
            yaxis_title=column.replace('_', ' ').title(),
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating moving averages chart: {e}")
        return None


def create_temporal_patterns_chart(data, column):
    """Create temporal patterns chart (day of week, hour of day)"""
    try:
        # Day of week pattern
        data_reset = data.reset_index()
        data_reset['day_of_week'] = data_reset['date'].dt.day_name()
        
        # Day of week aggregation
        dow_pattern = data_reset.groupby('day_of_week')[column].mean().reset_index()
        
        if dow_pattern.empty:
            return None
        
        # Ensure correct order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_pattern['day_of_week'] = pd.Categorical(
            dow_pattern['day_of_week'], 
            categories=day_order, 
            ordered=True
        )
        dow_pattern = dow_pattern.sort_values('day_of_week').dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dow_pattern['day_of_week'].astype(str),
            y=dow_pattern[column],
            name='Day of Week Pattern',
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title=f"Day of Week Pattern: {column.replace('_', ' ').title()}",
            xaxis_title="Day of Week",
            yaxis_title=f"Average {column.replace('_', ' ').title()}",
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating temporal patterns chart: {e}")
        return None


def create_correlation_heatmap(data, method):
    """Create correlation heatmap"""
    try:
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return None
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = numeric_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman')
        else:
            corr_matrix = numeric_data.corr(method='kendall')
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title=f"Feature Correlation Matrix ({method.title()})"
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return None


def detect_time_series_anomalies(time_series_df, contamination):
    """Detect anomalies in time series data"""
    try:
        df = time_series_df.copy()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Select numeric columns for anomaly detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Use 'incidents' column if available, otherwise first numeric column
        target_col = 'incidents' if 'incidents' in numeric_cols else numeric_cols[0]
        
        # Prepare data for isolation forest
        X = df[target_col].values.reshape(-1, 1)
        
        # Fit isolation forest
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        df['is_anomaly'] = isolation_forest.fit_predict(X) == -1
        df['anomaly_score'] = isolation_forest.score_samples(X)
        
        return df
    
    except Exception as e:
        logger.error(f"Error detecting time series anomalies: {e}")
        return None


def create_anomaly_visualization(anomaly_data):
    """Create anomaly visualization"""
    try:
        fig = go.Figure()
        
        # Normal points
        normal_data = anomaly_data[~anomaly_data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['incidents'] if 'incidents' in normal_data.columns else normal_data.iloc[:, 1],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6)
        ))
        
        # Anomaly points
        anomaly_points = anomaly_data[anomaly_data['is_anomaly']]
        if not anomaly_points.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_points['date'],
                y=anomaly_points['incidents'] if 'incidents' in anomaly_points.columns else anomaly_points.iloc[:, 1],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="Time Series Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating anomaly visualization: {e}")
        return None


# Additional helper functions would continue here...
# (Due to length constraints, I'm showing the main structure)

def prepare_correlation_data(routes_df, braking_df, swerving_df, time_series_df):
    """Prepare combined data for correlation analysis"""
    try:
        combined_data = pd.DataFrame()
        
        # Add time series data
        if time_series_df is not None:
            ts_numeric = time_series_df.select_dtypes(include=[np.number])
            combined_data = pd.concat([combined_data, ts_numeric], axis=1)
        
        # Add route aggregations
        if routes_df is not None:
            route_agg = routes_df.select_dtypes(include=[np.number]).mean().to_frame().T
            route_agg.columns = [f'route_{col}' for col in route_agg.columns]
            combined_data = pd.concat([combined_data, route_agg], axis=1)
        
        # Add hotspot aggregations
        if braking_df is not None:
            braking_agg = braking_df.select_dtypes(include=[np.number]).mean().to_frame().T
            braking_agg.columns = [f'braking_{col}' for col in braking_agg.columns]
            combined_data = pd.concat([combined_data, braking_agg], axis=1)
        
        if swerving_df is not None:
            swerving_agg = swerving_df.select_dtypes(include=[np.number]).mean().to_frame().T
            swerving_agg.columns = [f'swerving_{col}' for col in swerving_agg.columns]
            combined_data = pd.concat([combined_data, swerving_agg], axis=1)
        
        # Remove columns with all NaN values
        combined_data = combined_data.dropna(axis=1, how='all')
        
        return combined_data if not combined_data.empty else None
    
    except Exception as e:
        logger.error(f"Error preparing correlation data: {e}")
        return None


def find_significant_correlations(data, min_correlation, method):
    """Find significant correlations above threshold"""
    try:
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Extract significant correlations
        significant_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= min_correlation:
                    significant_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_value,
                        'Abs Correlation': abs(corr_value)
                    })
        
        # Create DataFrame and sort by absolute correlation
        significant_df = pd.DataFrame(significant_pairs)
        
        if not significant_df.empty:
            significant_df = significant_df.sort_values('Abs Correlation', ascending=False)
            significant_df = significant_df.drop('Abs Correlation', axis=1)
            significant_df['Correlation'] = significant_df['Correlation'].round(3)
        
        return significant_df
    
    except Exception as e:
        logger.error(f"Error finding significant correlations: {e}")
        return pd.DataFrame()


    except Exception as e:
        logger.error(f"Error creating forecast chart: {e}")
        return None, None


def create_forecast_chart(data, column, periods):
    """Create forecast chart using simple exponential smoothing"""
    try:
        # Simple exponential smoothing forecast
        if len(data) < periods:
            return None, None
        
        # Prepare data
        ts_data = data[column].dropna()
        
        if len(ts_data) < 10:
            return None, None
        
        # Split data for training and testing
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]
        
def create_forecast_chart(data, column, periods):
    """Create forecast chart using simple exponential smoothing"""
    try:
        # Prepare data
        ts_data = data[column].dropna()
        
        if len(ts_data) < 10:
            return None, None
        
        # Simple exponential smoothing forecast
        alpha = 0.3
        forecast_values = []
        
        # Use the last actual value as starting point
        last_value = ts_data.iloc[-1]
        
        # Generate forecast values
        for i in range(periods):
            if i == 0:
                # First forecast point
                forecast_val = alpha * last_value + (1 - alpha) * last_value
            else:
                # Subsequent forecast points
                forecast_val = alpha * forecast_values[-1] + (1 - alpha) * forecast_values[-1]
            
            forecast_values.append(forecast_val)
        
        # Create forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Forecast: {column.replace('_', ' ').title()}",
            xaxis_title="Date",
            yaxis_title=column.replace('_', ' ').title(),
            height=400
        )
        
        # Calculate simple forecast metrics
        forecast_metrics = {
            'Forecast Trend': 'Increasing' if forecast_values[-1] > forecast_values[0] else 'Decreasing',
            'Forecast Range': f"{min(forecast_values):.2f} - {max(forecast_values):.2f}",
            'Last Value': f"{last_value:.2f}"
        }
        
        return fig, forecast_metrics
    
    except Exception as e:
        logger.error(f"Error creating forecast chart: {e}")
        return None, None
    
    except Exception as e:
        logger.error(f"Error creating forecast chart: {e}")
        return None, None


def create_correlation_scatter(data, feature1, feature2, correlation):
    """Create scatter plot for correlation analysis"""
    try:
        fig = px.scatter(
            data, 
            x=feature1, 
            y=feature2,
            title=f"Correlation: {feature1} vs {feature2} (r={correlation:.3f})",
            trendline="ols"
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation scatter: {e}")
        return None


def detect_spatial_anomalies(braking_df, swerving_df, contamination):
    """Detect spatial anomalies in hotspot data"""
    try:
        # Combine braking and swerving data
        combined_spatial = pd.DataFrame()
        
        if braking_df is not None:
            braking_spatial = braking_df[['lat', 'lon']].copy()
            braking_spatial['type'] = 'braking'
            braking_spatial['intensity'] = braking_df.get('intensity', 1)
            combined_spatial = pd.concat([combined_spatial, braking_spatial])
        
        if swerving_df is not None:
            swerving_spatial = swerving_df[['lat', 'lon']].copy()
            swerving_spatial['type'] = 'swerving'
            swerving_spatial['intensity'] = swerving_df.get('intensity', 1)
            combined_spatial = pd.concat([combined_spatial, swerving_spatial])
        
        if combined_spatial.empty:
            return None
        
        # Prepare features for anomaly detection
        X = combined_spatial[['lat', 'lon', 'intensity']].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply isolation forest
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        combined_spatial['is_anomaly'] = isolation_forest.fit_predict(X_scaled) == -1
        combined_spatial['anomaly_score'] = isolation_forest.score_samples(X_scaled)
        
        return combined_spatial
    
    except Exception as e:
        logger.error(f"Error detecting spatial anomalies: {e}")
        return None


def create_spatial_anomaly_map(spatial_anomalies):
    """Create spatial anomaly map"""
    try:
        fig = go.Figure()
        
        # Normal points
        normal_points = spatial_anomalies[~spatial_anomalies['is_anomaly']]
        fig.add_trace(go.Scattermapbox(
            lat=normal_points['lat'],
            lon=normal_points['lon'],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Normal',
            text=normal_points['type']
        ))
        
        # Anomaly points
        anomaly_points = spatial_anomalies[spatial_anomalies['is_anomaly']]
        if not anomaly_points.empty:
            fig.add_trace(go.Scattermapbox(
                lat=anomaly_points['lat'],
                lon=anomaly_points['lon'],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name='Anomalies',
                text=anomaly_points['type']
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=spatial_anomalies['lat'].mean(), lon=spatial_anomalies['lon'].mean()),
                zoom=12
            ),
            title="Spatial Anomaly Detection",
            height=500
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating spatial anomaly map: {e}")
        return None


def prepare_modeling_data(time_series_df, routes_df):
    """Prepare data for predictive modeling"""
    try:
        # Use time series data as base
        modeling_data = time_series_df.copy()
        
        # Add date features
        if 'date' in modeling_data.columns:
            modeling_data['date'] = pd.to_datetime(modeling_data['date'])
            modeling_data['day_of_week'] = modeling_data['date'].dt.dayofweek
            modeling_data['month'] = modeling_data['date'].dt.month
            modeling_data['day_of_year'] = modeling_data['date'].dt.dayofyear
        
        # Add lagged features
        numeric_cols = modeling_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['day_of_week', 'month', 'day_of_year']:
                modeling_data[f'{col}_lag1'] = modeling_data[col].shift(1)
                modeling_data[f'{col}_lag7'] = modeling_data[col].shift(7)
        
        # Remove rows with NaN values
        modeling_data = modeling_data.dropna()
        
        return modeling_data if not modeling_data.empty else None
    
    except Exception as e:
        logger.error(f"Error preparing modeling data: {e}")
        return None


def render_linear_regression_analysis(modeling_data):
    """Render linear regression analysis"""
    try:
        st.markdown("#### 📊 Linear Regression Analysis")
        
        # Feature selection
        numeric_cols = modeling_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Target variable selection
        target_col = st.selectbox("Select Target Variable", numeric_cols, key="lr_target")
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Features",
            [col for col in numeric_cols if col != target_col],
            default=[col for col in numeric_cols if col != target_col][:5],
            key="lr_features"
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature")
            return
        
        # Prepare data
        X = modeling_data[feature_cols]
        y = modeling_data[target_col]
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MSE", f"{mse:.3f}")
        
        with col2:
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            st.markdown("**Feature Importance:**")
            st.dataframe(feature_importance[['Feature', 'Coefficient']], use_container_width=True)
        
        # Prediction vs Actual plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue')
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Predicted vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Error in linear regression analysis: {e}")
        st.error("Failed to perform linear regression analysis")


def render_random_forest_analysis(modeling_data):
    """Render random forest analysis"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        st.markdown("#### 🌲 Random Forest Analysis")
        
        # Feature selection
        numeric_cols = modeling_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Target variable selection
        target_col = st.selectbox("Select Target Variable", numeric_cols, key="rf_target")
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Features",
            [col for col in numeric_cols if col != target_col],
            default=[col for col in numeric_cols if col != target_col][:5],
            key="rf_features"
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature")
            return
        
        # Model parameters
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 3, 20, 10)
        
        # Prepare data
        X = modeling_data[feature_cols]
        y = modeling_data[target_col]
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MSE", f"{mse:.3f}")
        
        with col2:
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.markdown("**Feature Importance:**")
            st.dataframe(feature_importance, use_container_width=True)
        
        # Feature importance plot
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    except ImportError:
        st.error("Random Forest requires scikit-learn to be installed")
    except Exception as e:
        logger.error(f"Error in random forest analysis: {e}")
        st.error("Failed to perform random forest analysis")


def render_time_series_forecast(time_series_df, analytics_options):
    """Render time series forecasting"""
    try:
        st.markdown("#### 🔮 Time Series Forecasting")
        
        # Prepare data
        ts_data = prepare_time_series_data(time_series_df)
        
        if ts_data is None:
            st.error("Failed to prepare time series data")
            return
        
        # Select column to forecast
        numeric_cols = ts_data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Select Variable to Forecast", numeric_cols, key="ts_forecast")
        
        # Forecast parameters
        forecast_periods = analytics_options['forecast_periods']
        
        # Create forecast
        forecast_fig, forecast_metrics = create_forecast_chart(ts_data, target_col, forecast_periods)
        
        if forecast_fig is not None:
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            if forecast_metrics:
                metric_cols = st.columns(len(forecast_metrics))
                for i, (metric_name, metric_value) in enumerate(forecast_metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric_name, str(metric_value))
        else:
            st.error("Failed to create forecast")
    
    except Exception as e:
        logger.error(f"Error in time series forecasting: {e}")
        st.error("Failed to perform time series forecasting")


def prepare_statistical_test_data(time_series_df, braking_df, swerving_df):
    """Prepare data for statistical testing"""
    try:
        test_data = {}
        
        # Time series data
        if time_series_df is not None:
            test_data['time_series'] = time_series_df.copy()
        
        # Hotspot data
        if braking_df is not None:
            test_data['braking'] = braking_df.copy()
        
        if swerving_df is not None:
            test_data['swerving'] = swerving_df.copy()
        
        return test_data if test_data else None
    
    except Exception as e:
        logger.error(f"Error preparing statistical test data: {e}")
        return None


def render_t_test_analysis(test_data, analytics_options):
    """Render t-test analysis"""
    try:
        st.markdown("#### 📊 T-Test Analysis")
        
        # Select dataset
        available_datasets = list(test_data.keys())
        selected_dataset = st.selectbox("Select Dataset", available_datasets, key="ttest_dataset")
        
        data = test_data[selected_dataset]
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 1:
            st.warning("No numeric columns available for t-test")
            return
        
        # Select variable
        test_variable = st.selectbox("Select Variable", numeric_cols, key="ttest_variable")
        
        # Test type
        test_type = st.selectbox("Test Type", ["One-sample", "Two-sample"], key="ttest_type")
        
        if test_type == "One-sample":
            # One-sample t-test
            test_value = st.number_input("Test Value", value=0.0, key="ttest_value")
            
            sample_data = data[test_variable].dropna()
            t_stat, p_value = stats.ttest_1samp(sample_data, test_value)
            
            st.markdown("**Results:**")
            st.metric("t-statistic", f"{t_stat:.4f}")
            st.metric("p-value", f"{p_value:.4f}")
            
            # Interpretation
            alpha = 1 - analytics_options['confidence_level']
            if p_value < alpha:
                st.success(f"Reject null hypothesis (p < {alpha:.3f})")
            else:
                st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f})")
        
        elif test_type == "Two-sample":
            # Two-sample t-test
            if 'road_type' in data.columns:
                group_col = 'road_type'
            else:
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    group_col = st.selectbox("Select Grouping Variable", categorical_cols, key="ttest_group")
                else:
                    st.warning("No categorical columns available for grouping")
                    return
            
            # Get unique groups
            groups = data[group_col].unique()
            if len(groups) < 2:
                st.warning("Need at least 2 groups for two-sample t-test")
                return
            
            # Select two groups
            group1 = st.selectbox("Group 1", groups, key="ttest_group1")
            group2 = st.selectbox("Group 2", [g for g in groups if g != group1], key="ttest_group2")
            
            # Perform t-test
            sample1 = data[data[group_col] == group1][test_variable].dropna()
            sample2 = data[data[group_col] == group2][test_variable].dropna()
            
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            
            st.markdown("**Results:**")
            st.metric("t-statistic", f"{t_stat:.4f}")
            st.metric("p-value", f"{p_value:.4f}")
            
            # Group statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{group1} Mean", f"{sample1.mean():.4f}")
                st.metric(f"{group1} Std", f"{sample1.std():.4f}")
            
            with col2:
                st.metric(f"{group2} Mean", f"{sample2.mean():.4f}")
                st.metric(f"{group2} Std", f"{sample2.std():.4f}")
            
            # Interpretation
            alpha = 1 - analytics_options['confidence_level']
            if p_value < alpha:
                st.success(f"Reject null hypothesis (p < {alpha:.3f})")
            else:
                st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f})")
    
    except Exception as e:
        logger.error(f"Error in t-test analysis: {e}")
        st.error("Failed to perform t-test analysis")


def render_anova_analysis(test_data, analytics_options):
    """Render ANOVA analysis"""
    try:
        st.markdown("#### 📊 ANOVA Analysis")
        
        # Select dataset
        available_datasets = list(test_data.keys())
        selected_dataset = st.selectbox("Select Dataset", available_datasets, key="anova_dataset")
        
        data = test_data[selected_dataset]
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) < 1:
            st.warning("No numeric columns available for ANOVA")
            return
        
        if len(categorical_cols) < 1:
            st.warning("No categorical columns available for grouping")
            return
        
        # Select variables
        dependent_var = st.selectbox("Select Dependent Variable", numeric_cols, key="anova_dependent")
        grouping_var = st.selectbox("Select Grouping Variable", categorical_cols, key="anova_grouping")
        
        # Prepare data for ANOVA
        groups = []
        group_names = []
        
        for group_name in data[grouping_var].unique():
            group_data = data[data[grouping_var] == group_name][dependent_var].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(group_name)
        
        if len(groups) < 2:
            st.warning("Need at least 2 groups for ANOVA")
            return
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.markdown("**Results:**")
        st.metric("F-statistic", f"{f_stat:.4f}")
        st.metric("p-value", f"{p_value:.4f}")
        
        # Group statistics
        group_stats = []
        for i, group_name in enumerate(group_names):
            group_stats.append({
                'Group': group_name,
                'Mean': groups[i].mean(),
                'Std': groups[i].std(),
                'Count': len(groups[i])
            })
        
        group_stats_df = pd.DataFrame(group_stats)
        st.dataframe(group_stats_df, use_container_width=True)
        
        # Interpretation
        alpha = 1 - analytics_options['confidence_level']
        if p_value < alpha:
            st.success(f"Reject null hypothesis (p < {alpha:.3f}) - Groups have significantly different means")
        else:
            st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f}) - No significant difference between groups")
    
    except Exception as e:
        logger.error(f"Error in ANOVA analysis: {e}")
        st.error("Failed to perform ANOVA analysis")


def render_chi_square_analysis(test_data, analytics_options):
    """Render chi-square analysis"""
    try:
        st.markdown("#### 📊 Chi-Square Analysis")
        
        # Select dataset
        available_datasets = list(test_data.keys())
        selected_dataset = st.selectbox("Select Dataset", available_datasets, key="chi_dataset")
        
        data = test_data[selected_dataset]
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) < 2:
            st.warning("Need at least 2 categorical columns for chi-square test")
            return
        
        # Select variables
        var1 = st.selectbox("Select Variable 1", categorical_cols, key="chi_var1")
        var2 = st.selectbox("Select Variable 2", [col for col in categorical_cols if col != var1], key="chi_var2")
        
        # Create contingency table
        contingency_table = pd.crosstab(data[var1], data[var2])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        st.markdown("**Contingency Table:**")
        st.dataframe(contingency_table, use_container_width=True)
        
        st.markdown("**Results:**")
        st.metric("Chi-square statistic", f"{chi2_stat:.4f}")
        st.metric("p-value", f"{p_value:.4f}")
        st.metric("Degrees of freedom", dof)
        
        # Interpretation
        alpha = 1 - analytics_options['confidence_level']
        if p_value < alpha:
            st.success(f"Reject null hypothesis (p < {alpha:.3f}) - Variables are associated")
        else:
            st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f}) - No significant association")
    
    except Exception as e:
        logger.error(f"Error in chi-square analysis: {e}")
        st.error("Failed to perform chi-square analysis")


def render_ks_test_analysis(test_data, analytics_options):
    """Render Kolmogorov-Smirnov test analysis"""
    try:
        st.markdown("#### 📊 Kolmogorov-Smirnov Test")
        
        # Select dataset
        available_datasets = list(test_data.keys())
        selected_dataset = st.selectbox("Select Dataset", available_datasets, key="ks_dataset")
        
        data = test_data[selected_dataset]
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 1:
            st.warning("No numeric columns available for KS test")
            return
        
        # Select variable
        test_variable = st.selectbox("Select Variable", numeric_cols, key="ks_variable")
        
        # Test type
        test_type = st.selectbox("Test Type", ["Normality", "Two-sample"], key="ks_type")
        
        sample_data = data[test_variable].dropna()
        
        if test_type == "Normality":
            # Test for normality
            ks_stat, p_value = stats.kstest(sample_data, 'norm', args=(sample_data.mean(), sample_data.std()))
            
            st.markdown("**Results:**")
            st.metric("KS statistic", f"{ks_stat:.4f}")
            st.metric("p-value", f"{p_value:.4f}")
            
            # Interpretation
            alpha = 1 - analytics_options['confidence_level']
            if p_value < alpha:
                st.success(f"Reject null hypothesis (p < {alpha:.3f}) - Data is not normally distributed")
            else:
                st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f}) - Data may be normally distributed")
        
        elif test_type == "Two-sample":
            # Two-sample KS test
            if 'road_type' in data.columns:
                group_col = 'road_type'
            else:
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    group_col = st.selectbox("Select Grouping Variable", categorical_cols, key="ks_group")
                else:
                    st.warning("No categorical columns available for grouping")
                    return
            
            # Get unique groups
            groups = data[group_col].unique()
            if len(groups) < 2:
                st.warning("Need at least 2 groups for two-sample KS test")
                return
            
            # Select two groups
            group1 = st.selectbox("Group 1", groups, key="ks_group1")
            group2 = st.selectbox("Group 2", [g for g in groups if g != group1], key="ks_group2")
            
            # Perform KS test
            sample1 = data[data[group_col] == group1][test_variable].dropna()
            sample2 = data[data[group_col] == group2][test_variable].dropna()
            
            ks_stat, p_value = stats.ks_2samp(sample1, sample2)
            
            st.markdown("**Results:**")
            st.metric("KS statistic", f"{ks_stat:.4f}")
            st.metric("p-value", f"{p_value:.4f}")
            
            # Interpretation
            alpha = 1 - analytics_options['confidence_level']
            if p_value < alpha:
                st.success(f"Reject null hypothesis (p < {alpha:.3f}) - Samples come from different distributions")
            else:
                st.info(f"Fail to reject null hypothesis (p ≥ {alpha:.3f}) - Samples may come from same distribution")
    
    except Exception as e:
        logger.error(f"Error in KS test analysis: {e}")
        st.error("Failed to perform KS test analysis")
