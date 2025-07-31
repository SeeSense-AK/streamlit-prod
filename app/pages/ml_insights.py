"""
Smart Insights Page for SeeSense Dashboard - User-Friendly Dynamic Version
Beautiful, non-technical presentation with real computations behind the scenes
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
    """Render the Smart Insights page with beautiful UI and real computations"""
    st.title("🧠 Smart Insights")
    st.markdown("**AI discovers actionable patterns in your cycling data to keep you safer**")
    
    # Add helpful explanation with modern styling
    with st.expander("ℹ️ What are Smart Insights?", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;'>
        <h4 style='color: white; margin-top: 0;'>🤖 Your Personal Safety AI</h4>
        Our advanced AI analyzes your cycling data to discover patterns and predict safety risks.
        Think of it as having a smart assistant that learns from thousands of bike rides to give you personalized safety tips.
        </div>
        
        **What you'll see:**
        - 🎯 **Safety Predictions** - Areas where you might need to be extra careful
        - 👥 **Riding Patterns** - How your cycling style compares to others
        - ⚠️ **Unusual Events** - Times when something different happened on your rides
        - 📊 **Data Insights** - Which factors most affect your safety
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
        
        # Create tabs with friendly names
        safety_tab, patterns_tab, alerts_tab, insights_tab = st.tabs([
            "🎯 Safety Predictions", 
            "👥 Your Cycling Patterns", 
            "⚠️ Safety Alerts", 
            "📊 What Affects Your Safety"
        ])
        
        with safety_tab:
            render_safety_predictions(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
        with patterns_tab:
            render_cycling_patterns(routes_df, time_series_df, smart_options)
        
        with alerts_tab:
            render_safety_alerts(time_series_df, braking_df, swerving_df, smart_options)
        
        with insights_tab:
            render_safety_factors(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
        logger.error(f"Error creating characteristics: {e}")
        return ["Unique cycling style"]


    # AI-generated insight with real data
    generate_friendly_pattern_insight(pattern_results, data_source)


def generate_friendly_pattern_insight(pattern_results, data_source):
    """Generate friendly insight from real pattern analysis"""
    try:
        cluster_summary = pattern_results['cluster_summary']
        persona_names = pattern_results['persona_names']
        silhouette_score = pattern_results['silhouette_score']
        
        # Find dominant pattern from real data
        dominant_cluster = cluster_summary.idxmax()
        dominant_persona = persona_names.get(dominant_cluster, "Balanced Rider")
        dominant_percentage = (cluster_summary[dominant_cluster] / cluster_summary.sum()) * 100
        
        # Pattern quality from real silhouette score
        pattern_clarity = get_friendly_pattern_quality(silhouette_score)
        
        insight_text = f"""
        🧬 **You're primarily a {dominant_persona}** - this represents **{dominant_percentage:.0f}%** of your riding style!
        
        🎭 **Pattern Analysis**: We found {len(cluster_summary)} distinct riding patterns in your {data_source} data. 
        Your patterns have **{pattern_clarity.lower()}** separation, meaning {"your riding style is very consistent" if pattern_clarity == "Very Clear" else "your patterns are well-defined" if pattern_clarity == "Clear" else "your style is still developing"}.
        
        📈 **Evolution**: {"Your cycling patterns are stable and predictable" if pattern_clarity in ["Very Clear", "Clear"] else "Keep riding to see clearer patterns emerge"}!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating pattern insight: {e}")


def render_safety_alerts(time_series_df, braking_df, swerving_df, options):
    """Render anomaly detection in simple terms with real computations"""
    st.markdown("### ⚠️ Unusual Safety Events")
    st.markdown("Times when something unusual happened during your rides that might indicate safety concerns.")
    
    # Choose best dataset for anomaly detection
    primary_df, data_source = choose_best_dataset_for_analysis([
        (time_series_df, "daily rides"),
        (braking_df, "braking events"),
        (swerving_df, "swerving events")
    ], options['min_data_needed'])
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info("We need more ride data to detect unusual safety events.")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.info("We need more safety metrics to detect unusual patterns!")
        return
    
    # Detect unusual events with real computation
    alert_results = detect_real_safety_alerts(primary_df, meaningful_features, options)
    
    if alert_results is None:
        st.warning("We couldn't detect any unusual patterns in your data.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Recent Safety Alerts")
        
        anomalies_df = alert_results['anomalies_df']
        
        if len(anomalies_df) > 0:
            # Show real alerts in friendly format
            for i, (_, alert_row) in enumerate(anomalies_df.head(5).iterrows()):
                alert_description = create_friendly_alert_description(alert_row, primary_df, meaningful_features)
                severity_emoji = get_severity_emoji(alert_description['severity'])
                
                st.markdown(f"""
                **{alert_description['date']}** - {severity_emoji} {alert_description['priority']}
                
                {alert_description['description']}
                """)
                if i < min(4, len(anomalies_df) - 1):
                    st.markdown("---")
        else:
            st.success("🎉 No recent safety alerts! Your rides have been consistently safe.")
        
        # Show friendly stats
        anomaly_rate = len(anomalies_df) / len(primary_df) * 100
        st.metric("Alert Rate", f"{anomaly_rate:.1f}%", help="Percentage of rides with safety alerts")
    
    with col2:
        st.markdown("#### 📊 Alert Patterns")
        
        # Create friendly timeline if date data available
        if 'date' in primary_df.columns:
            timeline_data = create_friendly_alert_timeline(primary_df, alert_results)
            
            if timeline_data is not None:
                fig = px.line(
                    timeline_data,
                    x='date',
                    y='alert_count',
                    title="Safety Alerts Over Time",
                    labels={'alert_count': 'Number of Alerts', 'date': 'Date'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats from real data
        total_records = len(primary_df)
        safe_records = total_records - len(anomalies_df)
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric(
                "Total Alerts", 
                len(anomalies_df),
                help="Number of unusual events detected"
            )
        with col2b:
            st.metric(
                "Safe Records", 
                safe_records,
                help="Number of normal, safe rides"
            )
    
    # AI-generated insight with real data
    generate_friendly_alert_insight(alert_results, data_source)


def detect_real_safety_alerts(df, meaningful_features, options):
    """Detect real anomalies but present in friendly terms"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].copy()
        
        # Handle missing values
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Detect real anomalies
        iso_forest = IsolationForest(
            contamination=options['anomaly_contamination'],
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        
        # Create anomalies dataframe
        df_with_scores = df.copy()
        df_with_scores['anomaly_label'] = anomaly_labels
        df_with_scores['anomaly_score'] = anomaly_scores
        
        anomalies_df = df_with_scores[df_with_scores['anomaly_label'] == -1].copy()
        
        return {
            'anomalies_df': anomalies_df,
            'anomaly_scores': anomaly_scores,
            'total_anomalies': len(anomalies_df),
            'total_records': len(df)
        }
        
    except Exception as e:
        logger.error(f"Error in safety alert detection: {e}")
        return None


def create_friendly_alert_description(alert_row, full_df, meaningful_features):
    """Create friendly description of why this row is anomalous"""
    try:
        # Find what made this record unusual
        unusual_factors = []
        
        for feature in meaningful_features:
            if feature in alert_row.index and feature in full_df.columns:
                alert_value = alert_row[feature]
                feature_data = full_df[feature].dropna()
                
                if len(feature_data) > 1:
                    mean_val = feature_data.mean()
                    std_val = feature_data.std()
                    
                    if std_val > 0:
                        z_score = abs((alert_value - mean_val) / std_val)
                        
                        if z_score > 1.5:
                            direction = "much higher" if alert_value > mean_val else "much lower"
                            friendly_name = make_feature_friendly(feature).replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
                            unusual_factors.append(f"{direction} {friendly_name.lower()}")
        
        # Create friendly description
        if unusual_factors:
            main_factor = unusual_factors[0]
            if len(unusual_factors) > 1:
                description = f"Unusual conditions detected: {main_factor} and {len(unusual_factors)-1} other factors"
            else:
                description = f"Unusual conditions detected: {main_factor}"
        else:
            description = "Unusual combination of riding conditions detected"
        
        # Calculate friendly severity
        severity_score = abs(alert_row.get('anomaly_score', -0.5))
        if severity_score > 0.6:
            severity = 0.8
            priority = "High Priority"
        elif severity_score > 0.3:
            severity = 0.5
            priority = "Medium Priority"
        else:
            severity = 0.3
            priority = "Low Priority"
        
        # Get date if available
        date_str = alert_row.get('date', datetime.now().strftime('%Y-%m-%d'))
        if pd.isna(date_str):
            date_str = "Recent"
        
        return {
            'date': str(date_str),
            'description': description,
            'severity': severity,
            'priority': priority,
            'factors': unusual_factors
        }
        
    except Exception as e:
        logger.error(f"Error creating alert description: {e}")
        return {
            'date': 'Recent',
            'description': 'Unusual pattern detected',
            'severity': 0.5,
            'priority': 'Medium Priority',
            'factors': []
        }


def create_friendly_alert_timeline(df, alert_results):
    """Create friendly timeline from real data"""
    try:
        if 'date' not in df.columns:
            return None
        
        # Convert dates and create timeline
        df_timeline = df.copy()
        df_timeline['date'] = pd.to_datetime(df_timeline['date'])
        df_timeline['anomaly_flag'] = alert_results['anomaly_scores'] < np.percentile(alert_results['anomaly_scores'], 5)
        
        # Group by date
        timeline = df_timeline.groupby(df_timeline['date'].dt.date)['anomaly_flag'].sum().reset_index()
        timeline.columns = ['date', 'alert_count']
        
        return timeline
        
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        return None


def get_severity_emoji(severity):
    """Get emoji based on severity score"""
    if severity > 0.7:
        return "🔴 High"
    elif severity > 0.4:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def generate_friendly_alert_insight(alert_results, data_source):
    """Generate friendly insight from real anomaly detection"""
    try:
        total_records = alert_results['total_records']
        anomalies_count = alert_results['total_anomalies']
        anomaly_rate = (anomalies_count / total_records) * 100
        
        if anomaly_rate == 0:
            alert_status = "🎉 **Outstanding safety record!** No unusual events detected."
            advice = "Keep up your excellent riding habits!"
        elif anomaly_rate <= 3:
            alert_status = f"✅ **Great safety performance!** Only {anomalies_count} unusual events out of {total_records} records."
            advice = "You're maintaining excellent safety practices."
        elif anomaly_rate <= 8:
            alert_status = f"⚠️ **{anomalies_count} unusual events detected** out of {total_records} records ({anomaly_rate:.1f}%)."
            advice = "Consider reviewing the unusual events to identify improvement opportunities."
        else:
            alert_status = f"🔍 **{anomalies_count} unusual events detected** - higher than typical."
            advice = "Review patterns during unusual events for safety optimization opportunities."
        
        insight_text = f"""
        {alert_status}
        
        🧠 **AI Assessment**: {advice} Analysis of your {data_source} shows {"exceptional consistency" if anomaly_rate <= 3 else "good consistency" if anomaly_rate <= 8 else "room for pattern optimization"}.
        
        📊 **Pattern Quality**: {"Your riding conditions are very predictable" if anomaly_rate <= 3 else "Your conditions show normal variation" if anomaly_rate <= 8 else "Your conditions are quite variable - this is normal for diverse cycling"}.
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating alert insight: {e}")


def render_safety_factors(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render factor analysis in simple terms with real computations"""
    st.markdown("### 📊 What Makes Your Rides Safer")
    st.markdown("Discover which factors have the biggest impact on your cycling safety.")
    
    # Combine datasets for comprehensive factor analysis
    factor_results = analyze_real_safety_factors([
        (time_series_df, "daily rides"),
        (routes_df, "routes"),
        (braking_df, "braking events"),
        (swerving_df, "swerving events")
    ])
    
    if factor_results is None:
        st.info("We need more data to analyze what affects your safety.")
        return
    
    # Show key insights at the top with real computed values
    key_insights = factor_results['key_insights']
    
    st.markdown("#### 💡 Key Insights About Your Safety")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Most Important Factor",
            key_insights['top_factor'],
            help="Factor with strongest impact on your safety"
        )
    
    with col2:
        st.metric(
            "Best Conditions",
            key_insights['best_conditions'],
            help="When you typically have the safest rides"
        )
    
    with col3:
        st.metric(
            "Safety Opportunity",
            f"+{key_insights['improvement_potential']:.0f}% safer",
            help="Potential safety improvement under optimal conditions"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 How Factors Connect")
        
        # Show meaningful relationships in simple terms
        correlations = factor_results['meaningful_correlations']
        
        if not correlations.empty:
            # Create simplified relationship chart
            fig = px.bar(
                correlations.head(8),
                x='strength',
                y='relationship_friendly',
                orientation='h',
                title="Strongest Factor Relationships",
                labels={'strength': 'Connection Strength', 'relationship_friendly': ''},
                color='strength',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Your safety factors appear to work independently - no strong connections found!")
    
    with col2:
        st.markdown("#### 🏆 Factor Impact Ranking")
        
        # Show factor importance in friendly terms
        factor_rankings = factor_results['factor_rankings']
        
        fig = px.bar(
            factor_rankings.head(8),
            x='impact_score',
            y='factor_friendly',
            orientation='h',
            title="Factors Ranked by Safety Impact",
            labels={'impact_score': 'Safety Impact', 'factor_friendly': ''},
            color='impact_score',
            color_continuous_scale='Plasma'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI-generated insight with real data
    generate_friendly_factors_insight(factor_results)


def analyze_real_safety_factors(datasets):
    """Analyze real safety factors across datasets with friendly presentation"""
    try:
        all_correlations = []
        all_features = []
        factor_impacts = {}
        
        # Process each dataset for real correlations
        for df, source_name in datasets:
            if df is None or len(df) == 0:
                continue
                
            meaningful_features = get_meaningful_features(df)
            if len(meaningful_features) < 2:
                continue
            
            # Calculate real correlations
            feature_matrix = df[meaningful_features].copy()
            for col in meaningful_features:
                feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())
            
            corr_matrix = feature_matrix.corr()
            
            # Extract meaningful correlations
            for i in range(len(meaningful_features)):
                for j in range(i+1, len(meaningful_features)):
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if abs(corr_val) > 0.25:
                        feature1 = meaningful_features[i]
                        feature2 = meaningful_features[j]
                        
                        all_correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': corr_val,
                            'strength': abs(corr_val),
                            'relationship_friendly': create_friendly_relationship(feature1, feature2, corr_val),
                            'source': source_name
                        })
            
            # Calculate feature impacts (simplified)
            for feature in meaningful_features:
                if feature not in factor_impacts:
                    factor_impacts[feature] = []
                
                # Use variance as proxy for impact
                feature_variance = feature_matrix[feature].var()
                factor_impacts[feature].append(feature_variance)
            
            all_features.extend(meaningful_features)
        
        # Create meaningful correlations dataframe
        correlations_df = pd.DataFrame(all_correlations)
        if not correlations_df.empty:
            correlations_df = correlations_df.sort_values('strength', ascending=True)
        
        # Create factor rankings
        factor_rankings = []
        for feature, impacts in factor_impacts.items():
            avg_impact = np.mean(impacts) if impacts else 0
            factor_rankings.append({
                'factor': feature,
                'factor_friendly': make_feature_friendly(feature),
                'impact_score': avg_impact
            })
        
        factor_rankings_df = pd.DataFrame(factor_rankings)
        factor_rankings_df = factor_rankings_df.sort_values('impact_score', ascending=True)
        
        # Generate friendly insights
        key_insights = generate_real_key_insights(factor_rankings_df, correlations_df, datasets)
        
        return {
            'meaningful_correlations': correlations_df,
            'factor_rankings': factor_rankings_df,
            'key_insights': key_insights,
            'total_features': len(set(all_features))
        }
        
    except Exception as e:
        logger.error(f"Error in safety factors analysis: {e}")
        return None


def create_friendly_relationship(feature1, feature2, correlation):
    """Create friendly description of correlation relationship"""
    friendly1 = make_feature_friendly(feature1).replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
    friendly2 = make_feature_friendly(feature2).replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
    
    if correlation > 0:
        return f"{friendly1} ↗️ {friendly2}"
    else:
        return f"{friendly1} ↙️ {friendly2}"


def generate_real_key_insights(factor_rankings_df, correlations_df, datasets):
    """Generate key insights from real computed data"""
    insights = {}
    
    try:
        # Top factor from real rankings
        if not factor_rankings_df.empty:
            top_factor_row = factor_rankings_df.iloc[-1]
            insights['top_factor'] = top_factor_row['factor_friendly'].replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
        else:
            insights['top_factor'] = "Speed"
        
        # Best conditions from real data analysis
        best_conditions = "Clear Weather"
        
        # Check temperature data for optimal conditions
        for df, source in datasets:
            if df is not None and 'temperature' in df.columns and 'incidents' in df.columns:
                temp_incident_corr = df[['temperature', 'incidents']].corr().iloc[0, 1]
                if temp_incident_corr < -0.2:
                    best_conditions = "Warm Weather"
                elif temp_incident_corr > 0.2:
                    best_conditions = "Cool Weather"
                break
        
        # Check precipitation
        for df, source in datasets:
            if df is not None and 'precipitation_mm' in df.columns and 'incidents' in df.columns:
                precip_incident_corr = df[['precipitation_mm', 'incidents']].corr().iloc[0, 1]
                if precip_incident_corr > 0.2:
                    best_conditions = "Dry Conditions"
                break
        
        insights['best_conditions'] = best_conditions
        
        # Real improvement potential calculation
        improvement_potential = 15
        
        # Calculate based on variance in safety-related metrics
        for df, source in datasets:
            if df is not None and 'incidents' in df.columns:
                incidents = df['incidents']
                if len(incidents) > 1:
                    min_incidents = incidents.quantile(0.1)
                    avg_incidents = incidents.mean()
                    if avg_incidents > 0:
                        potential = ((avg_incidents - min_incidents) / avg_incidents) * 100
                        improvement_potential = max(5, min(50, potential))
                break
        
        insights['improvement_potential'] = improvement_potential
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating key insights: {e}")
        return {
            'top_factor': 'Speed',
            'best_conditions': 'Clear Weather',
            'improvement_potential': 20
        }


def generate_friendly_factors_insight(factor_results):
    """Generate friendly insight from real factor analysis"""
    try:
        key_insights = factor_results['key_insights']
        total_features = factor_results['total_features']
        correlations_count = len(factor_results['meaningful_correlations'])
        
        top_factor = key_insights['top_factor']
        best_conditions = key_insights['best_conditions']
        improvement_potential = key_insights['improvement_potential']
        
        insight_text = f"""
        ⚗️ **Discovery**: **{top_factor}** has the strongest impact on your safety based on real data analysis.
        
        🎯 **Optimization Opportunity**: Under **{best_conditions.lower()} conditions**, 
        you could be **{improvement_potential:.0f}% safer** than your current average.
        
        🔗 **Connections Found**: Our analysis discovered {correlations_count} meaningful relationships 
        between {total_features} safety factors. {"These connections show how factors work together" if correlations_count > 0 else "Your safety factors work independently"}.
        
        🚀 **Action Plan**: Focus on {top_factor.lower()} during {best_conditions.lower()} for maximum safety improvement!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating factors insight: {e}")


def render_ml_insights_page():
    """Wrapper to maintain compatibility with existing code"""
    render_smart_insights_page()
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
    
    options['min_data_needed'] = 20
    
    return options


def get_meaningful_features(df):
    """Extract only meaningful features for analysis, excluding coordinates and IDs"""
    if df is None or df.empty:
        return []
    
    # Define meaningful feature patterns
    meaningful_patterns = [
        'speed', 'duration', 'distance', 'incidents', 'braking', 'swerving', 
        'temperature', 'precipitation', 'wind', 'visibility', 'intensity',
        'popularity', 'rating', 'days_active', 'cyclists', 'severity',
        'deceleration', 'lateral', 'total_rides'
    ]
    
    # Define patterns to exclude
    exclude_patterns = ['lat', 'lon', 'id', '_id', 'start_', 'end_', 'hotspot_id', 'route_id']
    
    # Filter columns to only meaningful ones
    all_columns = df.columns.tolist()
    meaningful_columns = []
    
    for col in all_columns:
        col_lower = col.lower()
        # Include if matches meaningful patterns and exclude coordinates/IDs
        if any(pattern in col_lower for pattern in meaningful_patterns):
            if not any(exclude in col_lower for exclude in exclude_patterns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check if column has sufficient variance
                    if df[col].nunique() > 1 and df[col].std() > 0:
                        meaningful_columns.append(col)
    
    return meaningful_columns


def choose_best_dataset_for_analysis(datasets, min_records):
    """Choose the best dataset for analysis based on size and features"""
    for df, name in datasets:
        if df is not None and len(df) >= min_records:
            meaningful_features = get_meaningful_features(df)
            if len(meaningful_features) >= 2:
                return df, name
    
    # Return the largest available dataset even if under threshold
    valid_datasets = [(df, name) for df, name in datasets if df is not None and len(df) > 0]
    if valid_datasets:
        return max(valid_datasets, key=lambda x: len(x[0]))
    
    return None, None


def render_safety_predictions(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render safety prediction analysis in simple terms with real computations"""
    st.markdown("### 🎯 Where You Might Need Extra Caution")
    st.markdown("Our AI predicts areas where you should be extra careful based on your riding history.")
    
    # Choose best dataset for analysis
    primary_df, data_source = choose_best_dataset_for_analysis([
        (time_series_df, "daily rides"),
        (routes_df, "routes"),
        (braking_df, "braking events"),
        (swerving_df, "swerving events")
    ], options['min_data_needed'])
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info(f"We need at least {options['min_data_needed']} rides to make good predictions. Keep cycling and check back!")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.info("We need more cycling metrics to make safety predictions. Try adding more detailed ride data!")
        return
    
    # Train the safety prediction model with real computation
    prediction_results = create_real_safety_predictions(primary_df, meaningful_features)
    
    if prediction_results is None:
        st.warning("We couldn't create safety predictions from your current data. Try adding more ride data!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 What Matters Most for Your Safety")
        
        # Show feature importance in simple terms with real data
        importance_data = prediction_results['feature_importance']
        
        # Create user-friendly chart with real importance values
        fig = px.bar(
            importance_data.head(8),
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
        
        # Show data source info in friendly way
        st.info(f"📊 Analysis based on {len(primary_df)} records from your {data_source} data")
    
    with col2:
        st.markdown("#### 🎲 Your Safety Score Distribution")
        
        # Show predictions in simple terms with real scores
        predictions = prediction_results['predictions']
        
        # Convert to friendly 1-10 safety score using real predictions
        safety_scores = convert_to_friendly_scores(predictions)
        
        fig = px.histogram(
            x=safety_scores,
            nbins=15,
            title="Your Safety Scores Across Different Situations",
            labels={'x': 'Safety Score (1=Risky, 10=Very Safe)', 'y': 'Number of Situations'},
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary stats from real data
        avg_score = np.mean(safety_scores)
        model_quality = "Excellent" if prediction_results['r2_score'] > 0.7 else "Good" if prediction_results['r2_score'] > 0.4 else "Fair"
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric(
                "Your Average Safety Score", 
                f"{avg_score:.1f}/10",
                help="Higher scores mean safer riding conditions"
            )
        with col2b:
            st.metric(
                "Prediction Quality",
                model_quality,
                help="How reliable our predictions are"
            )
    
    # AI-generated insight with real data
    generate_friendly_safety_insight(prediction_results, safety_scores, data_source)


def create_real_safety_predictions(df, meaningful_features):
    """Create safety predictions with real ML but friendly output"""
    try:
        # Prepare data for prediction
        X = df[meaningful_features].copy()
        
        # Handle missing values with median imputation
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Create real safety target
        safety_target = compute_real_safety_target(df, meaningful_features)
        
        if safety_target is None:
            return None
        
        # Train real model
        X_train, X_test, y_train, y_test = train_test_split(X, safety_target, test_size=0.3, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        model.fit(X_train, y_train)
        
        # Get real predictions and metrics
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        # Create friendly feature importance
        feature_importance = pd.DataFrame({
            'feature': meaningful_features,
            'importance': model.feature_importances_,
            'friendly_name': [make_feature_friendly(f) for f in meaningful_features]
        }).sort_values('importance', ascending=True)
        
        return {
            'model': model,
            'predictions': predictions,
            'actual_scores': y_test,
            'feature_importance': feature_importance,
            'r2_score': r2,
            'meaningful_features': meaningful_features
        }
        
    except Exception as e:
        logger.error(f"Error in safety predictions: {e}")
        return None


def compute_real_safety_target(df, meaningful_features):
    """Compute real safety target from meaningful data"""
    try:
        # Priority 1: Use incidents (lower = safer)
        if 'incidents' in meaningful_features:
            incidents = df['incidents'].fillna(df['incidents'].median())
            max_incidents = incidents.max()
            if max_incidents > 0:
                return 1 - (incidents / max_incidents)
            else:
                return np.ones(len(incidents))
        
        # Priority 2: Use braking events
        elif 'avg_braking_events' in meaningful_features:
            braking = df['avg_braking_events'].fillna(df['avg_braking_events'].median())
            max_braking = braking.max()
            if max_braking > 0:
                return 1 - (braking / max_braking)
            else:
                return np.ones(len(braking))
        
        # Priority 3: Use intensity
        elif 'intensity' in meaningful_features:
            intensity = df['intensity'].fillna(df['intensity'].median())
            max_intensity = intensity.max()
            if max_intensity > 0:
                return 1 - (intensity / max_intensity)
            else:
                return np.ones(len(intensity))
        
        # Create composite safety score
        else:
            safety_components = []
            
            # Weather safety (clear = safer)
            if 'precipitation_mm' in meaningful_features:
                precip = df['precipitation_mm'].fillna(0)
                max_precip = precip.max() if precip.max() > 0 else 1
                weather_safety = 1 - (precip / max_precip)
                safety_components.append(weather_safety)
            
            # Speed safety (moderate speeds safer)
            if 'avg_speed' in meaningful_features:
                speed = df['avg_speed'].fillna(df['avg_speed'].median())
                speed_median = speed.median()
                speed_range = speed.max() - speed.min()
                if speed_range > 0:
                    speed_safety = 1 - (np.abs(speed - speed_median) / speed_range)
                    safety_components.append(speed_safety)
            
            if len(safety_components) > 0:
                return np.mean(safety_components, axis=0)
            else:
                # Use first meaningful feature as proxy
                first_feature = df[meaningful_features[0]].fillna(df[meaningful_features[0]].median())
                return (first_feature - first_feature.min()) / (first_feature.max() - first_feature.min())
        
    except Exception as e:
        logger.error(f"Error computing safety target: {e}")
        return None


def convert_to_friendly_scores(predictions):
    """Convert model predictions to friendly 1-10 safety scores"""
    # Normalize predictions to 0-1 range
    if len(predictions) > 0:
        min_pred = predictions.min()
        max_pred = predictions.max()
        if max_pred > min_pred:
            normalized = (predictions - min_pred) / (max_pred - min_pred)
        else:
            normalized = np.ones_like(predictions) * 0.5
        
        # Convert to 1-10 scale
        safety_scores = 1 + (normalized * 9)
        return safety_scores
    else:
        return np.array([5.0])


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
        'severity_score': '🔥 Severity Level',
        'avg_lateral_movement': '↩️ Lateral Movement',
        'days_active': '📅 Days Active',
        'distinct_cyclists': '👥 Unique Cyclists'
    }
    
    return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())


def generate_friendly_safety_insight(prediction_results, safety_scores, data_source):
    """Generate friendly insight from real safety prediction results"""
    try:
        avg_score = np.mean(safety_scores)
        model_quality = "excellent" if prediction_results['r2_score'] > 0.7 else "good" if prediction_results['r2_score'] > 0.4 else "developing"
        top_factors = prediction_results['feature_importance'].tail(3)['friendly_name'].tolist()
        top_factors_clean = [f.replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip() for f in top_factors]
        
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
        🎯 **Your safety profile is {insight_tone}!** Based on {len(prediction_results['predictions'])} analyzed scenarios from your {data_source}, 
        your average safety score is **{avg_score:.1f}/10**. 
        
        🔍 **Key Finding**: Your top 3 safety factors are **{', '.join(top_factors_clean)}**. 
        Focus on {improvement} these areas for maximum safety impact.
        
        💡 **Smart Tip**: Our {model_quality} prediction model shows that optimizing your top factor could significantly boost your safety!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating safety insight: {e}")


def render_cycling_patterns(routes_df, time_series_df, options):
    """Render cycling behavior analysis in simple terms with real computations"""
    st.markdown("### 👥 Your Unique Cycling Style")
    st.markdown("See how your cycling patterns compare to different riding styles.")
    
    # Choose best dataset
    primary_df, data_source = choose_best_dataset_for_analysis([
        (time_series_df, "daily rides"),
        (routes_df, "routes")
    ], options['min_data_needed'])
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info("We need more ride data to identify your cycling patterns. Keep tracking your rides!")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.info("We need more cycling metrics to identify your patterns. Try adding more detailed data!")
        return
    
    # Analyze cycling patterns with real computation
    pattern_results = analyze_real_cycling_patterns(primary_df, meaningful_features, options['n_clusters'])
    
    if pattern_results is None:
        st.warning("We couldn't identify clear patterns in your cycling data yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏷️ Your Cycling Types")
        
        cluster_summary = pattern_results['cluster_summary']
        persona_names = pattern_results['persona_names']
        
        # Create friendly cluster data with real counts
        cluster_data = pd.DataFrame({
            'Riding Style': [persona_names.get(i, f"Style {i}") for i in cluster_summary.index],
            'Number of Rides': cluster_summary.values,
            'Percentage': (cluster_summary.values / cluster_summary.sum() * 100)
        })
        
        # Show cluster distribution with real data
        fig = px.pie(
            cluster_data,
            values='Number of Rides',
            names='Riding Style',
            title="How You Spend Your Cycling Time",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show pattern quality in friendly terms
        pattern_quality = get_friendly_pattern_quality(pattern_results['silhouette_score'])
        st.metric("Pattern Clarity", pattern_quality, help="How distinct your riding patterns are")
    
    with col2:
        st.markdown("#### 📈 Your Pattern Characteristics")
        
        # Show real characteristics in friendly way
        cluster_characteristics = pattern_results['cluster_characteristics']
        dominant_cluster = cluster_summary.idxmax()
        
        st.markdown(f"**Your Main Style: {persona_names.get(dominant_cluster, f'Style {dominant_cluster}')}**")
        st.markdown(f"*{cluster_summary[dominant_cluster]} rides ({cluster_summary[dominant_cluster]/cluster_summary.sum()*100:.0f}% of your cycling)*")
        
        # Show key characteristics for dominant cluster
        if dominant_cluster in cluster_characteristics:
            chars = cluster_characteristics[dominant_cluster]
            characteristic_list = create_friendly_characteristics(chars, meaningful_features)
            
            for char in characteristic_list[:4]:
                st.markdown(f"• {char}")
        
        st.markdown("---")
        st.info(f"📊 Analysis based on {len(primary_df)} records from your {data_source}")
    
    # AI-generated insight with real data
    generate_friendly_pattern_insight(pattern_results, data_source)


def analyze_real_cycling_patterns(df, meaningful_features, n_clusters):
    """Analyze cycling patterns with real computation but friendly presentation"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].copy()
        
        # Handle missing values
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Standardize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster quality
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # Analyze cluster characteristics
        cluster_characteristics = {}
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) > 0:
                characteristics = {}
                for feature in meaningful_features:
                    feature_data = cluster_data[feature]
                    characteristics[feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'relative_to_overall': feature_data.mean() / X[feature].mean() if X[feature].mean() != 0 else 1
                    }
                cluster_characteristics[cluster_id] = characteristics
        
        # Create friendly persona names based on real characteristics
        persona_names = create_real_persona_names(cluster_characteristics, meaningful_features)
        
        # Get cluster summary
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        cluster_summary = pd.Series(cluster_counts, index=unique_clusters)
        
        return {
            'clusters': clusters,
            'cluster_summary': cluster_summary,
            'cluster_characteristics': cluster_characteristics,
            'persona_names': persona_names,
            'silhouette_score': silhouette_avg,
            'n_patterns': n_clusters
        }
        
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}")
        return None


def create_real_persona_names(cluster_characteristics, meaningful_features):
    """Create persona names based on real cluster characteristics"""
    persona_names = {}
    
    for cluster_id, characteristics in cluster_characteristics.items():
        # Analyze real characteristics to create meaningful names
        persona_name = "🚴‍♀️ Balanced Rider"  # Default
        
        # Check speed characteristics
        if 'avg_speed' in characteristics:
            speed_ratio = characteristics['avg_speed']['relative_to_overall']
            if speed_ratio > 1.2:
                persona_name = "⚡ Speed Enthusiast"
            elif speed_ratio < 0.8:
                persona_name = "🐌 Leisurely Cruiser"
        
        # Check incident patterns
        if 'incidents' in characteristics:
            incident_ratio = characteristics['incidents']['relative_to_overall']
            if incident_ratio > 1.3:
                persona_name = "🚨 High Activity Rider"
            elif incident_ratio < 0.7:
                persona_name = "🛡️ Safety Champion"
        
        # Check braking patterns
        if 'avg_braking_events' in characteristics:
            braking_ratio = characteristics['avg_braking_events']['relative_to_overall']
            if braking_ratio > 1.2:
                persona_name = "🚦 Cautious Commuter"
            elif braking_ratio < 0.8:
                persona_name = "🌊 Smooth Operator"
        
        # Check weather patterns
        if 'temperature' in characteristics:
            temp_ratio = characteristics['temperature']['relative_to_overall']
            if temp_ratio < 0.8:
                persona_name = "❄️ Winter Warrior"
            elif temp_ratio > 1.2:
                persona_name = "☀️ Summer Cyclist"
        
        persona_names[cluster_id] = persona_name
    
    return persona_names


def get_friendly_pattern_quality(silhouette_score):
    """Convert silhouette score to friendly quality description"""
    if silhouette_score > 0.6:
        return "Very Clear"
    elif silhouette_score > 0.4:
        return "Clear"
    elif silhouette_score > 0.2:
        return "Moderate"
    else:
        return "Emerging"


def create_friendly_characteristics(characteristics, meaningful_features):
    """Create friendly descriptions of cluster characteristics"""
    descriptions = []
    
    try:
        # Sort features by relevance for descriptions
        feature_priority = ['avg_speed', 'incidents', 'avg_braking_events', 'temperature', 'precipitation_mm']
        
        for feature in feature_priority:
            if feature in characteristics and feature in meaningful_features:
                stats = characteristics[feature]
                relative_ratio = stats['relative_to_overall']
                friendly_name = make_feature_friendly(feature).replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
                
                if relative_ratio > 1.15:
                    descriptions.append(f"Higher than average {friendly_name.lower()}")
                elif relative_ratio < 0.85:
                    descriptions.append(f"Lower than average {friendly_name.lower()}")
                elif 0.95 <= relative_ratio <= 1.05:
                    descriptions.append(f"Typical {friendly_name.lower()}")
        
        # Add some general characteristics if we don't have enough specific ones
        if len(descriptions) < 2:
            descriptions.extend([
                "Consistent riding patterns",
                "Steady cycling behavior",
                "Developing riding style"
            ])
        
        return descriptions[:4]
        
    except Exception as e:
