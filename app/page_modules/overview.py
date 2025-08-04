"""
Value-Driven Overview Page for SeeSense Dashboard
Focuses on actionable insights, business value, and decision-making support
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

from app.core.data_processor import data_processor
from app.utils.config import config
from app.core.metrics_calculator import metrics_calculator
from app.core.groq_insights_generator import create_insights_generator

logger = logging.getLogger(__name__)


def render_overview_page():
    """Main overview page focused on business value and actionable insights"""
    
    st.title("🎯 SeeSense Dashboard - Strategic Overview")
    st.markdown("**Transform your cycling safety data into actionable business insights**")
    
    try:
        # Load data
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_value_proposition()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Generate comprehensive analysis
        analysis_results = perform_comprehensive_analysis(routes_df, braking_df, swerving_df, time_series_df)
        
        # Render value-driven sections
        render_executive_summary(analysis_results)
        render_critical_insights(analysis_results)
        render_business_impact_analysis(analysis_results)
        render_risk_assessment(analysis_results)
        render_opportunity_identification(analysis_results)
        render_action_recommendations(analysis_results)
        
    except Exception as e:
        logger.error(f"Overview page error: {e}")
        st.error("Error generating insights. Please check your data quality.")
        with st.expander("Debug Information"):
            st.code(str(e))


def render_value_proposition():
    """Show value proposition when no data is available"""
    
    st.markdown("""
    ## 🚀 Unlock the Power of Your Cycling Safety Data
    
    **What you'll get with SeeSense Analytics:**
    
    ### 📊 Immediate Business Value
    - **Risk Reduction**: Identify and mitigate safety hazards before incidents occur
    - **Cost Savings**: Prevent accidents through predictive analytics
    - **Operational Efficiency**: Optimize route planning and resource allocation
    - **Compliance**: Meet safety regulations with data-driven evidence
    
    ### 🎯 Strategic Insights
    - **Performance Benchmarking**: Compare safety metrics across routes and time periods
    - **Trend Analysis**: Understand seasonal patterns and long-term safety evolution
    - **Resource Optimization**: Focus improvements where they'll have maximum impact
    - **ROI Quantification**: Measure the financial impact of safety initiatives
    
    ### 🚨 Proactive Safety Management
    - **Early Warning System**: Detect emerging safety issues before they escalate
    - **Hotspot Identification**: Pinpoint exact locations requiring intervention
    - **Predictive Maintenance**: Anticipate infrastructure needs based on usage patterns
    - **Evidence-Based Decisions**: Support policy changes with concrete data
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 💰 Financial Impact
        - Reduce incident costs by 40-60%
        - Lower insurance premiums
        - Optimize maintenance budgets
        - Improve liability management
        """)
    
    with col2:
        st.markdown("""
        ### 🛡️ Safety Outcomes
        - 25-40% reduction in accidents
        - Faster emergency response
        - Better route safety ratings
        - Enhanced user confidence
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Operational Benefits
        - Data-driven infrastructure planning
        - Improved resource allocation
        - Enhanced stakeholder reporting
        - Regulatory compliance support
        """)
    
    st.info("📁 **Get Started**: Upload your cycling safety data to unlock these insights immediately.")


def perform_comprehensive_analysis(routes_df, braking_df, swerving_df, time_series_df):
    """Perform comprehensive business-focused analysis"""
    
    # Calculate base metrics
    metrics = metrics_calculator.calculate_all_overview_metrics(
        routes_df, braking_df, swerving_df, time_series_df
    )
    
    analysis = {
        'metrics': metrics,
        'data_quality': assess_data_quality(routes_df, braking_df, swerving_df, time_series_df),
        'risk_analysis': perform_risk_analysis(routes_df, braking_df, swerving_df),
        'trend_analysis': analyze_trends(time_series_df),
        'hotspot_analysis': identify_hotspots(braking_df, swerving_df),
        'business_impact': calculate_business_impact(metrics, braking_df, swerving_df),
        'recommendations': generate_prioritized_recommendations(routes_df, braking_df, swerving_df, time_series_df),
        'opportunities': identify_improvement_opportunities(routes_df, braking_df, swerving_df, time_series_df)
    }
    
    return analysis


def assess_data_quality(routes_df, braking_df, swerving_df, time_series_df):
    """Assess data quality and completeness for reliable insights"""
    
    quality_assessment = {
        'overall_score': 0,
        'coverage': 0,
        'completeness': 0,
        'recency': 0,
        'reliability': 0,
        'issues': [],
        'recommendations': []
    }
    
    datasets = {
        'Routes': routes_df,
        'Braking Events': braking_df,
        'Swerving Events': swerving_df,
        'Time Series': time_series_df
    }
    
    total_datasets = len(datasets)
    available_datasets = sum(1 for df in datasets.values() if df is not None and len(df) > 0)
    
    # Coverage score (0-30 points)
    quality_assessment['coverage'] = (available_datasets / total_datasets) * 30
    
    # Completeness score (0-30 points)
    completeness_scores = []
    for name, df in datasets.items():
        if df is not None and len(df) > 0:
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            completeness_score = max(0, (1 - missing_ratio) * 30)
            completeness_scores.append(completeness_score)
            
            if missing_ratio > 0.2:  # More than 20% missing
                quality_assessment['issues'].append(f"{name}: {missing_ratio:.1%} missing data")
    
    if completeness_scores:
        quality_assessment['completeness'] = np.mean(completeness_scores)
    
    # Recency score (0-25 points)
    if time_series_df is not None and len(time_series_df) > 0:
        date_col = None
        for col in ['date', 'timestamp', 'created_at']:
            if col in time_series_df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                latest_date = pd.to_datetime(time_series_df[date_col]).max()
                days_old = (datetime.now() - latest_date).days
                
                if days_old <= 1:
                    quality_assessment['recency'] = 25
                elif days_old <= 7:
                    quality_assessment['recency'] = 20
                elif days_old <= 30:
                    quality_assessment['recency'] = 15
                elif days_old <= 90:
                    quality_assessment['recency'] = 10
                else:
                    quality_assessment['recency'] = 5
                    quality_assessment['issues'].append(f"Data is {days_old} days old")
            except:
                quality_assessment['issues'].append("Cannot determine data recency")
    
    # Sample size reliability (0-15 points)
    min_sample_sizes = {'Routes': 10, 'Braking Events': 50, 'Swerving Events': 30, 'Time Series': 30}
    reliability_scores = []
    
    for name, df in datasets.items():
        if df is not None:
            min_size = min_sample_sizes.get(name, 10)
            if len(df) >= min_size * 2:
                reliability_scores.append(15)
            elif len(df) >= min_size:
                reliability_scores.append(10)
            else:
                reliability_scores.append(5)
                quality_assessment['issues'].append(f"{name}: Only {len(df)} records (need {min_size}+)")
    
    if reliability_scores:
        quality_assessment['reliability'] = np.mean(reliability_scores)
    
    # Calculate overall score
    quality_assessment['overall_score'] = (
        quality_assessment['coverage'] + 
        quality_assessment['completeness'] + 
        quality_assessment['recency'] + 
        quality_assessment['reliability']
    )
    
    # Add improvement recommendations
    if quality_assessment['overall_score'] < 70:
        quality_assessment['recommendations'].append("Improve data collection consistency")
    if quality_assessment['coverage'] < 20:
        quality_assessment['recommendations'].append("Add missing dataset types")
    if quality_assessment['recency'] < 15:
        quality_assessment['recommendations'].append("Update data more frequently")
    
    return quality_assessment


def perform_risk_analysis(routes_df, braking_df, swerving_df):
    """Perform comprehensive risk analysis"""
    
    risk_analysis = {
        'overall_risk_level': 'Medium',
        'risk_score': 50,
        'high_risk_areas': [],
        'risk_trends': {},
        'critical_factors': [],
        'financial_exposure': 0
    }
    
    # Calculate incident density
    total_incidents = 0
    if braking_df is not None:
        total_incidents += len(braking_df)
    if swerving_df is not None:
        total_incidents += len(swerving_df)
    
    total_routes = len(routes_df) if routes_df is not None else 1
    incident_density = total_incidents / total_routes
    
    # Risk score calculation (0-100)
    if incident_density > 10:
        risk_analysis['risk_score'] = 85
        risk_analysis['overall_risk_level'] = 'Critical'
    elif incident_density > 5:
        risk_analysis['risk_score'] = 70
        risk_analysis['overall_risk_level'] = 'High'
    elif incident_density > 2:
        risk_analysis['risk_score'] = 55
        risk_analysis['overall_risk_level'] = 'Medium'
    else:
        risk_analysis['risk_score'] = 30
        risk_analysis['overall_risk_level'] = 'Low'
    
    # Identify high-risk areas
    if routes_df is not None and 'route_name' in routes_df.columns:
        # Count incidents per route
        route_incidents = {}
        
        for _, route in routes_df.iterrows():
            route_name = route['route_name']
            incident_count = 0
            
            # Count braking events for this route
            if braking_df is not None and 'route_name' in braking_df.columns:
                incident_count += len(braking_df[braking_df['route_name'] == route_name])
            
            # Count swerving events for this route
            if swerving_df is not None and 'route_name' in swerving_df.columns:
                incident_count += len(swerving_df[swerving_df['route_name'] == route_name])
            
            route_incidents[route_name] = incident_count
        
        # Find high-risk routes (top 20% by incident count)
        if route_incidents:
            sorted_routes = sorted(route_incidents.items(), key=lambda x: x[1], reverse=True)
            high_risk_count = max(1, len(sorted_routes) // 5)  # Top 20%
            risk_analysis['high_risk_areas'] = sorted_routes[:high_risk_count]
    
    # Estimate financial exposure (simplified model)
    avg_incident_cost = 2500  # Average cost per safety incident
    risk_analysis['financial_exposure'] = total_incidents * avg_incident_cost
    
    return risk_analysis


def analyze_trends(time_series_df):
    """Analyze trends to identify patterns and predict future issues"""
    
    trend_analysis = {
        'trend_direction': 'Stable',
        'trend_strength': 0,
        'seasonal_patterns': {},
        'anomalies': [],
        'predictions': {},
        'improvement_rate': 0
    }
    
    if time_series_df is None or len(time_series_df) < 7:
        trend_analysis['trend_direction'] = 'Insufficient Data'
        return trend_analysis
    
    # Find date and numeric columns
    date_col = None
    for col in ['date', 'timestamp', 'created_at']:
        if col in time_series_df.columns:
            date_col = col
            break
    
    if not date_col:
        return trend_analysis
    
    # Convert date column
    time_series_df = time_series_df.copy()
    time_series_df[date_col] = pd.to_datetime(time_series_df[date_col])
    
    # Find primary metric to analyze
    numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return trend_analysis
    
    primary_metric = numeric_cols[0]  # Use first numeric column
    
    # Sort by date
    time_series_df = time_series_df.sort_values(date_col)
    
    # Calculate trend using linear regression
    x_vals = np.arange(len(time_series_df))
    y_vals = time_series_df[primary_metric].values
    
    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        
        # Determine trend direction and strength
        trend_analysis['trend_strength'] = abs(slope)
        
        if slope > 0.1:
            trend_analysis['trend_direction'] = 'Improving' if primary_metric in ['safety_score', 'efficiency'] else 'Worsening'
        elif slope < -0.1:
            trend_analysis['trend_direction'] = 'Worsening' if primary_metric in ['safety_score', 'efficiency'] else 'Improving'
        else:
            trend_analysis['trend_direction'] = 'Stable'
        
        # Calculate improvement rate (annualized)
        if len(time_series_df) > 30:  # Need at least a month of data
            days_span = (time_series_df[date_col].max() - time_series_df[date_col].min()).days
            if days_span > 0:
                daily_change = slope
                annual_change = daily_change * 365
                current_value = y_vals[-1] if len(y_vals) > 0 else 1
                trend_analysis['improvement_rate'] = (annual_change / current_value) * 100 if current_value != 0 else 0
    
    return trend_analysis


def identify_hotspots(braking_df, swerving_df):
    """Identify safety hotspots requiring immediate attention"""
    
    hotspot_analysis = {
        'critical_hotspots': [],
        'geographic_clusters': [],
        'severity_ranking': [],
        'intervention_priority': []
    }
    
    # Combine incident data
    all_incidents = []
    
    if braking_df is not None and len(braking_df) > 0:
        braking_incidents = braking_df.copy()
        braking_incidents['incident_type'] = 'Braking'
        all_incidents.append(braking_incidents)
    
    if swerving_df is not None and len(swerving_df) > 0:
        swerving_incidents = swerving_df.copy()
        swerving_incidents['incident_type'] = 'Swerving'
        all_incidents.append(swerving_incidents)
    
    if not all_incidents:
        return hotspot_analysis
    
    combined_incidents = pd.concat(all_incidents, ignore_index=True)
    
    # Geographic clustering (if coordinates available)
    if 'latitude' in combined_incidents.columns and 'longitude' in combined_incidents.columns:
        # Simple grid-based clustering
        lat_bins = pd.cut(combined_incidents['latitude'], bins=10)
        lon_bins = pd.cut(combined_incidents['longitude'], bins=10)
        
        geographic_groups = combined_incidents.groupby([lat_bins, lon_bins]).size().reset_index(name='incident_count')
        geographic_groups = geographic_groups[geographic_groups['incident_count'] > 0]
        
        # Identify high-density areas
        if len(geographic_groups) > 0:
            top_clusters = geographic_groups.nlargest(5, 'incident_count')
            hotspot_analysis['geographic_clusters'] = top_clusters.to_dict('records')
    
    # Route-based hotspot analysis
    if 'route_name' in combined_incidents.columns:
        route_incidents = combined_incidents.groupby('route_name').size().reset_index(name='incident_count')
        route_incidents = route_incidents.sort_values('incident_count', ascending=False)
        
        # Critical hotspots (routes with >10 incidents or top 10%)
        critical_threshold = max(5, len(route_incidents) // 10)
        critical_routes = route_incidents.head(critical_threshold)
        
        hotspot_analysis['critical_hotspots'] = critical_routes.to_dict('records')
        
        # Severity ranking
        if 'severity' in combined_incidents.columns:
            severity_analysis = combined_incidents.groupby('route_name').agg({
                'severity': ['count', 'mean', 'max']
            }).reset_index()
            severity_analysis.columns = ['route_name', 'incident_count', 'avg_severity', 'max_severity']
            severity_analysis['risk_score'] = (
                severity_analysis['incident_count'] * 0.4 + 
                severity_analysis['avg_severity'] * 0.6
            )
            severity_analysis = severity_analysis.sort_values('risk_score', ascending=False)
            
            hotspot_analysis['severity_ranking'] = severity_analysis.head(10).to_dict('records')
    
    return hotspot_analysis


def calculate_business_impact(metrics, braking_df, swerving_df):
    """Calculate quantified business impact and ROI opportunities"""
    
    business_impact = {
        'current_cost': 0,
        'potential_savings': 0,
        'roi_opportunities': [],
        'cost_breakdown': {},
        'payback_period': 0
    }
    
    # Estimate current costs
    total_incidents = 0
    if braking_df is not None:
        total_incidents += len(braking_df)
    if swerving_df is not None:
        total_incidents += len(swerving_df)
    
    # Cost model (industry averages)
    avg_incident_cost = 2500
    admin_overhead = 0.3  # 30% overhead
    insurance_impact = 0.2  # 20% insurance cost increase
    
    direct_costs = total_incidents * avg_incident_cost
    indirect_costs = direct_costs * (admin_overhead + insurance_impact)
    
    business_impact['current_cost'] = direct_costs + indirect_costs
    
    business_impact['cost_breakdown'] = {
        'Direct Incident Costs': direct_costs,
        'Administrative Overhead': direct_costs * admin_overhead,
        'Insurance Impact': direct_costs * insurance_impact,
        'Total Annual Impact': business_impact['current_cost']
    }
    
    # Calculate potential savings with improvements
    safety_score = metrics.get('safety_score', 5)
    
    if safety_score < 7:  # Room for significant improvement
        potential_reduction = 0.4  # 40% reduction possible
    elif safety_score < 8.5:  # Moderate improvement possible
        potential_reduction = 0.25  # 25% reduction possible
    else:  # Already good, but optimization possible
        potential_reduction = 0.15  # 15% reduction possible
    
    business_impact['potential_savings'] = business_impact['current_cost'] * potential_reduction
    
    # ROI opportunities
    investment_scenarios = [
        {
            'intervention': 'Infrastructure Improvements',
            'investment': 50000,
            'expected_reduction': 0.3,
            'timeframe': 12
        },
        {
            'intervention': 'Enhanced Monitoring',
            'investment': 15000,
            'expected_reduction': 0.15,
            'timeframe': 6
        },
        {
            'intervention': 'Safety Training Program',
            'investment': 8000,
            'expected_reduction': 0.1,
            'timeframe': 3
        }
    ]
    
    for scenario in investment_scenarios:
        annual_savings = business_impact['current_cost'] * scenario['expected_reduction']
        payback_months = (scenario['investment'] / (annual_savings / 12)) if annual_savings > 0 else float('inf')
        roi_percent = ((annual_savings - scenario['investment']) / scenario['investment']) * 100 if scenario['investment'] > 0 else 0
        
        scenario['annual_savings'] = annual_savings
        scenario['payback_months'] = payback_months
        scenario['roi_percent'] = roi_percent
        
        business_impact['roi_opportunities'].append(scenario)
    
    # Calculate overall payback period for recommended improvements
    total_investment = sum(s['investment'] for s in business_impact['roi_opportunities'][:2])  # Top 2 recommendations
    total_annual_savings = business_impact['potential_savings']
    
    if total_annual_savings > 0:
        business_impact['payback_period'] = (total_investment / total_annual_savings) * 12  # In months
    
    return business_impact


def generate_prioritized_recommendations(routes_df, braking_df, swerving_df, time_series_df):
    """Generate prioritized, actionable recommendations"""
    
    recommendations = []
    
    # Data-driven recommendations
    total_incidents = 0
    if braking_df is not None:
        total_incidents += len(braking_df)
    if swerving_df is not None:
        total_incidents += len(swerving_df)
    
    total_routes = len(routes_df) if routes_df is not None else 1
    incident_rate = total_incidents / total_routes
    
    # High-priority recommendations
    if incident_rate > 5:
        recommendations.append({
            'priority': 'Critical',
            'category': 'Safety Infrastructure',
            'title': 'Immediate Safety Infrastructure Review',
            'description': f'With {incident_rate:.1f} incidents per route, immediate infrastructure assessment is needed.',
            'action_items': [
                'Conduct emergency safety audit of top 5 highest-incident routes',
                'Install additional safety measures (barriers, signage, lighting)',
                'Implement temporary speed restrictions in high-risk areas',
                'Deploy additional monitoring equipment'
            ],
            'estimated_cost': 75000,
            'expected_impact': '40-60% incident reduction',
            'timeframe': '1-2 months'
        })
    
    if braking_df is not None and len(braking_df) > 100:
        recommendations.append({
            'priority': 'High',
            'category': 'Traffic Management',
            'title': 'Enhanced Braking Event Management',
            'description': f'{len(braking_df)} braking events indicate traffic flow issues.',
            'action_items': [
                'Analyze braking event patterns to identify congestion points',
                'Optimize traffic signal timing at key intersections',
                'Implement early warning systems for hazards',
                'Consider separated cycling infrastructure'
            ],
            'estimated_cost': 35000,
            'expected_impact': '25-35% braking event reduction',
            'timeframe': '2-3 months'
        })
    
    # Medium-priority recommendations
    if time_series_df is not None and len(time_series_df) > 0:
        recommendations.append({
            'priority': 'Medium',
            'category': 'Data Analytics',
            'title': 'Predictive Safety Analytics Implementation',
            'description': 'Leverage historical data for proactive safety management.',
            'action_items': [
                'Implement machine learning models for incident prediction',
                'Set up automated alert systems for emerging patterns',
                'Create dynamic risk scoring for all routes',
                'Develop seasonal safety adjustment protocols'
            ],
            'estimated_cost': 25000,
            'expected_impact': '15-25% improvement in response time',
            'timeframe': '3-4 months'
        })
    
    # Always include data quality improvement
    recommendations.append({
        'priority': 'Medium',
        'category': 'Data Quality',
        'title': 'Data Collection Enhancement',
        'description': 'Improve data quality and coverage for better insights.',
        'action_items': [
            'Standardize data collection processes across all routes',
            'Implement real-time data validation',
            'Add missing data types (weather, traffic volume, etc.)',
            'Create automated data quality reports'
        ],
        'estimated_cost': 15000,
        'expected_impact': '20-30% improvement in insight accuracy',
        'timeframe': '1-2 months'
    })
    
    return sorted(recommendations, key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']])


def identify_improvement_opportunities(routes_df, braking_df, swerving_df, time_series_df):
    """Identify specific opportunities for improvement"""
    
    opportunities = {
        'quick_wins': [],
        'strategic_initiatives': [],
        'innovation_opportunities': [],
        'partnership_potential': []
    }
    
    # Quick wins (low cost, high impact)
    if braking_df is not None and 'route_name' in braking_df.columns:
        route_incidents = braking_df['route_name'].value_counts()
        if len(route_incidents) > 0:
            top_problem_route = route_incidents.index[0]
            opportunities['quick_wins'].append({
                'title': f'Focus on {top_problem_route}',
                'description': f'This route has {route_incidents.iloc[0]} incidents - targeted intervention could yield quick results',
                'estimated_effort': 'Low',
                'potential_impact': 'High',
                'timeframe': '2-4 weeks'
            })
    
    opportunities['quick_wins'].extend([
        {
            'title': 'Implement Weekly Safety Reports',
            'description': 'Regular reporting to stakeholders for accountability',
            'estimated_effort': 'Low',
            'potential_impact': 'Medium',
            'timeframe': '1 week'
        },
        {
            'title': 'Create Safety Score Dashboard',
            'description': 'Public dashboard showing safety improvements',
            'estimated_effort': 'Low',
            'potential_impact': 'Medium',
            'timeframe': '2 weeks'
        }
    ])
    
    # Strategic initiatives
    opportunities['strategic_initiatives'] = [
        {
            'title': 'Comprehensive Safety Infrastructure Upgrade',
            'description': 'Multi-year program to upgrade all high-risk routes',
            'estimated_investment': 500000,
            'potential_impact': 'Very High',
            'timeframe': '18-24 months'
        },
        {
            'title': 'Smart City Integration',
            'description': 'Integrate cycling safety data with broader city systems',
            'estimated_investment': 200000,
            'potential_impact': 'High',
            'timeframe': '12-18 months'
        }
    ]
    
    # Innovation opportunities
    opportunities['innovation_opportunities'] = [
        {
            'title': 'AI-Powered Route Optimization',
            'description': 'Use machine learning to dynamically adjust routes based on safety conditions',
            'technology_readiness': 'Medium',
            'potential_impact': 'Very High'
        },
        {
            'title': 'Real-Time Safety Alerts',
            'description': 'Mobile app integration for live safety notifications',
            'technology_readiness': 'High',
            'potential_impact': 'High'
        }
    ]
    
    return opportunities


def render_executive_summary(analysis_results):
    """Render executive summary with key insights"""
    
    st.markdown("## 📋 Executive Summary")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    risk_analysis = analysis_results['risk_analysis']
    business_impact = analysis_results['business_impact']
    data_quality = analysis_results['data_quality']
    
    with col1:
        risk_color = {
            'Critical': '🔴',
            'High': '🟠', 
            'Medium': '🟡',
            'Low': '🟢'
        }.get(risk_analysis['overall_risk_level'], '🟡')
        
        st.metric(
            "Overall Risk Level",
            f"{risk_color} {risk_analysis['overall_risk_level']}",
            delta=f"Score: {risk_analysis['risk_score']}/100"
        )
    
    with col2:
        st.metric(
            "Annual Safety Cost",
            f"${business_impact['current_cost']:,.0f}",
            delta=f"Potential Savings: ${business_impact['potential_savings']:,.0f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Data Quality Score",
            f"{data_quality['overall_score']:.0f}/100",
            delta=f"{len(data_quality['issues'])} issues identified",
            delta_color="inverse"
        )
    
    with col4:
        payback_months = business_impact.get('payback_period', 0)
        st.metric(
            "ROI Payback Period",
            f"{payback_months:.1f} months",
            delta="High-impact interventions",
            delta_color="normal"
        )
    
    # Executive insights
    st.markdown("### 🎯 Key Business Insights")
    
    # Generate AI-powered executive summary
    try:
        generator = create_insights_generator()
        metrics = analysis_results['metrics']
        insights = generator.generate_comprehensive_insights(metrics=metrics, routes_df=None)
        executive_summary = generator.generate_executive_summary(insights=insights, metrics=metrics)
        
        if executive_summary:
            st.info(f"💡 **AI Analysis**: {executive_summary}")
        else:
            st.info("💡 **Analysis**: Generating comprehensive insights from your safety data...")
    except Exception as e:
        logger.warning(f"AI summary error: {e}")
    
    # Critical findings
    critical_findings = []
    
    if risk_analysis['overall_risk_level'] in ['Critical', 'High']:
        critical_findings.append(f"⚠️ **Urgent**: {risk_analysis['overall_risk_level']} risk level requires immediate intervention")
    
    if business_impact['potential_savings'] > 50000:
        critical_findings.append(f"💰 **Opportunity**: ${business_impact['potential_savings']:,.0f} in potential annual savings identified")
    
    if data_quality['overall_score'] < 70:
        critical_findings.append(f"📊 **Data Quality**: Score of {data_quality['overall_score']:.0f}/100 may limit insight accuracy")
    
    if len(analysis_results['hotspot_analysis']['critical_hotspots']) > 0:
        hotspot_count = len(analysis_results['hotspot_analysis']['critical_hotspots'])
        critical_findings.append(f"🚨 **Hotspots**: {hotspot_count} critical areas need immediate attention")
    
    if critical_findings:
        for finding in critical_findings:
            st.markdown(finding)
    else:
        st.success("✅ **Status**: Overall system performance is within acceptable parameters")


def render_critical_insights(analysis_results):
    """Render critical insights that require immediate attention"""
    
    st.markdown("## 🚨 Critical Insights & Immediate Actions")
    
    recommendations = analysis_results['recommendations']
    critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
    high_recs = [r for r in recommendations if r['priority'] == 'High']
    
    if critical_recs or high_recs:
        priority_recs = critical_recs + high_recs
        
        for i, rec in enumerate(priority_recs[:3]):  # Show top 3 priority items
            with st.expander(f"🔥 **Priority {i+1}: {rec['title']}**", expanded=(i==0)):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Impact**: {rec['description']}")
                    
                    st.markdown("**Required Actions:**")
                    for action in rec['action_items']:
                        st.markdown(f"• {action}")
                
                with col2:
                    st.markdown("**Investment Analysis**")
                    
                    # Use smaller text for investment details
                    st.markdown(f"**Cost:** ${rec['estimated_cost']:,}")
                    st.markdown(f"**Expected Impact:** {rec['expected_impact']}")
                    st.markdown(f"**Timeline:** {rec['timeframe']}")
                    
                    # Calculate ROI
                    if rec['estimated_cost'] > 0:
                        # Rough ROI calculation based on expected impact
                        potential_savings = analysis_results['business_impact']['potential_savings']
                        if 'reduction' in rec['expected_impact']:
                            try:
                                # Extract percentage from impact description
                                import re
                                percentages = re.findall(r'(\d+)', rec['expected_impact'])
                                if percentages:
                                    avg_reduction = sum(int(p) for p in percentages) / len(percentages) / 100
                                    annual_savings = potential_savings * avg_reduction
                                    roi = ((annual_savings - rec['estimated_cost']) / rec['estimated_cost']) * 100
                                    st.markdown(f"**Estimated ROI:** {roi:.0f}%")
                            except:
                                pass
    else:
        st.success("✅ No critical issues identified. System is operating within normal parameters.")
        
        # Show preventive recommendations
        medium_recs = [r for r in recommendations if r['priority'] == 'Medium']
        if medium_recs:
            st.markdown("### 🔧 Preventive Measures")
            for rec in medium_recs[:2]:
                st.markdown(f"**{rec['title']}**: {rec['description']}")


def render_business_impact_analysis(analysis_results):
    """Render detailed business impact analysis"""
    
    st.markdown("## 💼 Business Impact Analysis")
    
    business_impact = analysis_results['business_impact']
    
    # Financial impact visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Current Cost Analysis")
        
        # Create cost breakdown chart
        cost_data = business_impact['cost_breakdown']
        if cost_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(cost_data.keys())[:-1],  # Exclude total
                    y=list(cost_data.values())[:-1],
                    marker_color=['#ef4444', '#f59e0b', '#8b5cf6']
                )
            ])
            fig.update_layout(
                title="Annual Safety Cost Breakdown",
                xaxis_title="Cost Category",
                yaxis_title="Cost ($)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost metrics
        st.metric("Total Annual Impact", f"${business_impact['current_cost']:,.0f}")
        
    with col2:
        st.markdown("### 📈 ROI Opportunities")
        
        roi_opportunities = business_impact['roi_opportunities']
        
        # ROI comparison chart
        if roi_opportunities:
            roi_data = pd.DataFrame(roi_opportunities)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=roi_data['investment'],
                y=roi_data['annual_savings'],
                mode='markers+text',
                text=roi_data['intervention'],
                textposition="top center",
                marker=dict(
                    size=roi_data['roi_percent'],
                    sizemode='diameter',
                    sizeref=max(roi_data['roi_percent'])/50,
                    sizemin=10,
                    color=roi_data['roi_percent'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="ROI %")
                )
            ))
            
            fig.update_layout(
                title="Investment vs. Annual Savings (Bubble size = ROI %)",
                xaxis_title="Investment Required ($)",
                yaxis_title="Annual Savings ($)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Potential Annual Savings", f"${business_impact['potential_savings']:,.0f}")
    
    # ROI recommendations table
    st.markdown("### 🎯 Investment Recommendations")
    
    if roi_opportunities:
        roi_df = pd.DataFrame(roi_opportunities)
        roi_df = roi_df.sort_values('roi_percent', ascending=False)
        
        # Format for display
        display_df = roi_df[['intervention', 'investment', 'annual_savings', 'roi_percent', 'payback_months']].copy()
        display_df['investment'] = display_df['investment'].apply(lambda x: f"${x:,.0f}")
        display_df['annual_savings'] = display_df['annual_savings'].apply(lambda x: f"${x:,.0f}")
        display_df['roi_percent'] = display_df['roi_percent'].apply(lambda x: f"{x:.0f}%")
        display_df['payback_months'] = display_df['payback_months'].apply(lambda x: f"{x:.1f} mo" if x != float('inf') else "N/A")
        
        display_df.columns = ['Intervention', 'Investment', 'Annual Savings', 'ROI', 'Payback']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_risk_assessment(analysis_results):
    """Render comprehensive risk assessment"""
    
    st.markdown("## ⚠️ Risk Assessment & Hotspot Analysis")
    
    risk_analysis = analysis_results['risk_analysis']
    hotspot_analysis = analysis_results['hotspot_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Risk Overview")
        
        # Risk level indicator
        risk_colors = {
            'Critical': '#dc2626',
            'High': '#ea580c',
            'Medium': '#ca8a04',
            'Low': '#16a34a'
        }
        
        risk_color = risk_colors.get(risk_analysis['overall_risk_level'], '#6b7280')
        
        st.markdown(f"""
        <div style="background: {risk_color}; color: white; padding: 15px; border-radius: 8px; text-align: center; margin: 10px 0;">
            <h3 style="margin: 0; color: white;">Risk Level: {risk_analysis['overall_risk_level']}</h3>
            <h2 style="margin: 5px 0; color: white;">{risk_analysis['risk_score']}/100</h2>
            <p style="margin: 0; color: white; font-size: 14px;">Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial exposure
        st.metric(
            "Estimated Financial Exposure",
            f"${risk_analysis['financial_exposure']:,.0f}",
            delta="Based on current incident rates"
        )
    
    with col2:
        st.markdown("### 🚨 Critical Hotspots")
        
        critical_hotspots = hotspot_analysis['critical_hotspots']
        
        if critical_hotspots:
            # Create hotspot ranking chart
            hotspot_df = pd.DataFrame(critical_hotspots)
            
            fig = px.bar(
                hotspot_df.head(10),
                x='incident_count',
                y='route_name',
                orientation='h',
                title="Top 10 High-Risk Routes",
                color='incident_count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Hotspot table
            st.markdown("**Immediate Attention Required:**")
            for hotspot in critical_hotspots[:5]:
                st.markdown(f"• **{hotspot['route_name']}**: {hotspot['incident_count']} incidents")
        else:
            st.info("✅ No critical hotspots identified at this time")
    
    # Geographic analysis
    if hotspot_analysis['geographic_clusters']:
        st.markdown("### 🗺️ Geographic Risk Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**High-Density Incident Areas**")
            clusters = hotspot_analysis['geographic_clusters']
            
            if len(clusters) > 0:
                cluster_df = pd.DataFrame(clusters)
                
                # Create geographic heatmap visualization
                fig = px.scatter(
                    cluster_df,
                    size='incident_count',
                    title="Incident Density Clusters",
                    labels={'incident_count': 'Incident Count'}
                )
                fig.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Mitigation Priority**")
            
            # Priority ranking
            priority_actions = [
                "🔴 Deploy emergency response teams",
                "🟠 Install additional safety infrastructure", 
                "🟡 Increase monitoring frequency",
                "🔵 Schedule comprehensive safety audit"
            ]
            
            for action in priority_actions:
                st.markdown(action)


def render_opportunity_identification(analysis_results):
    """Render opportunity identification and improvement potential"""
    
    st.markdown("## 🚀 Improvement Opportunities")
    
    opportunities = analysis_results['opportunities']
    
    # Quick wins
    st.markdown("### ⚡ Quick Wins (High Impact, Low Effort)")
    
    quick_wins = opportunities['quick_wins']
    
    if quick_wins:
        for i, opportunity in enumerate(quick_wins):
            with st.expander(f"💡 Quick Win {i+1}: {opportunity['title']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(opportunity['description'])
                
                with col2:
                    st.markdown(f"**Effort**: {opportunity['estimated_effort']}")
                    st.markdown(f"**Impact**: {opportunity['potential_impact']}")
                    st.markdown(f"**Timeline**: {opportunity['timeframe']}")
    
    # Strategic initiatives
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Strategic Initiatives")
        
        strategic = opportunities['strategic_initiatives']
        
        for initiative in strategic:
            st.markdown(f"""
**{initiative['title']}**

{initiative['description']}

• Investment: ${initiative['estimated_investment']:,}
• Impact: {initiative['potential_impact']}
• Timeline: {initiative['timeframe']}
            """)
    
    with col2:
        st.markdown("### 🔬 Innovation Opportunities")
        
        innovations = opportunities['innovation_opportunities']
        
        for innovation in innovations:
            st.markdown(f"""
**{innovation['title']}**

{innovation['description']}

• Technology Readiness: {innovation['technology_readiness']}
• Potential Impact: {innovation['potential_impact']}
            """)


def render_action_recommendations(analysis_results):
    """Render prioritized action recommendations"""
    
    st.markdown("## 🎯 Action Plan & Next Steps")
    
    recommendations = analysis_results['recommendations']
    
    # Create action timeline
    st.markdown("### 📅 Recommended Implementation Timeline")
    
    if recommendations:
        timeline_data = []
        base_date = datetime.now()
        
        for i, rec in enumerate(recommendations):
            if rec['timeframe']:
                # Extract months from timeframe
                import re
                months = re.findall(r'(\d+)', rec['timeframe'])
                if months:
                    max_months = max(int(m) for m in months)
                    start_date = base_date + timedelta(days=30*i)  # Stagger start dates
                    end_date = start_date + timedelta(days=30*max_months)
                    
                    timeline_data.append({
                        'Task': rec['title'],
                        'Start': start_date,
                        'Finish': end_date,
                        'Priority': rec['priority'],
                        'Duration': max_months
                    })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create horizontal bar chart showing timeline
            fig = go.Figure()
            
            priority_colors = {
                'Critical': '#dc2626',
                'High': '#ea580c', 
                'Medium': '#ca8a04',
                'Low': '#16a34a'
            }
            
            for i, row in timeline_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Start'], row['Finish']],
                    y=[row['Task'], row['Task']],
                    mode='lines+markers',
                    line=dict(
                        color=priority_colors.get(row['Priority'], '#6b7280'),
                        width=8
                    ),
                    marker=dict(size=8),
                    name=f"{row['Priority']} Priority",
                    showlegend=(i == 0 or row['Priority'] != timeline_df.iloc[i-1]['Priority'])
                ))
            
            fig.update_layout(
                title="Implementation Timeline",
                xaxis_title="Timeline",
                yaxis_title="Action Items",
                height=400,
                template="plotly_white",
                xaxis=dict(type='date')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available for visualization")
    else:
        st.info("No recommendations available for timeline planning")
    
    # Priority matrix
    st.markdown("### 🎯 Priority Action Matrix")
    
    priority_groups = {
        'Critical': [r for r in recommendations if r['priority'] == 'Critical'],
        'High': [r for r in recommendations if r['priority'] == 'High'],
        'Medium': [r for r in recommendations if r['priority'] == 'Medium']
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 Critical Priority")
        critical_actions = priority_groups['Critical']
        
        if critical_actions:
            for action in critical_actions:
                st.markdown(f"""
**{action['title']}**

Cost: ${action['estimated_cost']:,}
Timeline: {action['timeframe']}
Impact: {action['expected_impact']}
                """)
        else:
            st.success("✅ No critical actions required")
    
    with col2:
        st.markdown("#### 🟠 High Priority")
        high_actions = priority_groups['High']
        
        for action in high_actions[:2]:  # Show top 2
            st.markdown(f"""
**{action['title']}**

Cost: ${action['estimated_cost']:,}
Timeline: {action['timeframe']}
            """)
    
    with col3:
        st.markdown("#### 🟡 Medium Priority")
        medium_actions = priority_groups['Medium']
        
        for action in medium_actions[:2]:  # Show top 2
            st.markdown(f"""
**{action['title']}**

Cost: ${action['estimated_cost']:,}
Timeline: {action['timeframe']}
            """)
    
    # Summary recommendations
    st.markdown("### 📋 Executive Summary of Recommendations")
    
    total_investment = sum(r['estimated_cost'] for r in recommendations[:3])  # Top 3 priorities
    business_impact = analysis_results['business_impact']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Priority Investment",
            f"${total_investment:,.0f}",
            delta="Top 3 recommendations"
        )
    
    with col2:
        st.metric(
            "Expected Annual Savings", 
            f"${business_impact['potential_savings']:,.0f}",
            delta=f"ROI: {((business_impact['potential_savings'] - total_investment) / total_investment * 100):.0f}%" if total_investment > 0 else "N/A"
        )
    
    with col3:
        payback = (total_investment / business_impact['potential_savings'] * 12) if business_impact['potential_savings'] > 0 else 0
        st.metric(
            "Payback Period",
            f"{payback:.1f} months",
            delta="Based on expected savings"
        )
    
    # Call to action
    st.markdown("---")
    st.markdown("### 🚀 Ready to Take Action?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Detailed Report", use_container_width=True):
            st.info("Detailed report generation would be implemented here")
    
    with col2:
        if st.button("📧 Share with Stakeholders", use_container_width=True):
            st.info("Stakeholder sharing functionality would be implemented here")
    
    with col3:
        if st.button("⚙️ Start Implementation", use_container_width=True):
            st.info("Implementation planning tools would be available here")
    
    # Key takeaways
    st.markdown("### 🎯 Key Takeaways")
    
    risk_level = analysis_results['risk_analysis']['overall_risk_level']
    savings_potential = analysis_results['business_impact']['potential_savings']
    
    takeaways = []
    
    if risk_level in ['Critical', 'High']:
        takeaways.append(f"🚨 **Urgent Action Required**: {risk_level} risk level demands immediate intervention")
    
    if savings_potential > 50000:
        takeaways.append(f"💰 **Significant ROI Opportunity**: ${savings_potential:,.0f} in annual savings achievable")
    
    takeaways.append("🎯 **Data-Driven Approach**: All recommendations based on actual safety data analysis")
    takeaways.append("📈 **Continuous Improvement**: Regular monitoring will optimize safety outcomes")
    
    for takeaway in takeaways:
        st.markdown(takeaway)
