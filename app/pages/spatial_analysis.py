"""
Spatial Analysis Page for SeeSense Dashboard - USER-FRIENDLY VERSION
Simplified geospatial analysis with AI-generated insights for non-technical users
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging

# Optional imports with fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_spatial_analysis_page():
    """Render the user-friendly spatial analysis page"""
    st.title("🗺️ Where Are The Safety Issues?")
    st.markdown("**Discover the safest and riskiest places to cycle in your area**")
    
    # Show active filters
    show_active_filters()
    
    # Add helpful introduction
    render_intro_section()
    
    try:
        # Load all datasets
        all_data = data_processor.load_all_datasets()
        
        # Check if we have any data
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_spatial_data_message()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Add spatial analysis controls in sidebar
        spatial_options = render_user_friendly_controls()
        
        # Create user-friendly tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Hotspot Map", 
            "📊 Safety Zones", 
            "🛣️ Route Safety", 
            "💡 Smart Tips"
        ])
        
        with tab1:
            render_hotspot_discovery(braking_df, swerving_df, routes_df, spatial_options)
        
        with tab2:
            render_safety_zones(braking_df, swerving_df, spatial_options)
        
        with tab3:
            render_route_safety_analysis(routes_df, braking_df, swerving_df, spatial_options)
        
        with tab4:
            render_smart_recommendations(routes_df, braking_df, swerving_df, spatial_options)
        
    except Exception as e:
        logger.error(f"Error in spatial analysis page: {e}")
        st.error("⚠️ Something went wrong loading the map data.")
        st.info("Please check your data files and try refreshing the page.")


def show_active_filters():
    """Show current date filter status"""
    if 'date_filter' in st.session_state and st.session_state.date_filter:
        start_date, end_date = st.session_state.date_filter
        st.info(f"📅 **Viewing data from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}** | Change dates in Overview page")
    else:
        st.info("📅 **Viewing all available data** | Set date filters in the Overview page to focus on specific time periods")


def render_intro_section():
    """Render helpful introduction for non-tech users"""
    with st.expander("🎯 What am I looking at?", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;'>
        <h4 style='color: white; margin-top: 0;'>🗺️ Your Personal Safety Map</h4>
        We analyze where cyclists have to brake hard or swerve suddenly - these are signs of dangerous situations!
        </div>
        
        **What the colors mean:**
        - 🔴 **Red areas** = More dangerous (lots of sudden braking/swerving)
        - 🟡 **Yellow areas** = Moderate risk  
        - 🟢 **Green areas** = Safer cycling conditions
        - 📍 **Dots** = Exact locations where incidents happened
        
        **How to use this:**
        1. **Hotspot Map** - See dangerous areas on an interactive map
        2. **Safety Zones** - Find the safest and riskiest neighborhoods  
        3. **Route Safety** - Compare different cycling routes
        4. **Smart Tips** - Get personalized safety recommendations
        """)


def render_user_friendly_controls():
    """Render simplified, user-friendly controls"""
    st.sidebar.markdown("### 🎛️ Map Settings")
    
    options = {}
    
    # Simplified view options
    st.sidebar.markdown("**What to show:**")
    options['show_braking'] = st.sidebar.checkbox("🛑 Hard Braking Spots", value=True)
    options['show_swerving'] = st.sidebar.checkbox("↩️ Sudden Swerving Spots", value=True)
    options['show_routes'] = st.sidebar.checkbox("🛣️ Popular Routes", value=False)
    
    # Simplified sensitivity
    st.sidebar.markdown("**Map Detail:**")
    sensitivity = st.sidebar.radio(
        "How sensitive should the analysis be?",
        ["Show only major issues", "Balanced view", "Show all incidents"],
        index=1,
        help="Higher sensitivity shows more incidents but may include minor issues"
    )
    
    # Convert to technical parameters
    if sensitivity == "Show only major issues":
        options['cluster_eps'] = 300
        options['min_samples'] = 8
        options['density_radius'] = 600
    elif sensitivity == "Balanced view":
        options['cluster_eps'] = 200
        options['min_samples'] = 5
        options['density_radius'] = 400
    else:  # Show all incidents
        options['cluster_eps'] = 100
        options['min_samples'] = 3
        options['density_radius'] = 200
    
    return options


def render_hotspot_discovery(braking_df, swerving_df, routes_df, spatial_options):
    """Render user-friendly hotspot map with AI insights"""
    st.markdown("### 🎯 Danger Hotspot Map")
    st.markdown("**Interactive map showing where cycling gets risky**")
    
    # Check if we have data
    has_braking = braking_df is not None and len(braking_df) > 0
    has_swerving = swerving_df is not None and len(swerving_df) > 0
    
    if not (has_braking or has_swerving):
        st.warning("📍 No safety incident data available for mapping")
        st.info("Upload braking and swerving hotspot data to see dangerous areas on the map")
        return
    
    # Create the main map
    render_main_safety_map(braking_df, swerving_df, routes_df, spatial_options)
    
    # AI-Generated Insights Section
    render_ai_insights_hotspots(braking_df, swerving_df, spatial_options)


def render_main_safety_map(braking_df, swerving_df, routes_df, spatial_options):
    """Create the main interactive safety map"""
    
    # Combine all data for map center
    all_lats = []
    all_lons = []
    
    if braking_df is not None and len(braking_df) > 0:
        all_lats.extend(braking_df['lat'].tolist())
        all_lons.extend(braking_df['lon'].tolist())
    
    if swerving_df is not None and len(swerving_df) > 0:
        all_lats.extend(swerving_df['lat'].tolist())
        all_lons.extend(swerving_df['lon'].tolist())
    
    if not all_lats:
        st.error("No location data found")
        return
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # Create Plotly map
    fig = go.Figure()
    
    # Add braking incidents if enabled and available
    if spatial_options['show_braking'] and braking_df is not None and len(braking_df) > 0:
        hover_text = []
        for _, row in braking_df.iterrows():
            hover_text.append(
                f"🛑 Hard Braking Spot<br>" +
                f"Incidents: {row.get('incidents_count', 'N/A')}<br>" +
                f"Risk Level: {get_risk_level(row.get('intensity', 0))}<br>" +
                f"Location: {row.get('hotspot_id', 'Unknown')}"
            )
        
        fig.add_trace(go.Scattermapbox(
            lat=braking_df['lat'],
            lon=braking_df['lon'],
            mode='markers',
            marker=dict(
                size=braking_df.get('intensity', [10]*len(braking_df)) * 2 + 10,
                color='red',
                opacity=0.7,
                sizemode='diameter'
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='🛑 Hard Braking',
            showlegend=True
        ))
    
    # Add swerving incidents if enabled and available  
    if spatial_options['show_swerving'] and swerving_df is not None and len(swerving_df) > 0:
        hover_text = []
        for _, row in swerving_df.iterrows():
            hover_text.append(
                f"↩️ Sudden Swerving Spot<br>" +
                f"Incidents: {row.get('incidents_count', 'N/A')}<br>" +
                f"Risk Level: {get_risk_level(row.get('intensity', 0))}<br>" +
                f"Location: {row.get('hotspot_id', 'Unknown')}"
            )
        
        fig.add_trace(go.Scattermapbox(
            lat=swerving_df['lat'],
            lon=swerving_df['lon'],
            mode='markers',
            marker=dict(
                size=swerving_df.get('intensity', [8]*len(swerving_df)) * 2 + 8,
                color='orange',
                opacity=0.7,
                sizemode='diameter'
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='↩️ Sudden Swerving',
            showlegend=True
        ))
    
    # Add popular routes if enabled
    if spatial_options['show_routes'] and routes_df is not None and len(routes_df) > 0:
        # Add route start points
        fig.add_trace(go.Scattermapbox(
            lat=routes_df['start_lat'],
            lon=routes_df['start_lon'],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                opacity=0.5
            ),
            name='🛣️ Route Starts',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_ai_insights_hotspots(braking_df, swerving_df, spatial_options):
    """Generate and display AI insights for hotspots"""
    st.markdown("### 🤖 AI Safety Insights")
    
    # Generate insights based on actual data
    insights = generate_hotspot_insights(braking_df, swerving_df)
    
    # Display insights in cards
    col1, col2 = st.columns(2)
    
    with col1:
        render_insight_card(
            "🎯 Most Dangerous Area", 
            insights['most_dangerous'],
            "high"
        )
        
        render_insight_card(
            "📊 Safety Pattern", 
            insights['pattern'],
            "medium"
        )
    
    with col2:
        render_insight_card(
            "⚠️ Key Finding", 
            insights['key_finding'],
            "high"
        )
        
        render_insight_card(
            "💡 Recommendation", 
            insights['recommendation'],
            "low"
        )


def render_safety_zones(braking_df, swerving_df, spatial_options):
    """Render safety zone analysis"""
    st.markdown("### 📊 Safety Zone Analysis")
    st.markdown("**Which areas are safest and which should you avoid?**")
    
    # Check if we have data
    has_data = (braking_df is not None and len(braking_df) > 0) or (swerving_df is not None and len(swerving_df) > 0)
    
    if not has_data:
        st.warning("📊 No safety data available for zone analysis")
        return
    
    # Perform simplified clustering analysis
    safety_zones = analyze_safety_zones(braking_df, swerving_df, spatial_options)
    
    if safety_zones:
        # Display zone map
        render_safety_zone_map(safety_zones)
        
        # Display zone statistics
        render_zone_statistics(safety_zones)
        
        # AI insights for zones
        render_ai_insights_zones(safety_zones)
    else:
        st.info("⏳ Analyzing safety zones... Please wait")


def render_route_safety_analysis(routes_df, braking_df, swerving_df, spatial_options):
    """Render route safety analysis"""
    st.markdown("### 🛣️ Route Safety Comparison")
    st.markdown("**Compare the safety of different cycling routes**")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("🛣️ No route data available for safety analysis")
        st.info("Upload route data to compare the safety of different cycling paths")
        return
    
    # Analyze route safety
    route_analysis = analyze_route_safety_simplified(routes_df, braking_df, swerving_df)
    
    if route_analysis:
        # Display route comparison
        render_route_comparison(route_analysis)
        
        # AI insights for routes
        render_ai_insights_routes(route_analysis)
    else:
        st.info("⏳ Analyzing route safety... Please wait")


def render_smart_recommendations(routes_df, braking_df, swerving_df, spatial_options):
    """Render smart safety recommendations"""
    st.markdown("### 💡 Your Personal Safety Tips")
    st.markdown("**AI-powered recommendations to keep you safer**")
    
    # Generate comprehensive recommendations
    recommendations = generate_smart_recommendations(routes_df, braking_df, swerving_df, spatial_options)
    
    # Display recommendations in sections
    st.markdown("#### 🎯 Based on Your Data")
    for i, rec in enumerate(recommendations['personal'], 1):
        render_recommendation_card(f"{i}. {rec['title']}", rec['description'])
    
    st.markdown("#### 🏙️ Area-Specific Tips")
    for i, rec in enumerate(recommendations['area_specific'], 1):
        render_recommendation_card(f"{i}. {rec['title']}", rec['description'])
    
    st.markdown("#### 🚴 General Safety Tips")
    for i, rec in enumerate(recommendations['general'], 1):
        render_recommendation_card(f"{i}. {rec['title']}", rec['description'])


# Helper Functions

def get_risk_level(intensity):
    """Convert intensity to human-readable risk level"""
    if intensity > 7:
        return "🔴 Very High"
    elif intensity > 5:
        return "🟠 High"
    elif intensity > 3:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def render_insight_card(title, content, priority="medium"):
    """Render an AI insight card with styling"""
    if priority == "high":
        color = "#ff6b6b"
        icon = "🚨"
    elif priority == "low":
        color = "#51cf66"
        icon = "💡"
    else:
        color = "#ffd43b"
        icon = "📊"
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                padding: 15px; border-radius: 8px; margin: 10px 0; 
                border-left: 4px solid {color};'>
        <h5 style='margin: 0 0 8px 0; color: {color};'>{icon} {title}</h5>
        <p style='margin: 0; color: #333;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(title, description):
    """Render a recommendation card"""
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;'>
        <h6 style='margin: 0 0 8px 0; color: #28a745;'>{title}</h6>
        <p style='margin: 0; color: #333; font-size: 14px;'>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def generate_hotspot_insights(braking_df, swerving_df):
    """Generate AI insights based on hotspot data"""
    insights = {}
    
    # Count incidents
    braking_count = len(braking_df) if braking_df is not None else 0
    swerving_count = len(swerving_df) if swerving_df is not None else 0
    total_count = braking_count + swerving_count
    
    # Most dangerous area insight
    if braking_count > swerving_count:
        insights['most_dangerous'] = f"Hard braking incidents ({braking_count}) are more common than swerving ({swerving_count}). This suggests traffic-related safety issues."
    elif swerving_count > braking_count:
        insights['most_dangerous'] = f"Sudden swerving incidents ({swerving_count}) outnumber hard braking ({braking_count}). This indicates obstacle avoidance or road condition issues."
    else:
        insights['most_dangerous'] = f"Braking and swerving incidents are equally common ({braking_count} each). This suggests varied safety challenges."
    
    # Pattern analysis
    if total_count > 50:
        insights['pattern'] = "High incident density detected. Consider focusing on the most dangerous hotspots first."
    elif total_count > 20:
        insights['pattern'] = "Moderate incident spread across your area. Most routes have some level of risk."
    else:
        insights['pattern'] = "Low incident density. Your cycling area is relatively safe overall."
    
    # Key finding
    if braking_df is not None and 'intensity' in braking_df.columns:
        max_intensity = braking_df['intensity'].max()
        if max_intensity > 8:
            insights['key_finding'] = "Some locations have extremely high incident rates. These should be avoided if possible."
        else:
            insights['key_finding'] = "Incident intensities are moderate. Careful riding can help avoid most risks."
    else:
        insights['key_finding'] = "Safety incidents are present but appear manageable with awareness."
    
    # Recommendation
    if braking_count > swerving_count * 2:
        insights['recommendation'] = "Focus on traffic awareness - leave more following distance and watch for sudden stops."
    elif swerving_count > braking_count * 2:
        insights['recommendation'] = "Watch for road hazards - potholes, parked cars, and debris seem to be major issues."
    else:
        insights['recommendation'] = "Stay alert for both traffic and road hazards. Consider alternative routes through safer areas."
    
    return insights


def analyze_safety_zones(braking_df, swerving_df, spatial_options):
    """Analyze and categorize safety zones"""
    # Simplified safety zone analysis
    zones = []
    
    # Combine data
    all_incidents = []
    if braking_df is not None:
        for _, row in braking_df.iterrows():
            all_incidents.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'intensity': row.get('intensity', 1),
                'type': 'braking'
            })
    
    if swerving_df is not None:
        for _, row in swerving_df.iterrows():
            all_incidents.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'intensity': row.get('intensity', 1),
                'type': 'swerving'
            })
    
    if not all_incidents:
        return None
    
    # Simple grid-based analysis
    incidents_df = pd.DataFrame(all_incidents)
    
    # Create zones based on intensity
    high_risk = incidents_df[incidents_df['intensity'] > 6]
    medium_risk = incidents_df[(incidents_df['intensity'] > 3) & (incidents_df['intensity'] <= 6)]
    low_risk = incidents_df[incidents_df['intensity'] <= 3]
    
    return {
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'total_incidents': len(incidents_df)
    }


def render_safety_zone_map(safety_zones):
    """Render safety zone visualization"""
    fig = go.Figure()
    
    # Add high risk zones
    if len(safety_zones['high_risk']) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=safety_zones['high_risk']['lat'],
            lon=safety_zones['high_risk']['lon'],
            mode='markers',
            marker=dict(size=15, color='red', opacity=0.7),
            name='🔴 High Risk Zones',
            text=['High Risk Area'] * len(safety_zones['high_risk'])
        ))
    
    # Add medium risk zones
    if len(safety_zones['medium_risk']) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=safety_zones['medium_risk']['lat'],
            lon=safety_zones['medium_risk']['lon'],
            mode='markers',
            marker=dict(size=12, color='orange', opacity=0.6),
            name='🟡 Medium Risk Zones',
            text=['Medium Risk Area'] * len(safety_zones['medium_risk'])
        ))
    
    # Add low risk zones
    if len(safety_zones['low_risk']) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=safety_zones['low_risk']['lat'],
            lon=safety_zones['low_risk']['lon'],
            mode='markers',
            marker=dict(size=8, color='green', opacity=0.5),
            name='🟢 Lower Risk Zones',
            text=['Lower Risk Area'] * len(safety_zones['low_risk'])
        ))
    
    # Set map center
    all_lats = safety_zones['high_risk']['lat'].tolist() + safety_zones['medium_risk']['lat'].tolist() + safety_zones['low_risk']['lat'].tolist()
    all_lons = safety_zones['high_risk']['lon'].tolist() + safety_zones['medium_risk']['lon'].tolist() + safety_zones['low_risk']['lon'].tolist()
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=np.mean(all_lats), lon=np.mean(all_lons)),
            zoom=11
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_zone_statistics(safety_zones):
    """Display zone statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 High Risk", len(safety_zones['high_risk']))
    with col2:
        st.metric("🟡 Medium Risk", len(safety_zones['medium_risk']))
    with col3:
        st.metric("🟢 Lower Risk", len(safety_zones['low_risk']))
    with col4:
        st.metric("📊 Total Areas", safety_zones['total_incidents'])


def render_ai_insights_zones(safety_zones):
    """Generate AI insights for safety zones"""
    st.markdown("#### 🤖 Zone Analysis")
    
    high_count = len(safety_zones['high_risk'])
    medium_count = len(safety_zones['medium_risk'])
    low_count = len(safety_zones['low_risk'])
    total = high_count + medium_count + low_count
    
    if high_count > total * 0.3:
        insight = "⚠️ **High concentration of dangerous areas.** Consider using alternative routes or extra caution in red zones."
    elif high_count < total * 0.1:
        insight = "✅ **Most areas are relatively safe.** Your cycling environment has manageable risks overall."
    else:
        insight = "🎯 **Mixed safety landscape.** Plan routes to avoid red zones when possible."
    
    st.info(insight)


def analyze_route_safety_simplified(routes_df, braking_df, swerving_df):
    """Simplified route safety analysis"""
    if routes_df is None or len(routes_df) == 0:
        return None
    
    route_analysis = []
    
    for _, route in routes_df.iterrows():
        safety_score = calculate_route_safety_score(route, braking_df, swerving_df)
        route_analysis.append({
            'route_id': route.get('route_id', f"Route {len(route_analysis) + 1}"),
            'start_lat': route['start_lat'],
            'start_lon': route['start_lon'],
            'end_lat': route['end_lat'],
            'end_lon': route['end_lon'],
            'safety_score': safety_score,
            'risk_level': get_route_risk_level(safety_score)
        })
    
    return route_analysis


def calculate_route_safety_score(route, braking_df, swerving_df):
    """Calculate a simple safety score for a route"""
    # Simplified distance-based scoring
    risk_points = 0
    
    if braking_df is not None:
        for _, incident in braking_df.iterrows():
            dist_to_start = abs(route['start_lat'] - incident['lat']) + abs(route['start_lon'] - incident['lon'])
            dist_to_end = abs(route['end_lat'] - incident['lat']) + abs(route['end_lon'] - incident['lon'])
            min_dist = min(dist_to_start, dist_to_end)
            
            if min_dist < 0.01:  # Very close
                risk_points += incident.get('intensity', 1) * 2
            elif min_dist < 0.02:  # Moderately close
                risk_points += incident.get('intensity', 1)
    
    if swerving_df is not None:
        for _, incident in swerving_df.iterrows():
            dist_to_start = abs(route['start_lat'] - incident['lat']) + abs(route['start_lon'] - incident['lon'])
            dist_to_end = abs(route['end_lat'] - incident['lat']) + abs(route['end_lon'] - incident['lon'])
            min_dist = min(dist_to_start, dist_to_end)
            
            if min_dist < 0.01:
                risk_points += incident.get('intensity', 1) * 1.5
            elif min_dist < 0.02:
                risk_points += incident.get('intensity', 1) * 0.75
    
    # Convert to 0-100 safety score (higher = safer)
    safety_score = max(0, 100 - risk_points * 5)
    return round(safety_score, 1)


def get_route_risk_level(safety_score):
    """Convert safety score to risk level"""
    if safety_score >= 80:
        return "🟢 Low Risk"
    elif safety_score >= 60:
        return "🟡 Medium Risk"
    elif safety_score >= 40:
        return "🟠 High Risk"
    else:
        return "🔴 Very High Risk"


def render_route_comparison(route_analysis):
    """Render route comparison visualization"""
    if not route_analysis:
        return
    
    routes_df = pd.DataFrame(route_analysis)
    
    # Sort by safety score
    routes_df = routes_df.sort_values('safety_score', ascending=False)
    
    # Display top 5 safest and most dangerous routes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Safest Routes")
        safest_routes = routes_df.head(3)
        for _, route in safest_routes.iterrows():
            st.markdown(f"""
            <div style='background: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #28a745;'>
                <strong>{route['route_id']}</strong><br>
                Safety Score: {route['safety_score']}/100<br>
                Risk Level: {route['risk_level']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔴 Riskiest Routes")
        riskiest_routes = routes_df.tail(3)
        for _, route in riskiest_routes.iterrows():
            st.markdown(f"""
            <div style='background: #f8d7da; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #dc3545;'>
                <strong>{route['route_id']}</strong><br>
                Safety Score: {route['safety_score']}/100<br>
                Risk Level: {route['risk_level']}
            </div>
            """, unsafe_allow_html=True)
    
    # Route safety distribution chart
    st.markdown("#### 📊 Route Safety Distribution")
    fig = px.histogram(
        routes_df, 
        x='safety_score', 
        nbins=10,
        title="Distribution of Route Safety Scores",
        labels={'safety_score': 'Safety Score (0-100)', 'count': 'Number of Routes'},
        color_discrete_sequence=['#17a2b8']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_ai_insights_routes(route_analysis):
    """Generate AI insights for route analysis"""
    if not route_analysis:
        return
    
    routes_df = pd.DataFrame(route_analysis)
    
    st.markdown("#### 🤖 Route Safety Insights")
    
    avg_safety = routes_df['safety_score'].mean()
    safest_score = routes_df['safety_score'].max()
    riskiest_score = routes_df['safety_score'].min()
    
    # Generate insight based on data
    if avg_safety >= 75:
        insight = f"✅ **Good news!** Your routes are generally safe (average score: {avg_safety:.1f}/100). The safest route scores {safest_score}/100."
    elif avg_safety >= 50:
        insight = f"⚠️ **Mixed results.** Route safety varies significantly (average: {avg_safety:.1f}/100). Consider focusing on routes scoring above 70."
    else:
        insight = f"🚨 **Safety concern.** Many routes have low safety scores (average: {avg_safety:.1f}/100). Consider alternative paths or extra precautions."
    
    st.info(insight)
    
    # Additional specific recommendations
    high_risk_count = len(routes_df[routes_df['safety_score'] < 50])
    if high_risk_count > 0:
        st.warning(f"⚠️ {high_risk_count} routes have safety scores below 50. Consider avoiding these during peak traffic hours.")


def generate_smart_recommendations(routes_df, braking_df, swerving_df, spatial_options):
    """Generate comprehensive safety recommendations"""
    recommendations = {
        'personal': [],
        'area_specific': [],
        'general': []
    }
    
    # Analyze data for personal recommendations
    braking_count = len(braking_df) if braking_df is not None else 0
    swerving_count = len(swerving_df) if swerving_df is not None else 0
    route_count = len(routes_df) if routes_df is not None else 0
    
    # Personal recommendations based on data patterns
    if braking_count > swerving_count * 2:
        recommendations['personal'].append({
            'title': 'Focus on Traffic Awareness',
            'description': 'Your data shows more hard braking than swerving incidents. This suggests traffic-related risks. Maintain larger following distances and watch for sudden stops ahead.'
        })
    elif swerving_count > braking_count * 2:
        recommendations['personal'].append({
            'title': 'Watch for Road Hazards',
            'description': 'Swerving incidents dominate your data, indicating road obstacles. Stay alert for potholes, debris, parked cars, and pedestrians stepping into bike lanes.'
        })
    else:
        recommendations['personal'].append({
            'title': 'Stay Alert for Mixed Hazards',
            'description': 'Your incident data shows both traffic and road hazard issues. Maintain awareness of both vehicles and road conditions while cycling.'
        })
    
    if route_count > 5:
        recommendations['personal'].append({
            'title': 'Optimize Your Route Selection',
            'description': f'With {route_count} routes in your data, you have good options. Use the Route Safety tab to identify and prefer your safest paths.'
        })
    
    # Area-specific recommendations based on incident density
    total_incidents = braking_count + swerving_count
    if total_incidents > 100:
        recommendations['area_specific'].append({
            'title': 'High-Incident Area Strategy',
            'description': 'Your area has many safety incidents. Consider cycling during off-peak hours and using main roads with better bike infrastructure.'
        })
        recommendations['area_specific'].append({
            'title': 'Alternative Transportation',
            'description': 'For trips through high-risk zones, consider combining cycling with public transit or choosing completely different routes.'
        })
    elif total_incidents > 30:
        recommendations['area_specific'].append({
            'title': 'Moderate Risk Area Tips',
            'description': 'Your area has moderate safety challenges. Focus on defensive cycling and consider peak vs. off-peak timing for different routes.'
        })
    else:
        recommendations['area_specific'].append({
            'title': 'Low-Risk Area Advantage',
            'description': 'Your cycling area is relatively safe. Still maintain good safety practices, but you can be confident in your route choices.'
        })
    
    # Time-based recommendations
    recommendations['area_specific'].append({
        'title': 'Time Your Rides Strategically',
        'description': 'Avoid rush hours (7-9 AM, 5-7 PM) when possible. Early morning and mid-day rides typically have fewer traffic conflicts.'
    })
    
    # General safety recommendations
    recommendations['general'] = [
        {
            'title': 'Always Wear a Helmet',
            'description': 'A properly fitted helmet can reduce head injury risk by up to 85%. Make sure it meets safety standards and replace it after any impact.'
        },
        {
            'title': 'Increase Your Visibility',
            'description': 'Use bright colors during the day and reflective gear plus lights at night. Front white light and rear red light are often legally required.'
        },
        {
            'title': 'Follow Traffic Rules',
            'description': 'Ride predictably by following traffic laws. Signal turns, stop at red lights, and ride in the same direction as traffic.'
        },
        {
            'title': 'Maintain Your Bike',
            'description': 'Regular maintenance prevents mechanical failures. Check brakes, tires, and chain regularly. A well-maintained bike is a safer bike.'
        },
        {
            'title': 'Stay Alert and Focused',
            'description': 'Avoid phone use, earbuds at high volume, or other distractions. Keep your head up and scan constantly for potential hazards.'
        }
    ]
    
    return recommendations


def render_no_spatial_data_message():
    """Render user-friendly message when no spatial data is available"""
    st.warning("🗺️ No map data available yet")
    
    st.markdown("""
    ### What do I need to see the safety map?
    
    To unlock your personalized safety insights, upload data files with:
    
    **🎯 For Hotspot Mapping:**
    - **Braking hotspots** - locations where cyclists brake hard
    - **Swerving hotspots** - locations where cyclists swerve suddenly  
    - Each file should include latitude, longitude, and incident details
    
    **🛣️ For Route Analysis:**
    - **Route data** - start and end coordinates of cycling trips
    - **Popularity ratings** - how often routes are used (optional)
    
    **📁 Supported formats:** CSV, Excel (XLSX), JSON
    
    ### 🚀 Quick Start
    1. Go to the **Data Setup** page
    2. Upload your cycling data files
    3. Come back here to see your personalized safety map!
    
    ### 💡 Don't have data yet?
    Consider using cycling apps that track your rides and safety incidents, or contact your local cycling organization about available safety data.
    """)


# Additional utility functions for enhanced user experience

def create_summary_metrics(braking_df, swerving_df, routes_df):
    """Create summary metrics for the spatial analysis"""
    metrics = {}
    
    # Count totals
    metrics['total_braking'] = len(braking_df) if braking_df is not None else 0
    metrics['total_swerving'] = len(swerving_df) if swerving_df is not None else 0
    metrics['total_routes'] = len(routes_df) if routes_df is not None else 0
    metrics['total_incidents'] = metrics['total_braking'] + metrics['total_swerving']
    
    # Calculate risk areas
    if braking_df is not None and 'intensity' in braking_df.columns:
        metrics['high_risk_braking'] = len(braking_df[braking_df['intensity'] > 6])
    else:
        metrics['high_risk_braking'] = 0
        
    if swerving_df is not None and 'intensity' in swerving_df.columns:
        metrics['high_risk_swerving'] = len(swerving_df[swerving_df['intensity'] > 6])
    else:
        metrics['high_risk_swerving'] = 0
    
    metrics['total_high_risk'] = metrics['high_risk_braking'] + metrics['high_risk_swerving']
    
    return metrics


def display_summary_metrics():
    """Display summary metrics at the top of the page"""
    try:
        # Load data for metrics
        all_data = data_processor.load_all_datasets()
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        
        metrics = create_summary_metrics(braking_df, swerving_df, routes_df)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🛑 Braking Incidents",
                value=metrics['total_braking'],
                help="Total hard braking incidents in your data"
            )
        
        with col2:
            st.metric(
                label="↩️ Swerving Incidents", 
                value=metrics['total_swerving'],
                help="Total sudden swerving incidents in your data"
            )
        
        with col3:
            st.metric(
                label="🛣️ Routes Analyzed",
                value=metrics['total_routes'],
                help="Number of cycling routes in your data"
            )
        
        with col4:
            st.metric(
                label="🚨 High Risk Areas",
                value=metrics['total_high_risk'],
                help="Locations with high incident intensity"
            )
        
        # Add a separator
        st.markdown("---")
        
    except Exception as e:
        # Don't break the page if metrics fail
        logger.error(f"Error displaying summary metrics: {e}")


# Enhanced main function that includes summary metrics
def render_spatial_analysis_page_enhanced():
    """Enhanced version of the spatial analysis page with summary metrics"""
    st.title("🗺️ Where Are The Safety Issues?")
    st.markdown("**Discover the safest and riskiest places to cycle in your area**")
    
    # Show active filters
    show_active_filters()
    
    # Display summary metrics
    display_summary_metrics()
    
    # Add helpful introduction
    render_intro_section()
    
    # Continue with the rest of the original function...
    try:
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_spatial_data_message()
            return
        
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        spatial_options = render_user_friendly_controls()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Hotspot Map", 
            "📊 Safety Zones", 
            "🛣️ Route Safety", 
            "💡 Smart Tips"
        ])
        
        with tab1:
            render_hotspot_discovery(braking_df, swerving_df, routes_df, spatial_options)
        
        with tab2:
            render_safety_zones(braking_df, swerving_df, spatial_options)
        
        with tab3:
            render_route_safety_analysis(routes_df, braking_df, swerving_df, spatial_options)
        
        with tab4:
            render_smart_recommendations(routes_df, braking_df, swerving_df, spatial_options)
        
    except Exception as e:
        logger.error(f"Error in spatial analysis page: {e}")
        st.error("⚠️ Something went wrong loading the map data.")
        st.info("Please check your data files and try refreshing the page.")


# Use the enhanced version as the main function
render_spatial_analysis_page = render_spatial_analysis_page_enhanced
