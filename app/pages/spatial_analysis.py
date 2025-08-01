"""
Spatial Analysis Page for SeeSense Dashboard - Clean & Sensible Version
Simplified geospatial analysis with dynamic AI insights for non-technical users
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Optional imports with fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from sklearn.cluster import DBSCAN
from app.core.data_processor import data_processor

logger = logging.getLogger(__name__)


def render_spatial_analysis_page():
    """Render the spatial analysis page"""
    st.title("🗺️ Spatial Analysis")
    st.markdown("Visualize where cycling safety incidents occur and identify patterns")
    
    # Show current filters
    if 'date_filter' in st.session_state and st.session_state.date_filter:
        start_date, end_date = st.session_state.date_filter
        st.info(f"📅 Showing data from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")
    
    try:
        # Load datasets
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_message()
            return
        
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        
        # Quick stats
        render_quick_stats(braking_df, swerving_df, routes_df)
        
        # Main analysis tabs
        tab1, tab2, tab3 = st.tabs([
            "📍 Incident Map", 
            "📊 Hotspot Analysis", 
            "🛣️ Route Analysis"
        ])
        
        with tab1:
            render_incident_map(braking_df, swerving_df, routes_df)
        
        with tab2:
            render_hotspot_analysis(braking_df, swerving_df)
        
        with tab3:
            render_route_analysis(routes_df, braking_df, swerving_df)
        
    except Exception as e:
        logger.error(f"Error in spatial analysis: {e}")
        st.error("Error loading spatial analysis. Please check your data files.")


def render_quick_stats(braking_df, swerving_df, routes_df):
    """Show quick statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    braking_count = len(braking_df) if braking_df is not None else 0
    swerving_count = len(swerving_df) if swerving_df is not None else 0
    routes_count = len(routes_df) if routes_df is not None else 0
    
    with col1:
        st.metric("Braking Incidents", braking_count)
    with col2:
        st.metric("Swerving Incidents", swerving_count)
    with col3:
        st.metric("Total Incidents", braking_count + swerving_count)
    with col4:
        st.metric("Routes", routes_count)


def render_incident_map(braking_df, swerving_df, routes_df):
    """Render main incident map"""
    st.markdown("### Incident Locations")
    
    # Map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_braking = st.checkbox("Show Braking Incidents", value=True)
        show_swerving = st.checkbox("Show Swerving Incidents", value=True)
        show_routes = st.checkbox("Show Routes", value=False)
    
    with col1:
        # Create map
        fig = create_incident_map(braking_df, swerving_df, routes_df, 
                                show_braking, show_swerving, show_routes)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No location data available for mapping")
    
    # AI Insight
    render_map_insight(braking_df, swerving_df)


def create_incident_map(braking_df, swerving_df, routes_df, show_braking, show_swerving, show_routes):
    """Create the main incident map"""
    fig = go.Figure()
    
    # Collect all coordinates for map center
    all_lats, all_lons = [], []
    
    # Add braking incidents
    if show_braking and braking_df is not None and len(braking_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=braking_df['lat'],
            lon=braking_df['lon'],
            mode='markers',
            marker=dict(size=10, color='red', opacity=0.7),
            name='Braking Incidents',
            text=[f"Braking Incident<br>Intensity: {row.get('intensity', 'N/A')}" 
                  for _, row in braking_df.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        all_lats.extend(braking_df['lat'].tolist())
        all_lons.extend(braking_df['lon'].tolist())
    
    # Add swerving incidents
    if show_swerving and swerving_df is not None and len(swerving_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=swerving_df['lat'],
            lon=swerving_df['lon'],
            mode='markers',
            marker=dict(size=10, color='orange', opacity=0.7),
            name='Swerving Incidents',
            text=[f"Swerving Incident<br>Intensity: {row.get('intensity', 'N/A')}" 
                  for _, row in swerving_df.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        all_lats.extend(swerving_df['lat'].tolist())
        all_lons.extend(swerving_df['lon'].tolist())
    
    # Add routes
    if show_routes and routes_df is not None and len(routes_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=routes_df['start_lat'],
            lon=routes_df['start_lon'],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.5),
            name='Route Starts',
            text=['Route Start' for _ in range(len(routes_df))],
            hovertemplate='%{text}<extra></extra>'
        ))
        all_lats.extend(routes_df['start_lat'].tolist())
        all_lons.extend(routes_df['start_lon'].tolist())
    
    if not all_lats:
        return None
    
    # Set map center and zoom
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=12),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def render_hotspot_analysis(braking_df, swerving_df):
    """Render hotspot analysis"""
    st.markdown("### Hotspot Analysis")
    
    if (braking_df is None or len(braking_df) == 0) and (swerving_df is None or len(swerving_df) == 0):
        st.warning("No incident data available for hotspot analysis")
        return
    
    # Analysis controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analysis_type = st.selectbox(
            "Analyze",
            ["Braking Hotspots", "Swerving Hotspots", "Combined"]
        )
        
        cluster_distance = st.slider(
            "Cluster Distance (m)",
            min_value=50,
            max_value=500,
            value=200,
            help="Group incidents within this distance"
        )
    
    with col1:
        if analysis_type == "Braking Hotspots" and braking_df is not None:
            clusters = find_clusters(braking_df, cluster_distance)
            render_cluster_results("Braking", clusters)
        
        elif analysis_type == "Swerving Hotspots" and swerving_df is not None:
            clusters = find_clusters(swerving_df, cluster_distance)
            render_cluster_results("Swerving", clusters)
        
        elif analysis_type == "Combined":
            combined_df = combine_incident_data(braking_df, swerving_df)
            if combined_df is not None:
                clusters = find_clusters(combined_df, cluster_distance)
                render_cluster_results("Combined", clusters)
    
    # AI Insight
    render_hotspot_insight(braking_df, swerving_df, analysis_type)


def find_clusters(df, distance_m):
    """Find incident clusters using DBSCAN"""
    if df is None or len(df) < 3:
        return None
    
    # Convert distance to approximate degrees (rough conversion)
    eps_degrees = distance_m / 111000  # roughly 111km per degree
    
    coords = df[['lat', 'lon']].values
    
    try:
        clustering = DBSCAN(eps=eps_degrees, min_samples=2).fit(coords)
        df_clustered = df.copy()
        df_clustered['cluster'] = clustering.labels_
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_points = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_points),
                'center_lat': cluster_points['lat'].mean(),
                'center_lon': cluster_points['lon'].mean(),
                'avg_intensity': cluster_points.get('intensity', [1]*len(cluster_points)).mean()
            })
        
        return {
            'clustered_data': df_clustered,
            'stats': cluster_stats,
            'n_clusters': len([c for c in clustering.labels_ if c != -1])
        }
    
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return None


def render_cluster_results(incident_type, clusters):
    """Render clustering results"""
    if clusters is None:
        st.info(f"Not enough {incident_type.lower()} data for clustering")
        return
    
    st.markdown(f"**{incident_type} Clusters Found: {clusters['n_clusters']}**")
    
    if clusters['n_clusters'] > 0:
        # Show cluster map
        fig = create_cluster_map(clusters)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster table
        if clusters['stats']:
            cluster_df = pd.DataFrame(clusters['stats'])
            cluster_df = cluster_df.round(3)
            st.markdown("**Cluster Details:**")
            st.dataframe(cluster_df, use_container_width=True)


def create_cluster_map(clusters):
    """Create cluster visualization map"""
    fig = go.Figure()
    
    df = clusters['clustered_data']
    
    # Add clustered points
    for cluster_id in set(df['cluster']):
        if cluster_id == -1:  # Noise
            cluster_data = df[df['cluster'] == cluster_id]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_data['lat'],
                lon=cluster_data['lon'],
                mode='markers',
                marker=dict(size=8, color='gray', opacity=0.5),
                name='Individual Points',
                showlegend=True
            ))
        else:
            cluster_data = df[df['cluster'] == cluster_id]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_data['lat'],
                lon=cluster_data['lon'],
                mode='markers',
                marker=dict(size=12, opacity=0.8),
                name=f'Cluster {cluster_id}',
                showlegend=True
            ))
    
    # Add cluster centers
    if clusters['stats']:
        centers_df = pd.DataFrame(clusters['stats'])
        fig.add_trace(go.Scattermapbox(
            lat=centers_df['center_lat'],
            lon=centers_df['center_lon'],
            mode='markers',
            marker=dict(size=20, color='black', symbol='star'),
            name='Cluster Centers',
            showlegend=True
        ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
            zoom=13
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def render_route_analysis(routes_df, braking_df, swerving_df):
    """Render route analysis"""
    st.markdown("### Route Analysis")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("No route data available for analysis")
        return
    
    # Route safety scoring
    route_scores = calculate_route_safety_scores(routes_df, braking_df, swerving_df)
    
    if route_scores is not None:
        # Show route safety distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                route_scores, 
                x='safety_score',
                nbins=10,
                title="Route Safety Score Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show top/bottom routes
            sorted_routes = route_scores.sort_values('safety_score', ascending=False)
            
            st.markdown("**Safest Routes:**")
            st.dataframe(sorted_routes.head(3)[['route_id', 'safety_score']].round(1))
            
            st.markdown("**Riskiest Routes:**")
            st.dataframe(sorted_routes.tail(3)[['route_id', 'safety_score']].round(1))
        
        # AI Insight
        render_route_insight(route_scores)


def calculate_route_safety_scores(routes_df, braking_df, swerving_df):
    """Calculate safety scores for routes"""
    if routes_df is None:
        return None
    
    route_scores = []
    
    for idx, route in routes_df.iterrows():
        safety_score = 100  # Start with perfect score
        
        # Check proximity to incidents
        if braking_df is not None:
            for _, incident in braking_df.iterrows():
                distance = calculate_distance(
                    route['start_lat'], route['start_lon'],
                    incident['lat'], incident['lon']
                )
                if distance < 0.005:  # Within ~500m
                    penalty = incident.get('intensity', 1) * 5
                    safety_score -= penalty
        
        if swerving_df is not None:
            for _, incident in swerving_df.iterrows():
                distance = calculate_distance(
                    route['start_lat'], route['start_lon'],
                    incident['lat'], incident['lon']
                )
                if distance < 0.005:  # Within ~500m
                    penalty = incident.get('intensity', 1) * 3
                    safety_score -= penalty
        
        route_scores.append({
            'route_id': route.get('route_id', f'Route {idx}'),
            'safety_score': max(0, safety_score)
        })
    
    return pd.DataFrame(route_scores)


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate simple distance between two points"""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def combine_incident_data(braking_df, swerving_df):
    """Combine braking and swerving data"""
    combined = []
    
    if braking_df is not None and len(braking_df) > 0:
        braking_subset = braking_df[['lat', 'lon']].copy()
        if 'intensity' in braking_df.columns:
            braking_subset['intensity'] = braking_df['intensity']
        braking_subset['type'] = 'braking'
        combined.append(braking_subset)
    
    if swerving_df is not None and len(swerving_df) > 0:
        swerving_subset = swerving_df[['lat', 'lon']].copy()
        if 'intensity' in swerving_df.columns:
            swerving_subset['intensity'] = swerving_df['intensity']
        swerving_subset['type'] = 'swerving'
        combined.append(swerving_subset)
    
    if combined:
        return pd.concat(combined, ignore_index=True)
    return None


# AI Insight Functions
def render_map_insight(braking_df, swerving_df):
    """Generate AI insight for the map"""
    braking_count = len(braking_df) if braking_df is not None else 0
    swerving_count = len(swerving_df) if swerving_df is not None else 0
    
    if braking_count == 0 and swerving_count == 0:
        return
    
    st.markdown("#### 🤖 AI Insight")
    
    if braking_count > swerving_count * 1.5:
        insight = f"Your area shows more braking incidents ({braking_count}) than swerving ({swerving_count}), suggesting traffic-related safety issues are the primary concern."
    elif swerving_count > braking_count * 1.5:
        insight = f"Swerving incidents ({swerving_count}) outnumber braking incidents ({braking_count}), indicating road hazards like obstacles or poor road conditions."
    else:
        insight = f"Braking ({braking_count}) and swerving ({swerving_count}) incidents are fairly balanced, suggesting mixed safety challenges."
    
    st.info(insight)


def render_hotspot_insight(braking_df, swerving_df, analysis_type):
    """Generate AI insight for hotspot analysis"""
    st.markdown("#### 🤖 AI Insight")
    
    total_incidents = 0
    if braking_df is not None:
        total_incidents += len(braking_df)
    if swerving_df is not None:
        total_incidents += len(swerving_df)
    
    if total_incidents > 50:
        insight = "High incident density detected. Focus on the largest clusters for maximum safety impact."
    elif total_incidents > 20:
        insight = "Moderate incident clustering. Several hotspots identified that warrant attention."
    else:
        insight = "Low incident density. Your area appears relatively safe with isolated problem spots."
    
    st.info(insight)


def render_route_insight(route_scores):
    """Generate AI insight for route analysis"""
    st.markdown("#### 🤖 AI Insight")
    
    avg_score = route_scores['safety_score'].mean()
    min_score = route_scores['safety_score'].min()
    max_score = route_scores['safety_score'].max()
    
    if avg_score >= 80:
        insight = f"Route safety is generally good (average: {avg_score:.1f}/100). Focus on avoiding the lowest-scoring routes."
    elif avg_score >= 60:
        insight = f"Mixed route safety (average: {avg_score:.1f}/100). Significant differences between safest and riskiest routes."
    else:
        insight = f"Many routes show safety concerns (average: {avg_score:.1f}/100). Consider alternative paths or extra precautions."
    
    st.info(insight)


def render_no_data_message():
    """Show message when no data is available"""
    st.warning("No spatial data available")
    st.markdown("""
    **To use spatial analysis, you need:**
    - Braking or swerving hotspot data with latitude/longitude coordinates
    - Route data with start/end coordinates (optional)
    
    Upload your data files in the Data Setup page to get started.
    """)
