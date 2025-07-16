"""
Spatial Analysis Page for SeeSense Dashboard
Advanced geospatial analysis and mapping capabilities
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional imports with fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("Folium not available - some advanced map features will be limited")

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_route_intersection_analysis(routes_df, spatial_options):
    """Render route intersection analysis"""
    st.markdown("**Route Intersection Analysis**")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("No route data available")
        return
    
    # Calculate route intersections and overlaps
    intersection_stats = calculate_route_intersections(routes_df, spatial_options)
    
    if intersection_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("High-Traffic Intersections", intersection_stats.get('high_traffic_intersections', 0))
            st.metric("Route Overlap Areas", intersection_stats.get('overlap_areas', 0))
        
        with col2:
            st.metric("Average Routes per Area", f"{intersection_stats.get('avg_routes_per_area', 0):.1f}")
            st.metric("Peak Intersection Density", f"{intersection_stats.get('peak_density', 0):.1f}")
    
    # Visualize intersection hotspots
    render_intersection_map(routes_df, intersection_stats)


def render_clustering_metrics(df, spatial_options):
    """Render clustering metrics for a dataset"""
    clusters_df, cluster_stats = perform_spatial_clustering(
        df, spatial_options['cluster_eps'], spatial_options['min_samples']
    )
    
    if cluster_stats is not None:
        # Calculate clustering metrics
        n_clusters = len(cluster_stats[cluster_stats['cluster'] != -1])
        n_noise = len(cluster_stats[cluster_stats['cluster'] == -1])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clusters Found", n_clusters)
        
        with col2:
            st.metric("Noise Points", n_noise)
        
        with col3:
            if n_clusters > 0:
                avg_cluster_size = cluster_stats[cluster_stats['cluster'] != -1]['size'].mean()
                st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")


def calculate_proximity_statistics(routes_df, braking_df, swerving_df, spatial_options):
    """Calculate proximity statistics between routes and incidents"""
    try:
        buffer_distance = spatial_options['buffer_distance']
        stats = {}
        
        # Calculate routes near braking hotspots
        if braking_df is not None and len(braking_df) > 0:
            routes_near_braking = 0
            for _, route in routes_df.iterrows():
                min_distance = float('inf')
                for _, incident in braking_df.iterrows():
                    # Simple distance calculation (approximate)
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    min_distance = min(min_distance, distance)
                
                if min_distance <= buffer_distance:
                    routes_near_braking += 1
            
            stats['routes_near_braking'] = routes_near_braking
        
        # Calculate routes near swerving hotspots
        if swerving_df is not None and len(swerving_df) > 0:
            routes_near_swerving = 0
            for _, route in routes_df.iterrows():
                min_distance = float('inf')
                for _, incident in swerving_df.iterrows():
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    min_distance = min(min_distance, distance)
                
                if min_distance <= buffer_distance:
                    routes_near_swerving += 1
            
            stats['routes_near_swerving'] = routes_near_swerving
        
        # Calculate average distance to nearest incident
        all_distances = []
        all_incidents = []
        
        if braking_df is not None and len(braking_df) > 0:
            all_incidents.append(braking_df[['lat', 'lon']])
        if swerving_df is not None and len(swerving_df) > 0:
            all_incidents.append(swerving_df[['lat', 'lon']])
        
        if all_incidents:
            combined_incidents = pd.concat(all_incidents, ignore_index=True)
            
            for _, route in routes_df.iterrows():
                min_distance = float('inf')
                for _, incident in combined_incidents.iterrows():
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    min_distance = min(min_distance, distance)
                all_distances.append(min_distance)
            
            stats['avg_distance'] = np.mean(all_distances)
        
        # Calculate high-risk routes (routes near multiple incidents)
        high_risk_routes = 0
        if all_incidents:
            for _, route in routes_df.iterrows():
                nearby_incidents = 0
                for _, incident in combined_incidents.iterrows():
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    if distance <= buffer_distance:
                        nearby_incidents += 1
                
                if nearby_incidents >= 2:  # Routes near multiple incidents
                    high_risk_routes += 1
            
            stats['high_risk_routes'] = high_risk_routes
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating proximity statistics: {e}")
        return {}


def calculate_route_intersections(routes_df, spatial_options):
    """Calculate route intersection statistics"""
    try:
        stats = {}
        
        # Create a grid for intersection analysis
        lat_min, lat_max = routes_df['start_lat'].min(), routes_df['start_lat'].max()
        lon_min, lon_max = routes_df['start_lon'].min(), routes_df['start_lon'].max()
        
        # Expand to include end points
        lat_min = min(lat_min, routes_df['end_lat'].min())
        lat_max = max(lat_max, routes_df['end_lat'].max())
        lon_min = min(lon_min, routes_df['end_lon'].min())
        lon_max = max(lon_max, routes_df['end_lon'].max())
        
        # Create grid
        grid_size = 50
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        # Count routes passing through each grid cell
        grid_counts = np.zeros((grid_size, grid_size))
        
        for _, route in routes_df.iterrows():
            # Simple approximation: count grid cells near route start and end
            start_lat_idx = np.argmin(np.abs(lat_grid - route['start_lat']))
            start_lon_idx = np.argmin(np.abs(lon_grid - route['start_lon']))
            end_lat_idx = np.argmin(np.abs(lat_grid - route['end_lat']))
            end_lon_idx = np.argmin(np.abs(lon_grid - route['end_lon']))
            
            grid_counts[start_lat_idx, start_lon_idx] += 1
            grid_counts[end_lat_idx, end_lon_idx] += 1
        
        # Calculate statistics
        stats['high_traffic_intersections'] = np.sum(grid_counts > 5)
        stats['overlap_areas'] = np.sum(grid_counts > 1)
        stats['avg_routes_per_area'] = np.mean(grid_counts[grid_counts > 0])
        stats['peak_density'] = np.max(grid_counts)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating route intersections: {e}")
        return {}


def render_proximity_map(routes_df, braking_df, swerving_df, spatial_options):
    """Render proximity analysis map"""
    fig = go.Figure()
    
    # Add routes
    for _, route in routes_df.head(100).iterrows():  # Limit for performance
        fig.add_trace(go.Scattermapbox(
            lat=[route['start_lat'], route['end_lat']],
            lon=[route['start_lon'], route['end_lon']],
            mode='lines',
            line=dict(width=2, color='blue'),
            opacity=0.6,
            showlegend=False
        ))
    
    # Add braking incidents
    if braking_df is not None and len(braking_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=braking_df['lat'],
            lon=braking_df['lon'],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Braking Incidents',
            showlegend=True
        ))
    
    # Add swerving incidents
    if swerving_df is not None and len(swerving_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=swerving_df['lat'],
            lon=swerving_df['lon'],
            mode='markers',
            marker=dict(size=8, color='purple'),
            name='Swerving Incidents',
            showlegend=True
        ))
    
    # Set up map
    center_lat = routes_df['start_lat'].mean()
    center_lon = routes_df['start_lon'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Route-Incident Proximity Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_intersection_map(routes_df, intersection_stats):
    """Render route intersection map"""
    # Create a simple route intersection visualization
    fig = px.scatter_mapbox(
        routes_df.head(200),  # Limit for performance
        lat="start_lat",
        lon="start_lon",
        size="popularity_rating",
        color="route_type",
        hover_data=['route_id', 'distinct_cyclists'],
        zoom=12,
        mapbox_style="carto-positron",
        title="Route Intersection Points",
        size_max=15
    )
    
    # Add end points
    fig.add_trace(go.Scattermapbox(
        lat=routes_df.head(200)['end_lat'],
        lon=routes_df.head(200)['end_lon'],
        mode='markers',
        marker=dict(size=6, color='orange', opacity=0.7),
        name='Route End Points',
        showlegend=True
    ))
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def optimize_routes(routes_df, braking_df, swerving_df, optimization_type, risk_weight, max_detour):
    """Optimize routes based on safety and other factors"""
    try:
        # Create risk scores for routes
        routes_with_risk = routes_df.copy()
        
        # Calculate risk scores based on proximity to incidents
        risk_scores = []
        
        for _, route in routes_df.iterrows():
            route_risk = 0
            
            # Check proximity to braking hotspots
            if braking_df is not None and len(braking_df) > 0:
                for _, incident in braking_df.iterrows():
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    if distance < 500:  # Within 500m
                        route_risk += incident.get('intensity', 5) * (1 - distance / 500)
            
            # Check proximity to swerving hotspots
            if swerving_df is not None and len(swerving_df) > 0:
                for _, incident in swerving_df.iterrows():
                    distance = calculate_distance(
                        route['start_lat'], route['start_lon'],
                        incident['lat'], incident['lon']
                    )
                    if distance < 500:  # Within 500m
                        route_risk += incident.get('intensity', 5) * (1 - distance / 500)
            
            risk_scores.append(route_risk)
        
        routes_with_risk['risk_score'] = risk_scores
        
        # Calculate optimization score
        if optimization_type == "Minimize Risk":
            routes_with_risk['optimization_score'] = -routes_with_risk['risk_score']
        elif optimization_type == "Maximize Popularity":
            routes_with_risk['optimization_score'] = routes_with_risk['popularity_rating']
        else:  # Balance Both
            # Normalize scores
            max_risk = routes_with_risk['risk_score'].max()
            max_pop = routes_with_risk['popularity_rating'].max()
            
            normalized_risk = routes_with_risk['risk_score'] / max_risk if max_risk > 0 else 0
            normalized_pop = routes_with_risk['popularity_rating'] / max_pop if max_pop > 0 else 0
            
            routes_with_risk['optimization_score'] = (
                risk_weight * (-normalized_risk) + 
                (1 - risk_weight) * normalized_pop
            )
        
        # Rank routes
        routes_with_risk['rank'] = routes_with_risk['optimization_score'].rank(ascending=False)
        
        return routes_with_risk
        
    except Exception as e:
        logger.error(f"Error optimizing routes: {e}")
        return None


def render_route_optimization_results(optimized_routes, original_routes):
    """Render route optimization results"""
    # Top optimized routes
    top_routes = optimized_routes.head(10)
    
    # Create optimization results map
    fig = px.scatter_mapbox(
        optimized_routes,
        lat="start_lat",
        lon="start_lon",
        size="optimization_score",
        color="risk_score",
        hover_data=['route_id', 'popularity_rating', 'risk_score'],
        zoom=12,
        mapbox_style="carto-positron",
        title="Route Optimization Results",
        color_continuous_scale="RdYlGn_r",
        size_max=20
    )
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top optimized routes
    st.markdown("#### 🏆 Top Optimized Routes")
    
    display_cols = ['route_id', 'route_type', 'optimization_score', 'risk_score', 'popularity_rating']
    display_df = top_routes[display_cols].round(2)
    display_df.columns = ['Route ID', 'Type', 'Optimization Score', 'Risk Score', 'Popularity']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Optimization insights
    st.markdown("#### 💡 Optimization Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_risk_reduction = original_routes['popularity_rating'].mean() - optimized_routes['risk_score'].mean()
        st.metric("Average Risk Reduction", f"{avg_risk_reduction:.2f}")
        
        high_scoring_routes = len(optimized_routes[optimized_routes['optimization_score'] > 0])
        st.metric("High-Scoring Routes", high_scoring_routes)
    
    with col2:
        safety_improvement = len(optimized_routes[optimized_routes['risk_score'] < optimized_routes['risk_score'].median()])
        st.metric("Routes with Improved Safety", safety_improvement)
        
        avg_optimization_score = optimized_routes['optimization_score'].mean()
        st.metric("Average Optimization Score", f"{avg_optimization_score:.2f}")


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * cspatial_analysis_page():
    """Render the spatial analysis page"""
    st.title("🗺️ Spatial Analysis")
    st.markdown("Advanced geospatial analysis of cycling safety patterns")
    
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
        spatial_options = render_spatial_controls()
        
        # Create tabs for different spatial analyses
        spatial_tab1, spatial_tab2, spatial_tab3, spatial_tab4 = st.tabs([
            "🌡️ Density Analysis", 
            "🔍 Cluster Analysis", 
            "📏 Distance Analysis", 
            "🛣️ Route Optimization"
        ])
        
        with spatial_tab1:
            render_density_analysis(braking_df, swerving_df, routes_df, spatial_options)
        
        with spatial_tab2:
            render_cluster_analysis(braking_df, swerving_df, spatial_options)
        
        with spatial_tab3:
            render_distance_analysis(routes_df, braking_df, swerving_df, spatial_options)
        
        with spatial_tab4:
            render_route_optimization(routes_df, braking_df, swerving_df, spatial_options)
        
    except Exception as e:
        logger.error(f"Error in spatial analysis page: {e}")
        st.error("⚠️ An error occurred while loading spatial analysis.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Error Details"):
            st.code(str(e))


def render_no_spatial_data_message():
    """Render message when no spatial data is available"""
    st.warning("⚠️ No spatial data available for analysis.")
    st.markdown("""
    To use spatial analysis, you need:
    1. **Route data** with start/end coordinates
    2. **Hotspot data** with latitude/longitude coordinates
    3. **Sufficient data points** for meaningful analysis
    
    Please add your data files and refresh the page.
    """)


def render_spatial_controls():
    """Render spatial analysis controls in sidebar"""
    st.sidebar.markdown("### 🗺️ Spatial Analysis Settings")
    
    options = {}
    
    # Density analysis settings
    st.sidebar.markdown("**Density Analysis**")
    options['density_radius'] = st.sidebar.slider(
        "Density Radius (m)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Radius for density calculation"
    )
    
    options['grid_size'] = st.sidebar.slider(
        "Grid Resolution",
        min_value=20,
        max_value=100,
        value=50,
        help="Grid size for density heatmap"
    )
    
    # Clustering settings
    st.sidebar.markdown("**Cluster Analysis**")
    options['cluster_eps'] = st.sidebar.slider(
        "Cluster Distance (m)",
        min_value=50,
        max_value=1000,
        value=200,
        step=25,
        help="Maximum distance between points in same cluster"
    )
    
    options['min_samples'] = st.sidebar.slider(
        "Minimum Cluster Size",
        min_value=2,
        max_value=20,
        value=5,
        help="Minimum points required to form a cluster"
    )
    
    # Distance analysis settings
    st.sidebar.markdown("**Distance Analysis**")
    options['buffer_distance'] = st.sidebar.slider(
        "Buffer Distance (m)",
        min_value=50,
        max_value=1000,
        value=250,
        step=25,
        help="Buffer distance for proximity analysis"
    )
    
    return options


def render_density_analysis(braking_df, swerving_df, routes_df, spatial_options):
    """Render density analysis with heatmaps"""
    st.markdown("### 🌡️ Density Analysis")
    st.markdown("Visualize the spatial distribution and density of cycling safety incidents")
    
    # Check if we have location data
    has_braking = braking_df is not None and len(braking_df) > 0
    has_swerving = swerving_df is not None and len(swerving_df) > 0
    has_routes = routes_df is not None and len(routes_df) > 0
    
    if not (has_braking or has_swerving):
        st.warning("No hotspot data available for density analysis")
        return
    
    # Density analysis controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Braking Hotspots", "Swerving Hotspots", "Combined Hotspots", "Route Density"],
            help="Select what to analyze"
        )
        
        show_contours = st.checkbox("Show Density Contours", value=True)
        show_points = st.checkbox("Show Individual Points", value=True)
    
    with col1:
        if analysis_type == "Braking Hotspots" and has_braking:
            render_density_heatmap(braking_df, "Braking Incidents", "Reds", show_contours, show_points)
        
        elif analysis_type == "Swerving Hotspots" and has_swerving:
            render_density_heatmap(swerving_df, "Swerving Incidents", "Purples", show_contours, show_points)
        
        elif analysis_type == "Combined Hotspots":
            render_combined_density_analysis(braking_df, swerving_df, show_contours, show_points)
        
        elif analysis_type == "Route Density" and has_routes:
            render_route_density_analysis(routes_df, show_contours, show_points)
        
        else:
            st.info(f"No data available for {analysis_type}")
    
    # Density statistics
    render_density_statistics(braking_df, swerving_df, routes_df, spatial_options)


def render_density_heatmap(df, title, colorscale, show_contours=True, show_points=True):
    """Render density heatmap for a dataset"""
    if df is None or len(df) == 0:
        st.warning(f"No data available for {title}")
        return
    
    # Create density heatmap using plotly
    fig = px.density_mapbox(
        df, 
        lat='lat', 
        lon='lon', 
        z='intensity' if 'intensity' in df.columns else 'incidents_count',
        radius=20,
        center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
        zoom=12,
        mapbox_style="carto-positron",
        title=f"{title} Density Heatmap",
        color_continuous_scale=colorscale,
        opacity=0.7
    )
    
    # Add individual points if requested
    if show_points:
        fig.add_trace(go.Scattermapbox(
            lat=df['lat'],
            lon=df['lon'],
            mode='markers',
            marker=dict(
                size=8,
                color='white',
                opacity=0.8
            ),
            text=df.get('hotspot_id', 'Point'),
            name='Individual Points',
            showlegend=True
        ))
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_combined_density_analysis(braking_df, swerving_df, show_contours=True, show_points=True):
    """Render combined density analysis for both braking and swerving"""
    st.markdown("**Combined Hotspot Density Analysis**")
    
    # Create combined dataset
    combined_data = []
    
    if braking_df is not None and len(braking_df) > 0:
        braking_subset = braking_df[['lat', 'lon', 'intensity', 'incidents_count']].copy()
        braking_subset['type'] = 'Braking'
        combined_data.append(braking_subset)
    
    if swerving_df is not None and len(swerving_df) > 0:
        swerving_subset = swerving_df[['lat', 'lon', 'intensity', 'incidents_count']].copy()
        swerving_subset['type'] = 'Swerving'
        combined_data.append(swerving_subset)
    
    if not combined_data:
        st.warning("No hotspot data available for combined analysis")
        return
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Create scatter plot with different colors for each type
    fig = px.scatter_mapbox(
        combined_df,
        lat="lat",
        lon="lon",
        size="intensity",
        color="type",
        hover_data=['incidents_count'],
        color_discrete_map={'Braking': 'red', 'Swerving': 'purple'},
        zoom=12,
        center=dict(lat=combined_df['lat'].mean(), lon=combined_df['lon'].mean()),
        mapbox_style="carto-positron",
        title="Combined Hotspot Analysis",
        size_max=20
    )
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_route_density_analysis(routes_df, show_contours=True, show_points=True):
    """Render route density analysis"""
    st.markdown("**Route Density Analysis**")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("No route data available")
        return
    
    # Create route start point density
    fig = px.density_mapbox(
        routes_df,
        lat="start_lat",
        lon="start_lon",
        z="popularity_rating",
        radius=15,
        center=dict(lat=routes_df['start_lat'].mean(), lon=routes_df['start_lon'].mean()),
        zoom=12,
        mapbox_style="carto-positron",
        title="Route Start Point Density",
        color_continuous_scale="Viridis",
        opacity=0.7
    )
    
    # Add route end points
    fig.add_trace(go.Scattermapbox(
        lat=routes_df['end_lat'],
        lon=routes_df['end_lon'],
        mode='markers',
        marker=dict(
            size=6,
            color='orange',
            opacity=0.6
        ),
        name='Route End Points',
        showlegend=True
    ))
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_density_statistics(braking_df, swerving_df, routes_df, spatial_options):
    """Render density statistics"""
    st.markdown("#### 📊 Density Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if braking_df is not None and len(braking_df) > 0:
            braking_density = calculate_point_density(braking_df, spatial_options['density_radius'])
            st.metric("Braking Density", f"{braking_density:.2f} per km²")
            
            # Most dense area
            if 'intensity' in braking_df.columns:
                max_intensity_idx = braking_df['intensity'].idxmax()
                max_intensity = braking_df.loc[max_intensity_idx, 'intensity']
                st.metric("Peak Braking Intensity", f"{max_intensity:.1f}")
    
    with col2:
        if swerving_df is not None and len(swerving_df) > 0:
            swerving_density = calculate_point_density(swerving_df, spatial_options['density_radius'])
            st.metric("Swerving Density", f"{swerving_density:.2f} per km²")
            
            # Most dense area
            if 'intensity' in swerving_df.columns:
                max_intensity_idx = swerving_df['intensity'].idxmax()
                max_intensity = swerving_df.loc[max_intensity_idx, 'intensity']
                st.metric("Peak Swerving Intensity", f"{max_intensity:.1f}")
    
    with col3:
        if routes_df is not None and len(routes_df) > 0:
            route_density = calculate_point_density(routes_df, spatial_options['density_radius'], 
                                                   lat_col='start_lat', lon_col='start_lon')
            st.metric("Route Density", f"{route_density:.2f} per km²")
            
            # Coverage area
            coverage_area = calculate_coverage_area(routes_df)
            st.metric("Coverage Area", f"{coverage_area:.1f} km²")


def render_cluster_analysis(braking_df, swerving_df, spatial_options):
    """Render spatial cluster analysis"""
    st.markdown("### 🔍 Cluster Analysis")
    st.markdown("Identify spatial clusters of safety incidents using DBSCAN algorithm")
    
    # Check if we have data
    has_braking = braking_df is not None and len(braking_df) > 0
    has_swerving = swerving_df is not None and len(swerving_df) > 0
    
    if not (has_braking or has_swerving):
        st.warning("No hotspot data available for cluster analysis")
        return
    
    # Cluster analysis controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        cluster_type = st.selectbox(
            "Cluster Analysis Type",
            ["Braking Hotspots", "Swerving Hotspots", "Combined Analysis"],
            help="Select which incidents to cluster"
        )
        
        show_noise = st.checkbox("Show Noise Points", value=True)
        show_cluster_centers = st.checkbox("Show Cluster Centers", value=True)
    
    with col1:
        if cluster_type == "Braking Hotspots" and has_braking:
            clusters_df, cluster_stats = perform_spatial_clustering(
                braking_df, spatial_options['cluster_eps'], spatial_options['min_samples']
            )
            render_cluster_map(clusters_df, cluster_stats, "Braking Incidents", show_noise, show_cluster_centers)
        
        elif cluster_type == "Swerving Hotspots" and has_swerving:
            clusters_df, cluster_stats = perform_spatial_clustering(
                swerving_df, spatial_options['cluster_eps'], spatial_options['min_samples']
            )
            render_cluster_map(clusters_df, cluster_stats, "Swerving Incidents", show_noise, show_cluster_centers)
        
        elif cluster_type == "Combined Analysis":
            render_combined_cluster_analysis(braking_df, swerving_df, spatial_options, show_noise, show_cluster_centers)
        
        else:
            st.info(f"No data available for {cluster_type}")
    
    # Cluster statistics
    render_cluster_statistics(braking_df, swerving_df, spatial_options)


def render_distance_analysis(routes_df, braking_df, swerving_df, spatial_options):
    """Render distance and proximity analysis"""
    st.markdown("### 📏 Distance Analysis")
    st.markdown("Analyze spatial relationships and proximity between routes and safety incidents")
    
    # Check if we have sufficient data
    has_routes = routes_df is not None and len(routes_df) > 0
    has_incidents = (braking_df is not None and len(braking_df) > 0) or (swerving_df is not None and len(swerving_df) > 0)
    
    if not (has_routes and has_incidents):
        st.warning("Need both route data and incident data for distance analysis")
        return
    
    # Distance analysis controls
    analysis_tabs = st.tabs(["🎯 Route-Incident Proximity", "📐 Incident Clustering", "🛣️ Route Intersections"])
    
    with analysis_tabs[0]:
        render_route_incident_proximity(routes_df, braking_df, swerving_df, spatial_options)
    
    with analysis_tabs[1]:
        render_incident_clustering_analysis(braking_df, swerving_df, spatial_options)
    
    with analysis_tabs[2]:
        render_route_intersection_analysis(routes_df, spatial_options)


def render_route_optimization(routes_df, braking_df, swerving_df, spatial_options):
    """Render route optimization analysis"""
    st.markdown("### 🛣️ Route Optimization")
    st.markdown("Optimize cycling routes for safety using spatial analysis")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("No route data available for optimization")
        return
    
    # Route optimization controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        optimization_type = st.selectbox(
            "Optimization Goal",
            ["Minimize Risk", "Maximize Popularity", "Balance Both"],
            help="Select optimization objective"
        )
        
        risk_weight = st.slider(
            "Risk Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Weight for risk minimization (vs popularity)"
        )
        
        max_detour = st.slider(
            "Max Detour (%)",
            min_value=0,
            max_value=50,
            value=20,
            help="Maximum acceptable detour percentage"
        )
    
    with col1:
        # Route optimization analysis
        optimized_routes = optimize_routes(routes_df, braking_df, swerving_df, 
                                         optimization_type, risk_weight, max_detour)
        
        if optimized_routes is not None:
            render_route_optimization_results(optimized_routes, routes_df)
        else:
            st.info("Route optimization analysis in progress...")


# Helper functions for spatial analysis

def calculate_point_density(df, radius_m, lat_col='lat', lon_col='lon'):
    """Calculate point density per km²"""
    if df is None or len(df) == 0:
        return 0
    
    # Calculate approximate area coverage
    lat_range = df[lat_col].max() - df[lat_col].min()
    lon_range = df[lon_col].max() - df[lon_col].min()
    
    # Convert to approximate km² (rough calculation)
    area_km2 = lat_range * lon_range * 111.32 * 111.32 * np.cos(np.radians(df[lat_col].mean()))
    
    if area_km2 == 0:
        return 0
    
    return len(df) / area_km2


def calculate_coverage_area(routes_df):
    """Calculate the coverage area of routes"""
    if routes_df is None or len(routes_df) == 0:
        return 0
    
    # Get bounding box
    all_lats = pd.concat([routes_df['start_lat'], routes_df['end_lat']])
    all_lons = pd.concat([routes_df['start_lon'], routes_df['end_lon']])
    
    lat_range = all_lats.max() - all_lats.min()
    lon_range = all_lons.max() - all_lons.min()
    
    # Convert to approximate km²
    area_km2 = lat_range * lon_range * 111.32 * 111.32 * np.cos(np.radians(all_lats.mean()))
    
    return area_km2


def perform_spatial_clustering(df, eps_m, min_samples):
    """Perform DBSCAN clustering on spatial data"""
    if df is None or len(df) == 0:
        return None, None
    
    try:
        # Convert coordinates to approximate meters (rough conversion)
        coords = df[['lat', 'lon']].copy()
        coords['lat_m'] = coords['lat'] * 111320  # meters per degree lat
        coords['lon_m'] = coords['lon'] * 111320 * np.cos(np.radians(coords['lat']))  # meters per degree lon
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps_m, min_samples=min_samples).fit(coords[['lat_m', 'lon_m']])
        
        # Add cluster labels to dataframe
        clusters_df = df.copy()
        clusters_df['cluster'] = clustering.labels_
        
        # Calculate cluster statistics
        cluster_stats = clusters_df.groupby('cluster').agg({
            'lat': 'mean',
            'lon': 'mean',
            'intensity': 'mean' if 'intensity' in df.columns else 'count',
            'incidents_count': 'sum' if 'incidents_count' in df.columns else 'count'
        }).reset_index()
        
        # Add cluster size
        cluster_stats['size'] = clusters_df.groupby('cluster').size().values
        
        return clusters_df, cluster_stats
        
    except Exception as e:
        logger.error(f"Error performing spatial clustering: {e}")
        return None, None


def render_cluster_map(clusters_df, cluster_stats, title, show_noise=True, show_centers=True):
    """Render cluster analysis map"""
    if clusters_df is None or len(clusters_df) == 0:
        st.warning(f"No cluster data available for {title}")
        return
    
    # Create cluster visualization
    fig = px.scatter_mapbox(
        clusters_df,
        lat="lat",
        lon="lon",
        color="cluster",
        size="intensity" if 'intensity' in clusters_df.columns else "incidents_count",
        hover_data=['cluster', 'incidents_count'] if 'incidents_count' in clusters_df.columns else ['cluster'],
        zoom=12,
        center=dict(lat=clusters_df['lat'].mean(), lon=clusters_df['lon'].mean()),
        mapbox_style="carto-positron",
        title=f"{title} - Spatial Clusters",
        color_continuous_scale="Viridis"
    )
    
    # Add cluster centers
    if show_centers and cluster_stats is not None:
        # Filter out noise cluster (-1)
        centers = cluster_stats[cluster_stats['cluster'] != -1]
        if len(centers) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=centers['lat'],
                lon=centers['lon'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                name='Cluster Centers',
                showlegend=True
            ))
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show cluster statistics
    if cluster_stats is not None:
        st.markdown("**Cluster Statistics**")
        
        # Filter out noise if requested
        display_stats = cluster_stats.copy()
        if not show_noise:
            display_stats = display_stats[display_stats['cluster'] != -1]
        
        if len(display_stats) > 0:
            display_stats['cluster'] = display_stats['cluster'].apply(
                lambda x: f"Cluster {x}" if x != -1 else "Noise"
            )
            
            st.dataframe(
                display_stats[['cluster', 'size', 'intensity', 'incidents_count']].round(2),
                use_container_width=True
            )


def render_combined_cluster_analysis(braking_df, swerving_df, spatial_options, show_noise, show_centers):
    """Render combined cluster analysis"""
    st.markdown("**Combined Cluster Analysis**")
    
    # Combine datasets
    combined_data = []
    
    if braking_df is not None and len(braking_df) > 0:
        braking_subset = braking_df[['lat', 'lon', 'intensity', 'incidents_count']].copy()
        braking_subset['type'] = 'Braking'
        combined_data.append(braking_subset)
    
    if swerving_df is not None and len(swerving_df) > 0:
        swerving_subset = swerving_df[['lat', 'lon', 'intensity', 'incidents_count']].copy()
        swerving_subset['type'] = 'Swerving'
        combined_data.append(swerving_subset)
    
    if not combined_data:
        st.warning("No data available for combined cluster analysis")
        return
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Perform clustering
    clusters_df, cluster_stats = perform_spatial_clustering(
        combined_df, spatial_options['cluster_eps'], spatial_options['min_samples']
    )
    
    if clusters_df is not None:
        render_cluster_map(clusters_df, cluster_stats, "Combined Incidents", show_noise, show_centers)


def render_cluster_statistics(braking_df, swerving_df, spatial_options):
    """Render cluster statistics summary"""
    st.markdown("#### 📊 Cluster Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if braking_df is not None and len(braking_df) > 0:
            clusters_df, cluster_stats = perform_spatial_clustering(
                braking_df, spatial_options['cluster_eps'], spatial_options['min_samples']
            )
            
            if cluster_stats is not None:
                n_clusters = len(cluster_stats[cluster_stats['cluster'] != -1])
                n_noise = len(cluster_stats[cluster_stats['cluster'] == -1])
                
                st.metric("Braking Clusters", n_clusters)
                st.metric("Noise Points", n_noise)
    
    with col2:
        if swerving_df is not None and len(swerving_df) > 0:
            clusters_df, cluster_stats = perform_spatial_clustering(
                swerving_df, spatial_options['cluster_eps'], spatial_options['min_samples']
            )
            
            if cluster_stats is not None:
                n_clusters = len(cluster_stats[cluster_stats['cluster'] != -1])
                n_noise = len(cluster_stats[cluster_stats['cluster'] == -1])
                
                st.metric("Swerving Clusters", n_clusters)
                st.metric("Noise Points", n_noise)


def render_route_incident_proximity(routes_df, braking_df, swerving_df, spatial_options):
    """Render route-incident proximity analysis"""
    st.markdown("**Route-Incident Proximity Analysis**")
    
    if routes_df is None or len(routes_df) == 0:
        st.warning("No route data available")
        return
    
    # Calculate proximity statistics
    proximity_stats = calculate_proximity_statistics(routes_df, braking_df, swerving_df, spatial_options)
    
    if proximity_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Routes Near Braking Hotspots", proximity_stats.get('routes_near_braking', 0))
            st.metric("Routes Near Swerving Hotspots", proximity_stats.get('routes_near_swerving', 0))
        
        with col2:
            st.metric("Average Distance to Nearest Incident", f"{proximity_stats.get('avg_distance', 0):.0f}m")
            st.metric("High-Risk Routes", proximity_stats.get('high_risk_routes', 0))
    
    # Create proximity visualization
    render_proximity_map(routes_df, braking_df, swerving_df, spatial_options)


def render_incident_clustering_analysis(braking_df, swerving_df, spatial_options):
    """Render incident clustering analysis"""
    st.markdown("**Incident Clustering Analysis**")
    
    # Analyze clustering patterns
    if braking_df is not None and len(braking_df) > 0:
        st.markdown("*Braking Incident Clusters*")
        render_clustering_metrics(braking_df, spatial_options)
    
    if swerving_df is not None and len(swerving_df) > 0:
        st.markdown("*Swerving Incident Clusters*")
        render_clustering_metrics(swerving_df, spatial_options)


def render_
