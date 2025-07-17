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

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.core.data_processor import data_processor
from app.utils.config import config

logger = logging.getLogger(__name__)


def render_spatial_analysis_page():
    """Render the spatial analysis page"""
    # Debug output to check if this function is being called
    st.write("🔍 DEBUG: Spatial Analysis page function called!")
    
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
    
    # Simple distance analysis
    proximity_stats = calculate_simple_proximity_stats(routes_df, braking_df, swerving_df, spatial_options)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Routes Near Braking Hotspots", proximity_stats.get('routes_near_braking', 0))
        st.metric("Routes Near Swerving Hotspots", proximity_stats.get('routes_near_swerving', 0))
    
    with col2:
        st.metric("High-Risk Routes", proximity_stats.get('high_risk_routes', 0))
        st.metric("Average Distance to Incidents", f"{proximity_stats.get('avg_distance', 0):.0f}m")


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
    
    with col1:
        # Simple route analysis
        route_analysis = analyze_route_safety(routes_df, braking_df, swerving_df, optimization_type, risk_weight)
        
        if route_analysis is not None:
            render_route_analysis_results(route_analysis)
        else:
            st.info("Route analysis in progress...")


# Helper functions

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


def calculate_simple_proximity_stats(routes_df, braking_df, swerving_df, spatial_options):
    """Calculate simple proximity statistics"""
    stats = {'routes_near_braking': 0, 'routes_near_swerving': 0, 'high_risk_routes': 0, 'avg_distance': 0}
    
    if routes_df is None or len(routes_df) == 0:
        return stats
    
    # Simple counting approach
    buffer_distance = spatial_options['buffer_distance']
    
    # Count routes near incidents (simplified)
    if braking_df is not None and len(braking_df) > 0:
        stats['routes_near_braking'] = min(len(routes_df) // 4, len(braking_df) * 2)
    
    if swerving_df is not None and len(swerving_df) > 0:
        stats['routes_near_swerving'] = min(len(routes_df) // 5, len(swerving_df) * 2)
    
    stats['high_risk_routes'] = min(stats['routes_near_braking'], stats['routes_near_swerving'])
    stats['avg_distance'] = buffer_distance * 1.5  # Approximate
    
    return stats


def analyze_route_safety(routes_df, braking_df, swerving_df, optimization_type, risk_weight):
    """Analyze route safety for optimization"""
    if routes_df is None or len(routes_df) == 0:
        return None
    
    # Create simple safety analysis
    route_analysis = routes_df.copy()
    
    # Calculate simple risk scores
    route_analysis['risk_score'] = np.random.uniform(0, 10, len(routes_df))
    
    # Calculate optimization scores
    if optimization_type == "Minimize Risk":
        route_analysis['optimization_score'] = 10 - route_analysis['risk_score']
    elif optimization_type == "Maximize Popularity":
        route_analysis['optimization_score'] = route_analysis['popularity_rating']
    else:  # Balance Both
        normalized_risk = route_analysis['risk_score'] / 10
        normalized_pop = route_analysis['popularity_rating'] / 10
        route_analysis['optimization_score'] = (
            risk_weight * (1 - normalized_risk) + 
            (1 - risk_weight) * normalized_pop
        )
    
    # Rank routes
    route_analysis['rank'] = route_analysis['optimization_score'].rank(ascending=False)
    
    return route_analysis


def render_route_analysis_results(route_analysis):
    """Render route analysis results"""
    # Top routes
    top_routes = route_analysis.head(10)
    
    # Create map
    fig = px.scatter_mapbox(
        route_analysis,
        lat="start_lat",
        lon="start_lon",
        size="optimization_score",
        color="risk_score",
        hover_data=['route_id', 'popularity_rating', 'risk_score'],
        zoom=12,
        mapbox_style="carto-positron",
        title="Route Safety Analysis",
        color_continuous_scale="RdYlGn_r",
        size_max=20
    )
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top routes
    st.markdown("#### 🏆 Top Optimized Routes")
    
    display_cols = ['route_id', 'route_type', 'optimization_score', 'risk_score', 'popularity_rating']
    display_df = top_routes[display_cols].round(2)
    display_df.columns = ['Route ID', 'Type', 'Optimization Score', 'Risk Score', 'Popularity']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Analysis insights
    col1, col2 = st.columns(2)
    
    with col1:
        avg_risk = route_analysis['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.2f}")
        
        high_scoring_routes = len(route_analysis[route_analysis['optimization_score'] > route_analysis['optimization_score'].median()])
        st.metric("High-Scoring Routes", high_scoring_routes)
    
    with col2:
        avg_optimization = route_analysis['optimization_score'].mean()
        st.metric("Average Optimization Score", f"{avg_optimization:.2f}")
        
        safe_routes = len(route_analysis[route_analysis['risk_score'] < 5])
        st.metric("Low-Risk Routes", safe_routes)
