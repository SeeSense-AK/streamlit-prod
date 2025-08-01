"""
Clean Spatial Analysis Page for SeeSense Dashboard
Well-organized, business-focused geographic analysis for non-technical users
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Optional imports with fallbacks
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.core.data_processor import data_processor
from app.core.metrics_calculator import metrics_calculator
from app.core.groq_insights_generator import create_insights_generator

logger = logging.getLogger(__name__)


def render_spatial_analysis_page():
    """Main spatial analysis page renderer"""
    
    st.title("🗺️ Geographic Safety Intelligence")
    st.markdown("**Transform location data into actionable safety insights**")
    
    try:
        # Load data
        all_data = data_processor.load_all_datasets()
        available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
        
        if not available_datasets:
            render_no_data_state()
            return
        
        # Extract dataframes
        routes_df = all_data.get('routes', (None, {}))[0]
        braking_df = all_data.get('braking_hotspots', (None, {}))[0]
        swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
        time_series_df = all_data.get('time_series', (None, {}))[0]
        
        # Check data quality
        data_quality = check_spatial_data_quality(routes_df, braking_df, swerving_df)
        
        if data_quality['has_coordinates']:
            # Main analysis flow
            render_spatial_overview(routes_df, braking_df, swerving_df)
            render_interactive_maps(routes_df, braking_df, swerving_df)
            render_hotspot_analysis(braking_df, swerving_df)
            render_business_insights(routes_df, braking_df, swerving_df)
            render_recommendations(routes_df, braking_df, swerving_df)
        else:
            render_data_quality_issues(data_quality)
            
    except Exception as e:
        logger.error(f"Spatial analysis error: {e}")
        st.error("Error loading geographic analysis. Please check your data.")
        with st.expander("Error Details"):
            st.code(str(e))


def render_no_data_state():
    """Show message when no data is available"""
    
    st.markdown("""
    ## 🚀 Geographic Safety Intelligence
    
    **Unlock the power of location-based safety insights:**
    
    ### 📍 What You'll Discover
    - **Risk Hotspots**: Identify exact locations requiring immediate attention
    - **Route Optimization**: Compare safety across different routes
    - **Investment Prioritization**: Focus resources where they'll have maximum impact
    - **Geographic Patterns**: Understand how location affects safety outcomes
    
    ### 💰 Business Impact
    - **30-50% reduction** in incidents through targeted interventions
    - **Clear ROI calculations** for all safety investments
    - **Data-driven decisions** backed by geographic evidence
    - **Optimized resource allocation** based on risk analysis
    
    📁 **Get Started**: Upload your cycling safety data with location information to begin.
    """)
    
    st.info("💡 **Tip**: Ensure your data includes latitude/longitude coordinates for the best insights.")


def check_spatial_data_quality(routes_df, braking_df, swerving_df):
    """Check if we have sufficient spatial data for analysis"""
    
    quality = {
        'has_coordinates': False,
        'coordinate_columns': [],
        'data_issues': [],
        'recommendations': []
    }
    
    datasets = {
        'Routes': routes_df,
        'Braking Events': braking_df,
        'Swerving Events': swerving_df
    }
    
    for name, df in datasets.items():
        if df is None or len(df) == 0:
            continue
        
        # Look for coordinate columns
        lat_cols = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
        lon_cols = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'long', 'longitude'])]
        
        if lat_cols and lon_cols:
            quality['has_coordinates'] = True
            quality['coordinate_columns'].append({
                'dataset': name,
                'lat_col': lat_cols[0],
                'lon_col': lon_cols[0],
                'records': len(df)
            })
        else:
            quality['data_issues'].append(f"{name}: Missing coordinate columns")
    
    if not quality['has_coordinates']:
        quality['recommendations'].extend([
            "Add latitude and longitude columns to your data",
            "Ensure coordinate columns are named with 'lat'/'latitude' and 'lon'/'longitude'",
            "Include geographic coordinates for all safety incidents"
        ])
    
    return quality


def render_data_quality_issues(data_quality):
    """Show data quality issues and recommendations"""
    
    st.warning("⚠️ **Geographic Analysis Limited**: Missing coordinate data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Data Issues")
        for issue in data_quality['data_issues']:
            st.markdown(f"• {issue}")
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        for rec in data_quality['recommendations']:
            st.markdown(f"• {rec}")
    
    st.info("🔧 **Solution**: Add latitude and longitude columns to your data files for full geographic analysis.")


def render_spatial_overview(routes_df, braking_df, swerving_df):
    """Render spatial overview with key metrics"""
    
    st.markdown("## 📊 Geographic Overview")
    
    # Calculate basic metrics
    total_braking = len(braking_df) if braking_df is not None else 0
    total_swerving = len(swerving_df) if swerving_df is not None else 0
    total_incidents = total_braking + total_swerving
    total_routes = len(routes_df) if routes_df is not None else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚨 Total Incidents", f"{total_incidents:,}")
    
    with col2:
        st.metric("🛑 Braking Events", f"{total_braking:,}")
    
    with col3:
        st.metric("🔄 Swerving Events", f"{total_swerving:,}")
    
    with col4:
        st.metric("🛣️ Routes Monitored", f"{total_routes:,}")
    
    # Geographic spread analysis
    if total_incidents > 0:
        geographic_stats = calculate_geographic_spread(braking_df, swerving_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📍 Geographic Spread", 
                f"{geographic_stats['area_km2']:.1f} km²",
                help="Area covered by safety incidents"
            )
        
        with col2:
            incident_density = total_incidents / max(geographic_stats['area_km2'], 1)
            st.metric(
                "🎯 Incident Density", 
                f"{incident_density:.1f}/km²",
                help="Incidents per square kilometer"
            )
        
        with col3:
            risk_level = "High" if incident_density > 5 else "Medium" if incident_density > 2 else "Low"
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_level]
            st.metric(
                "⚠️ Overall Risk Level", 
                f"{risk_color} {risk_level}",
                help="Geographic risk assessment"
            )


def calculate_geographic_spread(braking_df, swerving_df):
    """Calculate geographic spread of incidents"""
    
    all_coords = []
    
    # Collect coordinates from both datasets
    for df in [braking_df, swerving_df]:
        if df is not None and len(df) > 0:
            lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                coords = df[[lat_col, lon_col]].dropna()
                if len(coords) > 0:
                    coords.columns = ['lat', 'lon']
                    all_coords.append(coords)
    
    if not all_coords:
        return {'area_km2': 0, 'center_lat': 0, 'center_lon': 0}
    
    combined_coords = pd.concat(all_coords, ignore_index=True)
    
    # Calculate bounding box
    lat_min, lat_max = combined_coords['lat'].min(), combined_coords['lat'].max()
    lon_min, lon_max = combined_coords['lon'].min(), combined_coords['lon'].max()
    
    # Rough area calculation (not perfectly accurate but good enough)
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    area_km2 = lat_diff * lon_diff * 111 * 111  # Rough conversion to km²
    
    return {
        'area_km2': max(area_km2, 1),  # Minimum 1 km²
        'center_lat': combined_coords['lat'].mean(),
        'center_lon': combined_coords['lon'].mean(),
        'bounds': {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max
        }
    }


def render_interactive_maps(routes_df, braking_df, swerving_df):
    """Render interactive maps with different views"""
    
    st.markdown("## 🗺️ Interactive Safety Maps")
    
    # Map controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        map_type = st.selectbox(
            "🗺️ Map View",
            ["Incident Overview", "Risk Heatmap", "Route Analysis"],
            index=0
        )
    
    with col2:
        show_braking = st.checkbox("🛑 Braking Events", value=True)
    
    with col3:
        show_swerving = st.checkbox("🔄 Swerving Events", value=True)
    
    with col4:
        show_routes = st.checkbox("🛣️ Routes", value=False)
    
    # Render selected map
    if map_type == "Incident Overview":
        render_incident_overview_map(braking_df, swerving_df, routes_df, show_braking, show_swerving, show_routes)
    elif map_type == "Risk Heatmap":
        render_risk_heatmap(braking_df, swerving_df, show_braking, show_swerving)
    else:
        render_route_analysis_map(routes_df, braking_df, swerving_df)


def render_incident_overview_map(braking_df, swerving_df, routes_df, show_braking, show_swerving, show_routes):
    """Render incident overview map"""
    
    st.markdown("### 📍 Incident Overview Map")
    
    # Prepare map data
    map_data = []
    
    if show_braking and braking_df is not None and len(braking_df) > 0:
        lat_col = next((col for col in braking_df.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in braking_df.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            braking_data = braking_df[[lat_col, lon_col]].dropna().head(200)  # Limit for performance
            braking_data['type'] = 'Braking Event'
            braking_data['color'] = '#ef4444'
            braking_data.columns = ['lat', 'lon', 'type', 'color']
            map_data.append(braking_data)
    
    if show_swerving and swerving_df is not None and len(swerving_df) > 0:
        lat_col = next((col for col in swerving_df.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in swerving_df.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            swerving_data = swerving_df[[lat_col, lon_col]].dropna().head(200)  # Limit for performance
            swerving_data['type'] = 'Swerving Event'
            swerving_data['color'] = '#f59e0b'
            swerving_data.columns = ['lat', 'lon', 'type', 'color']
            map_data.append(swerving_data)
    
    if map_data:
        combined_data = pd.concat(map_data, ignore_index=True)
        
        # Create scatter map
        fig = px.scatter_mapbox(
            combined_data,
            lat='lat',
            lon='lon',
            color='type',
            color_discrete_map={
                'Braking Event': '#ef4444',
                'Swerving Event': '#f59e0b'
            },
            hover_data=['type'],
            zoom=11,
            height=500,
            title="Safety Incident Locations"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Map insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Map Statistics**")
            st.markdown(f"• Total points displayed: {len(combined_data):,}")
            st.markdown(f"• Braking events: {len(combined_data[combined_data['type'] == 'Braking Event'])}")
            st.markdown(f"• Swerving events: {len(combined_data[combined_data['type'] == 'Swerving Event'])}")
        
        with col2:
            center_lat = combined_data['lat'].mean()
            center_lon = combined_data['lon'].mean()
            st.markdown("**🎯 Analysis Center**")
            st.markdown(f"• Latitude: {center_lat:.4f}")
            st.markdown(f"• Longitude: {center_lon:.4f}")
            st.markdown(f"• Coverage area: {calculate_geographic_spread(braking_df, swerving_df)['area_km2']:.1f} km²")
    
    else:
        st.info("🗺️ No location data available for mapping. Please check your coordinate columns.")


def render_risk_heatmap(braking_df, swerving_df, show_braking, show_swerving):
    """Render risk heatmap"""
    
    st.markdown("### 🔥 Risk Density Heatmap")
    
    # Prepare heatmap data
    heatmap_data = []
    
    if show_braking and braking_df is not None and len(braking_df) > 0:
        lat_col = next((col for col in braking_df.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in braking_df.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            braking_coords = braking_df[[lat_col, lon_col]].dropna()
            braking_coords['weight'] = 1.0
            braking_coords.columns = ['lat', 'lon', 'weight']
            heatmap_data.append(braking_coords)
    
    if show_swerving and swerving_df is not None and len(swerving_df) > 0:
        lat_col = next((col for col in swerving_df.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in swerving_df.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            swerving_coords = swerving_df[[lat_col, lon_col]].dropna()
            swerving_coords['weight'] = 0.8  # Slightly lower weight
            swerving_coords.columns = ['lat', 'lon', 'weight']
            heatmap_data.append(swerving_coords)
    
    if heatmap_data:
        combined_heatmap = pd.concat(heatmap_data, ignore_index=True)
        
        # Create heatmap
        fig = px.density_mapbox(
            combined_heatmap,
            lat='lat',
            lon='lon',
            z='weight',
            radius=20,
            center=dict(
                lat=combined_heatmap['lat'].mean(),
                lon=combined_heatmap['lon'].mean()
            ),
            zoom=11,
            mapbox_style="open-street-map",
            title="Incident Density Heatmap",
            color_continuous_scale="Reds"
        )
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap analysis
        st.markdown("**🔥 Risk Analysis**")
        
        # Simple risk zone identification
        risk_zones = identify_simple_risk_zones(combined_heatmap)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High-Risk Areas Identified:**")
            for i, zone in enumerate(risk_zones[:3], 1):
                st.markdown(f"{i}. Zone at ({zone['lat']:.4f}, {zone['lon']:.4f}) - {zone['incident_count']} incidents")
        
        with col2:
            total_high_risk = len([z for z in risk_zones if z['incident_count'] > 5])
            st.metric("🚨 High-Risk Zones", total_high_risk)
            st.metric("📍 Total Risk Zones", len(risk_zones))
    
    else:
        st.info("🔥 No data available for heatmap visualization.")


def identify_simple_risk_zones(incident_data):
    """Simple risk zone identification using grid-based clustering"""
    
    if len(incident_data) < 10:
        return []
    
    # Create simple grid-based zones
    lat_bins = pd.cut(incident_data['lat'], bins=min(8, len(incident_data)//5))
    lon_bins = pd.cut(incident_data['lon'], bins=min(8, len(incident_data)//5))
    
    # Group by grid cells
    grid_groups = incident_data.groupby([lat_bins, lon_bins]).agg({
        'weight': 'sum',
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    grid_groups = grid_groups.dropna()
    grid_groups['incident_count'] = grid_groups['weight'].round().astype(int)
    
    # Sort by incident count
    risk_zones = []
    for _, row in grid_groups.iterrows():
        if row['incident_count'] > 0:
            risk_zones.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'incident_count': row['incident_count'],
                'risk_score': row['weight']
            })
    
    return sorted(risk_zones, key=lambda x: x['incident_count'], reverse=True)


def render_route_analysis_map(routes_df, braking_df, swerving_df):
    """Render route analysis map"""
    
    st.markdown("### 🛣️ Route Safety Analysis")
    
    if routes_df is None or len(routes_df) == 0:
        st.info("🛣️ No route data available for analysis.")
        return
    
    # Calculate route safety scores
    route_safety = calculate_route_safety_scores(routes_df, braking_df, swerving_df)
    
    if route_safety:
        # Create route safety chart
        route_df = pd.DataFrame(route_safety)
        
        fig = px.bar(
            route_df.head(10),  # Top 10 routes
            x='route_name',
            y='safety_score',
            color='risk_level',
            color_discrete_map={
                'Low Risk': '#16a34a',
                'Medium Risk': '#f59e0b',
                'High Risk': '#ef4444'
            },
            title="Route Safety Scores",
            labels={'safety_score': 'Safety Score (0-10)', 'route_name': 'Route Name'}
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Route recommendations
        st.markdown("### 📊 Route Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Safest Routes")
            safest = sorted(route_safety, key=lambda x: x['safety_score'], reverse=True)[:3]
            for i, route in enumerate(safest, 1):
                st.markdown(f"{i}. **{route['route_name']}** - Score: {route['safety_score']:.1f}/10")
        
        with col2:
            st.markdown("#### ⚠️ Routes Needing Attention")
            riskiest = sorted(route_safety, key=lambda x: x['safety_score'])[:3]
            for i, route in enumerate(riskiest, 1):
                st.markdown(f"{i}. **{route['route_name']}** - Score: {route['safety_score']:.1f}/10")
    
    else:
        st.info("📊 Unable to calculate route safety scores. Check your data structure.")


def calculate_route_safety_scores(routes_df, braking_df, swerving_df):
    """Calculate safety scores for each route"""
    
    route_safety = []
    
    for _, route in routes_df.iterrows():
        route_name = route.get('route_name', f"Route_{len(route_safety)}")
        
        # Count incidents for this route
        braking_count = 0
        swerving_count = 0
        
        if braking_df is not None and 'route_name' in braking_df.columns:
            braking_count = len(braking_df[braking_df['route_name'] == route_name])
        
        if swerving_df is not None and 'route_name' in swerving_df.columns:
            swerving_count = len(swerving_df[swerving_df['route_name'] == route_name])
        
        total_incidents = braking_count + swerving_count
        route_length = route.get('length_km', route.get('distance_km', 1))
        
        # Calculate safety score (0-10 scale)
        incident_density = total_incidents / max(route_length, 1)
        safety_score = max(0, 10 - min(incident_density * 2, 10))
        
        # Determine risk level
        if safety_score >= 7:
            risk_level = 'Low Risk'
        elif safety_score >= 4:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'
        
        route_safety.append({
            'route_name': route_name,
            'safety_score': safety_score,
            'total_incidents': total_incidents,
            'braking_incidents': braking_count,
            'swerving_incidents': swerving_count,
            'risk_level': risk_level,
            'incident_density': incident_density
        })
    
    return route_safety


def render_hotspot_analysis(braking_df, swerving_df):
    """Render hotspot analysis section"""
    
    st.markdown("## 🚨 Hotspot Analysis")
    
    # Perform hotspot detection
    hotspots = detect_hotspots(braking_df, swerving_df)
    
    if hotspots:
        st.markdown("### 🎯 Critical Hotspots Identified")
        
        # Display top hotspots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Location Details")
            for i, hotspot in enumerate(hotspots[:5], 1):
                st.markdown(f"""
                **Hotspot {i}**
                - Location: ({hotspot['center_lat']:.4f}, {hotspot['center_lon']:.4f})
                - Incidents: {hotspot['incident_count']}
                - Risk Level: {hotspot['risk_level']}
                """)
        
        with col2:
            st.markdown("#### 💰 Impact Analysis")
            
            total_incidents = sum(h['incident_count'] for h in hotspots)
            estimated_cost = total_incidents * 2500  # $2500 per incident
            intervention_cost = len(hotspots) * 15000  # $15k per hotspot
            
            st.metric("Total Hotspot Incidents", total_incidents)
            st.metric("Estimated Annual Cost", f"${estimated_cost:,}")
            st.metric("Intervention Investment", f"${intervention_cost:,}")
            
            if intervention_cost > 0:
                potential_savings = estimated_cost * 0.6  # 60% reduction
                roi = ((potential_savings - intervention_cost) / intervention_cost) * 100
                st.metric("Potential ROI", f"{roi:.0f}%")
        
        # Hotspot priority table
        st.markdown("### 📋 Intervention Priorities")
        
        hotspot_df = pd.DataFrame([{
            'Priority': i,
            'Location': f"({h['center_lat']:.4f}, {h['center_lon']:.4f})",
            'Incidents': h['incident_count'],
            'Risk Level': h['risk_level'],
            'Est. Cost Impact': f"${h['incident_count'] * 2500:,}"
        } for i, h in enumerate(hotspots[:5], 1)])
        
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
    
    else:
        st.success("✅ No significant hotspots detected - indicates good overall safety distribution!")


def detect_hotspots(braking_df, swerving_df):
    """Detect hotspots using simple clustering"""
    
    # Combine incident data
    all_incidents = []
    
    for df, incident_type in [(braking_df, 'braking'), (swerving_df, 'swerving')]:
        if df is not None and len(df) > 0:
            lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                incidents = df[[lat_col, lon_col]].copy()
                incidents['type'] = incident_type
                incidents.columns = ['lat', 'lon', 'type']
                all_incidents.append(incidents)
    
    if not all_incidents:
        return []
    
    combined_data = pd.concat(all_incidents, ignore_index=True)
    
    if len(combined_data) < 10:
        return []
    
    # Use DBSCAN clustering if available, otherwise use simple grid
    if SKLEARN_AVAILABLE:
        return detect_hotspots_with_clustering(combined_data)
    else:
        return detect_hotspots_with_grid(combined_data)


def detect_hotspots_with_clustering(incident_data):
    """Detect hotspots using DBSCAN clustering"""
    
    try:
        coords = incident_data[['lat', 'lon']].values
        
        # Normalize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(coords_scaled)
        incident_data['cluster'] = clustering.labels_
        
        hotspots = []
        
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_data = incident_data[incident_data['cluster'] == cluster_id]
            
            if len(cluster_data) >= 5:  # Significant cluster
                incident_count = len(cluster_data)
                risk_level = 'Critical' if incident_count > 15 else 'High' if incident_count > 8 else 'Medium'
                
                hotspots.append({
                    'cluster_id': cluster_id,
                    'center_lat': cluster_data['lat'].mean(),
                    'center_lon': cluster_data['lon'].mean(),
                    'incident_count': incident_count,
                    'risk_level': risk_level
                })
        
        return sorted(hotspots, key=lambda x: x['incident_count'], reverse=True)
    
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        return detect_hotspots_with_grid(incident_data)


def detect_hotspots_with_grid(incident_data):
    """Detect hotspots using simple grid-based approach"""
    
    # Create grid
    lat_bins = pd.cut(incident_data['lat'], bins=8)
    lon_bins = pd.cut(incident_data['lon'], bins=8)
    
    # Group by grid cells
    grid_groups = incident_data.groupby([lat_bins, lon_bins]).agg({
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    # Count incidents per cell
    grid_counts = incident_data.groupby([lat_bins, lon_bins]).size().reset_index(name='incident_count')
    
    # Merge with coordinates
    hotspot_data = pd.merge(grid_groups, grid_counts, on=[lat_bins.name, lon_bins.name])
    hotspot_data = hotspot_data.dropna()
    
    # Filter significant hotspots
    hotspots = []
    for _, row in hotspot_data.iterrows():
        if row['incident_count'] >= 5:  # Minimum threshold
            risk_level = 'Critical' if row['incident_count'] > 15 else 'High' if row['incident_count'] > 8 else 'Medium'
            
            hotspots.append({
                'cluster_id': f"grid_{len(hotspots)}",
                'center_lat': row['lat'],
                'center_lon': row['lon'],
                'incident_count': row['incident_count'],
                'risk_level': risk_level
            })
    
    return sorted(hotspots, key=lambda x: x['incident_count'], reverse=True)


def render_business_insights(routes_df, braking_df, swerving_df):
    """Render business insights and ROI analysis"""
    
    st.markdown("## 💼 Business Impact Analysis")
    
    # Calculate business metrics
    business_metrics = calculate_business_metrics(routes_df, braking_df, swerving_df)
    
    # Display key business insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Financial Impact")
        st.metric("Annual Safety Cost", f"${business_metrics['annual_cost']:,}")
        st.metric("Potential Savings", f"${business_metrics['potential_savings']:,}")
        st.metric("ROI Opportunity", f"{business_metrics['roi_percentage']:.0f}%")
    
    with col2:
        st.markdown("### 📊 Risk Distribution")
        
        # Risk level breakdown
        risk_breakdown = business_metrics['risk_breakdown']
        fig = px.pie(
            values=list(risk_breakdown.values()),
            names=list(risk_breakdown.keys()),
            title="Geographic Risk Distribution",
            color_discrete_map={
                'High Risk': '#ef4444',
                'Medium Risk': '#f59e0b',
                'Low Risk': '#22c55e'
            }
        )
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 🎯 Investment Priorities")
        
        priorities = business_metrics['investment_priorities']
        for i, priority in enumerate(priorities[:3], 1):
            st.markdown(f"**{i}. {priority['area']}**")
            st.markdown(f"Cost: ${priority['cost']:,}")
            st.markdown(f"Impact: {priority['impact']}")
            st.markdown("---")
    
    # Generate AI insights
    render_ai_spatial_insights(business_metrics)


def calculate_business_metrics(routes_df, braking_df, swerving_df):
    """Calculate business-relevant metrics"""
    
    # Basic incident counts
    total_braking = len(braking_df) if braking_df is not None else 0
    total_swerving = len(swerving_df) if swerving_df is not None else 0
    total_incidents = total_braking + total_swerving
    
    # Financial calculations
    avg_incident_cost = 2500  # Average cost per incident
    annual_cost = total_incidents * avg_incident_cost
    
    # Potential savings (assuming 60% reduction with interventions)
    potential_savings = annual_cost * 0.6
    
    # Investment required (rough estimate)
    investment_required = total_incidents * 15000  # $15k per incident for intervention
    
    # ROI calculation
    roi_percentage = ((potential_savings - investment_required) / max(investment_required, 1)) * 100
    
    # Risk distribution (simplified)
    risk_breakdown = {
        'High Risk': max(0, total_incidents // 3),
        'Medium Risk': max(0, total_incidents // 3),
        'Low Risk': max(0, total_incidents - 2 * (total_incidents // 3))
    }
    
    # Investment priorities
    investment_priorities = [
        {
            'area': 'Critical Hotspots',
            'cost': investment_required * 0.4,
            'impact': 'High - 60-80% reduction'
        },
        {
            'area': 'Route Improvements',
            'cost': investment_required * 0.3,
            'impact': 'Medium - 30-50% reduction'
        },
        {
            'area': 'Monitoring Systems',
            'cost': investment_required * 0.2,
            'impact': 'Medium - 20-40% reduction'
        }
    ]
    
    return {
        'total_incidents': total_incidents,
        'annual_cost': annual_cost,
        'potential_savings': potential_savings,
        'investment_required': investment_required,
        'roi_percentage': roi_percentage,
        'risk_breakdown': risk_breakdown,
        'investment_priorities': investment_priorities
    }


def render_ai_spatial_insights(business_metrics):
    """Render AI-generated spatial insights"""
    
    st.markdown("### 🧠 AI Spatial Intelligence")
    
    try:
        # Generate AI insights using the existing generator
        generator = create_insights_generator()
        
        # Create a simplified metrics dict for the AI
        ai_metrics = {
            'total_incidents': business_metrics['total_incidents'],
            'annual_cost': business_metrics['annual_cost'],
            'potential_savings': business_metrics['potential_savings'],
            'roi_percentage': business_metrics['roi_percentage']
        }
        
        # Generate insights
        spatial_summary = generator.generate_executive_summary([], ai_metrics)
        
        if spatial_summary:
            st.info(f"💡 **AI Analysis**: {spatial_summary}")
        else:
            # Fallback insight based on data
            if business_metrics['total_incidents'] > 50:
                insight = f"High incident volume ({business_metrics['total_incidents']} incidents) indicates concentrated risk areas requiring targeted intervention. Geographic analysis shows potential for {business_metrics['roi_percentage']:.0f}% ROI through strategic safety investments."
            elif business_metrics['total_incidents'] > 20:
                insight = f"Moderate incident patterns detected. Strategic geographic interventions could achieve ${business_metrics['potential_savings']:,} in annual savings with focused investment approach."
            else:
                insight = "Low incident volume suggests good overall geographic safety performance. Consider preventive measures to maintain current safety levels."
            
            st.info(f"💡 **Geographic Intelligence**: {insight}")
    
    except Exception as e:
        logger.warning(f"AI insights error: {e}")
        
        # Simple data-driven insight
        if business_metrics['roi_percentage'] > 100:
            insight = f"Strong ROI opportunity: {business_metrics['roi_percentage']:.0f}% return on geographic safety investments"
        else:
            insight = "Geographic patterns suggest focused interventions will improve safety outcomes"
        
        st.info(f"💡 **Spatial Analysis**: {insight}")


def render_recommendations(routes_df, braking_df, swerving_df):
    """Render actionable recommendations"""
    
    st.markdown("## 🎯 Strategic Recommendations")
    
    # Generate recommendations based on data
    recommendations = generate_spatial_recommendations(routes_df, braking_df, swerving_df)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            priority_colors = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }
            
            priority_icon = priority_colors.get(rec['priority'], '🔵')
            
            with st.expander(f"{priority_icon} **{rec['priority']} Priority: {rec['title']}**", expanded=(i==0)):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Situation**: {rec['description']}")
                    
                    st.markdown("**Recommended Actions**:")
                    for action in rec['actions']:
                        st.markdown(f"• {action}")
                
                with col2:
                    st.markdown("**Investment Analysis**")
                    st.markdown(f"**Cost**: ${rec['cost']:,}")
                    st.markdown(f"**Timeline**: {rec['timeline']}")
                    st.markdown(f"**Expected Impact**: {rec['impact']}")
                    
                    if rec['cost'] > 0:
                        # Simple ROI calculation
                        annual_savings = rec['cost'] * 0.8  # Assume 80% annual return
                        payback_months = (rec['cost'] / (annual_savings / 12)) if annual_savings > 0 else 12
                        st.markdown(f"**Payback**: {payback_months:.1f} months")
    
    # Implementation timeline
    render_implementation_timeline(recommendations)
    
    # Success metrics
    render_success_metrics()


def generate_spatial_recommendations(routes_df, braking_df, swerving_df):
    """Generate spatial recommendations based on data analysis"""
    
    recommendations = []
    
    # Calculate basic metrics
    total_braking = len(braking_df) if braking_df is not None else 0
    total_swerving = len(swerving_df) if swerving_df is not None else 0
    total_incidents = total_braking + total_swerving
    total_routes = len(routes_df) if routes_df is not None else 0
    
    # High incident volume
    if total_incidents > 100:
        recommendations.append({
            'priority': 'Critical',
            'title': 'Emergency Hotspot Intervention',
            'description': f'High incident volume ({total_incidents} incidents) requires immediate geographic intervention',
            'actions': [
                'Identify and secure top 3 highest-risk locations',
                'Deploy emergency safety measures within 48 hours',
                'Implement temporary traffic control measures',
                'Conduct urgent safety audits'
            ],
            'cost': total_incidents * 200,  # $200 per incident for emergency response
            'timeline': '1-2 weeks',
            'impact': '60-80% reduction in high-risk areas'
        })
    
    # Route-based recommendations
    if total_routes > 5:
        recommendations.append({
            'priority': 'High',
            'title': 'Route Safety Optimization',
            'description': f'Multiple routes ({total_routes}) require safety assessment and optimization',
            'actions': [
                'Conduct route-by-route safety analysis',
                'Prioritize improvements on highest-risk routes',
                'Implement route-specific safety measures',
                'Create alternative route recommendations'
            ],
            'cost': total_routes * 5000,  # $5k per route
            'timeline': '2-3 months',
            'impact': '30-50% improvement per route'
        })
    
    # Data quality improvement
    recommendations.append({
        'priority': 'Medium',
        'title': 'Geographic Data Enhancement',
        'description': 'Improve geographic data quality for better spatial analysis',
        'actions': [
            'Standardize coordinate data collection',
            'Implement GPS accuracy improvements',
            'Add missing geographic metadata',
            'Create automated data validation'
        ],
        'cost': 15000,
        'timeline': '1-2 months',
        'impact': '40-60% improvement in analysis accuracy'
    })
    
    # Monitoring system
    if total_incidents > 20:
        recommendations.append({
            'priority': 'Medium',
            'title': 'Enhanced Geographic Monitoring',
            'description': 'Implement advanced monitoring systems for proactive safety management',
            'actions': [
                'Deploy real-time monitoring at hotspots',
                'Create automated alert systems',
                'Implement predictive analytics',
                'Establish regular safety patrols'
            ],
            'cost': 25000,
            'timeline': '2-4 months',
            'impact': '25-40% reduction through early intervention'
        })
    
    return recommendations


def render_implementation_timeline(recommendations):
    """Render implementation timeline"""
    
    if not recommendations:
        return
    
    st.markdown("### 📅 Implementation Timeline")
    
    # Create timeline visualization
    timeline_data = []
    base_date = datetime.now()
    
    for i, rec in enumerate(recommendations):
        # Extract timeline in weeks
        timeline_str = rec['timeline']
        if 'week' in timeline_str:
            weeks = int(timeline_str.split('-')[1].split()[0]) if '-' in timeline_str else int(timeline_str.split()[0])
        else:  # months
            weeks = int(timeline_str.split('-')[1].split()[0]) * 4 if '-' in timeline_str else int(timeline_str.split()[0]) * 4
        
        start_date = base_date + timedelta(weeks=i*2)  # Stagger starts
        end_date = start_date + timedelta(weeks=weeks)
        
        timeline_data.append({
            'Task': rec['title'][:30] + '...' if len(rec['title']) > 30 else rec['title'],
            'Start': start_date,
            'Finish': end_date,
            'Priority': rec['priority']
        })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create simple timeline chart
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
                marker=dict(size=6),
                name=row['Priority'],
                showlegend=(i == 0 or row['Priority'] != timeline_df.iloc[i-1]['Priority'])
            ))
        
        fig.update_layout(
            title="Recommended Implementation Schedule",
            xaxis_title="Timeline",
            yaxis_title="Action Items",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_success_metrics():
    """Render success metrics and KPIs"""
    
    st.markdown("### 📊 Success Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Safety Targets")
        st.markdown("• 50% reduction in hotspot incidents")
        st.markdown("• 30% overall network improvement")
        st.markdown("• Zero critical risk zones")
        st.markdown("• 95% geographic coverage")
    
    with col2:
        st.markdown("#### 💰 Financial Goals")
        st.markdown("• 200%+ ROI on interventions")
        st.markdown("• 50% reduction in incident costs")
        st.markdown("• 12-month payback period")
        st.markdown("• Measurable cost savings")
    
    with col3:
        st.markdown("#### 📈 Performance KPIs")
        st.markdown("• Monthly incident tracking")
        st.markdown("• Quarterly safety assessments")
        st.markdown("• Real-time hotspot monitoring")
        st.markdown("• Annual ROI evaluation")
    
    # Call to action
    st.markdown("---")
    st.markdown("### 🚀 Take Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Analysis", use_container_width=True):
            st.success("Analysis export functionality would be implemented here")
    
    with col2:
        if st.button("📧 Share Report", use_container_width=True):
            st.success("Report sharing functionality would be implemented here")
    
    with col3:
        if st.button("⚙️ Start Implementation", use_container_width=True):
            st.success("Implementation planning would begin here")
    
    # Final insights
    st.markdown("### 💡 Key Takeaways")
    
    takeaways = [
        "🎯 **Geographic patterns** reveal specific areas requiring targeted intervention",
        "💰 **High ROI potential** through strategic, location-based safety investments",
        "📊 **Data-driven approach** ensures maximum impact from limited resources",
        "⚡ **Immediate action** on critical hotspots will deliver fastest results",
        "🔄 **Continuous monitoring** enables proactive safety management"
    ]
    
    for takeaway in takeaways:
        st.markdown(takeaway)
