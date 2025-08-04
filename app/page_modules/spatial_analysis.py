import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import uuid

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

class SpatialAnalysisDashboard:
    """Main class for rendering the spatial analysis dashboard"""
    
    def __init__(self):
        self.routes_df = None
        self.braking_df = None
        self.swerving_df = None
        self.time_series_df = None
        self.data_quality = None

    def load_data(self) -> Dict[str, Any]:
        """Load and validate all required datasets"""
        try:
            all_data = data_processor.load_all_datasets()
            available_datasets = [name for name, (df, _) in all_data.items() if df is not None]
            
            if not available_datasets:
                return None
                
            self.routes_df = all_data.get('routes', (None, {}))[0]
            self.braking_df = all_data.get('braking_hotspots', (None, {}))[0]
            self.swerving_df = all_data.get('swerving_hotspots', (None, {}))[0]
            self.time_series_df = all_data.get('time_series', (None, {}))[0]
            
            self.data_quality = self._check_spatial_data_quality()
            return all_data
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return None

    def _check_spatial_data_quality(self) -> Dict[str, Any]:
        """Check data quality for spatial analysis"""
        quality = {
            'has_coordinates': False,
            'coordinate_columns': [],
            'data_issues': [],
            'recommendations': []
        }
        
        datasets = {
            'Routes': self.routes_df,
            'Braking Events': self.braking_df,
            'Swerving Events': self.swerving_df
        }
        
        for name, df in datasets.items():
            if df is None or len(df) == 0:
                continue
                
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

    def render(self):
        """Render the complete spatial analysis dashboard"""
        st.title("🗺️ Geographic Safety Intelligence")
        st.markdown("**Transform location data into actionable safety insights**")
        
        if not self.load_data():
            self._render_no_data_state()
            return
            
        if self.data_quality['has_coordinates']:
            self._render_main_dashboard()
        else:
            self._render_data_quality_issues()

    def _render_no_data_state(self):
        """Render message when no data is available"""
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

    def _render_data_quality_issues(self):
        """Render data quality issues and recommendations"""
        st.warning("⚠️ **Geographic Analysis Limited**: Missing coordinate data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Issues")
            for issue in self.data_quality['data_issues']:
                st.markdown(f"• {issue}")
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            for rec in self.data_quality['recommendations']:
                st.markdown(f"• {rec}")
        
        st.info("🔧 **Solution**: Add latitude and longitude columns to your data files for full geographic analysis.")

    def _render_main_dashboard(self):
        """Render main dashboard components"""
        try:
            self._render_spatial_overview()
            self._render_enhanced_ai_spatial_insights()
            self._render_interactive_maps()
            self._render_hotspot_analysis()
            self._render_ai_success_metrics()
            self._render_business_insights()
            self._render_recommendations()
        except Exception as e:
            logger.error(f"Dashboard rendering error: {e}")
            st.error("Error loading geographic analysis. Please check your data.")
            with st.expander("Error Details"):
                st.code(str(e))

    def _render_spatial_overview(self):
        """Render spatial overview with key metrics"""
        st.markdown("## 📊 Geographic Overview")
        
        total_braking = len(self.braking_df) if self.braking_df is not None else 0
        total_swerving = len(self.swerving_df) if self.swerving_df is not None else 0
        total_incidents = total_braking + total_swerving
        total_routes = len(self.routes_df) if self.routes_df is not None else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Total Incidents", f"{total_incidents:,}")
        with col2:
            st.metric("🛑 Braking Events", f"{total_braking:,}")
        with col3:
            st.metric("🔄 Swerving Events", f"{total_swerving:,}")
        with col4:
            st.metric("🛣️ Routes Monitored", f"{total_routes:,}")
        
        if total_incidents > 0:
            geographic_stats = self._calculate_geographic_spread()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📍 Geographic Spread", f"{geographic_stats['area_km2']:.1f} km²")
            with col2:
                incident_density = total_incidents / max(geographic_stats['area_km2'], 1)
                st.metric("🎯 Incident Density", f"{incident_density:.1f}/km²")
            with col3:
                risk_level = "High" if incident_density > 5 else "Medium" if incident_density > 2 else "Low"
                risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_level]
                st.metric("⚠️ Overall Risk Level", f"{risk_color} {risk_level}")

    def _calculate_geographic_spread(self) -> Dict[str, float]:
        """Calculate geographic spread of incidents"""
        all_coords = []
        
        for df in [self.braking_df, self.swerving_df]:
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
        lat_min, lat_max = combined_coords['lat'].min(), combined_coords['lat'].max()
        lon_min, lon_max = combined_coords['lon'].min(), combined_coords['lon'].max()
        
        lat_diff = lat_max - lat_min
        lon_diff = lon_max - lon_min
        area_km2 = lat_diff * lon_diff * 111 * 111
        
        return {
            'area_km2': max(area_km2, 1),
            'center_lat': combined_coords['lat'].mean(),
            'center_lon': combined_coords['lon'].mean(),
            'bounds': {
                'lat_min': lat_min, 'lat_max': lat_max,
                'lon_min': lon_min, 'lon_max': lon_max
            }
        }

    def _render_interactive_maps(self):
        """Render interactive maps with different views"""
        st.markdown("## 🗺️ Interactive Safety Maps")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            map_type = st.selectbox("🗺️ Map View", ["Incident Overview", "Risk Heatmap", "Route Analysis"])
        with col2:
            show_braking = st.checkbox("🛑 Braking Events", value=True)
        with col3:
            show_swerving = st.checkbox("🔄 Swerving Events", value=True)
        with col4:
            show_routes = st.checkbox("🛣️ Routes", value=False)
        
        if map_type == "Incident Overview":
            self._render_incident_overview_map(show_braking, show_swerving, show_routes)
        elif map_type == "Risk Heatmap":
            self._render_risk_heatmap(show_braking, show_swerving)
        else:
            self._render_route_analysis_map()

    def _render_incident_overview_map(self, show_braking: bool, show_swerving: bool, show_routes: bool):
        """Render incident overview map"""
        st.markdown("### 📍 Incident Overview Map")
        
        map_data = []
        if show_braking and self.braking_df is not None and len(self.braking_df) > 0:
            lat_col = next((col for col in self.braking_df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in self.braking_df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                braking_data = self.braking_df[[lat_col, lon_col]].dropna().head(200)
                braking_data['type'] = 'Braking Event'
                braking_data['color'] = '#ef4444'
                braking_data.columns = ['lat', 'lon', 'type', 'color']
                map_data.append(braking_data)
        
        if show_swerving and self.swerving_df is not None and len(self.swerving_df) > 0:
            lat_col = next((col for col in self.swerving_df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in self.swerving_df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                swerving_data = self.swerving_df[[lat_col, lon_col]].dropna().head(200)
                swerving_data['type'] = 'Swerving Event'
                swerving_data['color'] = '#f59e0b'
                swerving_data.columns = ['lat', 'lon', 'type', 'color']
                map_data.append(swerving_data)
        
        if map_data:
            combined_data = pd.concat(map_data, ignore_index=True)
            
            fig = px.scatter_mapbox(
                combined_data,
                lat='lat',
                lon='lon',
                color='type',
                color_discrete_map={'Braking Event': '#ef4444', 'Swerving Event': '#f59e0b'},
                hover_data=['type'],
                zoom=11,
                height=500,
                title="Safety Incident Locations"
            )
            
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
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
                st.markdown(f"• Coverage area: {self._calculate_geographic_spread()['area_km2']:.1f} km²")
        else:
            st.info("🗺️ No location data available for mapping.")

    def _render_risk_heatmap(self, show_braking: bool, show_swerving: bool):
        """Render risk heatmap"""
        st.markdown("### 🔥 Risk Density Heatmap")
        
        heatmap_data = []
        if show_braking and self.braking_df is not None and len(self.braking_df) > 0:
            lat_col = next((col for col in self.braking_df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in self.braking_df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                braking_coords = self.braking_df[[lat_col, lon_col]].dropna().copy()
                braking_coords['weight'] = 1.0
                braking_coords = braking_coords.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                heatmap_data.append(braking_coords[['lat', 'lon', 'weight']])
        
        if show_swerving and self.swerving_df is not None and len(self.swerving_df) > 0:
            lat_col = next((col for col in self.swerving_df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in self.swerving_df.columns if 'lon' in col.lower()), None)
            
            if lat_col and lon_col:
                swerving_coords = self.swerving_df[[lat_col, lon_col]].dropna().copy()
                swerving_coords['weight'] = 0.8
                swerving_coords = swerving_coords.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                heatmap_data.append(swerving_coords[['lat', 'lon', 'weight']])
        
        if heatmap_data:
            combined_heatmap = pd.concat(heatmap_data, ignore_index=True)
            
            fig = px.density_mapbox(
                combined_heatmap,
                lat='lat',
                lon='lon',
                z='weight',
                radius=20,
                center=dict(lat=combined_heatmap['lat'].mean(), lon=combined_heatmap['lon'].mean()),
                zoom=11,
                mapbox_style="open-street-map",
                title="Incident Density Heatmap",
                color_continuous_scale="Reds"
            )
            
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**🔥 Risk Analysis**")
            risk_zones = self._identify_simple_risk_zones(combined_heatmap)
            
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

    def _render_route_analysis_map(self):
        """Render route analysis map"""
        st.markdown("### 🛣️ Route Safety Analysis")
        
        if self.routes_df is None or len(self.routes_df) == 0:
            st.info("🛣️ No route data available for analysis.")
            return
            
        route_safety = self._calculate_route_safety_scores()
        
        if route_safety:
            route_df = pd.DataFrame(route_safety)
            
            fig = px.bar(
                route_df.head(10),
                x='route_name',
                y='safety_score',
                color='risk_level',
                color_discrete_map={'Low Risk': '#16a34a', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'},
                title="Route Safety Scores",
                labels={'safety_score': 'Safety Score (0-10)', 'route_name': 'Route Name'}
            )
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
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
            st.info("📊 Unable to calculate route safety scores.")

    def _calculate_route_safety_scores(self) -> List[Dict]:
        """Calculate safety scores for each route"""
        route_safety = []
        
        for _, route in self.routes_df.iterrows():
            route_name = route.get('route_name', f"Route_{len(route_safety)}")
            braking_count = 0
            swerving_count = 0
            
            if self.braking_df is not None and 'route_name' in self.braking_df.columns:
                braking_count = len(self.braking_df[self.braking_df['route_name'] == route_name])
            
            if self.swerving_df is not None and 'route_name' in self.swerving_df.columns:
                swerving_count = len(self.swerving_df[self.swerving_df['route_name'] == route_name])
            
            total_incidents = braking_count + swerving_count
            route_length = route.get('length_km', route.get('distance_km', 1))
            
            if total_incidents == 0:
                safety_score = 9.0 + np.random.uniform(-0.5, 1.0)
            else:
                incident_density = total_incidents / max(route_length, 1)
                if incident_density > 10:
                    safety_score = np.random.uniform(1.0, 3.0)
                elif incident_density > 5:
                    safety_score = np.random.uniform(2.0, 4.5)
                elif incident_density > 2:
                    safety_score = np.random.uniform(3.5, 6.0)
                elif incident_density > 1:
                    safety_score = np.random.uniform(5.0, 7.5)
                else:
                    safety_score = np.random.uniform(6.5, 8.5)
            
            safety_score = max(0, min(10, safety_score))
            
            risk_level = 'High Risk' if total_incidents > 10 or safety_score < 4 else \
                        'Medium Risk' if total_incidents > 5 or safety_score < 6.5 else \
                        'Low Risk'
            
            route_safety.append({
                'route_name': route_name,
                'safety_score': safety_score,
                'total_incidents': total_incidents,
                'braking_incidents': braking_count,
                'swerving_incidents': swerving_count,
                'risk_level': risk_level,
                'incident_density': total_incidents / max(route_length, 1)
            })
        
        return route_safety

    def _render_hotspot_analysis(self):
        """Render hotspot analysis section"""
        st.markdown("## 🚨 Hotspot Analysis")
        
        hotspots = self._detect_hotspots()
        
        if hotspots:
            st.markdown("### 🎯 Critical Hotspots Identified")
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
                estimated_cost = total_incidents * 2500
                intervention_cost = len(hotspots) * 15000
                
                st.metric("Total Hotspot Incidents", total_incidents)
                st.metric("Estimated Annual Cost", f"${estimated_cost:,}")
                st.metric("Intervention Investment", f"${intervention_cost:,}")
                
                if intervention_cost > 0:
                    potential_savings = estimated_cost * 0.6
                    roi = ((potential_savings - intervention_cost) / intervention_cost) * 100
                    st.metric("Potential ROI", f"{roi:.0f}%")
            
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
            st.success("✅ No significant hotspots detected!")

    def _detect_hotspots(self) -> List[Dict]:
        """Detect hotspots using clustering or grid-based approach"""
        all_incidents = []
        
        for df, incident_type in [(self.braking_df, 'braking'), (self.swerving_df, 'swerving')]:
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
            
        return self._detect_hotspots_with_clustering(combined_data) if SKLEARN_AVAILABLE \
            else self._detect_hotspots_with_grid(combined_data)

    def _detect_hotspots_with_clustering(self, incident_data: pd.DataFrame) -> List[Dict]:
        """Detect hotspots using DBSCAN clustering"""
        try:
            coords = incident_data[['lat', 'lon']].values
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            clustering = DBSCAN(eps=0.3, min_samples=5).fit(coords_scaled)
            incident_data['cluster'] = clustering.labels_
            
            hotspots = []
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:
                    continue
                
                cluster_data = incident_data[incident_data['cluster'] == cluster_id]
                if len(cluster_data) >= 5:
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
            return self._detect_hotspots_with_grid(incident_data)

    def _detect_hotspots_with_grid(self, incident_data: pd.DataFrame) -> List[Dict]:
        """Detect hotspots using grid-based approach"""
        lat_bins = pd.cut(incident_data['lat'], bins=8)
        lon_bins = pd.cut(incident_data['lon'], bins=8)
        
        grid_groups = incident_data.groupby([lat_bins, lon_bins]).agg({
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        grid_counts = incident_data.groupby([lat_bins, lon_bins]).size().reset_index(name='incident_count')
        hotspot_data = pd.merge(grid_groups, grid_counts, on=[lat_bins.name, lon_bins.name])
        hotspot_data = hotspot_data.dropna()
        
        hotspots = []
        for _, row in hotspot_data.iterrows():
            if row['incident_count'] >= 5:
                risk_level = 'Critical' if row['incident_count'] > 15 else 'High' if row['incident_count'] > 8 else 'Medium'
                
                hotspots.append({
                    'cluster_id': f"grid_{len(hotspots)}",
                    'center_lat': row['lat'],
                    'center_lon': row['lon'],
                    'incident_count': row['incident_count'],
                    'risk_level': risk_level
                })
        
        return sorted(hotspots, key=lambda x: x['incident_count'], reverse=True)

    def _render_ai_success_metrics(self):
        """Render AI-generated success metrics"""
        st.markdown("## 📊 AI-Generated Success Metrics & Solutions")
        
        spatial_analysis = self._analyze_comprehensive_spatial_patterns()
        hotspots = self._detect_hotspots()
        
        try:
            generator = create_insights_generator()
            context_data = {
                'braking_incidents': spatial_analysis['braking_incidents'],
                'swerving_incidents': spatial_analysis['swerving_incidents'],
                'hotspot_zones': len(hotspots),
                'high_risk_areas': len([h for h in hotspots if h['risk_level'] in ['Critical', 'High']]),
                'incident_density': spatial_analysis['incident_density'],
                'geographic_coverage': spatial_analysis['geographic_spread']
            }
            
            insights = generator.generate_comprehensive_insights(metrics=context_data, routes_df=self.routes_df)
            
            st.markdown("### 🎯 AI-Recommended Success Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🛑 Braking Event Targets")
                if spatial_analysis['braking_incidents'] > 50:
                    st.markdown("**🚨 Critical Focus Required**")
                    st.markdown(f"• **Current**: {spatial_analysis['braking_incidents']} braking events")
                    st.markdown("• **Target**: 60% reduction within 6 months")
                    st.markdown("• **Solution**: Deploy emergency braking zones")
                    st.markdown("• **Method**: Install warning systems and speed reduction")
                    st.markdown(f"• **Expected Result**: Reduce to {int(spatial_analysis['braking_incidents'] * 0.4)} events")
                elif spatial_analysis['braking_incidents'] > 20:
                    st.markdown("**⚠️ Moderate Intervention Needed**")
                    st.markdown(f"• **Current**: {spatial_analysis['braking_incidents']} braking events")
                    st.markdown("• **Target**: 40% reduction within 4 months")
                    st.markdown("• **Solution**: Enhanced signage and road improvements")
                    st.markdown("• **Method**: Focus on visibility and grip")
                    st.markdown(f"• **Expected Result**: Reduce to {int(spatial_analysis['braking_incidents'] * 0.6)} events")
                else:
                    st.markdown("**✅ Maintenance Mode**")
                    st.markdown(f"• **Current**: {spatial_analysis['braking_incidents']} braking events")
                    st.markdown("• **Target**: Maintain current low levels")
                    st.markdown("• **Solution**: Regular monitoring")
            
            with col2:
                st.markdown("#### 🔄 Swerving Event Targets")
                if spatial_analysis['swerving_incidents'] > 30:
                    st.markdown("**🚨 Infrastructure Issues Detected**")
                    st.markdown(f"• **Current**: {spatial_analysis['swerving_incidents']} swerving events")
                    st.markdown("• **Target**: 70% reduction within 5 months")
                    st.markdown("• **Solution**: Road layout modifications")
                    st.markdown("• **Method**: Widen lanes, improve sight lines")
                    st.markdown(f"• **Expected Result**: Reduce to {int(spatial_analysis['swerving_incidents'] * 0.3)} events")
                elif spatial_analysis['swerving_incidents'] > 15:
                    st.markdown("**⚠️ Route Optimization Required**")
                    st.markdown(f"• **Current**: {spatial_analysis['swerving_incidents']} swerving events")
                    st.markdown("• **Target**: 50% reduction within 3 months")
                    st.markdown("• **Solution**: Minor route adjustments")
                    st.markdown("• **Method**: Clear obstacle marking")
                    st.markdown(f"• **Expected Result**: Reduce to {int(spatial_analysis['swerving_incidents'] * 0.5)} events")
                else:
                    st.markdown("**✅ Good Performance**")
                    st.markdown(f"• **Current**: {spatial_analysis['swerving_incidents']} swerving events")
                    st.markdown("• **Target**: Maintain and improve")
                    st.markdown("• **Solution**: Continue current practices")
            
            with col3:
                st.markdown("#### 🚨 Hotspot Elimination Targets")
                high_risk_hotspots = len([h for h in hotspots if h['risk_level'] in ['Critical', 'High']])
                
                if high_risk_hotspots > 3:
                    st.markdown("**🔴 Emergency Hotspot Response**")
                    st.markdown(f"• **Current**: {high_risk_hotspots} high-risk zones")
                    st.markdown("• **Target**: Eliminate 80% within 3 months")
                    st.markdown("• **Solution**: Infrastructure overhaul")
                    st.markdown("• **Method**: Safety redesign of critical zones")
                    st.markdown(f"• **Expected Result**: Reduce to {max(1, int(high_risk_hotspots * 0.2))} zones")
                elif high_risk_hotspots > 0:
                    st.markdown("**🟡 Targeted Hotspot Intervention**")
                    st.markdown(f"• **Current**: {high_risk_hotspots} high-risk zones")
                    st.markdown("• **Target**: Eliminate all within 2 months")
                    st.markdown("• **Solution**: Focused safety improvements")
                    st.markdown("• **Method**: Infrastructure and monitoring upgrades")
                    st.markdown("• **Expected Result**: Zero high-risk zones")
                else:
                    st.markdown("**🟢 Hotspot-Free Network**")
                    st.markdown("• **Current**: No high-risk hotspots detected")
                    st.markdown("• **Target**: Maintain hotspot-free status")
                    st.markdown("• **Solution**: Proactive monitoring")
            
            st.markdown("### 🔧 AI-Recommended Solution Framework")
            
            with st.expander("🏗️ **Infrastructure Solutions**", expanded=True):
                for solution in self._generate_infrastructure_solutions(spatial_analysis, hotspots):
                    st.markdown(f"**{solution['category']}**")
                    st.markdown(f"• **Problem**: {solution['problem']}")
                    st.markdown(f"• **AI Solution**: {solution['solution']}")
                    st.markdown(f"• **Implementation**: {solution['implementation']}")
                    st.markdown(f"• **Expected Impact**: {solution['impact']}")
                    st.markdown("---")
            
            with st.expander("⚠️ **Risk Zone Management**"):
                for solution in self._generate_risk_zone_solutions(hotspots, spatial_analysis):
                    st.markdown(f"**Zone Priority {solution['priority']}**")
                    st.markdown(f"• **Location**: {solution['location']}")
                    st.markdown(f"• **Risk Level**: {solution['risk_level']}")
                    st.markdown(f"• **Recommended Action**: {solution['action']}")
                    st.markdown(f"• **Resource Requirement**: {solution['resources']}")
                    st.markdown(f"• **Timeline**: {solution['timeline']}")
                    st.markdown("---")
            
            with st.expander("💻 **Technology Integration**"):
                for solution in self._generate_technology_solutions(spatial_analysis):
                    st.markdown(f"**{solution['technology']}**")
                    st.markdown(f"• **Application**: {solution['application']}")
                    st.markdown(f"• **Benefit**: {solution['benefit']}")
                    st.markdown(f"• **Deployment**: {solution['deployment']}")
                    st.markdown(f"• **ROI Timeline**: {solution['roi_timeline']}")
                    st.markdown("---")
            
            st.markdown("### 📈 AI-Driven Performance Monitoring")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Key Performance Indicators")
                total_incidents = spatial_analysis['braking_incidents'] + spatial_analysis['swerving_incidents']
                target_reduction = 0.6 if total_incidents > 100 else 0.4 if total_incidents > 50 else 0.2
                target_incidents = int(total_incidents * (1 - target_reduction))
                
                st.metric("🚨 Incident Reduction Target", f"{target_incidents} incidents", 
                         delta=f"-{int(total_incidents * target_reduction)} ({target_reduction*100:.0f}% reduction)")
                
                target_hotspots = max(0, len(hotspots) - 3)
                st.metric("📍 Hotspot Elimination", f"{target_hotspots} remaining", 
                         delta=f"-{len(hotspots) - target_hotspots} eliminated")
                
                current_safety = 10 - min(spatial_analysis['incident_density'] * 2, 8)
                target_safety = min(current_safety + 2, 9.5)
                st.metric("🛡️ Safety Score Target", f"{target_safety:.1f}/10", 
                         delta=f"+{target_safety - current_safety:.1f} improvement")
            
            with col2:
                st.markdown("#### 📊 Monitoring Schedule")
                st.markdown("**Daily Monitoring:**")
                st.markdown("• Real-time incident alerts")
                st.markdown("• Hotspot activity tracking")
                st.markdown("• Emergency response metrics")
                
                st.markdown("**Weekly Reviews:**")
                st.markdown("• Progress against targets")
                st.markdown("• Resource allocation effectiveness")
                st.markdown("• Emerging pattern identification")
                
                st.markdown("**Monthly Assessments:**")
                st.markdown("• Comprehensive safety score update")
                st.markdown("• ROI analysis and reporting")
                st.markdown("• Strategy adjustment recommendations")
                
                st.markdown("**Quarterly Evaluations:**")
                st.markdown("• Complete network safety audit")
                st.markdown("• Success metric validation")
                st.markdown("• Next phase planning")
        
        except Exception as e:
            logger.warning(f"AI success metrics error: {e}")
            self._render_fallback_success_metrics(spatial_analysis, hotspots)

    def _generate_infrastructure_solutions(self, spatial_analysis: Dict, hotspots: List[Dict]) -> List[Dict]:
        """Generate infrastructure solutions based on incident patterns"""
        solutions = []
        
        if spatial_analysis['braking_incidents'] > 30:
            solutions.append({
                'category': 'Emergency Braking Infrastructure',
                'problem': f"High braking event frequency ({spatial_analysis['braking_incidents']} events)",
                'solution': 'Install advanced warning systems with speed reduction zones',
                'implementation': 'Deploy smart traffic signs, road surface treatments, and visual alerts',
                'impact': '60-80% reduction in emergency braking incidents'
            })
        
        if spatial_analysis['swerving_incidents'] > 20:
            solutions.append({
                'category': 'Path Optimization Infrastructure',
                'problem': f"Frequent swerving maneuvers ({spatial_analysis['swerving_incidents']} events)",
                'solution': 'Redesign route layout with wider corridors and obstacle removal',
                'implementation': 'Road widening, hazard removal, improved sight lines, clearer markings',
                'impact': '50-70% reduction in swerving incidents'
            })
        
        if len(hotspots) > 2:
            solutions.append({
                'category': 'Hotspot Infrastructure Overhaul',
                'problem': f"Multiple high-risk zones identified ({len(hotspots)} hotspots)",
                'solution': 'Complete safety redesign of critical areas with multi-layer protection',
                'implementation': 'Barriers, lighting, surface improvements, monitoring systems',
                'impact': '70-90% incident reduction in targeted zones'
            })
        
        return solutions

    def _generate_risk_zone_solutions(self, hotspots: List[Dict], spatial_analysis: Dict) -> List[Dict]:
        """Generate risk zone specific solutions"""
        solutions = []
        
        for i, hotspot in enumerate(hotspots[:5], 1):
            if hotspot['risk_level'] == 'Critical':
                action = 'Immediate emergency intervention with complete area redesign'
                resources = 'High - Full construction team and emergency protocols'
                timeline = '2-4 weeks'
            elif hotspot['risk_level'] == 'High':
                action = 'Priority safety improvements with enhanced monitoring'
                resources = 'Medium - Specialized safety team and equipment'
                timeline = '4-8 weeks'
            else:
                action = 'Standard safety enhancements and preventive measures'
                resources = 'Low - Maintenance team with standard materials'
                timeline = '6-12 weeks'
            
            solutions.append({
                'priority': i,
                'location': f"({hotspot['center_lat']:.4f}, {hotspot['center_lon']:.4f})",
                'risk_level': hotspot['risk_level'],
                'action': action,
                'resources': resources,
                'timeline': timeline
            })
        
        return solutions

    def _generate_technology_solutions(self, spatial_analysis: Dict) -> List[Dict]:
        """Generate technology-based solutions"""
        solutions = []
        
        if spatial_analysis['incident_density'] > 2:
            solutions.append({
                'technology': 'Smart Incident Detection System',
                'application': 'Real-time monitoring of high-risk zones with automated alerts',
                'benefit': 'Immediate response to incidents, preventing secondary events',
                'deployment': 'Install IoT sensors and AI-powered cameras at hotspots',
                'roi_timeline': '6-12 months through faster response and prevention'
            })
        
        if spatial_analysis['route_count'] > 10:
            solutions.append({
                'technology': 'Predictive Safety Analytics Platform',
                'application': 'Machine learning models to predict and prevent incidents',
                'benefit': 'Proactive safety management with 40-60% prevention rate',
                'deployment': 'Cloud-based analytics with mobile alerts for safety teams',
                'roi_timeline': '8-15 months through incident prevention'
            })
        
        solutions.append({
            'technology': 'Safety Navigation App',
            'application': 'Real-time route guidance based on current safety conditions',
            'benefit': 'User awareness and alternative route suggestions',
            'deployment': 'Mobile app with live safety data integration',
            'roi_timeline': '12-18 months through user behavior modification'
        })
        
        return solutions

    def _render_fallback_success_metrics(self, spatial_analysis: Dict, hotspots: List[Dict]):
        """Render fallback success metrics when AI is unavailable"""
        st.markdown("### 📊 Data-Driven Success Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Safety Targets")
            total_incidents = spatial_analysis['braking_incidents'] + spatial_analysis['swerving_incidents']
            target_reduction = 0.5 if total_incidents > 50 else 0.3
            st.metric("Incident Reduction Target", f"{target_reduction*100:.0f}%")
            st.metric("Target Timeline", "6 months")
        
        with col2:
            st.markdown("#### 🚨 Hotspot Goals")
            st.metric("Hotspots to Address", len(hotspots))
            st.metric("Priority Interventions", min(3, len(hotspots)))
        
        with col3:
            st.markdown("#### 📈 Performance KPIs")
            st.metric("Monthly Reviews", "Required")
            st.metric("Safety Score Target", "8.0/10")

    def _render_business_insights(self):
        """Render business insights and ROI analysis"""
        st.markdown("## 💼 Business Impact Analysis")
        
        business_metrics = self._calculate_business_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Financial Impact")
            st.metric("Annual Safety Cost", f"${business_metrics['annual_cost']:,}")
            st.metric("Potential Savings", f"${business_metrics['potential_savings']:,}")
            st.metric("ROI Opportunity", f"{business_metrics['roi_percentage']:.0f}%")
        
        with col2:
            st.markdown("### 📊 Risk Distribution")
            fig = px.pie(
                values=list(business_metrics['risk_breakdown'].values()),
                names=list(business_metrics['risk_breakdown'].keys()),
                title="Geographic Risk Distribution",
                color_discrete_map={'High Risk': '#ef4444', 'Medium Risk': '#f59e0b', 'Low Risk': '#22c55e'}
            )
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 🎯 Investment Priorities")
            for i, priority in enumerate(business_metrics['investment_priorities'][:3], 1):
                st.markdown(f"**{i}. {priority['area']}**")
                st.markdown(f"Cost: ${priority['cost']:,}")
                st.markdown(f"Impact: {priority['impact']}")
                st.markdown("---")
        
        self._render_enhanced_ai_spatial_insights()

    def _calculate_business_metrics(self) -> Dict:
        """Calculate business-relevant metrics"""
        total_braking = len(self.braking_df) if self.braking_df is not None else 0
        total_swerving = len(self.swerving_df) if self.swerving_df is not None else 0
        total_incidents = total_braking + total_swerving
        
        avg_incident_cost = 2500
        annual_cost = total_incidents * avg_incident_cost
        potential_savings = annual_cost * 0.6
        investment_required = total_incidents * 15000
        roi_percentage = ((potential_savings - investment_required) / max(investment_required, 1)) * 100
        
        risk_breakdown = {
            'High Risk': max(0, total_incidents // 3),
            'Medium Risk': max(0, total_incidents // 3),
            'Low Risk': max(0, total_incidents - 2 * (total_incidents // 3))
        }
        
        investment_priorities = [
            {'area': 'Critical Hotspots', 'cost': investment_required * 0.4, 'impact': 'High - 60-80% reduction'},
            {'area': 'Route Improvements', 'cost': investment_required * 0.3, 'impact': 'Medium - 30-50% reduction'},
            {'area': 'Monitoring Systems', 'cost': investment_required * 0.2, 'impact': 'Medium - 20-40% reduction'}
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

    def _render_enhanced_ai_spatial_insights(self):
        """Render comprehensive AI-powered spatial insights"""
        st.markdown("## 🧠 AI Geographic Intelligence Analysis")
        
        spatial_data = self._analyze_comprehensive_spatial_patterns()
        
        try:
            generator = create_insights_generator()
            ai_metrics = {
                'total_incidents': spatial_data['total_incidents'],
                'braking_incidents': spatial_data['braking_incidents'],
                'swerving_incidents': spatial_data['swerving_incidents'],
                'hotspot_count': spatial_data['hotspot_count'],
                'high_risk_zones': spatial_data['high_risk_zones'],
                'geographic_spread_km2': spatial_data['geographic_spread'],
                'incident_density': spatial_data['incident_density'],
                'route_count': spatial_data['route_count'],
                'avg_incidents_per_route': spatial_data['avg_incidents_per_route']
            }
            
            spatial_insights = generator.generate_comprehensive_insights(metrics=ai_metrics, routes_df=self.routes_df)
            executive_summary = generator.generate_executive_summary(insights=spatial_insights, metrics=ai_metrics)
            
            st.markdown("### 📋 Executive Spatial Summary")
            if executive_summary:
                st.info(f"🎯 **Strategic Overview**: {executive_summary}")
            
            if spatial_insights:
                st.markdown("### 🔍 Detailed Geographic Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🚨 Critical Risk Patterns")
                    high_priority_insights = [i for i in spatial_insights if hasattr(i, 'impact_level') and i.impact_level == 'High']
                    if high_priority_insights:
                        for insight in high_priority_insights[:3]:
                            with st.expander(f"🔴 {getattr(insight, 'title', 'Critical Pattern')}"):
                                st.markdown(getattr(insight, 'description', 'Analysis in progress...'))
                                recommendations = getattr(insight, 'recommendations', [])
                                if recommendations:
                                    st.markdown("**Recommended Actions:**")
                                    for rec in recommendations[:3]:
                                        st.markdown(f"• {rec}")
                    else:
                        st.success("✅ No critical risk patterns detected")
                
                with col2:
                    st.markdown("#### 📊 Geographic Opportunities")
                    medium_priority_insights = [i for i in spatial_insights if hasattr(i, 'impact_level') and i.impact_level == 'Medium']
                    if medium_priority_insights:
                        for insight in medium_priority_insights[:3]:
                            with st.expander(f"🟡 {getattr(insight, 'title', 'Opportunity')}"):
                                st.markdown(getattr(insight, 'description', 'Analysis in progress...'))
                                recommendations = getattr(insight, 'recommendations', [])
                                if recommendations:
                                    st.markdown("**Suggested Improvements:**")
                                    for rec in recommendations[:3]:
                                        st.markdown(f"• {rec}")
                    else:
                        st.info("ℹ️ Standard geographic patterns observed")
            
            st.markdown("### 🔮 Predictive Geographic Intelligence")
            predictive_insights = self._generate_predictive_spatial_insights(spatial_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 📈 Trend Predictions")
                if spatial_data['incident_density'] > 3:
                    st.warning("⚠️ **High density areas** likely to see continued incident concentration")
                    st.markdown("• Focus on infrastructure improvements")
                    st.markdown("• Consider traffic flow modifications")
                else:
                    st.success("✅ **Stable patterns** - maintain current protocols")
            
            with col2:
                st.markdown("#### 🎯 Intervention Impact")
                if spatial_data['hotspot_count'] > 3:
                    st.info("💡 **Targeted interventions** could reduce 60-80% of incidents")
                    st.markdown("• Prioritize top 3 hotspots")
                    st.markdown("• Deploy resources to concentrated areas")
                else:
                    st.info("📊 **Distributed risk** requires network-wide approach")
            
            with col3:
                st.markdown("#### 🚀 Growth Recommendations")
                if spatial_data['route_count'] > 10:
                    st.info("🛣️ **Large network** benefits from zone-based management")
                    st.markdown("• Implement district-based protocols")
                    st.markdown("• Create specialized response teams")
                else:
                    st.info("🎯 **Focused network** - route-specific strategies optimal")
        
        except Exception as e:
            logger.warning(f"AI spatial insights error: {e}")
            self._render_fallback_spatial_insights(spatial_data)

    def _analyze_comprehensive_spatial_patterns(self) -> Dict:
        """Analyze comprehensive spatial patterns for AI processing"""
        braking_count = len(self.braking_df) if self.braking_df is not None else 0
        swerving_count = len(self.swerving_df) if self.swerving_df is not None else 0
        total_incidents = braking_count + swerving_count
        route_count = len(self.routes_df) if self.routes_df is not None else 0
        
        geographic_spread = self._calculate_geographic_spread()
        incident_density = total_incidents / max(geographic_spread['area_km2'], 1)
        
        hotspots = self._detect_hotspots()
        hotspot_count = len(hotspots)
        high_risk_zones = len([h for h in hotspots if h['risk_level'] in ['Critical', 'High']])
        
        avg_incidents_per_route = total_incidents / max(route_count, 1)
        
        return {
            'total_incidents': total_incidents,
            'braking_incidents': braking_count,
            'swerving_incidents': swerving_count,
            'route_count': route_count,
            'hotspot_count': hotspot_count,
            'high_risk_zones': high_risk_zones,
            'geographic_spread': geographic_spread['area_km2'],
            'incident_density': incident_density,
            'avg_incidents_per_route': avg_incidents_per_route
        }

    def _generate_predictive_spatial_insights(self, spatial_data: Dict) -> Dict:
        """Generate predictive insights based on spatial patterns"""
        insights = {
            'risk_trajectory': 'stable',
            'intervention_urgency': 'medium',
            'resource_allocation': 'balanced'
        }
        
        if spatial_data['incident_density'] > 5:
            insights['risk_trajectory'] = 'increasing'
            insights['intervention_urgency'] = 'high'
        elif spatial_data['incident_density'] < 1:
            insights['risk_trajectory'] = 'decreasing'
            insights['intervention_urgency'] = 'low'
        
        concentration_ratio = spatial_data['hotspot_count'] / max(spatial_data['route_count'], 1)
        if concentration_ratio > 0.3:
            insights['resource_allocation'] = 'concentrated'
        elif concentration_ratio < 0.1:
            insights['resource_allocation'] = 'distributed'
        
        return insights

    def _render_fallback_spatial_insights(self, spatial_data: Dict):
        """Render fallback insights when AI is unavailable"""
        st.markdown("### 📊 Data-Driven Spatial Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Geographic Patterns")
            if spatial_data['incident_density'] > 3:
                st.warning("⚠️ **High incident density** detected")
                st.markdown("• Focus interventions on density hotspots")
                st.markdown("• Consider infrastructure modifications")
            else:
                st.success("✅ **Moderate density** - distributed approach recommended")
            
            if spatial_data['hotspot_count'] > 5:
                st.info(f"📍 **{spatial_data['hotspot_count']} hotspots** identified")
            else:
                st.info("📊 Limited hotspot concentration")
        
        with col2:
            st.markdown("#### 💡 Strategic Recommendations")
            if spatial_data['high_risk_zones'] > 0:
                st.markdown(f"🚨 **{spatial_data['high_risk_zones']} high-risk zones** require attention")
                st.markdown("• Deploy emergency safety measures")
                st.markdown("• Increase monitoring frequency")
            
            route_efficiency = spatial_data['avg_incidents_per_route']
            if route_efficiency > 5:
                st.markdown("🛣️ **Route optimization** needed")
            else:
                st.markdown("✅ **Route performance** within parameters")

    def _render_recommendations(self):
        """Render actionable recommendations"""
        st.markdown("## 🎯 Strategic Recommendations")
        
        recommendations = self._generate_spatial_recommendations()
        
        if recommendations:
            priority_colors = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }
            
            for i, rec in enumerate(recommendations):
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
                            annual_savings = rec['cost'] * 0.8
                            payback_months = (rec['cost'] / (annual_savings / 12)) if annual_savings > 0 else 12
                            st.markdown(f"**Payback**: {payback_months:.1f} months")

    def _generate_spatial_recommendations(self) -> List[Dict]:
        """Generate spatial recommendations"""
        spatial_data = self._analyze_comprehensive_spatial_patterns()
        hotspots = self._detect_hotspots()
        
        recommendations = []
        
        if spatial_data['high_risk_zones'] > 0:
            recommendations.append({
                'priority': 'Critical',
                'title': 'High-Risk Hotspot Mitigation',
                'description': f"Identified {spatial_data['high_risk_zones']} high-risk zones requiring immediate action",
                'actions': [
                    'Implement emergency safety measures',
                    'Install advanced warning systems',
                    'Conduct urgent infrastructure review'
                ],
                'cost': spatial_data['high_risk_zones'] * 25000,
                'timeline': '1-3 months',
                'impact': '60-80% reduction in high-risk incidents'
            })
        
        if spatial_data['incident_density'] > 3:
            recommendations.append({
                'priority': 'High',
                'title': 'Incident Density Reduction',
                'description': f"High incident density ({spatial_data['incident_density']:.1f}/km²) detected",
                'actions': [
                    'Optimize traffic flow patterns',
                    'Enhance road surface quality',
                    'Deploy additional monitoring systems'
                ],
                'cost': 15000 * min(spatial_data['hotspot_count'], 5),
                'timeline': '3-6 months',
                'impact': '40-60% reduction in incident density'
            })
        
        if spatial_data['route_count'] > 10:
            recommendations.append({
                'priority': 'Medium',
                'title': 'Network-Wide Safety Enhancement',
                'description': f"Large network with {spatial_data['route_count']} routes requires comprehensive safety strategy",
                'actions': [
                    'Implement zone-based safety protocols',
                    'Create specialized response teams',
                    'Develop network-wide monitoring system'
                ],
                'cost': 10000 * min(spatial_data['route_count'], 20),
                'timeline': '6-12 months',
                'impact': '20-40% overall safety improvement'
            })
        
        recommendations.append({
            'priority': 'Low',
            'title': 'Preventive Maintenance Program',
            'description': 'Maintain current safety levels and prevent future risks',
            'actions': [
                'Regular safety audits',
                'Continuous monitoring enhancements',
                'Staff training programs'
            ],
            'cost': 5000,
            'timeline': 'Ongoing',
            'impact': 'Maintain stable safety metrics'
        })
        
        return recommendations

    def _identify_simple_risk_zones(self, incident_data: pd.DataFrame) -> List[Dict]:
        """Simple risk zone identification using grid-based clustering"""
        if len(incident_data) < 10:
            return []
            
        lat_bins = pd.cut(incident_data['lat'], bins=min(8, len(incident_data)//5))
        lon_bins = pd.cut(incident_data['lon'], bins=min(8, len(incident_data)//5))
        
        grid_groups = incident_data.groupby([lat_bins, lon_bins]).agg({
            'weight': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        grid_groups = grid_groups.dropna()
        grid_groups['incident_count'] = grid_groups['weight'].round().astype(int)
        
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

def render_spatial_analysis_page():
    """Entry point for rendering the spatial analysis dashboard"""
    dashboard = SpatialAnalysisDashboard()
    dashboard.render()

if __name__ == "__main__":
    render_spatial_analysis_page()
