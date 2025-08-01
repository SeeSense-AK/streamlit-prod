def get_data_availability_message(df):
    """Get friendly message about data availability"""
    if df is None:
        return "No data available for this period."
    else:
        return f"Currently have {len(df)} records for analysis."


def create_dynamic_safety_predictions(df, meaningful_features):
    """Create safety predictions using real data with dynamic filtering"""
    try:
        # Prepare meaningful feature matrix
        X = df[meaningful_features].copy()
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Create intelligent safety target
        safety_target = create_intelligent_safety_target(df, meaningful_features)
        
        if safety_target is None:
            return None
        
        # Train model with cross-validation for smaller datasets
        if len(X) < 20:
            # Use all data for training when dataset is small
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            model.fit(X, safety_target)
            predictions = model.predict(X)
            r2 = r2_score(safety_target, predictions)
        else:
            # Use train-test split for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(X, safety_target, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
        
        # Create user-friendly feature names
        feature_importance = pd.DataFrame({
            'feature': meaningful_features,
            'importance': model.feature_importances_,
            'friendly_name': [make_feature_friendly(f) for f in meaningful_features]
        }).sort_values('importance', ascending=True)
        
        return {
            'model': model,
            'predictions': predictions,
            'feature_importance': feature_importance,
            'accuracy': r2,
            'meaningful_features': meaningful_features
        }
        
    except Exception as e:
        logger.error(f"Error in dynamic safety predictions: {e}")
        return None


def create_intelligent_safety_target(df, meaningful_features):
    """Create intelligent safety target based on meaningful variables"""
    try:
        # Prioritize incident-based targets
        if 'incidents' in meaningful_features:
            incidents = df['incidents'].fillna(df['incidents'].median())
            max_incidents = incidents.max()
            if max_incidents > 0:
                return 1 - (incidents / max_incidents)  # Inverse relationship
            else:
                return np.ones(len(incidents))
        
        elif 'avg_braking_events' in meaningful_features:
            braking = df['avg_braking_events'].fillna(df['avg_braking_events'].median())
            max_braking = braking.max()
            if max_braking > 0:
                return 1 - (braking / max_braking)
            else:
                return np.ones(len(braking))
        
        elif 'avg_swerving_events' in meaningful_features:
            swerving = df['avg_swerving_events'].fillna(df['avg_swerving_events'].median())
            max_swerving = swerving.max()
            if max_swerving > 0:
                return 1 - (swerving / max_swerving)
            else:
                return np.ones(len(swerving))
            
        elif 'intensity' in meaningful_features:
            intensity = df['intensity'].fillna(df['intensity'].median())
            max_intensity = intensity.max()
            if max_intensity > 0:
                return 1 - (intensity / max_intensity)
            else:
                return np.ones(len(intensity))
        
        else:
            # Use composite safety score
            safety_components = []
            
            if 'avg_speed' in meaningful_features:
                speed = df['avg_speed'].fillna(df['avg_speed'].median())
                speed_median = speed.median()
                speed_range = speed.max() - speed.min()
                if speed_range > 0:
                    speed_safety = 1 - (np.abs(speed - speed_median) / speed_range)
                    safety_components.append(speed_safety)
            
            if 'incidents_count' in meaningful_features:
                incidents = df['incidents_count'].fillna(df['incidents_count'].median())
                max_incidents = incidents.max()
                if max_incidents > 0:
                    safety_components.append(1 - (incidents / max_incidents))
            
            if 'precipitation_mm' in meaningful_features:
                precip = df['precipitation_mm'].fillna(0)
                max_precip = precip.max()
                if max_precip > 0:
                    weather_safety = 1 - (precip / max_precip)
                    safety_components.append(weather_safety)
            
            if len(safety_components) > 0:
                return np.mean(safety_components, axis=0)
            else:
                # Last resort: use first meaningful feature
                first_feature = df[meaningful_features[0]].fillna(df[meaningful_features[0]].median())
                feature_range = first_feature.max() - first_feature.min()
                if feature_range > 0:
                    return (first_feature - first_feature.min()) / feature_range
                else:
                    return np.ones(len(first_feature)) * 0.5
                
    except Exception as e:
        logger.error(f"Error creating safety target: {e}")
        return None


def analyze_dynamic_cycling_dna(df, meaningful_features, n_clusters):
    """Analyze cycling patterns dynamically based on filtered data"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].copy()
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Adjust number of clusters based on data size
        effective_clusters = min(n_clusters, len(df) // 3, 5)  # Max 5 clusters, min 3 samples per cluster
        if effective_clusters < 2:
            effective_clusters = 2
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score for cluster quality
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
        else:
            silhouette_avg = 0
        
        # Create cycling personas based on real data
        personas = create_dynamic_cycling_personas(df, meaningful_features, clusters, effective_clusters)
        
        # Analyze persona distribution
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        persona_distribution = pd.DataFrame({
            'persona': [personas.get(i, f"Style {i}") for i in unique_clusters],
            'count': cluster_counts,
            'percentage': cluster_counts / len(clusters) * 100
        })
        
        # Generate personality traits based on actual data
        personality_traits = generate_dynamic_personality_traits(df, meaningful_features, clusters)
        
        # Create timeline if date data available
        pattern_timeline = None
        if 'date' in df.columns and len(df) > 7:
            try:
                df_with_clusters = df.copy()
                df_with_clusters['cluster'] = clusters
                df_with_clusters['persona'] = [personas.get(c, f"Style {c}") for c in clusters]
                df_with_clusters['date'] = pd.to_datetime(df_with_clusters['date'])
                
                # Group by appropriate time period based on data range
                date_range = (df_with_clusters['date'].max() - df_with_clusters['date'].min()).days
                if date_range > 60:
                    period = 'W'  # Weekly
                elif date_range > 14:
                    period = 'W'  # Weekly
                else:
                    period = 'D'  # Daily
                
                grouped_patterns = df_with_clusters.groupby(
                    df_with_clusters['date'].dt.to_period(period)
                )['persona'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
                
                grouped_patterns['date'] = grouped_patterns['date'].dt.start_time
                grouped_patterns.columns = ['date', 'dominant_persona']
                
                pattern_timeline = grouped_patterns
            except Exception as e:
                logger.error(f"Error creating timeline: {e}")
                pattern_timeline = None
        
        return {
            'clusters': clusters,
            'personas': personas,
            'persona_distribution': persona_distribution,
            'personality_traits': personality_traits,
            'pattern_timeline': pattern_timeline,
            'n_patterns': effective_clusters,
            'silhouette_score': silhouette_avg
        }
        
    except Exception as e:
        logger.error(f"Error in dynamic cycling DNA analysis: {e}")
        return None


def create_dynamic_cycling_personas(df, meaningful_features, clusters, n_clusters):
    """Create meaningful cycling persona names based on real cluster characteristics"""
    personas = {}
    
    try:
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) == 0:
                personas[cluster_id] = f"Unique Style {cluster_id}"
                continue
            
            # Analyze cluster characteristics based on actual data
            persona_name = "🚴‍♀️ Balanced Rider"  # Default
            
            # Check speed patterns
            if 'avg_speed' in meaningful_features and len(df) > 1:
                cluster_speed = cluster_data['avg_speed'].mean()
                overall_speed = df['avg_speed'].mean()
                if cluster_speed > overall_speed * 1.15:
                    persona_name = "⚡ Speed Enthusiast"
                elif cluster_speed < overall_speed * 0.85:
                    persona_name = "🐌 Leisurely Cruiser"
            
            # Check incident patterns
            if 'incidents' in meaningful_features and len(df) > 1:
                cluster_incidents = cluster_data['incidents'].mean()
                overall_incidents = df['incidents'].mean()
                if cluster_incidents > overall_incidents * 1.3:
                    persona_name = "🚨 Risk Aware"  # More positive framing
                elif cluster_incidents < overall_incidents * 0.7:
                    persona_name = "🛡️ Safety Champion"
            
            # Check braking patterns
            if 'avg_braking_events' in meaningful_features and len(df) > 1:
                cluster_braking = cluster_data['avg_braking_events'].mean()
                overall_braking = df['avg_braking_events'].mean()
                if cluster_braking > overall_braking * 1.2:
                    persona_name = "🚦 Cautious Commuter"
                elif cluster_braking < overall_braking * 0.8:
                    persona_name = "🌊 Smooth Operator"
            
            # Check weather patterns
            if 'temperature' in meaningful_features and len(df) > 1:
                cluster_temp = cluster_data['temperature'].mean()
                temp_q75 = df['temperature'].quantile(0.75)
                temp_q25 = df['temperature'].quantile(0.25)
                if cluster_temp < temp_q25:
                    persona_name = "❄️ Winter Warrior"
                elif cluster_temp > temp_q75:
                    persona_name = "☀️ Summer Cyclist"
            
            personas[cluster_id] = persona_name
        
        return personas
        
    except Exception as e:
        logger.error(f"Error creating personas: {e}")
        return {i: f"Style {i}" for i in range(n_clusters)}


def generate_dynamic_personality_traits(df, meaningful_features, clusters):
    """Generate personality traits based on actual cycling patterns"""
    traits = []
    
    try:
        # Analyze overall patterns from real data
        if 'avg_speed' in meaningful_features and len(df) > 1:
            avg_speed = df['avg_speed'].mean()
            median_speed = df['avg_speed'].median()
            if avg_speed > median_speed * 1.1:
                traits.append("You prefer riding at above-average speeds")
            elif avg_speed < median_speed * 0.9:
                traits.append("You enjoy a comfortable, steady pace")
        
        if 'incidents' in meaningful_features and len(df) > 1:
            avg_incidents = df['incidents'].mean()
            median_incidents = df['incidents'].median()
            if avg_incidents < median_incidents:
                traits.append("You have fewer safety incidents than your typical rides")
            elif avg_incidents > median_incidents * 1.2:
                traits.append("You encounter more varied riding conditions")
        
        if 'avg_braking_events' in meaningful_features and len(df) > 1:
            avg_braking = df['avg_braking_events'].mean()
            median_braking = df['avg_braking_events'].median()
            if avg_braking < median_braking:
                traits.append("You brake smoothly and predictably")
            else:
                traits.append("You're responsive to changing conditions")
        
        if 'precipitation_mm' in meaningful_features:
            rides_in_rain = (df['precipitation_mm'] > 0).sum() if 'precipitation_mm' in df.columns else 0
            total_rides = len(df)
            if total_rides > 0:
                rain_percentage = rides_in_rain / total_rides * 100
                
                if rain_percentage > 25:
                    traits.append("You're a dedicated all-weather cyclist")
                elif rain_percentage < 5:
                    traits.append("You prefer fair weather riding")
        
        if 'temperature' in meaningful_features and len(df) > 1:
            temp_range = df['temperature'].max() - df['temperature'].min()
            if temp_range > 15:
                traits.append("You cycle across diverse weather conditions")
            elif df['temperature'].mean() > 20:
                traits.append("You prefer warmer cycling conditions")
            elif df['temperature'].mean() < 10:
                traits.append("You embrace cooler cycling weather")
        
        # Ensure we have at least some traits
        if len(traits) == 0:
            # Cluster-based fallback traits
            n_clusters = len(set(clusters))
            if n_clusters > 3:
                traits.append("You have a diverse cycling style with multiple patterns")
            elif n_clusters == 2:
                traits.append("You have two distinct cycling patterns")
            else:
                traits.append("You have a consistent cycling style")
            
            traits.append("Your riding patterns are developing")
        
        return traits[:4]  # Return top 4 traits
        
    except Exception as e:
        logger.error(f"Error generating traits: {e}")
        return ["You have a unique cycling style"]


def detect_dynamic_intelligent_alerts(df, meaningful_features, options):
    """Detect intelligent safety alerts using dynamic data"""
    try:
        # Prepare feature matrix
        X = df[meaningful_features].copy()
        for col in meaningful_features:
            X[col] = X[col].fillna(X[col].median())
        
        # Detect anomalies with adjusted contamination for smaller datasets
        effective_contamination = min(options['anomaly_contamination'], 0.2)  # Cap at 20%
        if len(df) < 20:
            effective_contamination = max(effective_contamination, 1/len(df))  # At least 1 anomaly
        
        isolation_forest = IsolationForest(
            contamination=effective_contamination,
            random_state=42
        )
        anomalies = isolation_forest.fit_predict(X)
        
        # Create intelligent alerts
        alert_mask = anomalies == -1
        alert_data = df[alert_mask].copy()
        
        if len(alert_data) == 0:
            return {
                'priority_alerts': [],
                'summary_stats': {'total_alerts': 0, 'safe_days': len(df), 'anomaly_rate': 0}
            }
        
        # Generate intelligent alert descriptions
        priority_alerts = []
        for _, row in alert_data.iterrows():
            alert = generate_dynamic_intelligent_alert_description(row, meaningful_features, df)
            priority_alerts.append(alert)
        
        # Sort by severity
        priority_alerts.sort(key=lambda x: x['severity'], reverse=True)
        
        # Create timeline
        alert_timeline = create_dynamic_alert_timeline(alert_data, df)
        
        # Calculate summary stats
        anomaly_rate = len(alert_data) / len(df) * 100
        summary_stats = {
            'total_alerts': len(alert_data),
            'safe_days': len(df) - len(alert_data),
            'anomaly_rate': anomaly_rate
        }
        
        return {
            'priority_alerts': priority_alerts,
            'alert_timeline': alert_timeline,
            'summary_stats': summary_stats
        }
        
    except Exception as e:
        logger.error(f"Error in dynamic intelligent alerts: {e}")
        return None


def analyze_dynamic_intelligent_safety_factors(df, meaningful_features):
    """Analyze safety factors using dynamic data"""
    try:
        # Calculate meaningful correlations
        feature_matrix = df[meaningful_features].copy()
        for col in meaningful_features:
            feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())
        
        correlation_matrix = feature_matrix.corr()
        
        # Create safety target for factor analysis
        safety_target = create_intelligent_safety_target(df, meaningful_features)
        
        if safety_target is None:
            return None
        
        # Calculate factor rankings based on correlation with safety
        factor_rankings = []
        for feature in meaningful_features:
            try:
                feature_values = feature_matrix[feature]
                if len(feature_values.unique()) > 1:  # Only if feature has variance
                    correlation_with_safety = np.corrcoef(feature_values, safety_target)[0, 1]
                    if not np.isnan(correlation_with_safety):
                        impact_score = abs(correlation_with_safety)
                        
                        factor_rankings.append({
                            'factor_name': make_feature_friendly(feature),
                            'impact_score': impact_score,
                            'correlation': correlation_with_safety
                        })
            except:
                continue
        
        factor_rankings_df = pd.DataFrame(factor_rankings)
        if not factor_rankings_df.empty:
            factor_rankings_df = factor_rankings_df.sort_values('impact_score', ascending=True)
        
        # Find smart correlations between meaningful factors
        smart_correlations = find_dynamic_smart_correlations(correlation_matrix, meaningful_features)
        
        # Generate key insights
        key_insights = generate_dynamic_factor_insights(factor_rankings_df, df, meaningful_features)
        
        return {
            'factor_rankings': factor_rankings_df,
            'smart_correlations': smart_correlations,
            'key_insights': key_insights,
            'correlation_matrix': correlation_matrix
        }
        
    except Exception as e:
        logger.error(f"Error in safety factors analysis: {e}")
        return None


def convert_predictions_to_safety_scores(predictions):
    """Convert model predictions to friendly 1-10 safety scores"""
    if len(predictions) == 0:
        return np.array([5.0])
    
    # Normalize predictions to 0-1 range
    min_pred = predictions.min()
    max_pred = predictions.max()
    
    if max_pred > min_pred:
        normalized = (predictions - min_pred) / (max_pred - min_pred)
    else:
        normalized = np.ones_like(predictions) * 0.5
    
    # Convert to 1-10 scale
    safety_scores = 1 + (normalized * 9)
    return safety_scores


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
        'severity_score': '🔥 Severity Level'
    }
    
    return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())


def find_dynamic_smart_correlations(correlation_matrix, meaningful_features):
    """Find meaningful correlations between factors using dynamic data"""
    correlations = []
    
    try:
        for i in range(len(meaningful_features)):
            for j in range(i+1, len(meaningful_features)):
                corr_val = correlation_matrix.iloc[i, j]
                
                if not np.isnan(corr_val) and abs(corr_val) > 0.3:  # Only meaningful correlations
                    factor1 = make_feature_friendly(meaningful_features[i])
                    factor2 = make_feature_friendly(meaningful_features[j])
                    
                    relationship_type = "Positive" if corr_val > 0 else "Negative"
                    
                    correlations.append({
                        'factor_pair': f"{factor1} ↔ {factor2}",
                        'factor_1_impact': abs(corr_val),
                        'factor_2_impact': abs(corr_val),
                        'connection_strength': abs(corr_val),
                        'relationship_type': relationship_type
                    })
        
        return pd.DataFrame(correlations).sort_values('connection_strength', ascending=False) if correlations else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error finding correlations: {e}")
        return pd.DataFrame()


def generate_dynamic_intelligent_alert_description(row, meaningful_features, full_df):
    """Generate intelligent, contextual alert descriptions using dynamic data"""
    try:
        date_str = row.get('date', datetime.now().strftime('%Y-%m-%d'))
        if pd.isna(date_str):
            date_str = "Recent"
        else:
            date_str = str(date_str)[:10]  # Format date
        
        # Analyze what made this record unusual compared to the dataset
        alert_reasons = []
        severity_factors = []
        
        for feature in meaningful_features:
            if feature in row.index and feature in full_df.columns:
                row_value = row[feature]
                if pd.isna(row_value):
                    continue
                    
                feature_data = full_df[feature].dropna()
                
                if len(feature_data) > 1:
                    mean_val = feature_data.mean()
                    std_val = feature_data.std()
                    
                    if std_val > 0:
                        z_score = abs((row_value - mean_val) / std_val)
                        severity_factors.append(z_score)
                        
                        if z_score > 1.5:  # Significantly different
                            direction = "much higher" if row_value > mean_val else "much lower"
                            friendly_name = make_feature_friendly(feature).replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip()
                            alert_reasons.append(f"{direction} {friendly_name.lower()} ({row_value:.1f} vs typical {mean_val:.1f})")
        
        # Calculate severity based on how unusual the combination is
        if severity_factors:
            avg_z_score = np.mean(severity_factors)
            severity = min(0.9, max(0.3, avg_z_score / 3))  # Scale to 0.3-0.9 range
        else:
            severity = 0.5
        
        # Create alert description
        if len(alert_reasons) > 0:
            main_reason = alert_reasons[0]
            title = "Unusual Riding Conditions"
            if len(alert_reasons) > 1:
                description = f"Detected {main_reason} and {len(alert_reasons)-1} other unusual factors"
            else:
                description = f"Detected {main_reason}"
        else:
            title = "Pattern Anomaly"
            description = "Unusual combination of riding conditions detected"
        
        return {
            'date': date_str,
            'title': title,
            'description': description,
            'severity': severity,
            'factors': alert_reasons
        }
        
    except Exception as e:
        logger.error(f"Error generating alert description: {e}")
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Safety Alert',
            'description': 'Unusual pattern detected',
            'severity': 0.5,
            'factors': []
        }


def create_dynamic_alert_timeline(alert_data, full_df):
    """Create timeline of alerts using dynamic data"""
    try:
        if 'date' in alert_data.columns and 'date' in full_df.columns:
            # Create timeline with actual dates
            alert_data_copy = alert_data.copy()
            alert_data_copy['date'] = pd.to_datetime(alert_data_copy['date'])
            
            timeline = alert_data_copy.groupby(alert_data_copy['date'].dt.date).size().reset_index()
            timeline.columns = ['date', 'alert_count']
            
            # Fill missing dates with 0 for the full date range
            full_df_copy = full_df.copy()
            full_df_copy['date'] = pd.to_datetime(full_df_copy['date'])
            date_range = pd.date_range(
                start=full_df_copy['date'].min().date(),
                end=full_df_copy['date'].max().date(),
                freq='D'
            )
            
            full_timeline = pd.DataFrame({'date': date_range.date})
            full_timeline = full_timeline.merge(timeline, on='date', how='left')
            full_timeline['alert_count'] = full_timeline['alert_count'].fillna(0)
            
            return full_timeline
        else:
            # Create simple timeline
            return pd.DataFrame({
                'date': [datetime.now().date()],
                'alert_count': [len(alert_data)]
            })
            
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        return pd.DataFrame({'date': [datetime.now().date()], 'alert_count': [0]})


def generate_dynamic_factor_insights(factor_rankings_df, df, meaningful_features):
    """Generate key insights about safety factors using dynamic data"""
    insights = {}
    
    try:
        # Primary safety factor from actual rankings
        if not factor_rankings_df.empty:
            top_factor = factor_rankings_df.iloc[-1]['factor_name']
            insights['primary_factor'] = top_factor.replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').replace('🌧️', '').strip()
        else:
            insights['primary_factor'] = 'Speed'
        
        # Determine optimal conditions from actual data
        optimal_conditions = "Clear Weather"  # Default
        
        if 'temperature' in meaningful_features and len(df) > 1:
            # Check if incidents correlate with temperature
            if 'incidents' in meaningful_features:
                temp_incident_corr = df[['temperature', 'incidents']].corr().iloc[0, 1]
                if not np.isnan(temp_incident_corr):
                    if temp_incident_corr < -0.2:  # Negative correlation = higher temp, fewer incidents
                        optimal_conditions = "Warm Weather"
                    elif temp_incident_corr > 0.2:
                        optimal_conditions = "Cool Weather"
        
        if 'precipitation_mm' in meaningful_features and len(df) > 1:
            avg_precip = df['precipitation_mm'].mean()
            if avg_precip < 1:
                optimal_conditions = "Dry Conditions"
            elif 'incidents' in meaningful_features:
                precip_incident_corr = df[['precipitation_mm', 'incidents']].corr().iloc[0, 1]
                if not np.isnan(precip_incident_corr) and precip_incident_corr > 0.2:
                    optimal_conditions = "Dry Conditions"
        
        insights['optimal_conditions'] = optimal_conditions
        
        # Calculate real improvement potential
        improvement_potential = 15  # Default
        
        if 'incidents' in meaningful_features and len(df) > 1:
            incidents = df['incidents']
            min_incidents = incidents.quantile(0.1)
            avg_incidents = incidents.mean()
            if avg_incidents > 0:
                potential = ((avg_incidents - min_incidents) / avg_incidents) * 100
                improvement_potential = max(5, min(50, potential))
        elif 'avg_braking_events' in meaningful_features and len(df) > 1:
            braking = df['avg_braking_events']
            min_braking = braking.quantile(0.1)
            avg_braking = braking.mean()
            if avg_braking > 0:
                potential = ((avg_braking - min_braking) / avg_braking) * 100
                improvement_potential = max(5, min(40, potential))
        
        insights['improvement_potential'] = improvement_potential
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {
            'primary_factor': 'Speed',
            'optimal_conditions': 'Clear Weather',
            'improvement_potential': 20
        }


# Dynamic AI-Generated Insight Functions

def generate_dynamic_safety_intelligence_insight(prediction_results, safety_scores, data_points):
    """Generate AI insight for safety intelligence using dynamic data"""
    try:
        avg_score = np.mean(safety_scores)
        accuracy = prediction_results['accuracy']
def generate_dynamic_safety_intelligence_insight(prediction_results, safety_scores, data_points):
    """Generate AI insight for safety intelligence using dynamic data"""
    try:
        avg_score = np.mean(safety_scores)
        accuracy = prediction_results['accuracy']
        top_factors = prediction_results['feature_importance'].tail(3)['friendly_name'].tolist()
        top_factors_clean = [f.replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').strip() for f in top_factors]
        
        # Determine model quality based on actual performance
        if accuracy > 0.7:
            model_quality = "excellent"
            confidence = "high confidence"
        elif accuracy > 0.4:
            model_quality = "good"
            confidence = "moderate confidence"
        else:
            model_quality = "developing"
            confidence = "preliminary insights"
        
        # Determine safety profile based on actual scores
        if avg_score > 7:
            insight_tone = "excellent"
            improvement = "fine-tuning"
        elif avg_score > 5:
            insight_tone = "good"
            improvement = "optimizing"
        else:
            insight_tone = "developing"
            improvement = "improving"
        
        # Get date range info
        date_info = ""
        if 'filter_start_date' in st.session_state and st.session_state.filter_start_date:
            date_info = " for the selected period"
        
        insight_text = f"""
        🎯 **Your safety profile is {insight_tone}!** Based on {data_points} analyzed scenarios{date_info}, 
        your average safety score is **{avg_score:.1f}/10**. 
        
        🔍 **Key Finding**: Your top 3 safety factors are **{', '.join(top_factors_clean)}**. 
        Focus on {improvement} these areas for maximum safety impact.
        
        💡 **AI Assessment**: Our {model_quality} prediction model (accuracy: {accuracy:.1%}) shows {confidence} in these insights. 
        {"Small improvements in your top factor could boost safety significantly!" if accuracy > 0.5 else "Collect more data for stronger predictions."}
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating safety insight: {e}")


def generate_dynamic_cycling_dna_insight(pattern_results, data_points):
    """Generate AI insight for cycling DNA using dynamic data"""
    try:
        if 'persona_distribution' in pattern_results and not pattern_results['persona_distribution'].empty:
            top_persona_row = pattern_results['persona_distribution'].loc[
                pattern_results['persona_distribution']['percentage'].idxmax()
            ]
            top_persona = top_persona_row['persona']
            percentage = top_persona_row['percentage']
        else:
            top_persona = "Balanced Rider"
            percentage = 60
        
        traits = pattern_results.get('personality_traits', [])
        trait_summary = traits[0] if traits else "You have a unique cycling style"
        
        silhouette_score = pattern_results.get('silhouette_score', 0)
        pattern_quality = "very distinct" if silhouette_score > 0.5 else "moderately distinct" if silhouette_score > 0.3 else "emerging"
        
        # Get date range info
        date_info = ""
        if 'filter_start_date' in st.session_state and st.session_state.filter_start_date:
            date_info = " during the selected period"
        
        insight_text = f"""
        🧬 **You're primarily a {top_persona}** - this represents **{percentage:.0f}%** of your riding style{date_info}!
        
        🎭 **Pattern Analysis**: We found {pattern_results['n_patterns']} {pattern_quality} riding patterns from {data_points} data points. 
        **{trait_summary}** This suggests you prioritize {"safety and consistency" if "safety" in trait_summary.lower() or "smooth" in trait_summary.lower() else "performance and efficiency" if "speed" in trait_summary.lower() else "adaptability and awareness" if "responsive" in trait_summary.lower() else "comfort and enjoyment"}.
        
        📈 **Pattern Evolution**: {"Your cycling patterns are very consistent and predictable" if pattern_quality == "very distinct" else "Your patterns are well-defined and stable" if pattern_quality == "moderately distinct" else "Your patterns are still developing - more data will reveal clearer clusters"}!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating DNA insight: {e}")


def generate_dynamic_smart_alerts_insight(alert_results, data_points):
    """Generate AI insight for smart alerts using dynamic data"""
    try:
        total_alerts = alert_results['summary_stats'].get('total_alerts', 0)
        safe_days = alert_results['summary_stats'].get('safe_days', 0)
        anomaly_rate = alert_results['summary_stats'].get('anomaly_rate', 0)
        
        # Determine alert status based on actual data
        if total_alerts == 0:
            alert_status = "🎉 **Outstanding safety record!** No alerts detected in the analyzed period."
            advice = "Keep up your excellent riding habits!"
            trend_assessment = "exceptional consistency"
        elif anomaly_rate <= 5:
            alert_status = f"✅ **Great safety performance!** Only {total_alerts} alerts out of {data_points} records ({anomaly_rate:.1f}%)."
            advice = "You're maintaining excellent safety practices."
            trend_assessment = "excellent consistency"
        elif anomaly_rate <= 15:
            alert_status = f"⚠️ **{total_alerts} alerts detected** out of {data_points} records ({anomaly_rate:.1f}%)."
            advice = "Consider reviewing the alert patterns to identify improvement opportunities."
            trend_assessment = "good consistency with room for optimization"
        else:
            alert_status = f"🔍 **{total_alerts} alerts detected** - higher than typical ({anomaly_rate:.1f}% anomaly rate)."
            advice = "Focus on identifying and addressing the key risk factors."
            trend_assessment = "variable patterns with optimization opportunities"
        
        # Get date range info
        date_info = ""
        if 'filter_start_date' in st.session_state and st.session_state.filter_start_date:
            date_info = " during the selected time period"
        
        insight_text = f"""
        {alert_status}
        
        🔥 **Safe Records**: {safe_days} out of {data_points} records{date_info} were flagged as safe! 
        
        🧠 **AI Assessment**: {advice} Analysis shows {trend_assessment}. 
        {"Your risk management is excellent" if anomaly_rate <= 5 else "Your safety patterns are developing well" if anomaly_rate <= 15 else "There are clear opportunities to improve safety consistency"}.
        
        📊 **Data Quality**: {"Strong statistical significance with reliable patterns" if data_points > 30 else "Moderate data size - trends are emerging" if data_points > 10 else "Limited data - collect more for stronger insights"}.
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating alerts insight: {e}")


def generate_dynamic_safety_factors_insight(factor_results, data_points):
    """Generate AI insight for safety factors using dynamic data"""
    try:
        if not factor_results['factor_rankings'].empty:
            top_factor_row = factor_results['factor_rankings'].iloc[-1]
            top_factor = top_factor_row['factor_name']
            top_impact = top_factor_row['impact_score']
        else:
            top_factor = "Speed Management"
            top_impact = 0.6
        
        key_insights = factor_results['key_insights']
        improvement_potential = key_insights.get('improvement_potential', 20)
        correlations_count = len(factor_results['smart_correlations'])
        
        # Remove emojis from factor name for cleaner text
        clean_top_factor = top_factor.replace('🚴‍♀️', '').replace('🏃‍♂️', '').replace('⚡', '').replace('🌧️', '').strip()
        
        # Assess correlation strength
        correlation_strength = "strong interconnections" if correlations_count > 3 else "moderate relationships" if correlations_count > 0 else "independent factors"
        
        # Get date range info
        date_info = ""
        if 'filter_start_date' in st.session_state and st.session_state.filter_start_date:
            date_info = " for the selected period"
        
        insight_text = f"""
        ⚗️ **Discovery**: **{clean_top_factor}** has the strongest impact on your safety (influence score: {top_impact:.3f}) based on {data_points} data points{date_info}.
        
        🎯 **Optimization Opportunity**: Under optimal conditions ({key_insights.get('optimal_conditions', 'clear weather').lower()}), 
        you could be **{improvement_potential:.0f}% safer** than your current average performance.
        
        🔗 **Factor Analysis**: Found {correlations_count} meaningful relationships between safety factors, indicating {correlation_strength}. 
        {"Understanding these connections is key to holistic safety improvement" if correlations_count > 0 else "Your safety factors work independently, allowing focused optimization"}.
        
        🚀 **Action Plan**: {"Focus on your top factor during optimal conditions for maximum safety ROI" if improvement_potential > 15 else "Fine-tune your approach to your top factor for marginal gains"}!
        """
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%); padding: 20px; border-radius: 15px; color: #333; margin: 20px 0;'>
        {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error generating factors insight: {e}")


# Keep the original function name for compatibility
def render_ml_insights_page():
    """Wrapper to maintain compatibility with existing code"""
    render_smart_insights_page()"""
Smart Insights Page for SeeSense Dashboard - Dynamic User-Friendly Version
Beautiful interface with real computations that respond to date filters
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
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
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
    """Render the Smart Insights page with dynamic data based on filters"""
    st.title("🧠 Smart Insights")
    st.markdown("**AI discovers actionable patterns in your cycling data to keep you safer**")
    
    # Add helpful explanation with modern styling
    with st.expander("ℹ️ What are Smart Insights?", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;'>
        <h4 style='color: white; margin-top: 0;'>🤖 Your Personal Safety AI</h4>
        Our advanced AI analyzes your cycling data to discover hidden patterns and predict safety risks.
        Think of it as having a smart cycling coach that learns from thousands of rides!
        </div>
        
        **What you'll discover:**
        - 🎯 **Safety Predictions** - Which conditions lead to higher risks
        - 👥 **Riding Patterns** - Your unique cycling personality and habits  
        - ⚠️ **Safety Alerts** - When conditions become unusually risky
        - 📊 **Smart Factors** - What really affects your safety (and what doesn't)
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
        
        # Apply date filters from session state (set by Overview page)
        routes_df, braking_df, swerving_df, time_series_df = apply_date_filters(
            routes_df, braking_df, swerving_df, time_series_df
        )
        
        # Add simple controls in sidebar
        smart_options = render_simple_controls()
        
        # Create modern tabs with emojis
        safety_tab, patterns_tab, alerts_tab, insights_tab = st.tabs([
            "🎯 Safety Intelligence", 
            "👥 Your Cycling DNA", 
            "⚠️ Smart Alerts", 
            "🧬 Safety Factors"
        ])
        
        with safety_tab:
            render_safety_intelligence(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
        with patterns_tab:
            render_cycling_dna(routes_df, time_series_df, smart_options)
        
        with alerts_tab:
            render_smart_alerts(time_series_df, braking_df, swerving_df, smart_options)
        
        with insights_tab:
            render_safety_factors_analysis(routes_df, braking_df, swerving_df, time_series_df, smart_options)
        
    except Exception as e:
        logger.error(f"Error in Smart Insights page: {e}")
        st.error("⚠️ Something went wrong while analyzing your data.")
        st.info("Please check your data files and try refreshing the page.")
        
        with st.expander("🔍 Technical Details"):
            st.code(str(e))


def apply_date_filters(routes_df, braking_df, swerving_df, time_series_df):
    """Apply date filters from session state to all dataframes"""
    try:
        # Get date filters from session state (set by Overview page)
        start_date = st.session_state.get('filter_start_date')
        end_date = st.session_state.get('filter_end_date')
        
        if start_date is None or end_date is None:
            return routes_df, braking_df, swerving_df, time_series_df
        
        # Convert to datetime for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter time series data
        if time_series_df is not None and 'date' in time_series_df.columns:
            time_series_df = time_series_df.copy()
            time_series_df['date'] = pd.to_datetime(time_series_df['date'])
            mask = (time_series_df['date'] >= start_date) & (time_series_df['date'] <= end_date)
            time_series_df = time_series_df[mask]
        
        # Filter braking hotspots data
        if braking_df is not None and 'date_recorded' in braking_df.columns:
            braking_df = braking_df.copy()
            braking_df['date_recorded'] = pd.to_datetime(braking_df['date_recorded'])
            mask = (braking_df['date_recorded'] >= start_date) & (braking_df['date_recorded'] <= end_date)
            braking_df = braking_df[mask]
        
        # Filter swerving hotspots data
        if swerving_df is not None and 'date_recorded' in swerving_df.columns:
            swerving_df = swerving_df.copy()
            swerving_df['date_recorded'] = pd.to_datetime(swerving_df['date_recorded'])
            mask = (swerving_df['date_recorded'] >= start_date) & (swerving_df['date_recorded'] <= end_date)
            swerving_df = swerving_df[mask]
        
        # Note: Routes data doesn't typically have dates, so we keep it as is
        # unless there's a specific date column
        
        return routes_df, braking_df, swerving_df, time_series_df
        
    except Exception as e:
        logger.error(f"Error applying date filters: {e}")
        return routes_df, braking_df, swerving_df, time_series_df


def render_no_data_message():
    """Render modern no-data message"""
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; color: white;'>
    <h2 style='color: white;'>🚀 Ready to Unlock Your Cycling Insights?</h2>
    <p style='font-size: 18px; margin: 20px 0;'>Upload your cycling data to discover amazing patterns!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 What We Need
    
    **📍 Route Data** - Where you've been cycling  
    **⏱️ Daily Stats** - Your ride history and metrics  
    **🚨 Safety Events** - Braking and swerving incidents
    
    Once you add your data files, our AI will reveal insights you never knew existed! 🎉
    """)


def render_simple_controls():
    """Render modern, user-friendly controls"""
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
    <h3 style='color: white; margin: 0;'>⚙️ AI Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    options = {}
    
    # Simplified controls with better UX
    options['sensitivity'] = st.sidebar.radio(
        "🔍 Alert Sensitivity",
        ["🟢 Relaxed", "🟡 Balanced", "🔴 Vigilant"],
        index=1,
        help="How sensitive should safety alerts be?"
    )
    
    # Convert to technical values
    sensitivity_map = {"🟢 Relaxed": 0.1, "🟡 Balanced": 0.05, "🔴 Vigilant": 0.02}
    options['anomaly_contamination'] = sensitivity_map[options['sensitivity']]
    
    options['prediction_period'] = st.sidebar.selectbox(
        "🔮 Prediction Horizon",
        ["📅 Next Week", "📊 Next 2 Weeks", "📈 Next Month", "🎯 Next Quarter"],
        index=2,
        help="How far ahead should we predict safety trends?"
    )
    
    # Convert to days
    period_map = {"📅 Next Week": 7, "📊 Next 2 Weeks": 14, "📈 Next Month": 30, "🎯 Next Quarter": 90}
    options['prediction_days'] = period_map[options['prediction_period']]
    
    options['pattern_detail'] = st.sidebar.selectbox(
        "🎨 Pattern Detail",
        ["🔍 Simple (2-3 patterns)", "⚖️ Moderate (4-5 patterns)", "🎯 Detailed (6-8 patterns)"],
        index=1,
        help="How detailed should pattern analysis be?"
    )
    
    # Convert to clusters
    detail_map = {"🔍 Simple (2-3 patterns)": 3, "⚖️ Moderate (4-5 patterns)": 4, "🎯 Detailed (6-8 patterns)": 6}
    options['n_clusters'] = detail_map[options['pattern_detail']]
    
    options['min_data_needed'] = 10  # Reduced for filtered data
    
    return options


def get_meaningful_features(df):
    """Extract only meaningful features for analysis, excluding coordinates and IDs"""
    if df is None or df.empty:
        return []
    
    # Define meaningful feature categories
    meaningful_patterns = [
        'speed', 'duration', 'distance', 'incidents', 'braking', 'swerving', 
        'temperature', 'precipitation', 'wind', 'visibility', 'intensity',
        'popularity', 'rating', 'days_active', 'cyclists', 'severity',
        'deceleration', 'lateral', 'total_rides'
    ]
    
    # Filter columns to only meaningful ones
    all_columns = df.columns.tolist()
    meaningful_columns = []
    
    for col in all_columns:
        col_lower = col.lower()
        # Include if matches meaningful patterns and exclude coordinates/IDs
        if any(pattern in col_lower for pattern in meaningful_patterns):
            if not any(exclude in col_lower for exclude in ['lat', 'lon', 'id', '_id', 'start_', 'end_']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique() > 1 and df[col].std() > 0:
                        meaningful_columns.append(col)
    
    return meaningful_columns


def render_safety_intelligence(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render advanced safety predictions with dynamic data"""
    st.markdown("### 🎯 Safety Intelligence")
    
    # Show date filter info if active
    show_filter_info()
    
    # Create AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🤖 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Analyzing your cycling patterns to predict when and where you're most at risk...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series data as primary source for meaningful analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info(f"🔄 Need at least {options['min_data_needed']} records for reliable predictions. {get_data_availability_message(primary_df)}")
        return
    
    # Get meaningful features only
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Not enough meaningful data for safety predictions. Add more cycling metrics!")
        return
    
    # Create safety predictions with meaningful variables
    prediction_results = create_dynamic_safety_predictions(primary_df, meaningful_features)
    
    if prediction_results is None:
        st.warning("🤔 Our AI couldn't find clear patterns in the filtered data. Try expanding the date range!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 What Drives Your Safety")
        
        importance_data = prediction_results['feature_importance']
        
        # Create beautiful, meaningful chart
        fig = px.bar(
            importance_data.head(8),  # Top 8 factors
            x='importance',
            y='friendly_name',
            orientation='h',
            title="Your Personal Safety Factors",
            labels={'importance': 'Impact Level', 'friendly_name': ''},
            color='importance',
            color_continuous_scale='Viridis',
            text='importance'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(height=400, showlegend=False, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎲 Your Safety Score Range")
        
        predictions = prediction_results['predictions']
        safety_scores = convert_predictions_to_safety_scores(predictions)
        
        # Create modern histogram
        fig = px.histogram(
            x=safety_scores,
            nbins=15,
            title="Distribution of Your Safety Scores",
            labels={'x': 'Safety Score (1=High Risk, 10=Very Safe)', 'y': 'Frequency'},
            color_discrete_sequence=['#6366f1']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add modern metrics
        avg_score = np.mean(safety_scores)
        score_std = np.std(safety_scores)
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric(
                "🏅 Average Safety Score", 
                f"{avg_score:.1f}/10",
                help="Your typical safety level across all conditions"
            )
        with col2b:
            consistency = "High" if score_std < 1 else "Medium" if score_std < 2 else "Variable"
            st.metric(
                "📊 Consistency",
                consistency,
                help="How consistent your safety scores are"
            )
    
    # AI-generated insight
    generate_dynamic_safety_intelligence_insight(prediction_results, safety_scores, len(primary_df))


def render_cycling_dna(routes_df, time_series_df, options):
    """Render personality-based cycling analysis with dynamic data"""
    st.markdown("### 👥 Your Cycling DNA")
    
    # Show date filter info if active
    show_filter_info()
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🧬 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Discovering your unique cycling personality and riding patterns...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series for richer pattern analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info(f"🧬 Need more data for cycling DNA analysis. {get_data_availability_message(primary_df)}")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Not enough cycling metrics to determine your patterns yet!")
        return
    
    # Analyze cycling patterns
    pattern_results = analyze_dynamic_cycling_dna(primary_df, meaningful_features, options['n_clusters'])
    
    if pattern_results is None:
        st.warning("🤔 Your cycling patterns are still emerging. Keep riding!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎭 Your Cycling Personas")
        
        if 'persona_distribution' in pattern_results:
            persona_data = pattern_results['persona_distribution']
            
            # Create stunning pie chart
            fig = px.pie(
                persona_data,
                values='percentage',
                names='persona',
                title="How You Spend Your Cycling Time",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Your Pattern Evolution")
        
        if 'pattern_timeline' in pattern_results and time_series_df is not None:
            timeline_data = pattern_results['pattern_timeline']
            
            fig = px.line(
                timeline_data,
                x='date',
                y='dominant_persona',
                title="How Your Cycling Style Evolves",
                labels={'dominant_persona': 'Primary Cycling Style', 'date': 'Date'},
                color_discrete_sequence=['#8b5cf6']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show personality traits instead
            st.markdown("**🎯 Your Cycling Traits:**")
            if 'personality_traits' in pattern_results:
                for trait in pattern_results['personality_traits']:
                    st.markdown(f"✨ {trait}")
    
    # AI-generated insight
    generate_dynamic_cycling_dna_insight(pattern_results, len(primary_df))


def render_smart_alerts(time_series_df, braking_df, swerving_df, options):
    """Render intelligent safety alerts with dynamic data"""
    st.markdown("### ⚠️ Smart Safety Alerts")
    
    # Show date filter info if active
    show_filter_info()
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>🔮 AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Monitoring unusual patterns and potential safety risks in real-time...</p>
        </div>
        """, unsafe_allow_html=True)
    
    if time_series_df is None or len(time_series_df) < options['min_data_needed']:
        st.info(f"⏳ Need more daily data to detect unusual patterns. {get_data_availability_message(time_series_df)}")
        return
    
    # Get meaningful features for anomaly detection
    meaningful_features = get_meaningful_features(time_series_df)
    
    if len(meaningful_features) < 2:
        st.warning("🔍 Need more safety metrics to detect unusual patterns!")
        return
    
    # Detect smart alerts
    alert_results = detect_dynamic_intelligent_alerts(time_series_df, meaningful_features, options)
    
    if alert_results is None:
        st.warning("🤔 No unusual patterns detected in your recent rides!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Recent Safety Alerts")
        
        if 'priority_alerts' in alert_results and len(alert_results['priority_alerts']) > 0:
            alerts = alert_results['priority_alerts']
            
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_color = "🔴" if alert['severity'] > 0.7 else "🟡" if alert['severity'] > 0.4 else "🟢"
                
                st.markdown(f"""
                <div style='border-left: 4px solid {"#ef4444" if alert["severity"] > 0.7 else "#f59e0b" if alert["severity"] > 0.4 else "#10b981"}; 
                           padding: 15px; margin: 10px 0; background: #f8fafc; border-radius: 0 8px 8px 0;'>
                <strong>{severity_color} {alert['title']}</strong><br>
                <span style='color: #64748b;'>{alert['description']}</span><br>
                <small style='color: #94a3b8;'>📅 {alert['date']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 15px;'>
            <h4 style='color: #155724; margin-top: 0;'>🎉 All Clear!</h4>
            <p style='color: #155724; margin-bottom: 0;'>No safety alerts detected in the selected period. Your rides have been consistently safe!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Alert Trends")
        
        if 'alert_timeline' in alert_results:
            timeline = alert_results['alert_timeline']
            
            fig = px.area(
                timeline,
                x='date',
                y='alert_count',
                title="Safety Alert Trends Over Time",
                labels={'alert_count': 'Daily Alerts', 'date': 'Date'},
                color_discrete_sequence=['#f59e0b']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Modern summary metrics
        if 'summary_stats' in alert_results:
            stats = alert_results['summary_stats']
            
            st.metric(
                "🚨 Alerts in Period", 
                stats.get('total_alerts', 0),
                help="Safety alerts in the selected time period"
            )
            
            safe_days = stats.get('safe_days', 0)
            st.metric(
                "🔥 Safe Days", 
                f"{safe_days} days",
                help="Days without safety alerts in selected period"
            )
    
    # AI-generated insight
    generate_dynamic_smart_alerts_insight(alert_results, len(time_series_df))


def render_safety_factors_analysis(routes_df, braking_df, swerving_df, time_series_df, options):
    """Render intelligent analysis of what affects safety with dynamic data"""
    st.markdown("### 🧬 What Really Affects Your Safety")
    
    # Show date filter info if active
    show_filter_info()
    
    # Modern AI insight card
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h4 style='margin-top: 0; color: #333;'>⚗️ AI Insight</h4>
        <p style='font-size: 16px; margin-bottom: 0; color: #555;'>Uncovering the hidden connections between conditions and your safety...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use time series for comprehensive factor analysis
    primary_df = time_series_df if time_series_df is not None and len(time_series_df) > options['min_data_needed'] else routes_df
    
    if primary_df is None or len(primary_df) < options['min_data_needed']:
        st.info(f"🔬 Need more data for factor analysis. {get_data_availability_message(primary_df)}")
        return
    
    # Get meaningful features
    meaningful_features = get_meaningful_features(primary_df)
    
    if len(meaningful_features) < 3:
        st.warning("🔍 Need more safety metrics to analyze factor relationships!")
        return
    
    # Analyze safety factors with meaningful variables only
    factor_results = analyze_dynamic_intelligent_safety_factors(primary_df, meaningful_features)
    
    if factor_results is None:
        st.warning("🤔 Couldn't find clear relationships between safety factors yet!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Safety Factor Rankings")
        
        if 'factor_rankings' in factor_results:
            rankings = factor_results['factor_rankings']
            
            # Create modern ranking chart
            fig = px.bar(
                rankings.head(10),
                x='impact_score',
                y='factor_name',
                orientation='h',
                title="Factors Ranked by Safety Impact",
                labels={'impact_score': 'Safety Impact Score', 'factor_name': ''},
                color='impact_score',
                color_continuous_scale='Plasma',
                text='impact_score'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔗 Smart Factor Connections")
        
        if 'smart_correlations' in factor_results:
            correlations = factor_results['smart_correlations']
            
            if not correlations.empty:
                # Create network-style correlation chart
                fig = px.scatter(
                    correlations,
                    x='factor_1_impact',
                    y='factor_2_impact', 
                    size='connection_strength',
                    color='relationship_type',
                    title="How Safety Factors Connect",
                    labels={
                        'factor_1_impact': 'Factor 1 Impact',
                        'factor_2_impact': 'Factor 2 Impact',
                        'connection_strength': 'Connection Strength'
                    },
                    hover_data=['factor_pair']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔍 No strong factor connections found in the selected period!")
    
    # Key insights summary
    if 'key_insights' in factor_results:
        insights = factor_results['key_insights']
        
        st.markdown("#### 💡 Key Discoveries")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏆 Top Safety Factor",
                insights.get('primary_factor', 'Speed'),
                help="The single most important factor for your safety"
            )
        
        with col2:
            st.metric(
                "🌟 Best Conditions",
                insights.get('optimal_conditions', 'Clear Weather'),
                help="When you're typically safest"
            )
        
        with col3:
            improvement = insights.get('improvement_potential', 0)
            st.metric(
                "🚀 Improvement Potential",
                f"{improvement:.0f}% safer",
                help="How much safer you could be with optimal conditions"
            )
    
    # AI-generated insight
    generate_dynamic_safety_factors_insight(factor_results, len(primary_df))


# Helper functions for dynamic data processing

def show_filter_info():
    """Show information about active date filters"""
    start_date = st.session_state.get('filter_start_date')
    end_date = st.session_state.get('filter_end_date')
    
    if start_date and end_date:
        st.info(f"📅 **Filtered Analysis:** {start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date} to {end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date}")


def get_data_availability_message(df):
    """Get friendly message about data availability"""
    if df is None:
        return "No data available for this period."
    else:
        return f"Currently
