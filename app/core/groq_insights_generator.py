"""
Groq-Powered Insights Generator for SeeSense Dashboard
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not installed. Install with: pip install groq")

logger = logging.getLogger(__name__)

@dataclass
class InsightSummary:
    """Represents a data-driven insight with context"""
    title: str
    description: str
    impact_level: str  # 'High', 'Medium', 'Low'
    category: str  # 'Safety', 'Infrastructure', 'Behavior', 'Operational'
    data_points: List[str]
    recommendations: List[str]
    confidence_score: float
    priority_rank: int

class GroqInsightsGenerator:
    """
    Generate intelligent insights using Groq AI for non-technical clients
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq insights generator
        """
        # Try Streamlit secrets first, then environment variables
        try:
            import streamlit as st
            self.api_key = api_key or st.secrets.get("GROQ_API_KEY") or os.getenv('GROQ_API_KEY')
        except:
            self.api_key = api_key or os.getenv('GROQ_API_KEY')
    
        self.client = None
        self.logger = logging.getLogger(__name__)
    
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.logger.info("Groq client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            self.logger.warning("Groq not available - using rule-based insights")
    
    def generate_comprehensive_insights(self, 
                                      metrics: Dict[str, Any], 
                                      routes_df: pd.DataFrame = None,
                                      hotspots_data: Dict[str, Any] = None,
                                      time_series_df: pd.DataFrame = None) -> List[InsightSummary]:
        """
        Generate comprehensive insights from all available data
        """
        insights = []
        
        # Generate different types of insights
        insights.extend(self._generate_safety_insights(metrics))
        insights.extend(self._generate_infrastructure_insights(metrics))
        insights.extend(self._generate_operational_insights(metrics))
        
        # Rank insights by importance
        insights = self._rank_insights_by_importance(insights)
        
        return insights
    
    def _generate_safety_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate safety-focused insights"""
        insights = []
        
        # Safety score insights
        if metrics.get('safety_score', 0) > 0:
            safety_score = metrics['safety_score']
            safety_delta = metrics.get('safety_delta', 'N/A')
            
            if safety_score >= 7:
                impact_level = 'Low'
                description = f"Your network maintains a good safety score of {safety_score:.1f}/10."
            elif safety_score >= 5:
                impact_level = 'Medium'
                description = f"Your network has a moderate safety score of {safety_score:.1f}/10."
            else:
                impact_level = 'High'
                description = f"Your network has a concerning safety score of {safety_score:.1f}/10."
            
            if safety_delta != 'N/A':
                description += f" Recent trend: {safety_delta}."
            
            recommendations = []
            if safety_score < 7:
                recommendations.append("Focus on addressing high-priority hotspots")
                recommendations.append("Implement targeted safety interventions")
            
            insights.append(InsightSummary(
                title="Network Safety Performance",
                description=description,
                impact_level=impact_level,
                category="Safety",
                data_points=[f"Safety Score: {safety_score:.1f}/10", f"Trend: {safety_delta}"],
                recommendations=recommendations,
                confidence_score=0.9,
                priority_rank=1
            ))
        
        return insights
    
    def _generate_infrastructure_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate infrastructure-focused insights"""
        insights = []
        
        # Infrastructure coverage
        if metrics.get('infrastructure_coverage', 0) >= 0:
            coverage = metrics['infrastructure_coverage']
            coverage_delta = metrics.get('infrastructure_delta', 'N/A')
            
            if coverage >= 80:
                impact_level = 'Low'
                description = f"Excellent infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
            elif coverage >= 50:
                impact_level = 'Medium'
                description = f"Moderate infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
            else:
                impact_level = 'High'
                description = f"Low infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
            
            if coverage_delta != 'N/A':
                description += f" Recent progress: {coverage_delta}."
            
            recommendations = []
            if coverage < 70:
                recommendations.append("Prioritize bike lane development on high-traffic routes")
                recommendations.append("Assess infrastructure gaps in popular cycling areas")
            
            insights.append(InsightSummary(
                title="Infrastructure Coverage Analysis",
                description=description,
                impact_level=impact_level,
                category="Infrastructure",
                data_points=[f"Coverage: {coverage:.1f}%", f"Trend: {coverage_delta}"],
                recommendations=recommendations,
                confidence_score=0.88,
                priority_rank=2
            ))
        
        return insights
    
    def _generate_operational_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate operational insights"""
        insights = []
        
        # Cycling volume analysis
        if metrics.get('avg_daily_rides', 0) > 0:
            daily_rides = metrics['avg_daily_rides']
            rides_delta = metrics.get('rides_delta', 'N/A')
            
            if daily_rides >= 1000:
                impact_level = 'Low'
                description = f"High cycling volume with {daily_rides:,} average daily rides."
            elif daily_rides >= 500:
                impact_level = 'Medium'
                description = f"Moderate cycling volume with {daily_rides:,} average daily rides."
            else:
                impact_level = 'High'
                description = f"Low cycling volume with {daily_rides:,} average daily rides."
            
            if rides_delta != 'N/A':
                description += f" Volume trend: {rides_delta}."
            
            recommendations = []
            if daily_rides < 500:
                recommendations.append("Investigate barriers to cycling adoption")
                recommendations.append("Consider promotional campaigns to increase ridership")
            
            insights.append(InsightSummary(
                title="Cycling Volume Analysis",
                description=description,
                impact_level=impact_level,
                category="Operational",
                data_points=[f"Daily Rides: {daily_rides:,}", f"Trend: {rides_delta}"],
                recommendations=recommendations,
                confidence_score=0.82,
                priority_rank=3
            ))
        
        return insights
    
    def _rank_insights_by_importance(self, insights: List[InsightSummary]) -> List[InsightSummary]:
        """Rank insights by importance and impact"""
        # Sort by impact level (High > Medium > Low) and then by confidence score
        impact_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        
        insights.sort(
            key=lambda x: (impact_weights[x.impact_level], x.confidence_score),
            reverse=True
        )
        
        # Update priority ranks
        for i, insight in enumerate(insights):
            insight.priority_rank = i + 1
        
        return insights
    
    def generate_executive_summary(self, insights: List[InsightSummary], 
                                 metrics: Dict[str, Any]) -> str:
        """Generate an executive summary"""
        if not self.client:
            return self._generate_fallback_summary(insights, metrics)
        
        try:
            # Create summary prompt
            prompt = f"""
            Create an executive summary for cycling safety data analysis.
            
            Network Overview:
            - Safety Score: {metrics.get('safety_score', 'N/A')}/10
            - Total Routes: {metrics.get('total_routes', 'N/A')}
            - Daily Rides: {metrics.get('avg_daily_rides', 'N/A'):,}
            - Infrastructure Coverage: {metrics.get('infrastructure_coverage', 'N/A')}%
            - Active Safety Hotspots: {metrics.get('total_hotspots', 'N/A')}
            
            Key Findings:
            """
            
            for insight in insights[:3]:  # Top 3 insights
                prompt += f"- {insight.title}: {insight.description}\n"
            
            prompt += """
            
            Please provide a concise executive summary (150-200 words) that:
            1. Highlights the current state of cycling safety
            2. Identifies top 3 priorities for improvement
            3. Suggests next steps for city planners
            4. Uses business-friendly language suitable for municipal leaders
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary with Groq: {e}")
            return self._generate_fallback_summary(insights, metrics)
    
    def _generate_fallback_summary(self, insights: List[InsightSummary], 
                                 metrics: Dict[str, Any]) -> str:
        """Generate a fallback summary without Groq"""
        safety_score = metrics.get('safety_score', 0)
        total_routes = metrics.get('total_routes', 0)
        daily_rides = metrics.get('avg_daily_rides', 0)
        
        # Determine overall status
        if safety_score >= 7:
            status = "good"
        elif safety_score >= 5:
            status = "moderate"
        else:
            status = "concerning"
        
        summary = f"""
        **Network Status**: Your cycling network shows {status} safety performance with a {safety_score:.1f}/10 safety score across {total_routes} routes serving {daily_rides:,} daily rides.
        
        **Key Priorities**:
        """
        
        # Add top 3 insights
        for i, insight in enumerate(insights[:3]):
            summary += f"\n{i+1}. {insight.title}: {insight.description}"
        
        summary += """
        
        **Recommended Actions**: Focus on addressing high-impact safety issues, improving infrastructure coverage, and implementing data-driven interventions to enhance cyclist safety and network performance.
        """
        
        return summary


# Initialize the insights generator
def create_insights_generator(groq_api_key: Optional[str] = None) -> GroqInsightsGenerator:
    """Create and return a configured insights generator"""
    return GroqInsightsGenerator(api_key=groq_api_key)
