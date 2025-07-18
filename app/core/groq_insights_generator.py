"""
FIXED Groq-Powered Insights Generator for SeeSense Dashboard
Handles version compatibility issues with Groq client
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import Groq with better error handling
GROQ_AVAILABLE = False
GROQ_VERSION = None

try:
    import groq
    from groq import Groq
    GROQ_AVAILABLE = True
    GROQ_VERSION = getattr(groq, '__version__', 'unknown')
    logging.info(f"Groq library loaded successfully, version: {GROQ_VERSION}")
except ImportError as e:
    logging.warning(f"Groq not available: {e}")
except Exception as e:
    logging.error(f"Error importing Groq: {e}")

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
    FIXED VERSION with better Groq client handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the insights generator with improved Groq handling
        """
        self.provided_api_key = api_key
        self._client = None
        self._api_key = None
        self._initialization_attempted = False
        self._initialization_error = None
        self.logger = logging.getLogger(__name__)
        
        # Log Groq availability
        self.logger.info(f"Groq available: {GROQ_AVAILABLE}, version: {GROQ_VERSION}")
    
    def _ensure_client_initialized(self) -> bool:
        """
        Ensure Groq client is initialized with better error handling
        Returns True if client is ready, False if fallback should be used
        """
        if self._client is not None:
            return True
            
        if self._initialization_attempted and self._client is None:
            return False
        
        self._initialization_attempted = True
        
        # Step 1: Check if Groq is available
        if not GROQ_AVAILABLE:
            self._initialization_error = "Groq library not available"
            self.logger.warning("Groq library not available - using rule-based insights")
            return False
        
        # Step 2: Get API key
        self._api_key = self._get_api_key()
        if not self._api_key:
            self._initialization_error = "No API key found"
            self.logger.warning("No API key available - using rule-based insights")
            return False
        
        # Step 3: Initialize client with multiple methods
        return self._initialize_groq_client()
    
    def _initialize_groq_client(self) -> bool:
        """Initialize Groq client with multiple fallback methods"""
        
        # Method 1: Basic initialization (most common)
        try:
            self._client = Groq(api_key=self._api_key)
            self.logger.info("Groq client initialized with basic method")
            return self._test_client_connection()
        except Exception as e:
            self.logger.debug(f"Basic Groq initialization failed: {e}")
        
        # Method 2: Initialize with explicit parameters
        try:
            self._client = Groq(
                api_key=self._api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.logger.info("Groq client initialized with explicit base URL")
            return self._test_client_connection()
        except Exception as e:
            self.logger.debug(f"Explicit URL Groq initialization failed: {e}")
        
        # Method 3: Initialize with minimal parameters
        try:
            # Create client with only essential parameters
            import groq
            self._client = groq.Client(api_key=self._api_key)
            self.logger.info("Groq client initialized with groq.Client")
            return self._test_client_connection()
        except Exception as e:
            self.logger.debug(f"groq.Client initialization failed: {e}")
        
        # Method 4: Try older initialization method
        try:
            from groq import Groq as GroqClient
            self._client = GroqClient(api_key=self._api_key)
            self.logger.info("Groq client initialized with GroqClient alias")
            return self._test_client_connection()
        except Exception as e:
            self.logger.debug(f"GroqClient alias initialization failed: {e}")
        
        # All methods failed
        self._initialization_error = "All Groq initialization methods failed"
        self.logger.error("Failed to initialize Groq client with any method - using rule-based insights")
        return False
    
    def _test_client_connection(self) -> bool:
        """Test the Groq client connection"""
        try:
            # Use a minimal test request
            response = self._client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.1,
                max_tokens=5
            )
            
            if response and response.choices:
                self.logger.info("Groq API connection test successful")
                return True
            else:
                self.logger.warning("Groq API returned empty response")
                return False
                
        except Exception as e:
            self.logger.error(f"Groq API connection test failed: {e}")
            self._client = None
            self._initialization_error = f"API test failed: {str(e)}"
            return False
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        
        # Source 1: Provided key
        if self.provided_api_key:
            self.logger.debug("Using provided API key")
            return self.provided_api_key
        
        # Source 2: Environment variables
        env_key = os.getenv('GROQ_API_KEY')
        if env_key:
            self.logger.debug("Using environment variable API key")
            return env_key
        
        # Source 3: Streamlit secrets
        try:
            import streamlit as st
            
            # Check if we're in Streamlit context
            try:
                _ = st.session_state
                
                # Try direct access
                streamlit_key = st.secrets.get("GROQ_API_KEY")
                if streamlit_key:
                    self.logger.debug("Using Streamlit secrets API key")
                    return streamlit_key
                    
            except Exception as e:
                self.logger.debug(f"Streamlit secrets access failed: {e}")
                
        except ImportError:
            self.logger.debug("Streamlit not available")
        except Exception as e:
            self.logger.debug(f"Streamlit context check failed: {e}")
        
        self.logger.warning("No API key found in any source")
        return None
    
    @property
    def client(self) -> Optional[Groq]:
        """Get the Groq client, initializing if necessary"""
        if self._ensure_client_initialized():
            return self._client
        return None
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key"""
        if self._api_key is None:
            self._api_key = self._get_api_key()
        return self._api_key
    
    @property
    def initialization_error(self) -> Optional[str]:
        """Get the initialization error if any"""
        return self._initialization_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information for debugging"""
        return {
            'groq_available': GROQ_AVAILABLE,
            'groq_version': GROQ_VERSION,
            'api_key_available': self.api_key is not None,
            'client_initialized': self._client is not None,
            'initialization_attempted': self._initialization_attempted,
            'initialization_error': self._initialization_error
        }
    
    def generate_comprehensive_insights(self, 
                                      metrics: Dict[str, Any], 
                                      routes_df: pd.DataFrame = None,
                                      hotspots_data: Dict[str, Any] = None,
                                      time_series_df: pd.DataFrame = None) -> List[InsightSummary]:
        """
        Generate comprehensive insights from all available data
        """
        try:
            insights = []
            
            # Generate different types of insights (these don't require AI)
            insights.extend(self._generate_safety_insights(metrics))
            insights.extend(self._generate_infrastructure_insights(metrics))
            insights.extend(self._generate_operational_insights(metrics))
            
            # Rank insights by importance
            insights = self._rank_insights_by_importance(insights)
            
            self.logger.info(f"Generated {len(insights)} comprehensive insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive insights: {e}")
            return self._generate_fallback_insights(metrics)
    
    def _generate_safety_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate safety-focused insights"""
        insights = []
        
        safety_score = metrics.get('safety_score', 0)
        if safety_score > 0:
            safety_delta = metrics.get('safety_delta', 'N/A')
            
            if safety_score >= 7:
                impact_level = 'Low'
                description = f"Your network maintains a good safety score of {safety_score:.1f}/10."
                recommendations = ["Continue monitoring current safety protocols", "Consider expanding successful safety measures"]
            elif safety_score >= 5:
                impact_level = 'Medium'
                description = f"Your network has a moderate safety score of {safety_score:.1f}/10."
                recommendations = ["Identify and address medium-risk areas", "Implement targeted safety interventions"]
            else:
                impact_level = 'High'
                description = f"Your network has a concerning safety score of {safety_score:.1f}/10."
                recommendations = ["Immediate safety audit required", "Focus on high-priority hotspots"]
            
            if safety_delta != 'N/A':
                description += f" Recent trend: {safety_delta}."
            
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
        
        coverage = metrics.get('infrastructure_coverage', 0)
        if coverage >= 0:
            coverage_delta = metrics.get('infrastructure_delta', 'N/A')
            
            if coverage >= 80:
                impact_level = 'Low'
                description = f"Excellent infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
                recommendations = ["Maintain current infrastructure quality", "Focus on route connectivity"]
            elif coverage >= 50:
                impact_level = 'Medium'
                description = f"Moderate infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
                recommendations = ["Prioritize infrastructure expansion", "Develop improvement plan"]
            else:
                impact_level = 'High'
                description = f"Low infrastructure coverage at {coverage:.1f}% of routes with bike lanes."
                recommendations = ["Urgent infrastructure development needed", "Apply for cycling infrastructure grants"]
            
            if coverage_delta != 'N/A':
                description += f" Recent progress: {coverage_delta}."
            
            insights.append(InsightSummary(
                title="Infrastructure Coverage Analysis",
                description=description,
                impact_level=impact_level,
                category="Infrastructure",
                data_points=[f"Coverage: {coverage:.1f}%", f"Progress: {coverage_delta}"],
                recommendations=recommendations,
                confidence_score=0.85,
                priority_rank=2
            ))
        
        return insights
    
    def _generate_operational_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate operational insights"""
        insights = []
        
        daily_rides = metrics.get('avg_daily_rides', 0)
        if daily_rides > 0:
            rides_delta = metrics.get('rides_delta', 'N/A')
            
            if daily_rides >= 1000:
                impact_level = 'Low'
                description = f"Strong cycling adoption with {daily_rides:,} average daily rides."
                recommendations = ["Maintain service quality", "Plan for capacity expansion"]
            elif daily_rides >= 500:
                impact_level = 'Medium'
                description = f"Moderate cycling volume with {daily_rides:,} average daily rides."
                recommendations = ["Develop rider engagement programs", "Improve route accessibility"]
            else:
                impact_level = 'High'
                description = f"Low cycling volume with {daily_rides:,} average daily rides."
                recommendations = ["Investigate barriers to cycling adoption", "Launch promotional campaigns"]
            
            if rides_delta != 'N/A':
                description += f" Volume trend: {rides_delta}."
            
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
    
    def _generate_fallback_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate basic fallback insights"""
        return [
            InsightSummary(
                title="System Status",
                description=f"Dashboard is operating with rule-based analytics. AI insights unavailable: {self._initialization_error or 'Unknown error'}",
                impact_level="Medium",
                category="Operational",
                data_points=["AI service unavailable"],
                recommendations=["Check Groq client configuration", "Verify API key", "Check network connectivity"],
                confidence_score=0.5,
                priority_rank=1
            )
        ]
    
    def _rank_insights_by_importance(self, insights: List[InsightSummary]) -> List[InsightSummary]:
        """Rank insights by importance and impact"""
        impact_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        
        insights.sort(
            key=lambda x: (impact_weights[x.impact_level], x.confidence_score),
            reverse=True
        )
        
        for i, insight in enumerate(insights):
            insight.priority_rank = i + 1
        
        return insights
    
    def generate_executive_summary(self, insights: List[InsightSummary], 
                                 metrics: Dict[str, Any]) -> str:
        """Generate an executive summary using AI when available"""
        
        # Try AI summary if client is available
        if self.client:
            try:
                return self._generate_ai_summary(insights, metrics)
            except Exception as e:
                self.logger.error(f"AI summary generation failed: {e}")
        
        # Fall back to rule-based summary
        return self._generate_fallback_summary(insights, metrics)
    
    def _generate_ai_summary(self, insights: List[InsightSummary], 
                           metrics: Dict[str, Any]) -> str:
        """Generate AI-powered executive summary"""
        prompt = f"""
        Create an executive summary for cycling safety data analysis.
        
        Network Overview:
        - Safety Score: {metrics.get('safety_score', 'N/A')}/10
        - Total Routes: {metrics.get('total_routes', 'N/A')}
        - Daily Rides: {metrics.get('avg_daily_rides', 'N/A'):,}
        - Infrastructure Coverage: {metrics.get('infrastructure_coverage', 'N/A')}%
        
        Key Findings:
        """
        
        for insight in insights[:3]:
            prompt += f"- {insight.title}: {insight.description}\n"
        
        prompt += """
        
        Provide a concise executive summary (150-200 words) for municipal leaders.
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _generate_fallback_summary(self, insights: List[InsightSummary], 
                                 metrics: Dict[str, Any]) -> str:
        """Generate a fallback summary without Groq"""
        safety_score = metrics.get('safety_score', 0)
        total_routes = metrics.get('total_routes', 0)
        daily_rides = metrics.get('avg_daily_rides', 0)
        
        status = "good" if safety_score >= 7 else "moderate" if safety_score >= 5 else "concerning"
        
        summary = f"""
        **Network Status**: Your cycling network shows {status} safety performance with a {safety_score:.1f}/10 safety score across {total_routes} routes serving {daily_rides:,} daily rides.
        
        **Key Priorities**:
        """
        
        for i, insight in enumerate(insights[:3]):
            summary += f"\n{i+1}. {insight.title}: {insight.description}"
        
        summary += """
        
        **Recommended Actions**: Focus on addressing high-impact safety issues, improving infrastructure coverage, and implementing data-driven interventions to enhance cyclist safety and network performance.
        """
        
        return summary

# Factory function
def create_insights_generator(groq_api_key: Optional[str] = None) -> GroqInsightsGenerator:
    """Create and return a configured insights generator"""
    return GroqInsightsGenerator(api_key=groq_api_key)
