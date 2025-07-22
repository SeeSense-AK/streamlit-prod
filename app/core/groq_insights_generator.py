"""
Groq-Powered Insights Generator for SeeSense Dashboard
Clean version without problematic caching
"""
import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.info("python-dotenv not available, using system environment variables only")

# Import Groq with comprehensive error handling
GROQ_AVAILABLE = False
GROQ_VERSION = None
GROQ_CLIENT_CLASS = None

try:
    # Try the standard import first
    from groq import Groq
    GROQ_CLIENT_CLASS = Groq
    GROQ_AVAILABLE = True
    
    # Try to get version info
    try:
        import groq
        GROQ_VERSION = getattr(groq, '__version__', 'unknown')
    except:
        GROQ_VERSION = 'unknown'
    
    logging.info(f"Groq library loaded successfully, version: {GROQ_VERSION}")
    
except ImportError as e:
    logging.warning(f"Groq library not available: {e}")
    logging.info("AI insights will not be available - falling back to rule-based insights")
except Exception as e:
    logging.error(f"Error importing Groq: {e}")

# Try alternative import methods if standard method failed
if not GROQ_AVAILABLE:
    try:
        import groq
        GROQ_CLIENT_CLASS = groq.Client
        GROQ_AVAILABLE = True
        GROQ_VERSION = getattr(groq, '__version__', 'unknown')
        logging.info(f"Groq library loaded with alternative method, version: {GROQ_VERSION}")
    except:
        pass

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
    Generate intelligent insights using Groq AI for cycling safety analysis.
    Falls back to rule-based insights when AI is not available.
    Clean version without caching complications.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the insights generator
        
        Args:
            api_key: Optional API key. If not provided, will use environment variables
        """
        self.provided_api_key = api_key
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._api_key = None
        self._initialization_attempted = False
        self._initialization_error = None
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get API key from multiple sources in order of priority:
        1. Provided API key (constructor parameter)
        2. Environment variable GROQ_API_KEY
        3. Streamlit secrets (if available)
        """
        
        # Source 1: Provided API key
        if self.provided_api_key:
            self.logger.debug("Using provided API key")
            return self.provided_api_key
        
        # Source 2: Environment variable
        env_key = os.getenv('GROQ_API_KEY')
        if env_key:
            self.logger.debug("Using environment variable API key")
            return env_key
        
        # Source 3: Streamlit secrets (as fallback)
        try:
            import streamlit as st
            
            # Check if we're in Streamlit context and secrets are available
            if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
                streamlit_key = st.secrets["GROQ_API_KEY"]
                if streamlit_key:
                    self.logger.debug("Using Streamlit secrets API key")
                    return streamlit_key
            else:
                self.logger.debug("GROQ_API_KEY not found in Streamlit secrets")
                
        except ImportError:
            self.logger.debug("Streamlit not available")
        except Exception as e:
            self.logger.debug(f"Could not access Streamlit secrets: {e}")
        
        # No API key found
        self.logger.warning("No GROQ_API_KEY found in environment variables, provided parameter, or Streamlit secrets")
        return None
    
    def _initialize_client(self) -> bool:
        """
        Initialize the Groq client with proper error handling
        
        Returns:
            bool: True if client initialized successfully, False otherwise
        """
        if self._client is not None:
            return True
            
        if self._initialization_attempted and self._client is None:
            return False
        
        self._initialization_attempted = True
        
        # Check if Groq is available
        if not GROQ_AVAILABLE:
            self._initialization_error = "Groq library not installed"
            self.logger.warning("Groq library not available - install with: pip install groq")
            return False
        
        # Get API key
        self._api_key = self._get_api_key()
        if not self._api_key:
            self._initialization_error = "No API key available"
            self.logger.warning("No Groq API key found - AI insights will not be available")
            return False
        
        # Initialize Groq client
        try:
            self._client = Groq(api_key=self._api_key)
            
            # Test connection
            if self._test_connection():
                self.logger.info("Groq client initialized successfully")
                return True
            else:
                self._client = None
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {e}")
            self._initialization_error = f"Client initialization failed: {str(e)}"
            return False
    
    def _test_connection(self) -> bool:
        """
        Test the Groq API connection with a minimal request
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            response = self._client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.1,
                max_tokens=1
            )
            
            if response and response.choices and len(response.choices) > 0:
                self.logger.info("Groq API connection test successful")
                return True
            else:
                self.logger.warning("Groq API test returned empty response")
                self._initialization_error = "API test returned empty response"
                return False
                
        except Exception as e:
            self.logger.error(f"Groq API connection test failed: {e}")
            self._initialization_error = f"API connection test failed: {str(e)}"
            self._client = None
            return False
    
    @property
    def client(self) -> Optional[Groq]:
        """Get the Groq client, initializing if necessary"""
        if self._initialize_client():
            return self._client
        return None
    
    @property
    def api_key_available(self) -> bool:
        """Check if API key is available"""
        if self._api_key is None:
            self._api_key = self._get_api_key()
        return self._api_key is not None
    
    @property
    def is_ready(self) -> bool:
        """Check if the generator is ready to use AI features"""
        return self.client is not None
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get detailed status information for debugging"""
        return {
            'groq_available': GROQ_AVAILABLE,
            'groq_version': GROQ_VERSION,
            'api_key_available': self.api_key_available,
            'client_ready': self.is_ready,
            'initialization_attempted': self._initialization_attempted,
            'initialization_error': self._initialization_error
        }
    
    def generate_comprehensive_insights(self, 
                                      metrics: Dict[str, Any], 
                                      routes_df: pd.DataFrame = None,
                                      hotspots_data: Dict[str, Any] = None,
                                      time_series_df: pd.DataFrame = None,
                                      use_cache: bool = False,  # Parameter kept for compatibility
                                      force_refresh: bool = False) -> List[InsightSummary]:
        """
        Generate comprehensive insights from cycling safety data
        """
        return self._generate_insights_direct(metrics, routes_df, hotspots_data, time_series_df)
    
    def _generate_insights_direct(self, 
                                 metrics: Dict[str, Any], 
                                 routes_df: pd.DataFrame = None,
                                 hotspots_data: Dict[str, Any] = None,
                                 time_series_df: pd.DataFrame = None) -> List[InsightSummary]:
        """
        Generate insights directly without caching
        """
        try:
            insights = []
            
            # Generate rule-based insights (always available)
            insights.extend(self._generate_safety_insights(metrics))
            insights.extend(self._generate_infrastructure_insights(metrics))
            insights.extend(self._generate_operational_insights(metrics))
            
            # Try to enhance with AI insights if available
            if self.is_ready:
                try:
                    ai_insights = self._generate_ai_insights(metrics, insights)
                    if ai_insights:
                        insights.extend(ai_insights)
                        self.logger.info("Added AI-generated insights")
                except Exception as e:
                    self.logger.warning(f"AI insight generation failed: {e}")
            
            # Rank and return insights
            insights = self._rank_insights_by_importance(insights)
            self.logger.info(f"Generated {len(insights)} total insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return self._generate_fallback_insights(metrics)
    
    def _generate_safety_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate safety-focused insights"""
        insights = []
        
        safety_score = metrics.get('safety_score', 0)
        if safety_score > 0:
            if safety_score >= 7:
                insights.append(InsightSummary(
                    title="Strong Safety Performance",
                    description=f"Network maintains good safety score of {safety_score:.1f}/10",
                    impact_level="Low",
                    category="Safety",
                    data_points=[f"Safety score: {safety_score:.1f}/10"],
                    recommendations=["Continue current safety protocols", "Monitor for any declining trends"],
                    confidence_score=0.8,
                    priority_rank=3
                ))
            elif safety_score >= 5:
                insights.append(InsightSummary(
                    title="Moderate Safety Concerns",
                    description=f"Safety score of {safety_score:.1f}/10 indicates room for improvement",
                    impact_level="Medium",
                    category="Safety",
                    data_points=[f"Safety score: {safety_score:.1f}/10"],
                    recommendations=["Review high-risk areas", "Implement targeted safety interventions"],
                    confidence_score=0.7,
                    priority_rank=2
                ))
            else:
                insights.append(InsightSummary(
                    title="Critical Safety Issues",
                    description=f"Low safety score of {safety_score:.1f}/10 requires immediate attention",
                    impact_level="High",
                    category="Safety",
                    data_points=[f"Safety score: {safety_score:.1f}/10"],
                    recommendations=["Immediate safety audit", "Emergency response protocol review"],
                    confidence_score=0.9,
                    priority_rank=1
                ))
        
        # Check for hotspot concentration
        total_hotspots = metrics.get('total_hotspots', 0)
        if total_hotspots > 10:
            insights.append(InsightSummary(
                title="High Hotspot Concentration",
                description=f"Network shows {total_hotspots} safety hotspots requiring attention",
                impact_level="Medium" if total_hotspots <= 25 else "High",
                category="Safety",
                data_points=[f"Total hotspots: {total_hotspots}"],
                recommendations=["Prioritize hotspot remediation", "Deploy targeted interventions"],
                confidence_score=0.8,
                priority_rank=2
            ))
        
        return insights
    
    def _generate_infrastructure_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate infrastructure-focused insights"""
        insights = []
        
        total_routes = metrics.get('total_routes', 0)
        infrastructure_coverage = metrics.get('infrastructure_coverage', 0)
        
        if total_routes > 0:
            if infrastructure_coverage < 60:
                insights.append(InsightSummary(
                    title="Insufficient Infrastructure Coverage",
                    description=f"Only {infrastructure_coverage:.1f}% infrastructure coverage across {total_routes} routes",
                    impact_level="High" if infrastructure_coverage < 40 else "Medium",
                    category="Infrastructure",
                    data_points=[f"Coverage: {infrastructure_coverage:.1f}%", f"Routes: {total_routes}"],
                    recommendations=["Expand cycling infrastructure", "Prioritize high-usage routes"],
                    confidence_score=0.8,
                    priority_rank=1 if infrastructure_coverage < 40 else 2
                ))
            elif infrastructure_coverage >= 80:
                insights.append(InsightSummary(
                    title="Excellent Infrastructure Coverage",
                    description=f"Strong {infrastructure_coverage:.1f}% infrastructure coverage supports {total_routes} routes",
                    impact_level="Low",
                    category="Infrastructure",
                    data_points=[f"Coverage: {infrastructure_coverage:.1f}%", f"Routes: {total_routes}"],
                    recommendations=["Maintain current infrastructure", "Focus on optimization"],
                    confidence_score=0.7,
                    priority_rank=3
                ))
        
        return insights
    
    def _generate_operational_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate operational insights"""
        insights = []
        
        avg_daily_rides = metrics.get('avg_daily_rides', 0)
        network_efficiency = metrics.get('network_efficiency', 0)
        
        if avg_daily_rides > 0:
            if avg_daily_rides > 1000:
                insights.append(InsightSummary(
                    title="High Network Utilization",
                    description=f"Network serves {avg_daily_rides:,.0f} daily rides, indicating strong adoption",
                    impact_level="Low",
                    category="Operational",
                    data_points=[f"Daily rides: {avg_daily_rides:,.0f}"],
                    recommendations=["Monitor capacity", "Plan for growth"],
                    confidence_score=0.7,
                    priority_rank=3
                ))
            elif avg_daily_rides < 100:
                insights.append(InsightSummary(
                    title="Low Network Utilization",
                    description=f"Only {avg_daily_rides:,.0f} daily rides suggests underutilization",
                    impact_level="Medium",
                    category="Operational",
                    data_points=[f"Daily rides: {avg_daily_rides:,.0f}"],
                    recommendations=["Investigate usage barriers", "Improve route promotion"],
                    confidence_score=0.6,
                    priority_rank=2
                ))
        
        if network_efficiency > 0:
            if network_efficiency < 60:
                insights.append(InsightSummary(
                    title="Network Efficiency Concerns",
                    description=f"Network efficiency of {network_efficiency:.1f}% indicates optimization opportunities",
                    impact_level="Medium",
                    category="Operational",
                    data_points=[f"Efficiency: {network_efficiency:.1f}%"],
                    recommendations=["Route optimization analysis", "Capacity rebalancing"],
                    confidence_score=0.7,
                    priority_rank=2
                ))
        
        return insights
    
    def _generate_ai_insights(self, metrics: Dict[str, Any], existing_insights: List[InsightSummary]) -> List[InsightSummary]:
        """Generate AI-powered insights"""
        if not self.is_ready:
            return []
        
        try:
            # Create a comprehensive prompt for AI analysis
            prompt = f"""
            Analyze this cycling safety network data and provide 2-3 additional insights:
            
            Metrics:
            - Safety Score: {metrics.get('safety_score', 'N/A')}/10
            - Daily Rides: {metrics.get('avg_daily_rides', 'N/A'):,}
            - Total Routes: {metrics.get('total_routes', 'N/A')}
            - Infrastructure Coverage: {metrics.get('infrastructure_coverage', 'N/A')}%
            - Network Efficiency: {metrics.get('network_efficiency', 'N/A')}%
            - Total Hotspots: {metrics.get('total_hotspots', 'N/A')}
            
            Existing insights cover: {[i.category for i in existing_insights]}
            
            Provide insights as JSON array with format:
            [{{
                "title": "Insight Title",
                "description": "Brief description",
                "impact_level": "High/Medium/Low",
                "category": "Category",
                "recommendations": ["rec1", "rec2"],
                "confidence_score": 0.8
            }}]
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            import json
            insights_data = json.loads(content)
            
            ai_insights = []
            for i, insight_data in enumerate(insights_data[:3]):  # Limit to 3 insights
                ai_insights.append(InsightSummary(
                    title=insight_data.get('title', 'AI Insight'),
                    description=insight_data.get('description', ''),
                    impact_level=insight_data.get('impact_level', 'Medium'),
                    category=insight_data.get('category', 'Analysis'),
                    data_points=[f"AI Analysis Confidence: {insight_data.get('confidence_score', 0.5):.1f}"],
                    recommendations=insight_data.get('recommendations', ['Review findings']),
                    confidence_score=insight_data.get('confidence_score', 0.5),
                    priority_rank=2
                ))
            
            return ai_insights
            
        except Exception as e:
            self.logger.warning(f"AI insights generation failed: {e}")
            return []
    
    def _rank_insights_by_importance(self, insights: List[InsightSummary]) -> List[InsightSummary]:
        """Rank insights by importance"""
        # Sort by priority rank (lower number = higher priority), then by impact level
        impact_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        
        def sort_key(insight):
            impact_weight = impact_weights.get(insight.impact_level, 1)
            return (insight.priority_rank, -impact_weight, -insight.confidence_score)
        
        return sorted(insights, key=sort_key)
    
    def _generate_fallback_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate basic fallback insights when other methods fail"""
        return [
            InsightSummary(
                title="System Operational",
                description="Basic insights available. Consider enabling AI insights for deeper analysis.",
                impact_level="Medium",
                category="Operational",
                data_points=["System operational"],
                recommendations=["Review metric trends", "Enable AI features for enhanced insights"],
                confidence_score=0.5,
                priority_rank=1
            )
        ]
    
    def generate_executive_summary(self, 
                                 insights: List[InsightSummary], 
                                 metrics: Dict[str, Any], 
                                 use_cache: bool = False,  # Parameter kept for compatibility
                                 force_refresh: bool = False) -> str:
        """Generate an executive summary for stakeholders"""
        return self._generate_executive_summary_direct(insights, metrics)
    
    def _generate_executive_summary_direct(self, insights: List[InsightSummary], metrics: Dict[str, Any]) -> str:
        """Generate executive summary directly"""
        
        # Try AI summary if available
        if self.is_ready:
            try:
                return self._generate_ai_summary(insights, metrics)
            except Exception as e:
                self.logger.error(f"AI summary generation failed: {e}")
        
        # Fall back to rule-based summary
        return self._generate_rule_based_summary(insights, metrics)
    
    def _generate_ai_summary(self, insights: List[InsightSummary], metrics: Dict[str, Any]) -> str:
        """Generate AI-powered executive summary"""
        prompt = f"""
        Create an executive summary for cycling safety network analysis:
        
        Network Overview:
        - Safety Score: {metrics.get('safety_score', 'N/A')}/10
        - Total Routes: {metrics.get('total_routes', 'N/A')}
        - Daily Rides: {metrics.get('avg_daily_rides', 'N/A'):,}
        - Infrastructure Coverage: {metrics.get('infrastructure_coverage', 'N/A')}%
        
        Key Findings:
        """
        
        for insight in insights[:3]:
            prompt += f"- {insight.title}: {insight.description}\n"
        
        prompt += "\nProvide a concise executive summary (150-200 words) for municipal leaders."
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _generate_rule_based_summary(self, insights: List[InsightSummary], metrics: Dict[str, Any]) -> str:
        """Generate rule-based executive summary"""
        safety_score = metrics.get('safety_score', 0)
        total_routes = metrics.get('total_routes', 0)
        daily_rides = metrics.get('avg_daily_rides', 0)
        
        status = "good" if safety_score >= 7 else "moderate" if safety_score >= 5 else "concerning"
        
        summary = f"""
**Network Overview**: Your cycling network shows {status} performance with a {safety_score:.1f}/10 safety score across {total_routes} routes serving {daily_rides:,} daily rides.

**Key Priorities**:
        """
        
        for i, insight in enumerate(insights[:3]):
            summary += f"\n{i+1}. **{insight.title}**: {insight.description}"
        
        summary += f"""

**Recommendations**: Focus on {insights[0].category.lower()} improvements, monitor trends, and implement data-driven safety interventions to enhance network performance.
        """
        
        return summary


# Factory function for easy instantiation
def create_insights_generator(api_key: Optional[str] = None) -> GroqInsightsGenerator:
    """
    Create and return a configured insights generator
    
    Args:
        api_key: Optional Groq API key. If not provided, will use environment variables
        
    Returns:
        GroqInsightsGenerator instance
    """
    return GroqInsightsGenerator(api_key=api_key)


# Simplified cache control for compatibility
def add_cache_controls():
    """Add simplified cache control UI elements to the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🧠 AI Insights**")
        st.caption("Real-time generation")


def get_insights_with_cache(metrics: Dict[str, Any],
                           routes_df: pd.DataFrame = None,
                           force_refresh: bool = False) -> Tuple[List[InsightSummary], str]:
    """
    Get both insights and executive summary (compatibility function)
    """
    # Generate insights
    generator = create_insights_generator()
    insights = generator.generate_comprehensive_insights(
        metrics=metrics,
        routes_df=routes_df
    )
    
    # Generate executive summary
    executive_summary = generator.generate_executive_summary(
        insights=insights,
        metrics=metrics
    )
    
    return insights, executive_summary
