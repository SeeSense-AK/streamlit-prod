"""
Groq-Powered Insights Generator for SeeSense Dashboard
REWRITTEN VERSION with reliable environment variable handling
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the insights generator
        
        Args:
            api_key: Optional API key. If not provided, will try environment variables
        """
        self.provided_api_key = api_key
        self._client = None
        self._api_key = None
        self._initialization_attempted = False
        self._initialization_error = None
        self.logger = logging.getLogger(__name__)
        
        # Log configuration
        self.logger.info(f"Groq available: {GROQ_AVAILABLE}")
        if GROQ_AVAILABLE:
            self.logger.info(f"Groq version: {GROQ_VERSION}")
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get API key from multiple sources in priority order:
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
        
        # Initialize Groq client with multiple compatibility methods
        initialization_methods = [
            ("Standard Groq()", lambda: Groq(api_key=self._api_key)),
            ("Groq with explicit params", lambda: Groq(api_key=self._api_key, timeout=30)),
            ("groq.Client()", lambda: __import__('groq').Client(api_key=self._api_key)),
            ("GROQ_CLIENT_CLASS", lambda: GROQ_CLIENT_CLASS(api_key=self._api_key) if GROQ_CLIENT_CLASS else None)
        ]
        
        for method_name, method_func in initialization_methods:
            try:
                self.logger.debug(f"Trying {method_name}")
                client = method_func()
                if client is not None:
                    self._client = client
                    self.logger.info(f"Groq client initialized successfully with {method_name}")
                    return self._test_connection()
                    
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    self.logger.debug(f"{method_name} failed due to parameter mismatch: {e}")
                    continue
                else:
                    self.logger.debug(f"{method_name} failed with TypeError: {e}")
                    continue
            except Exception as e:
                self.logger.debug(f"{method_name} failed: {e}")
                continue
        
        # All methods failed
        self.logger.error("All Groq client initialization methods failed")
        self._initialization_error = "All initialization methods failed - check Groq version compatibility"
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
                                      time_series_df: pd.DataFrame = None) -> List[InsightSummary]:
        """
        Generate comprehensive insights from cycling safety data
        
        Args:
            metrics: Dictionary of calculated metrics
            routes_df: Optional routes dataframe
            hotspots_data: Optional hotspots data
            time_series_df: Optional time series dataframe
            
        Returns:
            List of InsightSummary objects
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
            self.logger.error(f"Error generating comprehensive insights: {e}")
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
                    priority_rank=0
                ))
            elif safety_score >= 5:
                insights.append(InsightSummary(
                    title="Moderate Safety Concerns",
                    description=f"Network safety score of {safety_score:.1f}/10 indicates room for improvement",
                    impact_level="Medium",
                    category="Safety",
                    data_points=[f"Safety score: {safety_score:.1f}/10"],
                    recommendations=["Focus on medium-risk areas", "Implement targeted safety interventions"],
                    confidence_score=0.9,
                    priority_rank=0
                ))
            else:
                insights.append(InsightSummary(
                    title="Critical Safety Issues",
                    description=f"Low safety score of {safety_score:.1f}/10 requires immediate attention",
                    impact_level="High",
                    category="Safety",
                    data_points=[f"Safety score: {safety_score:.1f}/10"],
                    recommendations=["Immediate safety audit required", "Address high-priority hotspots first"],
                    confidence_score=0.95,
                    priority_rank=0
                ))
        
        return insights
    
    def _generate_infrastructure_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate infrastructure-focused insights"""
        insights = []
        
        coverage = metrics.get('infrastructure_coverage', 0)
        if coverage > 0:
            if coverage >= 80:
                insights.append(InsightSummary(
                    title="Excellent Infrastructure Coverage",
                    description=f"High infrastructure coverage at {coverage:.1f}% provides good cycling support",
                    impact_level="Low",
                    category="Infrastructure",
                    data_points=[f"Coverage: {coverage:.1f}%"],
                    recommendations=["Maintain current infrastructure", "Focus on quality improvements"],
                    confidence_score=0.8,
                    priority_rank=0
                ))
            elif coverage >= 60:
                insights.append(InsightSummary(
                    title="Infrastructure Expansion Needed",
                    description=f"Infrastructure coverage at {coverage:.1f}% has room for strategic expansion",
                    impact_level="Medium",
                    category="Infrastructure",
                    data_points=[f"Coverage: {coverage:.1f}%"],
                    recommendations=["Identify high-traffic areas without infrastructure", "Plan targeted expansions"],
                    confidence_score=0.85,
                    priority_rank=0
                ))
            else:
                insights.append(InsightSummary(
                    title="Critical Infrastructure Gaps",
                    description=f"Low infrastructure coverage at {coverage:.1f}% limits cycling safety",
                    impact_level="High",
                    category="Infrastructure",
                    data_points=[f"Coverage: {coverage:.1f}%"],
                    recommendations=["Develop comprehensive infrastructure plan", "Prioritize high-risk corridors"],
                    confidence_score=0.9,
                    priority_rank=0
                ))
        
        return insights
    
    def _generate_operational_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate operational insights"""
        insights = []
        
        daily_rides = metrics.get('avg_daily_rides', 0)
        total_routes = metrics.get('total_routes', 0)
        
        if daily_rides > 0 and total_routes > 0:
            rides_per_route = daily_rides / total_routes
            
            insights.append(InsightSummary(
                title="Network Utilization Analysis",
                description=f"Average of {rides_per_route:.1f} rides per route daily across {total_routes} routes",
                impact_level="Medium",
                category="Operational",
                data_points=[f"Daily rides: {daily_rides:,}", f"Routes: {total_routes}", f"Rides per route: {rides_per_route:.1f}"],
                recommendations=["Monitor route efficiency", "Consider rebalancing popular routes"],
                confidence_score=0.7,
                priority_rank=0
            ))
        
        return insights
    
    def _generate_ai_insights(self, metrics: Dict[str, Any], existing_insights: List[InsightSummary]) -> List[InsightSummary]:
        """Generate AI-powered insights using Groq"""
        if not self.client:
            return []
        
        try:
            # Create prompt for AI analysis
            prompt = self._create_analysis_prompt(metrics, existing_insights)
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            ai_content = response.choices[0].message.content
            
            # Parse AI response into insights
            ai_insight = InsightSummary(
                title="AI-Generated Strategic Insight",
                description=ai_content[:200] + "..." if len(ai_content) > 200 else ai_content,
                impact_level="Medium",
                category="Strategic",
                data_points=["AI analysis of network patterns"],
                recommendations=["Consider AI recommendations", "Validate with local expertise"],
                confidence_score=0.7,
                priority_rank=0
            )
            
            return [ai_insight]
            
        except Exception as e:
            self.logger.error(f"AI insight generation failed: {e}")
            return []
    
    def _create_analysis_prompt(self, metrics: Dict[str, Any], insights: List[InsightSummary]) -> str:
        """Create prompt for AI analysis"""
        prompt = f"""
        Analyze this cycling safety network data and provide strategic insights:
        
        Metrics:
        - Safety Score: {metrics.get('safety_score', 'N/A')}/10
        - Total Routes: {metrics.get('total_routes', 'N/A')}
        - Daily Rides: {metrics.get('avg_daily_rides', 'N/A')}
        - Infrastructure Coverage: {metrics.get('infrastructure_coverage', 'N/A')}%
        
        Current Analysis Shows:
        """
        
        for insight in insights[:3]:
            prompt += f"- {insight.title}: {insight.impact_level} impact\n"
        
        prompt += "\nProvide one strategic recommendation for municipal planners (max 150 words):"
        
        return prompt
    
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
    
    def _generate_fallback_insights(self, metrics: Dict[str, Any]) -> List[InsightSummary]:
        """Generate basic fallback insights when everything else fails"""
        return [
            InsightSummary(
                title="Data Analysis Complete",
                description="Basic metrics analysis completed. Consider enabling AI insights for deeper analysis.",
                impact_level="Medium",
                category="Operational",
                data_points=["System operational"],
                recommendations=["Review metric trends", "Enable AI features for enhanced insights"],
                confidence_score=0.5,
                priority_rank=1
            )
        ]
    
    def generate_executive_summary(self, insights: List[InsightSummary], metrics: Dict[str, Any]) -> str:
        """Generate an executive summary for stakeholders"""
        
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
