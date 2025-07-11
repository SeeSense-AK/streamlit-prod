"""
Core business logic modules for SeeSense Dashboard
Data processing, ML models, analytics, and insights generation
"""

from .data_processor import data_processor, DataProcessor

# When we implement these modules, we'll add imports like:
# from .ml_models import RiskPredictor, SafetyClusterer, AnomalyDetector
# from .analytics import TimeSeriesAnalyzer, CorrelationAnalyzer
# from .spatial_analyzer import HotspotAnalyzer, RouteOptimizer
# from .insights_generator import InsightsEngine, RecommendationEngine

__all__ = [
    "data_processor",
    "DataProcessor",
    
    # Placeholder for future ML/Analytics modules
    # "RiskPredictor",
    # "SafetyClusterer", 
    # "AnomalyDetector",
    # "TimeSeriesAnalyzer",
    # "CorrelationAnalyzer",
    # "HotspotAnalyzer",
    # "RouteOptimizer",
    # "InsightsEngine",
    # "RecommendationEngine"
]
