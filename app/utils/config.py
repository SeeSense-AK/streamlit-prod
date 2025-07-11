"""
Configuration management for SeeSense Dashboard
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class Config:
    """Configuration manager for the dashboard application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (defaults to config/settings.yaml)
        """
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = None
        self._schema_data = None
        self.load_config()
        self.load_schema()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file)
                logging.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.warning(f"Configuration file not found: {self.config_path}")
            self._config_data = self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            self._config_data = self._get_default_config()
    
    def load_schema(self) -> None:
        """Load data schema from YAML file"""
        try:
            schema_path = self.config_path.parent / "data_schema.yaml"
            with open(schema_path, 'r', encoding='utf-8') as file:
                self._schema_data = yaml.safe_load(file)
                logging.info(f"Schema loaded from {schema_path}")
        except FileNotFoundError:
            logging.warning(f"Schema file not found: {schema_path}")
            self._schema_data = {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing schema file: {e}")
            self._schema_data = {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., 'app.title')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_schema(self, dataset: str) -> Dict[str, Any]:
        """
        Get schema for a specific dataset
        
        Args:
            dataset: Dataset name (e.g., 'routes', 'braking_hotspots')
            
        Returns:
            Schema dictionary for the dataset
        """
        return self._schema_data.get(dataset, {})
    
    def get_data_path(self, path_type: str = "raw") -> Path:
        """
        Get data directory path
        
        Args:
            path_type: Type of data path ('raw' or 'processed')
            
        Returns:
            Path object for data directory
        """
        project_root = Path(__file__).parent.parent.parent
        
        if path_type == "raw":
            return project_root / self.get('data.raw_data_path', 'data/raw')
        elif path_type == "processed":
            return project_root / self.get('data.processed_data_path', 'data/processed')
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def get_assets_path(self) -> Path:
        """Get assets directory path"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "assets"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'app': {
                'title': 'SeeSense Safety Analytics Platform',
                'icon': '🚲',
                'layout': 'wide'
            },
            'data': {
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed',
                'cache_ttl': 3600
            },
            'visualization': {
                'default_map_center': {'lat': 51.5074, 'lon': -0.1278},
                'default_zoom': 12
            }
        }
    
    @property
    def app_title(self) -> str:
        """Get application title"""
        return self.get('app.title', 'SeeSense Dashboard')
    
    @property
    def app_icon(self) -> str:
        """Get application icon"""
        return self.get('app.icon', '🚲')
    
    @property
    def cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self.get('data.cache_ttl', 3600)
    
    @property
    def max_file_size_mb(self) -> int:
        """Get maximum file size in MB"""
        return self.get('data.max_file_size_mb', 100)
    
    @property
    def default_map_center(self) -> Dict[str, float]:
        """Get default map center coordinates"""
        return self.get('visualization.default_map_center', {'lat': 51.5074, 'lon': -0.1278})
    
    @property
    def default_zoom(self) -> int:
        """Get default map zoom level"""
        return self.get('visualization.default_zoom', 12)


# Global configuration instance
config = Config()


def setup_logging() -> None:
    """Setup logging configuration"""
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "dashboard.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# Environment-specific configurations
def get_environment() -> str:
    """Get current environment (development, staging, production)"""
    return os.getenv('DASHBOARD_ENV', 'development')


def is_development() -> bool:
    """Check if running in development environment"""
    return get_environment() == 'development'


def is_production() -> bool:
    """Check if running in production environment"""
    return get_environment() == 'production'
