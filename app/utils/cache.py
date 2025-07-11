"""
Caching utilities for SeeSense Dashboard
"""
import streamlit as st
import pandas as pd
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from functools import wraps
import json
import gzip

from .config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """Enhanced cache manager for dashboard data and computations"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.cache_dir = config.get_data_path("processed") / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = config.cache_ttl
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key for function call"""
        # Create a string representation of arguments
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key not in self.cache_index:
            return None
        
        cache_info = self.cache_index[key]
        cache_file = self.cache_dir / f"{key}.pkl.gz"
        
        # Check if cache file exists
        if not cache_file.exists():
            logger.warning(f"Cache file missing for key {key}")
            del self.cache_index[key]
            self._save_cache_index()
            return None
        
        # Check TTL
        created_time = datetime.fromisoformat(cache_info['created'])
        ttl = cache_info.get('ttl', self.default_ttl)
        
        if datetime.now() - created_time > timedelta(seconds=ttl):
            logger.info(f"Cache expired for key {key}")
            self.delete(key)
            return None
        
        # Load from cache
        try:
            with gzip.open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for key {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cache for key {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_file = self.cache_dir / f"{key}.pkl.gz"
        
        try:
            # Save data
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            self.cache_index[key] = {
                'created': datetime.now().isoformat(),
                'ttl': ttl,
                'size_bytes': cache_file.stat().st_size
            }
            self._save_cache_index()
            
            logger.debug(f"Cached data for key {key}")
            
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete item from cache"""
        cache_file = self.cache_dir / f"{key}.pkl.gz"
        
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self.cache_index:
            del self.cache_index[key]
            self._save_cache_index()
        
        logger.debug(f"Deleted cache for key {key}")
    
    def clear(self) -> None:
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl.gz"):
            cache_file.unlink()
        
        self.cache_index = {}
        self._save_cache_index()
        
        logger.info("Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(info.get('size_bytes', 0) for info in self.cache_index.values())
        
        return {
            'total_items': len(self.cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'items': [
                {
                    'key': key,
                    'created': info['created'],
                    'ttl_seconds': info['ttl'],
                    'size_mb': info.get('size_bytes', 0) / (1024 * 1024)
                }
                for key, info in self.cache_index.items()
            ]
        }


# Global cache manager instance
cache_manager = CacheManager()


def cached_function(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds (uses default if None)
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
            cache_key = cache_manager._generate_cache_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def streamlit_cache_data(func: Optional[Callable] = None, *, ttl: Optional[int] = None):
    """
    Enhanced Streamlit cache decorator with custom TTL
    
    Args:
        func: Function to cache
        ttl: Time to live in seconds
    """
    def decorator(f: Callable) -> Callable:
        # Use Streamlit's built-in caching with custom TTL
        cache_decorator = st.cache_data(ttl=ttl if ttl else config.cache_ttl)
        return cache_decorator(f)
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def clear_streamlit_cache():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
    cache_manager.clear()
    logger.info("Cleared all caches")


def get_cache_info() -> Dict[str, Any]:
    """Get comprehensive cache information"""
    cache_stats = cache_manager.get_cache_stats()
    
    # Add Streamlit cache info if available
    streamlit_info = {
        'streamlit_cache_available': hasattr(st, 'cache_data'),
        'manual_cache_items': cache_stats['total_items'],
        'manual_cache_size_mb': cache_stats['total_size_mb']
    }
    
    return {
        **cache_stats,
        **streamlit_info
    }


# Utility functions for common caching patterns
def cache_dataframe(key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> None:
    """Cache a DataFrame with compression"""
    cache_manager.set(key, df, ttl)


def get_cached_dataframe(key: str) -> Optional[pd.DataFrame]:
    """Get a cached DataFrame"""
    return cache_manager.get(key)


def cache_computation_result(key: str, result: Any, ttl: Optional[int] = None) -> None:
    """Cache computation result"""
    cache_manager.set(key, result, ttl)


def get_cached_computation(key: str) -> Optional[Any]:
    """Get cached computation result"""
    return cache_manager.get(key)
