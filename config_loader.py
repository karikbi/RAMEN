#!/usr/bin/env python3
"""
RAMEN Pipeline - Configuration Loader

Loads configuration from config.yaml with environment variable overrides.
Provides type-safe access to configuration values.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("wallpaper_curator")


@dataclass
class QualityWeights:
    """Quality scoring component weights."""
    visual: float = 0.40
    composition: float = 0.30
    aesthetic: float = 0.20
    suitability: float = 0.10


@dataclass
class QualityConfig:
    """Quality scoring configuration."""
    threshold: float = 5.5  # Aesthetic V2.5 native 1-10 scale
    weights: QualityWeights = field(default_factory=QualityWeights)


@dataclass
class SubredditConfig:
    """Configuration for a single subreddit."""
    name: str
    fetch_count: int = 50
    min_upvotes: int = 5000


@dataclass
class RedditConfig:
    """Reddit source configuration."""
    enabled: bool = True
    subreddits: list[SubredditConfig] = field(default_factory=list)


@dataclass
class SourcesConfig:
    """All source configurations."""
    reddit: RedditConfig = field(default_factory=RedditConfig)
    unsplash_enabled: bool = True
    unsplash_count: int = 30
    pexels_enabled: bool = True
    pexels_count: int = 20


@dataclass
class FilterConfig:
    """Hard filter configuration."""
    min_width: int = 2560
    min_height: int = 1440
    min_file_size_kb: int = 200
    max_file_size_mb: int = 15
    min_aspect_ratio: float = 1.77
    max_aspect_ratio: float = 2.33
    max_text_coverage: float = 0.30
    phash_threshold: int = 10


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay_sec: float = 2.0
    max_delay_sec: float = 60.0


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    api_call_sec: int = 30
    model_inference_sec: int = 60
    upload_sec: int = 120
    max_runtime_minutes: int = 50


class ConfigLoader:
    """
    Load configuration from YAML with environment variable overrides.
    
    Environment variables use the format: RAMEN_<SECTION>_<KEY>
    Examples:
        RAMEN_QUALITY_THRESHOLD=0.90
        RAMEN_FILTERS_MIN_WIDTH=3840
        RAMEN_TIMEOUTS_MAX_RUNTIME_MINUTES=45
    """
    
    ENV_PREFIX = "RAMEN_"
    
    def __init__(self, config_path: Path = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.yaml file. Defaults to ./config.yaml
        """
        self.config_path = config_path or Path("./config.yaml")
        self.raw_config: dict = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.raw_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.raw_config = {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Expand environment variable references in values
        self._expand_env_vars()
    
    def _apply_env_overrides(self) -> None:
        """Override configuration values from RAMEN_* environment variables."""
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue
            
            # Parse key: RAMEN_SECTION_KEY -> section.key
            parts = key[len(self.ENV_PREFIX):].lower().split("_")
            
            if len(parts) >= 2:
                section = parts[0]
                config_key = "_".join(parts[1:])
                
                # Convert value to appropriate type
                typed_value = self._parse_value(value)
                
                if section not in self.raw_config:
                    self.raw_config[section] = {}
                
                if isinstance(self.raw_config[section], dict):
                    self.raw_config[section][config_key] = typed_value
                    logger.debug(f"Config override: {section}.{config_key} = {typed_value}")
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate Python type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _expand_env_vars(self, obj: Any = None) -> Any:
        """Expand ${VAR} references in configuration values."""
        if obj is None:
            obj = self.raw_config
        
        if isinstance(obj, str):
            # Match ${VAR_NAME} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                obj = obj.replace(f"${{{var_name}}}", env_value)
            return obj
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        return obj
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path.
        
        Args:
            path: Dot-separated path (e.g., 'quality.threshold')
            default: Default value if path not found
        
        Returns:
            Configuration value or default
        
        Examples:
            config.get('quality.threshold')  # 5.5 (1-10 scale)
            config.get('filters.min_width')  # 2560
            config.get('sources.reddit.enabled')  # True
        """
        parts = path.split(".")
        value = self.raw_config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration as dataclass."""
        quality = self.raw_config.get("quality", {})
        weights_dict = quality.get("weights", {})
        
        return QualityConfig(
            threshold=quality.get("threshold", 5.5),
            weights=QualityWeights(
                visual=weights_dict.get("visual", 0.40),
                composition=weights_dict.get("composition", 0.30),
                aesthetic=weights_dict.get("aesthetic", 0.20),
                suitability=weights_dict.get("suitability", 0.10),
            )
        )
    
    def get_filter_config(self) -> FilterConfig:
        """Get filter configuration as dataclass."""
        filters = self.raw_config.get("filters", {})
        
        return FilterConfig(
            min_width=filters.get("min_width", 2560),
            min_height=filters.get("min_height", 1440),
            min_file_size_kb=filters.get("min_file_size_kb", 200),
            max_file_size_mb=filters.get("max_file_size_mb", 15),
            min_aspect_ratio=filters.get("min_aspect_ratio", 1.77),
            max_aspect_ratio=filters.get("max_aspect_ratio", 2.33),
            max_text_coverage=filters.get("max_text_coverage", 0.30),
            phash_threshold=filters.get("phash_threshold", 10),
        )
    
    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration as dataclass."""
        retry = self.raw_config.get("retry", {})
        
        return RetryConfig(
            max_attempts=retry.get("max_attempts", 3),
            base_delay_sec=retry.get("base_delay_sec", 2.0),
            max_delay_sec=retry.get("max_delay_sec", 60.0),
        )
    
    def get_timeout_config(self) -> TimeoutConfig:
        """Get timeout configuration as dataclass."""
        timeouts = self.raw_config.get("timeouts", {})
        
        return TimeoutConfig(
            api_call_sec=timeouts.get("api_call_sec", 30),
            model_inference_sec=timeouts.get("model_inference_sec", 60),
            upload_sec=timeouts.get("upload_sec", 120),
            max_runtime_minutes=timeouts.get("max_runtime_minutes", 50),
        )
    
    def get_subreddit_configs(self) -> list[SubredditConfig]:
        """Get list of subreddit configurations."""
        reddit = self.raw_config.get("sources", {}).get("reddit", {})
        subreddits = reddit.get("subreddits", [])
        
        return [
            SubredditConfig(
                name=sr.get("name", ""),
                fetch_count=sr.get("fetch_count", 50),
                min_upvotes=sr.get("min_upvotes", 5000),
            )
            for sr in subreddits
            if sr.get("name")
        ]
    
    def get_smoke_test_config(self) -> dict:
        """Get smoke test configuration."""
        return self.raw_config.get("smoke_test", {
            "reddit_count": 3,
            "unsplash_count": 2,
            "pexels_count": 0,
            "skip_upload": True,
            "max_runtime_sec": 120,
        })
    
    def is_smoke_test_mode(self) -> bool:
        """Check if running in smoke test mode via env var."""
        return os.environ.get("RAMEN_SMOKE_TEST", "").lower() in ("true", "1", "yes")


# Global config instance (lazy loaded)
_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config(config_path: Path = None) -> ConfigLoader:
    """Reload configuration from file."""
    global _config
    _config = ConfigLoader(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    import json
    
    config = ConfigLoader()
    
    print("Configuration loaded successfully!\n")
    print(f"Quality threshold: {config.get('quality.threshold')}")
    print(f"Min resolution: {config.get('filters.min_width')}x{config.get('filters.min_height')}")
    print(f"Max runtime: {config.get('timeouts.max_runtime_minutes')} minutes")
    
    print("\nSubreddits:")
    for sr in config.get_subreddit_configs():
        print(f"  - {sr.name}: fetch={sr.fetch_count}, min_upvotes={sr.min_upvotes}")
    
    print("\n--- Full Raw Config ---")
    print(json.dumps(config.raw_config, indent=2, default=str))
