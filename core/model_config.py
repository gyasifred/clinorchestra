#!/usr/bin/env python3
"""
Model Configuration Handler
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration data class"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.01
    max_tokens: int = 4096
    max_seq_length: int = 16384
    model_type: str = "chat"
    
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    google_project_id: Optional[str] = None
    
    local_model_path: Optional[str] = None
    quantization: Optional[str] = "4bit"
    gpu_layers: int = -1
    
    def validate(self) -> tuple[bool, str]:
        """Validate configuration completeness"""
        if not self.provider:
            return False, "Provider must be specified"
        
        if not self.model_name:
            return False, "Model name must be specified"
        
        if self.provider in ["openai", "anthropic", "google"]:
            if not self.api_key:
                return False, f"API key required for {self.provider}"
        
        if self.provider == "azure":
            if not all([self.api_key, self.azure_endpoint, self.azure_deployment]):
                return False, "Azure requires API key, endpoint, and deployment name"
        
        if self.provider == "local":
            if not self.local_model_path and not self.model_name:
                return False, "Local provider requires model path or model name"
        
        if not (0.0 <= self.temperature <= 2.0):
            return False, "Temperature must be between 0.0 and 2.0"
        
        if self.max_tokens < 1:
            return False, "Max tokens must be positive"
        
        return True, "Configuration valid"


class ModelConfigManager:
    """Manages model configurations with save/load capabilities"""
    
    SUPPORTED_MODELS = {
        "openai": [
            "gpt-4o-2024-11-20",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini"
        ],
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "google": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "azure": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-4-32k",
            "gpt-35-turbo",
            "gpt-35-turbo-16k"
        ],
        "local": [
            "unsloth/Phi-4",
            "unsloth/Phi-3.5-mini-instruct",
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            "unsloth/Meta-Llama-3.1-70B-Instruct",
            "unsloth/Mistral-7B-Instruct-v0.3",
            "unsloth/Mixtral-8x7B-Instruct-v0.1",
            "unsloth/Qwen2.5-7B-Instruct",
            "unsloth/Qwen2.5-14B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Custom (specify path)"
        ]
    }
    
    MODEL_CONTEXT_LIMITS = {
        "gpt-4o-2024-11-20": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "o1-preview": 128000,
        "o1-mini": 128000,
        "claude-sonnet-4-20250514": 200000,
        "claude-opus-4-20250514": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "gemini-2.0-flash-exp": 1000000,
        "gemini-1.5-pro-002": 2000000,
        "gemini-1.5-flash-002": 1000000,
        "gemini-1.5-pro": 2000000,
        "gemini-1.5-flash": 1000000,
        "unsloth/Phi-4": 16384,
        "unsloth/Phi-3.5-mini-instruct": 131072,
        "unsloth/Meta-Llama-3.1-8B-Instruct": 131072,
        "unsloth/Meta-Llama-3.1-70B-Instruct": 131072,
        "unsloth/Mistral-7B-Instruct-v0.3": 32768,
        "unsloth/Qwen2.5-7B-Instruct": 131072,
        "unsloth/Qwen2.5-14B-Instruct": 131072,
        "meta-llama/Llama-3.1-8B-Instruct": 131072,
        "mistralai/Mistral-7B-Instruct-v0.3": 32768
    }
    
    def __init__(self, config_dir: str = "./configs"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[ModelConfig] = None
        logger.info(f"ModelConfigManager initialized")
    
    def get_supported_models(self, provider: str) -> List[str]:
        """Get list of supported models for a provider"""
        return self.SUPPORTED_MODELS.get(provider, [])
    
    def get_model_context_limit(self, model_name: str) -> int:
        """Get context window limit for a model"""
        if model_name in self.MODEL_CONTEXT_LIMITS:
            return self.MODEL_CONTEXT_LIMITS[model_name]
        
        for key, limit in self.MODEL_CONTEXT_LIMITS.items():
            if key in model_name or model_name in key:
                return limit
        
        return 16384
    
    def validate_model(self, provider: str, model_name: str) -> tuple[bool, str]:
        """Validate if model is supported for provider"""
        supported = self.SUPPORTED_MODELS.get(provider, [])
        
        if not supported:
            return False, f"Unknown provider: {provider}"
        
        if model_name in supported:
            return True, "Model supported"
        
        if provider == "local":
            return True, "Local model (custom path allowed)"
        
        for supported_model in supported:
            if supported_model in model_name or model_name in supported_model:
                return True, "Model supported (version match)"
        
        return False, f"Model '{model_name}' not in supported list for {provider}"
    
    def create_config(self, **kwargs) -> ModelConfig:
        """Create a new model configuration"""
        if 'max_seq_length' not in kwargs and 'model_name' in kwargs:
            kwargs['max_seq_length'] = self.get_model_context_limit(kwargs['model_name'])
        
        config = ModelConfig(**kwargs)
        self.current_config = config
        return config
    
    def save_config(self, config: ModelConfig, profile_name: str) -> tuple[bool, str]:
        """Save configuration to JSON file"""
        try:
            is_valid, message = config.validate()
            if not is_valid:
                return False, f"Validation failed: {message}"
            
            filepath = self.config_dir / f"{profile_name}.json"
            
            config_dict = asdict(config)
            config_dict['api_key'] = "***REDACTED***" if config.api_key else None
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved: {profile_name}")
            return True, f"Configuration saved to {filepath}"
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False, f"Error saving configuration: {e}"
    
    def load_config(self, profile_name: str) -> tuple[bool, Optional[ModelConfig], str]:
        """Load configuration from JSON file"""
        try:
            filepath = self.config_dir / f"{profile_name}.json"
            
            if not filepath.exists():
                return False, None, f"Configuration file not found: {filepath}"
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            if config_dict.get('api_key') == "***REDACTED***":
                config_dict['api_key'] = None
            
            config = ModelConfig(**config_dict)
            self.current_config = config
            
            logger.info(f"Configuration loaded: {profile_name}")
            return True, config, "Configuration loaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False, None, f"Error loading configuration: {e}"
    
    def list_profiles(self) -> List[str]:
        """List all saved configuration profiles"""
        profiles = []
        for filepath in self.config_dir.glob("*.json"):
            profiles.append(filepath.stem)
        return sorted(profiles)
    
    def export_config(self, config: ModelConfig) -> str:
        """Export configuration as JSON string"""
        config_dict = asdict(config)
        config_dict['api_key'] = "***REDACTED***" if config.api_key else None
        return json.dumps(config_dict, indent=2)
    
    def import_config(self, json_string: str) -> tuple[bool, Optional[ModelConfig], str]:
        """Import configuration from JSON string"""
        try:
            config_dict = json.loads(json_string)
            
            if config_dict.get('api_key') == "***REDACTED***":
                config_dict['api_key'] = None
            
            config = ModelConfig(**config_dict)
            is_valid, message = config.validate()
            
            if not is_valid:
                return False, None, f"Invalid configuration: {message}"
            
            self.current_config = config
            return True, config, "Configuration imported successfully"
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {e}"
        except Exception as e:
            return False, None, f"Error importing configuration: {e}"
    
    def get_api_key_from_env(self, provider: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_KEY"
        }
        
        env_var = env_vars.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for current configuration"""
        if self.current_config and self.current_config.provider == provider:
            self.current_config.api_key = api_key
            return True
        return False
    
    def get_config_summary(self, config: ModelConfig) -> Dict[str, Any]:
        """Get human-readable configuration summary"""
        has_key = bool(config.api_key)
        
        summary = {
            "Provider": config.provider.title(),
            "Model": config.model_name,
            "Temperature": config.temperature,
            "Max Tokens": config.max_tokens,
            "Model Type": config.model_type,
            "API Key": "Set" if has_key else "Not Set"
        }
        
        if config.provider == "azure":
            summary["Endpoint"] = config.azure_endpoint or "Not set"
            summary["Deployment"] = config.azure_deployment or "Not set"
        
        if config.provider == "google":
            summary["Project ID"] = config.google_project_id or "Not set"
        
        if config.provider == "local":
            summary["Model Path"] = config.local_model_path or "Using model name"
            summary["Max Seq Length"] = config.max_seq_length
            summary["Quantization"] = config.quantization or "None"
            summary["GPU Layers"] = config.gpu_layers
        
        context_limit = self.get_model_context_limit(config.model_name)
        summary["Context Window"] = f"{context_limit:,} tokens"
        
        return summary