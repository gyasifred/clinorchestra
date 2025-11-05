#!/usr/bin/env python3
"""
LLM Manager - Universal LLM interface for cloud and local models
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

import os
import torch
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from core.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


class LLMManager:
    """Unified LLM manager for cloud and local models"""
    
    PROVIDER_PARAMS = {
        'openai': ['provider', 'model_name', 'api_key', 'temperature', 'max_tokens'],
        'anthropic': ['provider', 'model_name', 'api_key', 'temperature', 'max_tokens'],
        'google': ['provider', 'model_name', 'api_key', 'temperature', 'max_tokens', 'google_project_id'],
        'azure': ['provider', 'model_name', 'api_key', 'temperature', 'max_tokens', 'azure_endpoint', 'azure_deployment'],
        'local': ['provider', 'model_name', 'temperature', 'max_tokens', 'local_model_path', 'quantization', 'gpu_layers', 'max_seq_length']
    }
    
    # Models that use max_completion_tokens instead of max_tokens
    COMPLETION_TOKEN_MODELS = [
        'o1-preview', 'o1-mini', 'o1',
        'gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-11-20'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM manager"""
        self.provider = config.get('provider', 'openai').lower()
        
        allowed_params = self.PROVIDER_PARAMS.get(self.provider, [])
        filtered_config = {k: v for k, v in config.items() if k in allowed_params}
        
        self.model_name = filtered_config.get('model_name')
        self.api_key = filtered_config.get('api_key') or self._get_api_key_from_env()
        self.temperature = filtered_config.get('temperature', 0.01)
        self.max_tokens = filtered_config.get('max_tokens', 4096)
        self.max_seq_length = filtered_config.get('max_seq_length', 16384)
        
        self.azure_endpoint = filtered_config.get('azure_endpoint')
        self.azure_deployment = filtered_config.get('azure_deployment')
        self.google_project_id = filtered_config.get('google_project_id')
        
        self.local_model_path = filtered_config.get('local_model_path') or self.model_name
        self.quantization = filtered_config.get('quantization', '4bit')
        self.gpu_layers = filtered_config.get('gpu_layers', -1)
        
        self.client = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._initialize_client()
        
        logger.info(f"LLMManager initialized: {self.provider}/{self.model_name}")
        if self.provider == 'local':
            logger.info(f"Local model context window: {self.max_seq_length}")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_KEY"
        }
        env_var = env_vars.get(self.provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def _initialize_client(self):
        """Initialize client"""
        try:
            if self.provider == "openai":
                self._initialize_openai_client()
            elif self.provider == "anthropic":
                self._initialize_anthropic_client()
            elif self.provider == "google":
                self._initialize_google_client()
            elif self.provider == "azure":
                self._initialize_azure_client()
            elif self.provider == "local":
                self._initialize_local_model()
            else:
                raise ValueError(f"Provider {self.provider} not supported")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider}: {e}")
            raise
    
    def _initialize_openai_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI client initialized: {self.model_name}")
    
    def _initialize_anthropic_client(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Anthropic client initialized: {self.model_name}")
    
    def _initialize_google_client(self):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        if not self.api_key:
            raise ValueError("Google API key required")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f"Google client initialized: {self.model_name}")
    
    def _initialize_azure_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        if not self.api_key or not self.azure_endpoint or not self.azure_deployment:
            raise ValueError("Azure requires API key, endpoint, and deployment")
        
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=self.azure_endpoint
        )
        logger.info(f"Azure client initialized: {self.azure_deployment}")
    
    def _initialize_local_model(self):
        """Initialize local model with proper context window"""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("unsloth package not installed")
        
        logger.info(f"Loading local model: {self.local_model_path}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        
        dtype = torch.float16
        load_in_4bit = self.quantization == "4bit"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.local_model_path,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        FastLanguageModel.for_inference(self.model)
        
        chat_template = self._detect_chat_template(self.local_model_path)
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == 'cuda' and not load_in_4bit:
            self.model.to(self.device)
            torch.cuda.empty_cache()
        
        logger.info(f"Local model loaded: device={self.device}, context={self.max_seq_length}, quantized={load_in_4bit}")
    
    def _detect_chat_template(self, model_name: str) -> str:
        """Determine chat template"""
        model_lower = model_name.lower()
        
        if 'phi-4' in model_lower:
            return 'phi-4'
        elif 'phi-3' in model_lower or 'phi' in model_lower:
            return 'phi-3'
        elif 'llama-3' in model_lower or 'llama3' in model_lower:
            return 'llama-3.1'
        elif 'llama' in model_lower:
            return 'llama-2'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'qwen2.5' in model_lower:
            return 'qwen-2.5'
        elif 'qwen' in model_lower:
            return 'qwen2'
        elif 'gemma' in model_lower:
            return 'gemma'
        else:
            logger.warning(f"Unknown model, using chatml")
            return 'chatml'
    
    def _uses_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens parameter"""
        if self.provider != 'openai':
            return False
        return any(model_id in self.model_name for model_id in self.COMPLETION_TOKEN_MODELS)
    
    def _is_qwen_model(self) -> bool:
        """Check if current model is a Qwen model"""
        if self.provider != 'local':
            return False
        model_name_lower = self.local_model_path.lower()
        return 'qwen' in model_name_lower
    
    def _is_llama_model(self) -> bool:
        """Check if current model is a Llama model"""
        if self.provider != 'local':
            return False
        model_name_lower = self.local_model_path.lower()
        return 'llama' in model_name_lower
    
    def _should_use_json_mode_for_qwen(self, prompt: str) -> bool:
        """Determine if JSON mode should be enabled for Qwen models"""
        if not self._is_qwen_model():
            return False
        
        prompt_lower = prompt.lower()
        json_indicators = ['json', 'JSON', '{"', 'return only json', 'output format']
        return any(indicator in prompt or indicator in prompt_lower for indicator in json_indicators)
    
    def _should_use_json_mode_for_llama(self, prompt: str) -> bool:
        """Determine if JSON mode should be enabled for Llama models"""
        if not self._is_llama_model():
            return False
        
        prompt_lower = prompt.lower()
        json_indicators = ['json', 'JSON', '{"', 'return only json', 'output format']
        return any(indicator in prompt or indicator in prompt_lower for indicator in json_indicators)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text from prompt"""
        max_tok = max_tokens or self.max_tokens
        
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, max_tok)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, max_tok)
            elif self.provider == "google":
                return self._generate_google(prompt, max_tok)
            elif self.provider == "azure":
                return self._generate_azure(prompt, max_tok)
            elif self.provider == "local":
                return self._generate_local(prompt, max_tok)
            else:
                raise ValueError(f"Provider {self.provider} not supported")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        """Generate with OpenAI"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature
            }
            
            # Use correct parameter based on model
            if self._uses_completion_tokens():
                params["max_completion_tokens"] = max_tokens
                logger.debug(f"Using max_completion_tokens={max_tokens} for {self.model_name}")
            else:
                params["max_tokens"] = max_tokens
                logger.debug(f"Using max_tokens={max_tokens} for {self.model_name}")
            
            # Add JSON mode for supported models
            if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
                if not any(model_id in self.model_name for model_id in self.COMPLETION_TOKEN_MODELS):
                    params["response_format"] = {"type": "json_object"}
                    logger.debug(f"JSON mode enabled for {self.model_name}")
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _generate_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Generate with Anthropic"""
        try:
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = self.client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _generate_google(self, prompt: str, max_tokens: int) -> str:
        """Generate with Google"""
        try:
            generation_config = {
                'temperature': self.temperature,
                'max_output_tokens': max_tokens,
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            raise
    
    def _generate_azure(self, prompt: str, max_tokens: int) -> str:
        """Generate with Azure"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            params = {
                "model": self.azure_deployment,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens
            }
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure generation failed: {e}")
            raise
    
    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Generate with local model"""
        try:
            # Check if JSON mode is needed for Qwen or Llama models
            use_json_mode_for_qwen = self._should_use_json_mode_for_qwen(prompt)
            use_json_mode_for_llama = self._should_use_json_mode_for_llama(prompt)
            
            messages = []
            
            # Add system message for Qwen models
            if use_json_mode_for_qwen:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant that provides responses exclusively in JSON format. Never include any text before or after the JSON object."
                })
                logger.info("JSON mode enabled for Qwen model via system message")
            
            # Add system message for Llama models
            if use_json_mode_for_llama:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant that provides responses exclusively in JSON format. Never include any text before or after the JSON object."
                })
                logger.info("JSON mode enabled for Llama model via system message")
            
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length - max_tokens
            )
            
            # Move to device if CUDA
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generation parameters
            generation_params = {
                "max_new_tokens": max_tokens,
                "temperature": self.temperature,
                "do_sample": True if self.temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.provider == "local" and self.model is not None:
            if self.device and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
        
        logger.info("LLM Manager cleaned up")