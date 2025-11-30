#!/usr/bin/env python3
"""
LLM Manager - Universal LLM interface for cloud and local models
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

import os
import torch
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from core.logging_config import get_logger
from core.llm_cache import LLMResponseCache
from core.model_profiles import MODEL_PROFILES
from core.adaptive_retry import AdaptiveRetryManager, create_retry_context, RetryStrategy

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

        # Apply model profile if available for optimized defaults
        profile = MODEL_PROFILES.get(self.model_name)
        if profile and profile.provider == self.provider:
            self.temperature = filtered_config.get('temperature', profile.temperature)
            self.max_tokens = filtered_config.get('max_tokens', profile.max_tokens)
            logger.info(f" Using optimized profile for {self.model_name} ({profile.optimization_level})")
        else:
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

        # Initialize LLM response cache (400x faster for repeated queries)
        cache_enabled = config.get('llm_cache_enabled', True)
        cache_db_path = config.get('llm_cache_db_path', 'cache/llm_responses.db')
        self.cache_bypass = config.get('llm_cache_bypass', False)  # Bypass cache to force fresh calls
        self.prompt_config_hash = config.get('prompt_config_hash', '')  # Config hash for cache invalidation
        self.llm_cache = LLMResponseCache(
            cache_db_path=cache_db_path,
            enabled=cache_enabled
        )

        # Initialize adaptive retry manager for generation failures
        retry_config = config.get('adaptive_retry_config')
        metrics_tracker = None

        if retry_config and retry_config.track_retry_metrics:
            from core.retry_metrics import get_retry_metrics_tracker
            metrics_tracker = get_retry_metrics_tracker(
                db_path=retry_config.track_retry_metrics if isinstance(retry_config.track_retry_metrics, str) else "cache/retry_metrics.db"
            )

        max_retries = retry_config.max_retry_attempts if retry_config else 5
        self.retry_manager = AdaptiveRetryManager(
            max_retries=max_retries,
            config=retry_config,
            metrics_tracker=metrics_tracker
        )
        self.enable_adaptive_retry = config.get('enable_adaptive_retry', True)
        if retry_config:
            self.enable_adaptive_retry = retry_config.enabled

        self._initialize_client()

        logger.info(f"LLMManager initialized: {self.provider}/{self.model_name}")
        if cache_enabled:
            if self.cache_bypass:
                logger.info(f" LLM caching: ENABLED but BYPASSED (forcing fresh calls)")
            else:
                logger.info(f" LLM caching: ENABLED (400x faster for cached queries)")
        if self.enable_adaptive_retry:
            logger.info(f" Adaptive retry: ENABLED (auto-recovery from LLM failures)")
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
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, enable_retry: bool = None,
                 system_prompt: Optional[str] = None, enable_prompt_caching: bool = False) -> str:
        """
        Generate text from prompt with caching and optional adaptive retry

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            enable_retry: Override adaptive retry setting (None = use default)
            system_prompt: Optional system prompt (Anthropic only, will be cached if caching enabled)
            enable_prompt_caching: Enable Anthropic prompt caching (reduces cost by 90% for cached content)

        Returns:
            Generated text
        """
        max_tok = max_tokens or self.max_tokens

        # Check cache first (unless bypassed)
        # Note: System prompt is included in cache key for Anthropic
        cache_key_suffix = f"_sys:{hash(system_prompt)}" if system_prompt else ""
        if not self.cache_bypass:
            cached_response = self.llm_cache.get(
                prompt=prompt + cache_key_suffix,
                model_name=f"{self.provider}:{self.model_name}",
                temperature=self.temperature,
                max_tokens=max_tok,
                config_hash=self.prompt_config_hash  # Invalidates cache when config changes
            )
            if cached_response:
                logger.debug(f" Cache HIT (400x faster)")
                return cached_response
        else:
            logger.debug(f" Cache BYPASSED - forcing fresh LLM call")

        # Determine if adaptive retry should be used
        use_retry = enable_retry if enable_retry is not None else self.enable_adaptive_retry

        # Cache miss or bypass - generate response
        if use_retry:
            # Use adaptive retry for robust generation
            return self._generate_with_adaptive_retry(prompt, max_tok, system_prompt, enable_prompt_caching)
        else:
            # Direct generation without retry
            return self._generate_direct(prompt, max_tok, system_prompt, enable_prompt_caching)

    def _generate_direct(self, prompt: str, max_tokens: int, system_prompt: Optional[str] = None,
                        enable_prompt_caching: bool = False) -> str:
        """Generate without adaptive retry (original behavior)"""
        try:
            if self.provider == "openai":
                response = self._generate_openai(prompt, max_tokens)
            elif self.provider == "anthropic":
                response = self._generate_anthropic(prompt, max_tokens, system_prompt, enable_prompt_caching)
            elif self.provider == "google":
                response = self._generate_google(prompt, max_tokens)
            elif self.provider == "azure":
                response = self._generate_azure(prompt, max_tokens)
            elif self.provider == "local":
                response = self._generate_local(prompt, max_tokens)
            else:
                raise ValueError(f"Provider {self.provider} not supported")

            # Store in cache (unless bypassed)
            cache_key_suffix = f"_sys:{hash(system_prompt)}" if system_prompt else ""
            if not self.cache_bypass:
                self.llm_cache.put(
                    prompt=prompt + cache_key_suffix,
                    model_name=f"{self.provider}:{self.model_name}",
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    response=response,
                    config_hash=self.prompt_config_hash
                )

            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _generate_with_adaptive_retry(self, prompt: str, max_tokens: int, system_prompt: Optional[str] = None,
                                      enable_prompt_caching: bool = False) -> str:
        """
        Generate with adaptive retry on failures

        Implements progressive prompt reduction if generation fails
        """
        # Create retry context with configuration
        retry_context = create_retry_context(
            clinical_text=prompt,
            max_attempts=self.retry_manager.max_retries,
            config=self.retry_manager.config,
            provider=self.provider,
            model_name=self.model_name
        )

        # Define generation function that uses modified prompt
        def generation_func():
            # Use potentially truncated prompt from retry context
            current_prompt = retry_context.clinical_text

            # Call provider-specific generation
            if self.provider == "openai":
                response = self._generate_openai(current_prompt, max_tokens)
            elif self.provider == "anthropic":
                response = self._generate_anthropic(current_prompt, max_tokens, system_prompt, enable_prompt_caching)
            elif self.provider == "google":
                response = self._generate_google(current_prompt, max_tokens)
            elif self.provider == "azure":
                response = self._generate_azure(current_prompt, max_tokens)
            elif self.provider == "local":
                response = self._generate_local(current_prompt, max_tokens)
            else:
                raise ValueError(f"Provider {self.provider} not supported")

            return response

        # Execute with retry
        try:
            response = self.retry_manager.execute_with_retry(
                generation_func,
                retry_context
            )

            # Store in cache (use original prompt as key)
            cache_key_suffix = f"_sys:{hash(system_prompt)}" if system_prompt else ""
            if not self.cache_bypass:
                self.llm_cache.put(
                    prompt=prompt + cache_key_suffix,
                    model_name=f"{self.provider}:{self.model_name}",
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    response=response,
                    config_hash=self.prompt_config_hash
                )

            return response

        except Exception as e:
            logger.error(f"Generation failed after {retry_context.attempt} attempts: {e}")
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
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, system_prompt: Optional[str] = None,
                            enable_prompt_caching: bool = False) -> str:
        """
        Generate with Anthropic

        Args:
            prompt: User prompt
            max_tokens: Max tokens to generate
            system_prompt: Optional system prompt (will be cached if caching enabled)
            enable_prompt_caching: Enable prompt caching (reduces cost by 90% for cached content)
        """
        try:
            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }

            # Add system prompt with cache control if provided
            if system_prompt:
                if enable_prompt_caching and "claude-3" in self.model_name.lower():
                    # Use prompt caching for system prompts (reduces cost by 90%)
                    params["system"] = [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    logger.debug("Prompt caching ENABLED for system prompt")
                else:
                    params["system"] = system_prompt

            response = self.client.messages.create(**params)

            # Log cache usage if available
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens > 0:
                    logger.info(f" Cache HIT: {usage.cache_read_input_tokens} tokens read from cache "
                              f"(saved 90% on {usage.cache_read_input_tokens} tokens)")
                if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens > 0:
                    logger.debug(f"Cache WRITE: {usage.cache_creation_input_tokens} tokens cached for future use")

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

            # Log input size for debugging
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"[GENERATION] Input tokens: {input_length}, Max output: {max_tokens}, Total: {input_length + max_tokens}")

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

            # Generate with progress indication
            logger.info(f"[GENERATION] Starting model inference (this may take 10-60s for complex tasks)...")
            import time
            start_gen_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)

            gen_duration = time.time() - start_gen_time
            output_length = outputs[0].shape[0] - input_length
            logger.info(f"[GENERATION] Completed in {gen_duration:.2f}s. Generated {output_length} tokens ({output_length/gen_duration:.1f} tokens/s)")
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()

        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise

    def generate_with_tool_calling(self, messages: list, tools: list, max_tokens: Optional[int] = None) -> dict:
        """
        Generate with native tool calling support (OpenAI/Anthropic)

        Args:
            messages: Conversation history in API format
            tools: List of tool definitions in OpenAI function calling format
            max_tokens: Max tokens to generate

        Returns:
            dict with:
                - content: str (response content)
                - tool_calls: list (tool calls if any)
                - finish_reason: str ('stop', 'tool_calls', etc.)
        """
        max_tok = max_tokens or self.max_tokens

        try:
            if self.provider == "openai":
                return self._generate_with_tools_openai(messages, tools, max_tok)
            elif self.provider == "anthropic":
                return self._generate_with_tools_anthropic(messages, tools, max_tok)
            else:
                # Fallback for providers that don't support native tool calling
                logger.warning(f"Provider {self.provider} doesn't support native tool calling, using text generation")
                # Convert messages to text
                prompt = self._messages_to_text(messages)
                response = self.generate(prompt, max_tok)
                return {
                    'content': response,
                    'tool_calls': [],
                    'finish_reason': 'stop'
                }
        except Exception as e:
            logger.error(f"Tool calling generation failed: {e}")
            raise

    def _generate_with_tools_openai(self, messages: list, tools: list, max_tokens: int) -> dict:
        """Generate with OpenAI tool calling"""
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature
            }

            # Add tools if provided
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            # Use correct parameter based on model
            if self._uses_completion_tokens():
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Parse response
            result = {
                'content': message.content or '',
                'tool_calls': [],
                'finish_reason': finish_reason
            }

            # Check for tool calls
            if message.tool_calls:
                for tc in message.tool_calls:
                    result['tool_calls'].append({
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    })

            return result

        except Exception as e:
            logger.error(f"OpenAI tool calling failed: {e}")
            raise

    def _generate_with_tools_anthropic(self, messages: list, tools: list, max_tokens: int) -> dict:
        """Generate with Anthropic tool calling"""
        try:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            if tools:
                for tool in tools:
                    anthropic_tools.append({
                        'name': tool['function']['name'],
                        'description': tool['function']['description'],
                        'input_schema': tool['function']['parameters']
                    })

            # Separate system message from conversation
            system_msg = None
            conv_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                else:
                    conv_messages.append(msg)

            params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": conv_messages
            }

            if system_msg:
                params["system"] = system_msg

            if anthropic_tools:
                params["tools"] = anthropic_tools

            response = self.client.messages.create(**params)

            # Parse response
            result = {
                'content': '',
                'tool_calls': [],
                'finish_reason': response.stop_reason
            }

            # Extract content and tool calls
            for block in response.content:
                if block.type == 'text':
                    result['content'] += block.text
                elif block.type == 'tool_use':
                    result['tool_calls'].append({
                        'id': block.id,
                        'type': 'function',
                        'function': {
                            'name': block.name,
                            'arguments': json.dumps(block.input)
                        }
                    })

            return result

        except Exception as e:
            logger.error(f"Anthropic tool calling failed: {e}")
            raise

    def _messages_to_text(self, messages: list) -> str:
        """Convert message history to text (fallback for non-tool-calling providers)"""
        lines = []
        for msg in messages:
            role = msg.get('role', '').upper()
            content = msg.get('content', '')
            if content:
                lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

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