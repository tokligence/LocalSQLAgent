"""
LLM Configuration Management for Tokligence LocalSQLAgent
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger("LLMConfig")

class LLMConfig:
    """Manage LLM configuration settings"""

    CONFIG_FILE = Path.home() / ".tokligence" / "llm_config.json"

    PROVIDERS = {
        "ollama": {
            "name": "Ollama (Local)",
            "base_url": "http://localhost:11434",
            "models_endpoint": "/api/tags",
            "generate_endpoint": "/api/generate",
            "default_model": "qwen2.5-coder:7b"
        },
        "openai": {
            "name": "OpenAI API",
            "base_url": "https://api.openai.com/v1",
            "models_endpoint": "/models",
            "generate_endpoint": "/chat/completions",
            "default_model": "gpt-3.5-turbo"
        }
    }

    def __init__(self):
        """Initialize LLM configuration"""
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config = None

        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        if not config:
            # Default configuration
            config = {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "qwen2.5-coder:7b",
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            }

        # Environment overrides (useful for containers)
        env_provider = os.getenv("LLM_PROVIDER")
        env_ollama_base = os.getenv("OLLAMA_BASE_URL")
        env_ollama_model = os.getenv("OLLAMA_MODEL")
        env_openai_model = os.getenv("OPENAI_MODEL")
        env_ollama_temp = os.getenv("OLLAMA_TEMPERATURE")
        env_ollama_max_tokens = os.getenv("OLLAMA_MAX_TOKENS")
        env_openai_temp = os.getenv("OPENAI_TEMPERATURE")
        env_openai_max_tokens = os.getenv("OPENAI_MAX_TOKENS")

        if env_provider:
            config["provider"] = env_provider
        if env_ollama_base:
            config["ollama"]["base_url"] = env_ollama_base
        if env_ollama_model:
            config["ollama"]["model"] = env_ollama_model
        if env_ollama_temp:
            try:
                config["ollama"]["temperature"] = float(env_ollama_temp)
            except ValueError:
                pass
        if env_ollama_max_tokens:
            try:
                config["ollama"]["max_tokens"] = int(env_ollama_max_tokens)
            except ValueError:
                pass
        if env_openai_model:
            config["openai"]["model"] = env_openai_model
        if env_openai_temp:
            try:
                config["openai"]["temperature"] = float(env_openai_temp)
            except ValueError:
                pass
        if env_openai_max_tokens:
            try:
                config["openai"]["max_tokens"] = int(env_openai_max_tokens)
            except ValueError:
                pass

        # Ensure defaults exist for optional keys
        config.setdefault("ollama", {})
        config.setdefault("openai", {})
        config["ollama"].setdefault("temperature", 0.1)
        config["ollama"].setdefault("max_tokens", 500)
        config["openai"].setdefault("temperature", 0.1)
        config["openai"].setdefault("max_tokens", 500)

        return config

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)

            self.config = config
            logger.info(f"Configuration saved to {self.CONFIG_FILE}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def test_ollama_connection(self) -> tuple[bool, str, List[str]]:
        """Test Ollama connection and get available models"""
        try:
            base_url = self.config["ollama"]["base_url"]
            response = requests.get(f"{base_url}/api/tags", timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if not model_names:
                    return False, "Ollama is running but no models are installed", []

                return True, f"Connected to Ollama with {len(model_names)} models", model_names
            else:
                return False, f"Ollama returned status {response.status_code}", []

        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Please ensure Ollama is running on port 11434", []
        except Exception as e:
            return False, f"Error connecting to Ollama: {str(e)}", []

    def test_openai_connection(self) -> tuple[bool, str]:
        """Test OpenAI API connection"""
        api_key = self.config["openai"]["api_key"]

        if not api_key:
            return False, "OpenAI API key not configured"

        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )

            if response.status_code == 200:
                return True, "Successfully connected to OpenAI API"
            elif response.status_code == 401:
                return False, "Invalid OpenAI API key"
            else:
                return False, f"OpenAI API returned status {response.status_code}"

        except Exception as e:
            return False, f"Error connecting to OpenAI: {str(e)}"

    def get_current_provider(self) -> str:
        """Get the current LLM provider"""
        return self.config.get("provider", "ollama")

    def get_current_model(self) -> str:
        """Get the current model for the active provider"""
        provider = self.get_current_provider()
        if provider == "ollama":
            return self.config["ollama"]["model"]
        else:
            return self.config["openai"]["model"]

    def call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> str:
        """Call the configured LLM with the given prompt"""
        active_provider = provider or self.get_current_provider()

        if active_provider == "ollama":
            temp = temperature if temperature is not None else self.config["ollama"].get("temperature", 0.1)
            tokens = max_tokens if max_tokens is not None else self.config["ollama"].get("max_tokens", 500)
            return self._call_ollama(prompt, temp, tokens, model=model)
        if active_provider == "openai":
            temp = temperature if temperature is not None else self.config["openai"].get("temperature", 0.1)
            tokens = max_tokens if max_tokens is not None else self.config["openai"].get("max_tokens", 500)
            return self._call_openai(prompt, temp, tokens, model=model)
        raise ValueError(f"Unknown provider: {active_provider}")

    def _call_ollama(self, prompt: str, temperature: float, max_tokens: int, model: Optional[str] = None) -> str:
        """Call Ollama API"""
        try:
            base_url = self.config["ollama"]["base_url"]
            model_name = model or self.config["ollama"]["model"]

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60  # Longer timeout for generation
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            raise Exception("Cannot connect to Ollama. Please ensure it's running.")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int, model: Optional[str] = None) -> str:
        """Call OpenAI API"""
        api_key = self.config["openai"]["api_key"]

        if not api_key:
            raise Exception("OpenAI API key not configured")

        try:
            model_name = model or self.config["openai"]["model"]

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                raise Exception("Invalid OpenAI API key")
            else:
                logger.error(f"OpenAI returned status {response.status_code}: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

# Singleton instance
_llm_config = None

def get_llm_config() -> LLMConfig:
    """Get the singleton LLM configuration instance"""
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config
