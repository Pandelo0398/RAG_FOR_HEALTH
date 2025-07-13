"""
Cliente LLM para interacción con Qwen
"""

import asyncio
import json
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..config import config
from ..models.medical import MedicalProcessingResult


class LLMClientInterface(ABC):
    """Interface para clientes LLM"""
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> Optional[str]:
        """Completar texto usando el LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verificar si el LLM está disponible"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        pass


class QwenClient(LLMClientInterface):
    """Cliente para interactuar con Qwen LLM"""
    
    def __init__(self, 
                 api_base: str = None,
                 model: str = None,
                 timeout: int = None,
                 max_retries: int = None):
        
        self.api_base = api_base or config.LLM_API_BASE
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.max_retries = max_retries or config.LLM_MAX_RETRIES
        
        self.session = requests.Session()
        self._last_health_check = None
        self._is_healthy = False
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del servidor Qwen"""
        try:
            # Health check con cache de 30 segundos
            now = datetime.now()
            if (self._last_health_check and 
                (now - self._last_health_check).total_seconds() < 30):
                return self._is_healthy
            
            # Intentar health endpoint
            health_url = f"{self.api_base}/health"
            response = self.session.get(health_url, timeout=5)
            
            if response.status_code == 200:
                self._is_healthy = True
                self._last_health_check = now
                return True
            
            # Intentar endpoint de modelos como fallback
            models_url = f"{self.api_base}/v1/models"
            response = self.session.get(models_url, timeout=5)
            
            self._is_healthy = response.status_code == 200
            self._last_health_check = now
            return self._is_healthy
            
        except Exception:
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    async def complete(self, prompt: str, **kwargs) -> Optional[str]:
        """Completar texto usando Qwen"""
        
        if not self.is_available():
            print(f"❌ Servidor Qwen no disponible en {self.api_base}")
            return None
        
        # Parámetros por defecto
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", config.LLM_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", config.LLM_MAX_TOKENS),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            **kwargs.get("extra_params", {})
        }
        
        # Intentos con backoff exponencial
        delay = 1.0
        last_error = None
        
        for intento in range(self.max_retries):
            try:
                print(f"🔄 Intento {intento + 1}/{self.max_retries} con Qwen...")
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(
                        f"{self.api_base}/v1/chat/completions",
                        json=params,
                        timeout=self.timeout
                    )
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"✅ Qwen respondió exitosamente")
                    return content
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    
            except Exception as e:
                last_error = str(e)
                print(f"❌ Error en intento {intento + 1}: {str(e)[:100]}...")
            
            # Esperar antes del siguiente intento
            if intento < self.max_retries - 1:
                print(f"⏳ Esperando {delay:.1f}s antes del siguiente intento...")
                await asyncio.sleep(delay)
                delay *= 1.5
        
        print(f"💥 Todos los intentos fallaron. Último error: {last_error}")
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "is_available": self.is_available(),
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Probar conexión con mensaje simple"""
        test_prompt = "Responde solo 'OK' si me puedes escuchar."
        
        start_time = datetime.now()
        response = await self.complete(test_prompt)
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        return {
            "success": response is not None,
            "response": response,
            "response_time_seconds": response_time,
            "timestamp": end_time.isoformat()
        }


class FallbackLLMClient(LLMClientInterface):
    """Cliente LLM de fallback que siempre devuelve None"""
    
    def __init__(self):
        self.name = "FallbackClient"
    
    async def complete(self, prompt: str, **kwargs) -> Optional[str]:
        """Siempre devuelve None para forzar el uso de fallback"""
        print("🛠️ Usando cliente LLM de fallback")
        return None
    
    def is_available(self) -> bool:
        """Siempre disponible como fallback"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del cliente fallback"""
        return {
            "model": "fallback",
            "type": "rule_based",
            "is_available": True
        }
