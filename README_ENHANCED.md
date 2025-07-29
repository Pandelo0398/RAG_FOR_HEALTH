# 🏥 Sistema RAG Médico Mejorado con LlamaIndex

Sistema completo de procesamiento de documentos médicos que integra las capacidades avanzadas de **LlamaIndex** para proporcionar análisis médico inteligente, agentes especializados y evaluación de calidad automática.

## 🚀 Nuevas Características con LlamaIndex

### ✨ **Integraciones Principales**

- **🔧 LlamaIndex Core**: Migración completa a la arquitectura optimizada de LlamaIndex
- **🤖 Agentes Médicos Inteligentes**: Agentes especializados para diagnóstico, tratamiento e investigación
- **📊 Sistema de Evaluación**: Evaluación automática de calidad de respuestas médicas
- **🔄 Workflows Médicos**: Procesamiento automatizado con pasos especializados
- **📈 Monitoreo Avanzado**: Métricas detalladas de rendimiento y calidad

### 🎯 **Mejoras Específicas**

1. **Vector Store Optimizado**: Integración nativa con ChromaDB a través de LlamaIndex
2. **Embeddings Mejorados**: Configuración optimizada para texto médico
3. **Retrieval Inteligente**: Búsqueda semántica avanzada con filtros médicos
4. **Agentes Especializados**: 
   - Agente de Diagnóstico
   - Agente de Tratamiento  
   - Agente de Investigación Médica
5. **Evaluación de Calidad**: Múltiples criterios médicos (corrección, relevancia, fidelidad)
6. **Workflows Automatizados**: Procesamiento paso a paso con validación

## 📦 Instalación

### Requisitos Previos

```bash
# Python 3.8+
python --version

# Instalar dependencias base
pip install -r requirements.txt
```

### Configuración del Entorno

```bash
# Variables de entorno (opcional)
export EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
export LLM_API_BASE="http://localhost:8000"
export LLM_MODEL="Qwen/Qwen3-14B"
export CHROMA_PERSIST_DIR="./medical_chroma_db"
```

## 🏗️ Arquitectura Mejorada

```
medical_rag_system/
├── infrastructure/
│   ├── llama_index_integration.py    # 🆕 Integración completa con LlamaIndex
│   ├── embedding_service.py          # Mejorado con LlamaIndex
│   ├── vector_store.py               # Mejorado con LlamaIndex
│   └── llm_client.py                 # Compatible con LlamaIndex
├── services/
│   ├── medical_agent.py              # 🆕 Sistema de agentes médicos
│   ├── evaluation_service.py         # 🆕 Evaluación de calidad
│   ├── medical_processor.py          # Mejorado
│   ├── document_manager.py           # Mejorado
│   └── fallback_processor.py         # Mantenido
├── application/
│   ├── medical_system_enhanced.py    # 🆕 Sistema principal mejorado
│   ├── enhanced_demo.py              # 🆕 Demostración completa
│   └── medical_system.py             # Sistema original
└── models/
    ├── medical.py                    # Mejorado
    ├── document.py                   # Mejorado
    └── patient.py                    # Mantenido
```

## 🚀 Uso Rápido

### 1. Sistema Básico con LlamaIndex

```python
from medical_rag_system.application.medical_system_enhanced import EnhancedMedicalRAGSystem

# Inicializar sistema mejorado
system = EnhancedMedicalRAGSystem(
    use_llama_index=True,
    enable_agents=False,  # Opcional
    enable_evaluation=False  # Opcional
)

# Agregar documento médico
doc_id = await system.add_medical_document_enhanced(
    text="Protocolo de tratamiento para diabetes...",
    document_type=DocumentType.TREATMENT,
    title="Guía Diabetes",
    service="Endocrinología"
)

# Procesar consulta
result = await system.process_diagnosis_enhanced(
    text="¿Cuál es el tratamiento de la diabetes?",
    patient_age=65,
    patient_sex="Masculino"
)

print(f"Respuesta: {result.diagnosis_analysis}")
print(f"Confianza: {result.confidence_score}")
```

### 2. Sistema Completo con Agentes

```python
# Inicializar con todas las características
system = EnhancedMedicalRAGSystem(
    use_llama_index=True,
    enable_agents=True,
    enable_evaluation=True
)

# Usar agente de diagnóstico
diagnosis_result = await system.query_with_agent(
    query="Analiza este caso: paciente con dolor torácico",
    agent_type="diagnosis",
    patient_context={"age": 70, "symptoms": ["dolor torácico"]}
)

# Usar agente de tratamiento
treatment_result = await system.query_with_agent(
    query="Sugiere tratamiento para diabetes",
    agent_type="treatment",
    patient_context={"diagnosis": "diabetes mellitus tipo 2"}
)
```

### 3. Evaluación de Calidad

```python
# Evaluar respuesta individual
evaluation = await system.evaluation_service.evaluate_medical_response(
    query="¿Cuál es el tratamiento de la diabetes?",
    response="La metformina es el tratamiento de primera línea...",
    context_nodes=retrieved_nodes
)

print(f"Score: {evaluation['average_score']}")
print(f"Calidad: {evaluation['quality_level']}")

# Evaluar rendimiento del sistema
performance = await system.evaluate_system_performance([
    "¿Cuál es el tratamiento de la diabetes?",
    "¿Cuáles son los síntomas del infarto?",
    "¿Cómo se diagnostica la hipertensión?"
])
```

## 🎯 Demostración Completa

### Ejecutar Demostración

```bash
# Demostración completa
python -m medical_rag_system.application.enhanced_demo

# Prueba rápida
python -m medical_rag_system.application.enhanced_demo quick
```

### Características de la Demostración

1. **🔧 Inicialización del Sistema**: Verificación de componentes
2. **📚 Carga de Documentos**: Protocolos médicos de ejemplo
3. **🔍 Capacidades Básicas**: Búsqueda y procesamiento
4. **🤖 Agentes Especializados**: Diagnóstico y tratamiento
5. **📊 Evaluación de Calidad**: Métricas de rendimiento
6. **👨‍⚕️ Casos Clínicos**: Escenarios complejos
7. **📈 Estadísticas**: Métricas completas del sistema

## 🔧 Configuración Avanzada

### Configuración de Agentes

```python
# Configurar agentes personalizados
from medical_rag_system.services.medical_agent import MedicalAgentSystem

agent_system = MedicalAgentSystem(llama_rag)

# Agentes disponibles:
# - "diagnosis": Análisis de diagnósticos
# - "treatment": Sugerencias de tratamiento
# - "research": Investigación médica
```

### Configuración de Evaluación

```python
# Configurar evaluadores personalizados
from medical_rag_system.services.evaluation_service import MedicalEvaluationService

evaluation_service = MedicalEvaluationService(llama_rag)

# Evaluadores disponibles:
# - "correctness": Precisión médica
# - "relevancy": Relevancia clínica
# - "faithfulness": Fidelidad a fuentes
# - "answer_relevancy": Relevancia de respuesta
# - "context_relevancy": Relevancia de contexto
```

### Configuración de LlamaIndex

```python
# Configuración avanzada de LlamaIndex
from medical_rag_system.infrastructure.llama_index_integration import LlamaIndexMedicalRAG

llama_rag = LlamaIndexMedicalRAG(
    persist_dir="./custom_chroma_db",
    collection_name="medical_docs_custom",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_config={
        "model": "Qwen/Qwen3-14B",
        "api_base": "http://localhost:8000",
        "max_tokens": 2048,
        "temperature": 0.1
    }
)
```

## 📊 Métricas y Monitoreo

### Estadísticas del Sistema

```python
# Obtener estadísticas completas
stats = system.get_system_stats_enhanced()

print("📊 Estadísticas del Sistema:")
print(f"   Documentos: {stats['system']['documents_processed']}")
print(f"   Diagnósticos: {stats['system']['diagnoses_processed']}")
print(f"   Consultas con agentes: {stats['system']['agent_queries']}")
print(f"   Evaluaciones: {stats['system']['evaluations_performed']}")

# Estadísticas de LlamaIndex
llama_stats = stats['llama_index']
print(f"   Nodos totales: {llama_stats['index']['total_nodes']}")
print(f"   Modelo embeddings: {llama_stats['index']['embedding_model']}")

# Estadísticas de agentes
agent_stats = stats['agents']
print(f"   Agentes disponibles: {agent_stats['total_agents']}")
print(f"   Herramientas: {agent_stats['total_tools']}")
```

### Reportes de Evaluación

```python
# Exportar reporte de evaluación
report_path = system.evaluation_service.export_evaluation_report(
    output_path="medical_evaluation_report.json",
    format="json"
)

print(f"📄 Reporte exportado: {report_path}")
```

## 🔍 Comparación: Sistema Original vs Mejorado

| Característica | Sistema Original | Sistema Mejorado |
|----------------|------------------|------------------|
| **Arquitectura** | Custom implementation | LlamaIndex Core |
| **Vector Store** | ChromaDB directo | ChromaDB + LlamaIndex |
| **Embeddings** | SentenceTransformers | HuggingFace + LlamaIndex |
| **Agentes** | ❌ No disponible | ✅ 3 agentes especializados |
| **Evaluación** | ❌ No disponible | ✅ 5 evaluadores médicos |
| **Workflows** | ❌ No disponible | ✅ Workflows automatizados |
| **Monitoreo** | Básico | ✅ Métricas avanzadas |
| **Retrieval** | Búsqueda simple | ✅ Retrieval inteligente |
| **Calidad** | Sin evaluación | ✅ Evaluación automática |

## 🎯 Casos de Uso

### 1. **Análisis de Diagnósticos**
```python
# Procesar diagnóstico con agente especializado
result = await system.query_with_agent(
    query="Paciente con dolor torácico y presión alta",
    agent_type="diagnosis",
    patient_context={"age": 65, "symptoms": ["dolor torácico", "presión alta"]}
)
```

### 2. **Sugerencias de Tratamiento**
```python
# Obtener recomendaciones de tratamiento
result = await system.query_with_agent(
    query="Tratamiento para diabetes mellitus tipo 2",
    agent_type="treatment",
    patient_context={"diagnosis": "diabetes mellitus tipo 2", "age": 58}
)
```

### 3. **Investigación Médica**
```python
# Buscar información médica relevante
result = await system.query_with_agent(
    query="Últimas guías sobre hipertensión arterial",
    agent_type="research"
)
```

### 4. **Evaluación de Calidad**
```python
# Evaluar calidad de respuesta médica
evaluation = await system.evaluation_service.evaluate_medical_response(
    query="¿Cuál es el tratamiento de la diabetes?",
    response="La metformina es el tratamiento de primera línea...",
    context_nodes=retrieved_nodes
)
```

## 🛠️ Desarrollo y Extensión

### Agregar Nuevos Agentes

```python
# Crear agente personalizado
@step()
def custom_medical_agent(context: Context, query: str) -> str:
    """Agente médico personalizado"""
    # Lógica personalizada
    return "Respuesta personalizada"

# Registrar en el sistema
agent_system.agents["custom"] = ReActAgent.from_tools(
    tools=[custom_tool],
    llm=llama_rag.llm,
    system_prompt="Eres un agente médico personalizado..."
)
```

### Agregar Nuevos Evaluadores

```python
# Crear evaluador personalizado
custom_evaluator = CustomMedicalEvaluator(
    llm=llama_rag.llm,
    criteria={
        "custom_criterion": "Descripción del criterio personalizado"
    }
)

# Agregar al servicio de evaluación
evaluation_service.evaluators["custom"] = custom_evaluator
```

## 📈 Rendimiento y Optimización

### Optimizaciones Implementadas

1. **Lazy Loading**: Carga diferida de modelos y componentes
2. **Caching**: Cache de embeddings y respuestas
3. **Batch Processing**: Procesamiento por lotes
4. **Async/Await**: Operaciones asíncronas
5. **Connection Pooling**: Reutilización de conexiones

### Métricas de Rendimiento

```python
# Monitorear rendimiento
health = await system.test_system_health_enhanced()

print("🏥 Estado del Sistema:")
for component, status in health['components'].items():
    print(f"   {component}: {'✅' if status['available'] else '❌'}")
```

## 🔒 Seguridad y Privacidad

### Características de Seguridad

- **Validación de Entrada**: Validación robusta con Pydantic
- **Sanitización**: Limpieza de texto médico
- **Logging Seguro**: Logs sin información sensible
- **Fallback Seguro**: Sistema de respaldo robusto

### Configuración de Privacidad

```python
# Configurar logging seguro
import logging
logging.getLogger('medical_rag_system').setLevel(logging.WARNING)

# Configurar sanitización
system = EnhancedMedicalRAGSystem(
    use_llama_index=True,
    custom_config={
        "LOG_LEVEL": "WARNING",
        "SANITIZE_INPUT": True
    }
)
```

## 🤝 Contribución

### Guías de Contribución

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva característica'`)
4. **Push** a la rama (`git push origin feature/nueva-caracteristica`)
5. **Crea** un Pull Request

### Estándares de Código

- **PEP 8**: Estilo de código Python
- **Type Hints**: Anotaciones de tipo obligatorias
- **Docstrings**: Documentación completa
- **Tests**: Cobertura de pruebas mínima 80%

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE.md` para más detalles.

## 🆘 Soporte

### Problemas Comunes

1. **Error de importación de LlamaIndex**
   ```bash
   pip install llama-index-core llama-index-llms-openai-like
   ```

2. **Error de ChromaDB**
   ```bash
   pip install chromadb
   ```

3. **Error de embeddings**
   ```bash
   pip install sentence-transformers
   ```

### Contacto

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/RAG_FOR_HEALTH/issues)
- **Documentación**: [Wiki del Proyecto](https://github.com/tu-usuario/RAG_FOR_HEALTH/wiki)
- **Email**: soporte@medical-rag.com

---

## 🎉 ¡Gracias por usar el Sistema RAG Médico Mejorado!

Este sistema representa un avance significativo en la aplicación de IA para el procesamiento de información médica, combinando las mejores prácticas de LlamaIndex con especialización médica. 