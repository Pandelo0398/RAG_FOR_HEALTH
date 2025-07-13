# 🏥 Sistema RAG Médico Optimizado

Sistema completo de procesamiento de documentos médicos con arquitectura de capas, integración de LLM Qwen, ChromaDB para almacenamiento vectorial y procesamiento resiliente con fallback.

## 🚀 Características Principales

- **Arquitectura de Capas**: Separación clara de responsabilidades
- **LLM Qwen Integrado**: Procesamiento inteligente con fallback automático  
- **ChromaDB Persistente**: Almacenamiento vectorial optimizado
- **Chunking Inteligente**: Procesamiento eficiente de documentos largos
- **Retrieval Semántico**: Contexto relevante antes de consultar LLM
- **Validación Robusta**: Modelos Pydantic para datos estructurados
- **Sistema Resiliente**: Funciona con o sin servidor LLM

## 📁 Estructura del Proyecto

```
medical_rag_system/
├── __init__.py                 # Paquete principal
├── config.py                  # Configuración centralizada
├── models/                    # Modelos de datos
│   ├── __init__.py
│   ├── patient.py            # Datos del paciente
│   ├── medical.py            # Modelos médicos y validación
│   └── document.py           # Gestión de documentos
├── infrastructure/           # Capa de infraestructura
│   ├── __init__.py
│   ├── llm_client.py        # Cliente LLM Qwen
│   ├── vector_store.py      # ChromaDB wrapper
│   └── embedding_service.py # Servicio de embeddings
├── services/                # Lógica de negocio
│   ├── __init__.py
│   ├── medical_processor.py # Procesador médico principal
│   ├── document_manager.py  # Gestor de documentos
│   └── fallback_processor.py # Procesador de fallback
└── application/             # Capa de aplicación
    ├── __init__.py
    ├── medical_system.py    # Sistema principal
    └── demo_runner.py       # Ejecutor de demos
```

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd RAG_FOR_HEALTH
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

### Variables de Entorno (Opcional)

```bash
# LLM Configuration
export LLM_API_BASE="http://localhost:8000"
export LLM_MODEL="Qwen/Qwen3-14B"

# ChromaDB Configuration  
export CHROMA_PERSIST_DIR="./medical_chroma_db"

# Embedding Configuration
export EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Application Configuration
export APP_DEBUG="false"
export LOG_LEVEL="INFO"
```

### Servidor Qwen (Opcional)

Para usar el LLM Qwen, debe estar ejecutándose en el puerto configurado:

```bash
# Ejemplo con vLLM
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --port 8000
```

**Nota**: El sistema funciona perfectamente sin Qwen usando el procesador de fallback.

## 🚀 Uso Básico

### Inicialización del Sistema

```python
import asyncio
from medical_rag_system.application import MedicalRAGSystem

# Inicializar sistema
system = MedicalRAGSystem(use_qwen=True)

# Verificar salud del sistema
health = await system.test_system_health()
print("Sistema saludable:", health["system_healthy"])
```

### Procesamiento de Diagnósticos

```python
# Procesar diagnóstico médico
result = await system.process_diagnosis(
    text="Paciente de 65 años con DM tipo 2 descompensada y HTA secundaria",
    patient_age=65,
    patient_sex="M",
    service="medicina interna"
)

print("Diagnóstico principal:", result.main_diagnosis)
print("Códigos CIE-10:", result.cie10_codes)
print("Síntomas:", result.symptoms)
print("Confianza:", result.get_overall_confidence())
```

### Gestión de Documentos

```python
# Agregar documento médico
doc_id = system.add_medical_document(
    text="Protocolo de manejo de diabetes mellitus...",
    document_type=DocumentType.CLINICAL_NOTE,
    title="Protocolo DM",
    service="endocrinología"
)

# Buscar documentos similares
results = system.search_medical_documents(
    query="diabetes tratamiento",
    top_k=5
)
```

## 🎯 Demo Completo

```python
from medical_rag_system.application import DemoRunner

# Crear y ejecutar demo
demo = DemoRunner(system)
results = await demo.run_complete_demo()

# Demo de caso individual
case_result = await demo.run_single_case_demo(case_index=0)
```

## 📊 Arquitectura de Capas

### 1. **Capa de Configuración** (`config.py`)
- Configuración centralizada y validada
- Variables de entorno y valores por defecto
- Validación de parámetros del sistema

### 2. **Capa de Modelos** (`models/`)
- **Patient**: Datos del paciente con validación
- **Medical**: Resultados médicos con Pydantic
- **Document**: Metadatos y gestión de documentos

### 3. **Capa de Infraestructura** (`infrastructure/`)
- **LLM Client**: Interfaz con Qwen y fallback
- **Vector Store**: ChromaDB para almacenamiento vectorial
- **Embedding Service**: Generación de embeddings multiidioma

### 4. **Capa de Servicios** (`services/`)
- **Medical Processor**: Lógica de procesamiento médico
- **Document Manager**: Gestión de documentos médicos
- **Fallback Processor**: Procesamiento basado en reglas

### 5. **Capa de Aplicación** (`application/`)
- **Medical System**: Orquestación del sistema completo
- **Demo Runner**: Ejecución de demostraciones

## 🔧 Características Técnicas

### Procesamiento Resiliente
- **Reintentos automáticos** con backoff exponencial
- **Fallback inteligente** cuando LLM no está disponible
- **Validación JSON** robusta de respuestas LLM

### Almacenamiento Vectorial
- **Chunking inteligente** con overlap configurable
- **Embeddings multiidioma** optimizados para español médico
- **Búsqueda semántica** con filtros de metadatos

### Validación de Datos
- **Modelos Pydantic** para validación automática
- **Códigos CIE-10** validados
- **Entidades clínicas** estructuradas

## 📈 Rendimiento

- **Tiempo típico**: 2-5 segundos por diagnóstico
- **Chunking**: 512 caracteres con overlap de 50
- **Top-K**: 5 contextos más relevantes por defecto
- **Confianza**: 80-90% con LLM, 60% con fallback

## 🧪 Testing

```python
# Verificar salud del sistema
health = await system.test_system_health()

# Obtener estadísticas
stats = system.get_system_stats()
print("Documentos procesados:", stats["system"]["documents_processed"])
print("Diagnósticos procesados:", stats["system"]["diagnoses_processed"])
```

## 📝 Casos de Uso

1. **Análisis de informes clínicos**
2. **Codificación automática CIE-10**
3. **Expansión de acrónimos médicos**
4. **Extracción de entidades clínicas**
5. **Identificación de factores de riesgo**
6. **Búsqueda semántica en documentos médicos**

## 🔒 Consideraciones de Seguridad

- Los datos médicos se procesan localmente
- No se envían datos a servicios externos (excepto LLM local)
- Validación estricta de entradas
- Logs configurables para auditoría

## 🤝 Contribuciones

1. Fork del repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

## 🆘 Soporte

Para reportar bugs o solicitar características, crear un issue en el repositorio de GitHub.

---

**Sistema RAG Médico v1.0.0** - Procesamiento inteligente de documentos médicos con arquitectura de capas empresarial.
