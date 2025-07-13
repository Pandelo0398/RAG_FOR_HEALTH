# Medical RAG System - Setup and Installation

# Archivo para configuración automática del entorno
import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    print("📦 Instalando dependencias...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False


def setup_directories():
    """Crear directorios necesarios"""
    print("📁 Creando directorios...")
    
    directories = [
        "medical_chroma_db",
        "embedding_cache",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio creado: {directory}")


def create_env_file():
    """Crear archivo .env de ejemplo"""
    print("⚙️ Creando archivo .env de ejemplo...")
    
    env_content = """# Medical RAG System Configuration

# LLM Configuration
LLM_API_BASE=http://localhost:8000
LLM_MODEL=Qwen/Qwen3-14B

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./medical_chroma_db

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Application Configuration
APP_DEBUG=false
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env.example")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("✅ Archivo .env.example creado")


def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requerido")
        return False
    
    print(f"✅ Python {sys.version} detectado")
    return True


def main():
    """Configuración principal"""
    print("🏥 Configurando Sistema RAG Médico")
    print("=" * 40)
    
    # Verificar Python
    if not check_python_version():
        return
    
    # Crear directorios
    setup_directories()
    
    # Crear archivo de configuración
    create_env_file()
    
    # Instalar dependencias
    if install_requirements():
        print("\n🎉 ¡Configuración completada exitosamente!")
        print("\n📝 Próximos pasos:")
        print("1. Ejecutar: python example.py")
        print("2. O importar: from medical_rag_system.application import MedicalRAGSystem")
        print("\n💡 Opcional: Configurar servidor Qwen para funcionalidad completa")
    else:
        print("\n❌ Error en la configuración")


if __name__ == "__main__":
    main()
