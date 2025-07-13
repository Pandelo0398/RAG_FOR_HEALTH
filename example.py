"""
Script de ejemplo para usar el Sistema RAG Médico
"""

import asyncio
from medical_rag_system.application import MedicalRAGSystem, DemoRunner
from medical_rag_system.models.document import DocumentType


async def main():
    """Función principal de ejemplo"""
    
    print("🏥 Iniciando Sistema RAG Médico de Ejemplo")
    print("=" * 50)
    
    # 1. Inicializar sistema
    print("⚙️ Inicializando sistema...")
    system = MedicalRAGSystem(use_qwen=True)
    
    # 2. Verificar salud del sistema
    print("\n🔍 Verificando salud del sistema...")
    health = await system.test_system_health()
    print(f"Sistema saludable: {health['system_healthy']}")
    
    # 3. Ejemplo de procesamiento de diagnóstico
    print("\n🔬 Ejemplo de procesamiento de diagnóstico...")
    
    diagnosis_text = """
    Paciente masculino de 68 años con antecedente de DM tipo 2 e HTA, 
    acude por dolor torácico opresivo de 2 horas de evolución, 
    acompañado de disnea y diaforesis. ECG muestra elevación del ST 
    en derivaciones V2-V6. Troponinas elevadas.
    """
    
    result = await system.process_diagnosis(
        text=diagnosis_text,
        patient_age=68,
        patient_sex="M",
        service="cardiología"
    )
    
    print(f"✅ Diagnóstico procesado:")
    print(f"   Método: {result.processing_method}")
    print(f"   Diagnóstico: {result.main_diagnosis}")
    print(f"   CIE-10: {result.cie10_codes}")
    print(f"   Síntomas: {result.symptoms}")
    print(f"   Factores de riesgo: {result.risk_factors}")
    print(f"   Confianza: {result.get_overall_confidence():.1%}")
    
    # 4. Ejemplo de agregar documento
    print("\n📄 Agregando documento de ejemplo...")
    
    sample_doc = """
    PROTOCOLO DE MANEJO DE INFARTO AGUDO DE MIOCARDIO
    
    CRITERIOS DIAGNÓSTICOS:
    - Dolor torácico típico >30 minutos
    - Elevación de biomarcadores cardíacos (troponinas)
    - Cambios electrocardiográficos compatibles
    
    TRATAMIENTO INMEDIATO:
    1. Doble antiagregación (ASA + P2Y12)
    2. Anticoagulación (heparina)
    3. Estatinas de alta intensidad
    4. Betabloqueadores si no contraindicados
    
    REPERFUSIÓN:
    - Angioplastia primaria <90 minutos (gold standard)
    - Fibrinólisis si angioplastia no disponible <120 minutos
    """
    
    doc_id = system.add_medical_document(
        text=sample_doc,
        document_type=DocumentType.CLINICAL_NOTE,
        title="Protocolo IAM",
        service="cardiología"
    )
    
    if doc_id:
        print(f"✅ Documento agregado: {doc_id}")
    
    # 5. Ejemplo de búsqueda
    print("\n🔍 Ejemplo de búsqueda de documentos...")
    
    search_results = system.search_medical_documents(
        query="infarto miocardio tratamiento",
        top_k=3
    )
    
    print(f"✅ Encontrados {len(search_results)} documentos:")
    for i, result in enumerate(search_results, 1):
        print(f"   {i}. Similitud: {result['similarity_score']:.1%}")
        print(f"      Texto: {result['text'][:100]}...")
    
    # 6. Estadísticas del sistema
    print("\n📊 Estadísticas del sistema:")
    stats = system.get_system_stats()
    print(f"   Documentos procesados: {stats['system']['documents_processed']}")
    print(f"   Diagnósticos procesados: {stats['system']['diagnoses_processed']}")
    
    if 'processing' in stats:
        proc_stats = stats['processing']
        print(f"   Tasa de éxito: {proc_stats.get('success_rate', 0):.1%}")
        print(f"   LLM disponible: {proc_stats.get('llm_available', False)}")
    
    print("\n🎉 ¡Ejemplo completado exitosamente!")


async def run_demo():
    """Ejecutar demo completo del sistema"""
    
    print("🚀 Ejecutando Demo Completo del Sistema")
    print("=" * 50)
    
    # Inicializar sistema y demo
    system = MedicalRAGSystem(use_qwen=True)
    demo = DemoRunner(system)
    
    # Ejecutar demo completo
    results = await demo.run_complete_demo()
    
    print("\n📊 RESUMEN DEL DEMO:")
    print(f"✅ Demo completado exitosamente")
    print(f"⏱️ Tiempo total: Se completó en {results.get('end_time', 'N/A')}")


if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Ejecutar ejemplo básico")
    print("2. Ejecutar demo completo")
    
    try:
        choice = input("Opción (1 o 2): ").strip()
        
        if choice == "1":
            asyncio.run(main())
        elif choice == "2":
            asyncio.run(run_demo())
        else:
            print("Opción inválida. Ejecutando ejemplo básico...")
            asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener todas las dependencias instaladas")
