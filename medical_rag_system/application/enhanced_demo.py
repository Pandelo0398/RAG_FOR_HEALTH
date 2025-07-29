"""
Demostración Mejorada del Sistema RAG Médico con LlamaIndex
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from .medical_system_enhanced import EnhancedMedicalRAGSystem
from ..models.document import DocumentType


class EnhancedMedicalDemo:
    """Demostración completa del sistema RAG médico mejorado"""
    
    def __init__(self):
        """Inicializar demostración"""
        
        self.system = None
        self.demo_data = self._load_demo_data()
    
    def _load_demo_data(self) -> Dict[str, Any]:
        """Cargar datos de demostración"""
        
        return {
            "medical_documents": [
                {
                    "title": "Protocolo de Tratamiento - Diabetes Mellitus",
                    "text": """
                    DIABETES MELLITUS TIPO 2 - PROTOCOLO DE TRATAMIENTO
                    
                    CRITERIOS DIAGNÓSTICOS:
                    - Glucemia en ayunas ≥ 126 mg/dL
                    - Glucemia postprandial ≥ 200 mg/dL
                    - Hemoglobina glicosilada ≥ 6.5%
                    
                    TRATAMIENTO FARMACOLÓGICO:
                    1. Metformina: 500-2000 mg/día (primera línea)
                    2. Sulfonilureas: Glibenclamida, Gliclazida
                    3. Inhibidores DPP-4: Sitagliptina, Vildagliptina
                    4. Agonistas GLP-1: Liraglutida, Dulaglutida
                    5. Inhibidores SGLT2: Empagliflozina, Dapagliflozina
                    
                    SEGUIMIENTO:
                    - Control glucémico cada 3 meses
                    - Hemoglobina glicosilada objetivo < 7%
                    - Evaluación de complicaciones anual
                    
                    COMPLICACIONES:
                    - Retinopatía diabética
                    - Nefropatía diabética
                    - Neuropatía diabética
                    - Enfermedad cardiovascular
                    """,
                    "type": DocumentType.TREATMENT,
                    "service": "Endocrinología"
                },
                {
                    "title": "Guía Clínica - Hipertensión Arterial",
                    "text": """
                    HIPERTENSIÓN ARTERIAL - GUÍA CLÍNICA
                    
                    DEFINICIÓN:
                    Presión arterial sistólica ≥ 140 mmHg y/o diastólica ≥ 90 mmHg
                    
                    CLASIFICACIÓN:
                    - Normal: < 120/80 mmHg
                    - Prehipertensión: 120-139/80-89 mmHg
                    - Hipertensión Estadio 1: 140-159/90-99 mmHg
                    - Hipertensión Estadio 2: ≥ 160/100 mmHg
                    
                    TRATAMIENTO:
                    1. Modificaciones del estilo de vida
                    2. Inhibidores de la ECA: Enalapril, Lisinopril
                    3. Antagonistas de receptores de angiotensina: Losartán, Valsartán
                    4. Bloqueadores de canales de calcio: Amlodipino, Nifedipino
                    5. Diuréticos tiazídicos: Hidroclorotiazida
                    
                    OBJETIVOS DE TRATAMIENTO:
                    - < 140/90 mmHg en población general
                    - < 130/80 mmHg en diabéticos y enfermedad renal
                    """,
                    "type": DocumentType.TREATMENT,
                    "service": "Cardiología"
                },
                {
                    "title": "Caso Clínico - Infarto Agudo de Miocardio",
                    "text": """
                    CASO CLÍNICO: INFARTO AGUDO DE MIOCARDIO
                    
                    PACIENTE: Varón de 65 años
                    ANTECEDENTES: Hipertensión arterial, diabetes mellitus tipo 2
                    
                    CLÍNICA:
                    - Dolor torácico opresivo de 2 horas de evolución
                    - Irradiación a brazo izquierdo y mandíbula
                    - Sudoración profusa y náuseas
                    
                    EXPLORACIÓN:
                    - TA: 180/110 mmHg
                    - FC: 95 lpm, rítmico
                    - Auscultación: S4, sin soplos
                    
                    ECG:
                    - Elevación del segmento ST en cara anterior
                    - Ondas Q patológicas en V1-V4
                    
                    DIAGNÓSTICO:
                    Infarto agudo de miocardio con elevación del segmento ST (IAMCEST)
                    de cara anterior
                    
                    TRATAMIENTO INMEDIATO:
                    1. Aspirina 300 mg
                    2. Clopidogrel 600 mg
                    3. Heparina no fraccionada
                    4. Angioplastia primaria
                    
                    EVOLUCIÓN:
                    Reperfusión exitosa con stent en arteria descendente anterior
                    """,
                    "type": DocumentType.DIAGNOSIS,
                    "service": "Cardiología"
                }
            ],
            "test_queries": [
                "¿Cuál es el tratamiento de primera línea para diabetes mellitus tipo 2?",
                "¿Cuáles son los criterios diagnósticos de hipertensión arterial?",
                "¿Qué complicaciones puede tener un paciente diabético?",
                "¿Cuál es el manejo inicial del infarto agudo de miocardio?",
                "¿Qué medicamentos se usan para tratar la hipertensión arterial?"
            ],
            "patient_scenarios": [
                {
                    "age": 58,
                    "sex": "Femenino",
                    "diagnosis": "Paciente con diabetes mellitus tipo 2 de 5 años de evolución, mal controlada con hemoglobina glicosilada de 8.5%",
                    "query": "¿Qué tratamiento farmacológico recomiendas para mejorar el control glucémico?"
                },
                {
                    "age": 72,
                    "sex": "Masculino",
                    "diagnosis": "Paciente hipertenso con presión arterial de 160/95 mmHg y antecedentes de diabetes",
                    "query": "¿Cuál es el objetivo de presión arterial y qué medicamentos son más apropiados?"
                }
            ]
        }
    
    async def run_enhanced_demo(self):
        """Ejecutar demostración completa del sistema mejorado"""
        
        print("🚀 INICIANDO DEMOSTRACIÓN MEJORADA DEL SISTEMA RAG MÉDICO")
        print("=" * 80)
        
        # 1. Inicializar sistema mejorado
        await self._initialize_enhanced_system()
        
        # 2. Cargar documentos médicos
        await self._load_medical_documents()
        
        # 3. Demostrar capacidades básicas
        await self._demonstrate_basic_capabilities()
        
        # 4. Demostrar agentes especializados
        await self._demonstrate_agents()
        
        # 5. Demostrar evaluación de calidad
        await self._demonstrate_evaluation()
        
        # 6. Demostrar casos clínicos complejos
        await self._demonstrate_clinical_cases()
        
        # 7. Mostrar estadísticas finales
        await self._show_final_statistics()
        
        print("\n✅ DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    
    async def _initialize_enhanced_system(self):
        """Inicializar sistema mejorado"""
        
        print("\n🔧 1. INICIALIZANDO SISTEMA MEJORADO")
        print("-" * 50)
        
        try:
            self.system = EnhancedMedicalRAGSystem(
                use_llama_index=True,
                enable_agents=True,
                enable_evaluation=True
            )
            
            # Verificar características disponibles
            features = self.system.get_available_features()
            print("✅ Características disponibles:")
            for feature, available in features.items():
                status = "✅" if available else "❌"
                print(f"   {status} {feature}")
            
            # Test de salud del sistema
            health = await self.system.test_system_health_enhanced()
            print(f"\n🏥 Estado del sistema: {'✅ SALUDABLE' if health['system_healthy'] else '❌ PROBLEMAS'}")
            
        except Exception as e:
            print(f"❌ Error inicializando sistema: {e}")
            raise
    
    async def _load_medical_documents(self):
        """Cargar documentos médicos de demostración"""
        
        print("\n📚 2. CARGANDO DOCUMENTOS MÉDICOS")
        print("-" * 50)
        
        for i, doc in enumerate(self.demo_data["medical_documents"], 1):
            try:
                doc_id = await self.system.add_medical_document_enhanced(
                    text=doc["text"],
                    document_type=doc["type"],
                    title=doc["title"],
                    service=doc["service"]
                )
                
                print(f"✅ Documento {i} cargado: {doc['title']}")
                print(f"   ID: {doc_id}")
                print(f"   Tipo: {doc['type'].value}")
                print(f"   Servicio: {doc['service']}")
                
            except Exception as e:
                print(f"❌ Error cargando documento {i}: {e}")
    
    async def _demonstrate_basic_capabilities(self):
        """Demostrar capacidades básicas del sistema"""
        
        print("\n🔍 3. DEMOSTRANDO CAPACIDADES BÁSICAS")
        print("-" * 50)
        
        # Búsqueda de documentos
        print("\n📖 Búsqueda de documentos médicos:")
        search_results = self.system.search_medical_documents_enhanced(
            query="tratamiento diabetes",
            top_k=3
        )
        
        print(f"Encontrados {len(search_results)} documentos relevantes:")
        for i, result in enumerate(search_results[:2], 1):
            print(f"   {i}. Similitud: {result['similarity_score']:.3f}")
            print(f"      Texto: {result['text'][:100]}...")
            print(f"      Tipo: {result['metadata'].get('document_type', 'N/A')}")
        
        # Procesamiento de diagnóstico básico
        print("\n🏥 Procesamiento de diagnóstico básico:")
        diagnosis_result = await self.system.process_diagnosis_enhanced(
            text="Paciente con diabetes mellitus tipo 2 mal controlada",
            patient_age=65,
            patient_sex="Masculino",
            service="Endocrinología",
            use_agents=False,
            evaluate_response=True
        )
        
        print(f"✅ Diagnóstico procesado en {diagnosis_result.processing_time:.2f}s")
        print(f"   Confianza: {diagnosis_result.confidence_score:.3f}")
        print(f"   Análisis: {diagnosis_result.diagnosis_analysis[:200]}...")
    
    async def _demonstrate_agents(self):
        """Demostrar sistema de agentes especializados"""
        
        print("\n🤖 4. DEMOSTRANDO AGENTES ESPECIALIZADOS")
        print("-" * 50)
        
        # Agente de diagnóstico
        print("\n🔬 Agente de Diagnóstico:")
        diagnosis_agent_result = await self.system.query_with_agent(
            query="Analiza este caso: paciente de 70 años con dolor torácico y presión alta",
            agent_type="diagnosis",
            patient_context={"age": 70, "symptoms": ["dolor torácico", "presión alta"]}
        )
        
        if "error" not in diagnosis_agent_result:
            print("✅ Respuesta del agente de diagnóstico:")
            print(f"   {diagnosis_agent_result['response'][:300]}...")
        else:
            print(f"❌ Error con agente de diagnóstico: {diagnosis_agent_result['error']}")
        
        # Agente de tratamiento
        print("\n💊 Agente de Tratamiento:")
        treatment_agent_result = await self.system.query_with_agent(
            query="Sugiere tratamiento para diabetes mellitus tipo 2",
            agent_type="treatment",
            patient_context={"diagnosis": "diabetes mellitus tipo 2"}
        )
        
        if "error" not in treatment_agent_result:
            print("✅ Respuesta del agente de tratamiento:")
            print(f"   {treatment_agent_result['response'][:300]}...")
        else:
            print(f"❌ Error con agente de tratamiento: {treatment_agent_result['error']}")
    
    async def _demonstrate_evaluation(self):
        """Demostrar sistema de evaluación"""
        
        print("\n📊 5. DEMOSTRANDO SISTEMA DE EVALUACIÓN")
        print("-" * 50)
        
        # Evaluar respuestas de prueba
        test_queries = self.demo_data["test_queries"][:3]
        
        print(f"\n🔍 Evaluando {len(test_queries)} consultas de prueba:")
        
        evaluation_result = await self.system.evaluate_system_performance(test_queries)
        
        if "error" not in evaluation_result:
            stats = evaluation_result.get("system_evaluation", {}).get("batch_statistics", {})
            print(f"✅ Resultados de evaluación:")
            print(f"   Score promedio: {stats.get('average_score', 0):.3f}")
            print(f"   Score mínimo: {stats.get('min_score', 0):.3f}")
            print(f"   Score máximo: {stats.get('max_score', 0):.3f}")
            
            distribution = stats.get('quality_distribution', {})
            print(f"   Distribución de calidad:")
            for quality, count in distribution.items():
                if count > 0:
                    print(f"     {quality}: {count}")
        else:
            print(f"❌ Error en evaluación: {evaluation_result['error']}")
    
    async def _demonstrate_clinical_cases(self):
        """Demostrar casos clínicos complejos"""
        
        print("\n👨‍⚕️ 6. DEMOSTRANDO CASOS CLÍNICOS COMPLEJOS")
        print("-" * 50)
        
        for i, scenario in enumerate(self.demo_data["patient_scenarios"], 1):
            print(f"\n📋 Caso Clínico {i}:")
            print(f"   Paciente: {scenario['age']} años, {scenario['sex']}")
            print(f"   Diagnóstico: {scenario['diagnosis']}")
            print(f"   Consulta: {scenario['query']}")
            
            # Procesar con agentes
            result = await self.system.process_diagnosis_enhanced(
                text=scenario['diagnosis'],
                patient_age=scenario['age'],
                patient_sex=scenario['sex'],
                use_agents=True,
                evaluate_response=True
            )
            
            print(f"✅ Respuesta del sistema:")
            print(f"   Tiempo: {result.processing_time:.2f}s")
            print(f"   Confianza: {result.confidence_score:.3f}")
            print(f"   Análisis: {result.diagnosis_analysis[:200]}...")
            
            if result.treatment_suggestions:
                print(f"   Tratamiento: {result.treatment_suggestions[:200]}...")
    
    async def _show_final_statistics(self):
        """Mostrar estadísticas finales"""
        
        print("\n📈 7. ESTADÍSTICAS FINALES DEL SISTEMA")
        print("-" * 50)
        
        stats = self.system.get_system_stats_enhanced()
        
        print("📊 Estadísticas del sistema:")
        system_stats = stats.get("system", {})
        print(f"   Documentos procesados: {system_stats.get('documents_processed', 0)}")
        print(f"   Diagnósticos procesados: {system_stats.get('diagnoses_processed', 0)}")
        print(f"   Consultas con agentes: {system_stats.get('agent_queries', 0)}")
        print(f"   Evaluaciones realizadas: {system_stats.get('evaluations_performed', 0)}")
        
        # Estadísticas de LlamaIndex
        if "llama_index" in stats:
            llama_stats = stats["llama_index"]
            print(f"\n🔧 Estadísticas de LlamaIndex:")
            index_stats = llama_stats.get("index", {})
            print(f"   Nodos totales: {index_stats.get('total_nodes', 0)}")
            print(f"   Modelo de embeddings: {index_stats.get('embedding_model', 'N/A')}")
            print(f"   Vector store: {index_stats.get('vector_store_type', 'N/A')}")
        
        # Estadísticas de agentes
        if "agents" in stats:
            agent_stats = stats["agents"]
            print(f"\n🤖 Estadísticas de Agentes:")
            print(f"   Agentes disponibles: {agent_stats.get('total_agents', 0)}")
            print(f"   Herramientas disponibles: {agent_stats.get('total_tools', 0)}")
        
        # Estadísticas de evaluación
        if "evaluation" in stats:
            eval_stats = stats["evaluation"]
            print(f"\n📊 Estadísticas de Evaluación:")
            print(f"   Evaluaciones totales: {eval_stats.get('total_evaluations', 0)}")
            print(f"   Score promedio: {eval_stats.get('overall_average_score', 0):.3f}")
    
    async def run_quick_test(self):
        """Ejecutar prueba rápida del sistema"""
        
        print("⚡ PRUEBA RÁPIDA DEL SISTEMA RAG MÉDICO")
        print("=" * 50)
        
        try:
            # Inicializar sistema
            self.system = EnhancedMedicalRAGSystem(
                use_llama_index=True,
                enable_agents=False,  # Deshabilitar para prueba rápida
                enable_evaluation=False
            )
            
            # Cargar un documento
            doc = self.demo_data["medical_documents"][0]
            await self.system.add_medical_document_enhanced(
                text=doc["text"],
                document_type=doc["type"],
                title=doc["title"]
            )
            
            # Procesar una consulta
            result = await self.system.process_diagnosis_enhanced(
                text="¿Cuál es el tratamiento de la diabetes?",
                use_agents=False
            )
            
            print(f"✅ Prueba exitosa:")
            print(f"   Respuesta: {result.diagnosis_analysis[:100]}...")
            print(f"   Confianza: {result.confidence_score:.3f}")
            
        except Exception as e:
            print(f"❌ Error en prueba rápida: {e}")


async def main():
    """Función principal para ejecutar la demostración"""
    
    demo = EnhancedMedicalDemo()
    
    # Ejecutar demostración completa o prueba rápida
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        await demo.run_quick_test()
    else:
        await demo.run_enhanced_demo()


if __name__ == "__main__":
    asyncio.run(main()) 