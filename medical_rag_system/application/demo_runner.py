"""
Demo Runner - Ejecutor de demostraciones del sistema
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime

from ..models.patient import PatientData
from ..models.document import DocumentType
from .medical_system import MedicalRAGSystem


class DemoRunner:
    """Ejecutor de demostraciones del sistema médico"""
    
    def __init__(self, medical_system: MedicalRAGSystem):
        self.system = medical_system
        
        # Casos de prueba predefinidos
        self.test_cases = [
            {
                "description": "Caso diabético con hipertensión",
                "diagnosis": "Paciente de 65 años con DM tipo 2 descompensada y HTA secundaria, presenta poliuria y polidipsia",
                "patient": {"age": 65, "sex": "M", "service": "medicina interna"}
            },
            {
                "description": "Caso cardiológico complejo",
                "diagnosis": "IAM anterior extenso con FA de nueva aparición, IC funcional clase III, requiere cateterismo urgente",
                "patient": {"age": 58, "sex": "M", "service": "cardiología"}
            },
            {
                "description": "Caso respiratorio",
                "diagnosis": "EPOC reagudizado con disnea de grandes esfuerzos, sibilancias difusas y expectoración purulenta",
                "patient": {"age": 70, "sex": "M", "service": "neumología"}
            },
            {
                "description": "Caso neurológico",
                "diagnosis": "ACV isquémico en territorio de ACM izquierda con hemiparesia derecha y afasia de Broca",
                "patient": {"age": 72, "sex": "F", "service": "neurología"}
            },
            {
                "description": "Caso ginecológico",
                "diagnosis": "Paciente de 35 años con dolor pélvico crónico, dismenorrea severa, sospecha de endometriosis",
                "patient": {"age": 35, "sex": "F", "service": "ginecología"}
            }
        ]
        
        # Documentos médicos de ejemplo
        self.sample_documents = [
            {
                "title": "Protocolo Diabetes Mellitus",
                "type": DocumentType.CLINICAL_NOTE,
                "text": """
                PROTOCOLO DE MANEJO DE DIABETES MELLITUS TIPO 2
                
                DIAGNÓSTICO:
                - Glucemia en ayunas ≥126 mg/dL en dos ocasiones
                - HbA1c ≥6.5%
                - Glucemia al azar ≥200 mg/dL con síntomas
                
                TRATAMIENTO INICIAL:
                1. Metformina 500mg cada 12 horas
                2. Dieta hipocalórica 1500-1800 kcal/día
                3. Ejercicio aeróbico 150 min/semana
                
                SEGUIMIENTO:
                - Control glucémico cada 3 meses
                - HbA1c semestral
                - Fondo de ojo anual
                - Función renal cada 6 meses
                
                COMPLICACIONES:
                - Nefropatía diabética
                - Retinopatía diabética
                - Neuropatía periférica
                - Enfermedad cardiovascular
                """,
                "service": "endocrinología"
            },
            {
                "title": "Guía Manejo IAM",
                "type": DocumentType.CLINICAL_NOTE,
                "text": """
                GUÍA DE MANEJO INFARTO AGUDO DE MIOCARDIO
                
                DIAGNÓSTICO:
                - Dolor torácico típico >30 minutos
                - Elevación troponinas
                - Cambios electrocardiográficos
                - Elevación ST ≥1mm en 2 derivaciones contiguas
                
                TRATAMIENTO INMEDIATO:
                1. Aspirina 300mg masticable
                2. Clopidogrel 600mg dosis carga
                3. Atorvastatina 80mg
                4. Metoprolol 25mg c/12h
                
                REPERFUSIÓN:
                - Angioplastia primaria <90 minutos
                - Trombolisis si no disponible angioplastia
                
                COMPLICACIONES:
                - Arritmias ventriculares
                - Insuficiencia cardíaca
                - Shock cardiogénico
                - Ruptura cardíaca
                """,
                "service": "cardiología"
            }
        ]
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Ejecutar demostración completa del sistema"""
        
        print("🚀 DEMOSTRACIÓN COMPLETA DEL SISTEMA RAG MÉDICO")
        print("=" * 60)
        
        demo_results = {
            "start_time": datetime.now().isoformat(),
            "system_health": {},
            "document_ingestion": {},
            "diagnosis_processing": {},
            "search_demo": {},
            "performance_metrics": {}
        }
        
        # 1. Verificar salud del sistema
        print("\n🔍 1. VERIFICANDO SALUD DEL SISTEMA")
        print("-" * 40)
        
        health_status = await self.system.test_system_health()
        demo_results["system_health"] = health_status
        
        if health_status["system_healthy"]:
            print("✅ Sistema saludable")
        else:
            print("⚠️ Sistema con problemas")
        
        # 2. Ingesta de documentos
        print("\n📚 2. INGESTA DE DOCUMENTOS MÉDICOS")
        print("-" * 40)
        
        ingestion_results = await self._demo_document_ingestion()
        demo_results["document_ingestion"] = ingestion_results
        
        # 3. Procesamiento de diagnósticos
        print("\n🏥 3. PROCESAMIENTO DE DIAGNÓSTICOS")
        print("-" * 40)
        
        processing_results = await self._demo_diagnosis_processing()
        demo_results["diagnosis_processing"] = processing_results
        
        # 4. Demo de búsqueda
        print("\n🔍 4. DEMOSTRACIÓN DE BÚSQUEDA")
        print("-" * 40)
        
        search_results = await self._demo_search_functionality()
        demo_results["search_demo"] = search_results
        
        # 5. Métricas de rendimiento
        print("\n📊 5. MÉTRICAS DE RENDIMIENTO")
        print("-" * 40)
        
        performance_metrics = self._collect_performance_metrics()
        demo_results["performance_metrics"] = performance_metrics
        
        demo_results["end_time"] = datetime.now().isoformat()
        
        # Resumen final
        self._print_demo_summary(demo_results)
        
        return demo_results
    
    async def _demo_document_ingestion(self) -> Dict[str, Any]:
        """Demostrar ingesta de documentos"""
        
        results = {
            "documents_added": [],
            "errors": [],
            "total_added": 0
        }
        
        for i, doc in enumerate(self.sample_documents, 1):
            try:
                print(f"📄 Agregando documento {i}: {doc['title']}")
                
                doc_id = self.system.add_medical_document(
                    text=doc["text"],
                    document_type=doc["type"],
                    title=doc["title"],
                    service=doc["service"]
                )
                
                if doc_id:
                    results["documents_added"].append({
                        "id": doc_id,
                        "title": doc["title"],
                        "type": doc["type"].value
                    })
                    print(f"✅ Documento agregado: {doc_id}")
                else:
                    results["errors"].append(f"Error agregando {doc['title']}")
                    
            except Exception as e:
                error_msg = f"Error en {doc['title']}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        results["total_added"] = len(results["documents_added"])
        print(f"\n📊 Documentos agregados: {results['total_added']}")
        
        return results
    
    async def _demo_diagnosis_processing(self) -> Dict[str, Any]:
        """Demostrar procesamiento de diagnósticos"""
        
        results = {
            "processed_cases": [],
            "errors": [],
            "summary_stats": {}
        }
        
        processing_times = []
        success_count = 0
        llm_used_count = 0
        
        for i, case in enumerate(self.test_cases, 1):
            try:
                print(f"\n🔬 CASO {i}: {case['description']}")
                print(f"📝 Diagnóstico: {case['diagnosis'][:80]}...")
                print(f"👤 Paciente: {case['patient']['age']} años, {case['patient']['sex']}, {case['patient']['service']}")
                print("-" * 50)
                
                start_time = datetime.now()
                
                # Procesar diagnóstico
                result = await self.system.process_diagnosis(
                    text=case["diagnosis"],
                    patient_age=case["patient"]["age"],
                    patient_sex=case["patient"]["sex"],
                    service=case["patient"]["service"]
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                processing_times.append(processing_time)
                
                if result.success:
                    success_count += 1
                    if result.processing_method == "llm_processing":
                        llm_used_count += 1
                
                # Mostrar resultados
                print(f"\n📊 RESULTADOS:")
                print(f"   ⏱️ Tiempo: {processing_time:.2f}s")
                print(f"   🔧 Método: {result.processing_method}")
                print(f"   ✅ Éxito: {result.success}")
                print(f"   📋 Diagnóstico: {result.main_diagnosis[:60]}...")
                print(f"   🏷️ CIE-10: {', '.join(result.cie10_codes) if result.cie10_codes else 'N/A'}")
                print(f"   🔤 Acrónimos: {len(result.expanded_acronyms)}")
                print(f"   🩺 Síntomas: {', '.join(result.symptoms[:3])}{'...' if len(result.symptoms) > 3 else ''}")
                print(f"   ⚠️ Factores riesgo: {len(result.risk_factors)}")
                print(f"   🎯 Confianza: {result.get_overall_confidence():.1%}")
                
                case_result = {
                    "case_number": i,
                    "description": case["description"],
                    "processing_time": processing_time,
                    "method": result.processing_method,
                    "success": result.success,
                    "confidence": result.get_overall_confidence(),
                    "cie10_count": len(result.cie10_codes),
                    "symptoms_count": len(result.symptoms),
                    "entities_count": len(result.clinical_entities)
                }
                
                results["processed_cases"].append(case_result)
                
            except Exception as e:
                error_msg = f"Error en caso {i}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Estadísticas resumen
        if processing_times:
            results["summary_stats"] = {
                "total_cases": len(self.test_cases),
                "successful_cases": success_count,
                "llm_used": llm_used_count,
                "fallback_used": success_count - llm_used_count,
                "average_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "success_rate": success_count / len(self.test_cases)
            }
        
        return results
    
    async def _demo_search_functionality(self) -> Dict[str, Any]:
        """Demostrar funcionalidad de búsqueda"""
        
        if not self.system.document_manager:
            return {"error": "Document manager no disponible"}
        
        search_queries = [
            "diabetes tratamiento",
            "infarto miocardio",
            "dolor torácico",
            "complicaciones cardiovasculares"
        ]
        
        results = {
            "search_results": [],
            "total_searches": len(search_queries)
        }
        
        for query in search_queries:
            try:
                print(f"🔍 Buscando: '{query}'")
                
                search_results = self.system.search_medical_documents(
                    query=query,
                    top_k=3
                )
                
                print(f"✅ Encontrados {len(search_results)} resultados")
                
                for i, result in enumerate(search_results, 1):
                    print(f"   {i}. Similitud: {result['similarity_score']:.2%}")
                    print(f"      Texto: {result['text'][:100]}...")
                
                results["search_results"].append({
                    "query": query,
                    "results_count": len(search_results),
                    "avg_similarity": sum(r["similarity_score"] for r in search_results) / len(search_results) if search_results else 0
                })
                
            except Exception as e:
                print(f"❌ Error buscando '{query}': {e}")
        
        return results
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de rendimiento"""
        
        # Obtener estadísticas del sistema
        system_stats = self.system.get_system_stats()
        
        # Métricas específicas
        metrics = {
            "system_stats": system_stats,
            "component_status": {
                "llm_available": self.system.llm_client.is_available(),
                "vector_store_available": self.system.vector_store is not None,
                "document_manager_available": self.system.document_manager is not None
            }
        }
        
        return metrics
    
    def _print_demo_summary(self, results: Dict[str, Any]):
        """Imprimir resumen de la demostración"""
        
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE LA DEMOSTRACIÓN")
        print("=" * 60)
        
        # Salud del sistema
        health = results["system_health"]
        print(f"🏥 Sistema: {'✅ Saludable' if health.get('system_healthy') else '⚠️ Con problemas'}")
        
        # Documentos
        docs = results["document_ingestion"]
        print(f"📚 Documentos agregados: {docs.get('total_added', 0)}")
        
        # Procesamiento
        processing = results["diagnosis_processing"]
        if "summary_stats" in processing:
            stats = processing["summary_stats"]
            print(f"🔬 Casos procesados: {stats.get('successful_cases', 0)}/{stats.get('total_cases', 0)}")
            print(f"🧠 Uso LLM: {stats.get('llm_used', 0)} casos")
            print(f"🛠️ Uso fallback: {stats.get('fallback_used', 0)} casos")
            print(f"⏱️ Tiempo promedio: {stats.get('average_processing_time', 0):.2f}s")
            print(f"🎯 Tasa de éxito: {stats.get('success_rate', 0):.1%}")
        
        # Búsqueda
        search = results["search_demo"]
        if "search_results" in search:
            print(f"🔍 Búsquedas realizadas: {search.get('total_searches', 0)}")
        
        print("\n🎉 ¡Demostración completada exitosamente!")
    
    async def run_single_case_demo(self, case_index: int = 0) -> Dict[str, Any]:
        """Ejecutar demo de un solo caso"""
        
        if case_index >= len(self.test_cases):
            raise ValueError(f"Índice de caso inválido: {case_index}")
        
        case = self.test_cases[case_index]
        
        print(f"🔬 DEMO CASO INDIVIDUAL: {case['description']}")
        print("=" * 50)
        
        result = await self.system.process_diagnosis(
            text=case["diagnosis"],
            patient_age=case["patient"]["age"],
            patient_sex=case["patient"]["sex"],
            service=case["patient"]["service"]
        )
        
        # Mostrar resultado detallado
        print(f"\n📋 RESULTADO DETALLADO:")
        print(f"   Diagnóstico principal: {result.main_diagnosis}")
        print(f"   Diagnósticos secundarios: {result.secondary_diagnoses}")
        print(f"   Códigos CIE-10: {result.cie10_codes}")
        print(f"   Acrónimos expandidos: {result.expanded_acronyms}")
        print(f"   Entidades clínicas: {len(result.clinical_entities)}")
        print(f"   Síntomas: {result.symptoms}")
        print(f"   Factores de riesgo: {result.risk_factors}")
        print(f"   Método de procesamiento: {result.processing_method}")
        print(f"   Confianza: {result.get_overall_confidence():.1%}")
        
        return {
            "case": case,
            "result": result.dict(),
            "summary": result.to_summary_dict()
        }
