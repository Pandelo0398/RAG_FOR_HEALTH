"""
Servicio de Evaluación y Monitoreo del Sistema RAG Médico
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import json
from pathlib import Path

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator
)
from llama_index.core.evaluation.eval_utils import get_responses
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.schema import Document

from ..infrastructure.llama_index_integration import LlamaIndexMedicalRAG
from ..models.medical import MedicalProcessingResult

logger = logging.getLogger(__name__)


class MedicalEvaluationService:
    """Servicio de evaluación para el sistema RAG médico"""
    
    def __init__(self, llama_rag: LlamaIndexMedicalRAG):
        """
        Inicializar servicio de evaluación
        
        Args:
            llama_rag: Instancia del sistema RAG con LlamaIndex
        """
        
        self.llama_rag = llama_rag
        self.evaluators = {}
        self.evaluation_history = []
        
        self._setup_evaluators()
    
    def _setup_evaluators(self):
        """Configurar evaluadores especializados para medicina"""
        
        # Evaluador de corrección médica
        self.evaluators["correctness"] = CorrectnessEvaluator(
            llm=self.llama_rag.llm,
            criteria={
                "medical_accuracy": "La respuesta debe ser médicamente precisa y basada en evidencia científica",
                "clinical_relevance": "La respuesta debe ser relevante para el contexto clínico",
                "safety": "La respuesta debe priorizar la seguridad del paciente",
                "completeness": "La respuesta debe ser completa y abordar todos los aspectos de la consulta"
            }
        )
        
        # Evaluador de relevancia
        self.evaluators["relevancy"] = RelevancyEvaluator(
            llm=self.llama_rag.llm,
            criteria={
                "medical_relevance": "La respuesta debe ser relevante para la consulta médica",
                "clinical_context": "La respuesta debe considerar el contexto clínico del paciente",
                "diagnostic_relevance": "La respuesta debe ser relevante para el diagnóstico o tratamiento"
            }
        )
        
        # Evaluador de fidelidad (faithfulness)
        self.evaluators["faithfulness"] = FaithfulnessEvaluator(
            llm=self.llama_rag.llm,
            criteria={
                "source_faithfulness": "La respuesta debe ser fiel a las fuentes médicas consultadas",
                "evidence_based": "La respuesta debe basarse en evidencia médica",
                "no_hallucination": "La respuesta no debe contener información no respaldada por las fuentes"
            }
        )
        
        # Evaluador de relevancia de respuesta
        self.evaluators["answer_relevancy"] = AnswerRelevancyEvaluator(
            llm=self.llama_rag.llm,
            criteria={
                "medical_answer_relevance": "La respuesta debe responder directamente a la consulta médica",
                "clinical_utility": "La respuesta debe ser útil para la práctica clínica",
                "patient_centered": "La respuesta debe estar centrada en el paciente"
            }
        )
        
        # Evaluador de relevancia de contexto
        self.evaluators["context_relevancy"] = ContextRelevancyEvaluator(
            llm=self.llama_rag.llm,
            criteria={
                "medical_context_relevance": "El contexto recuperado debe ser relevante para la consulta médica",
                "diagnostic_context": "El contexto debe ser relevante para el diagnóstico",
                "treatment_context": "El contexto debe ser relevante para el tratamiento"
            }
        )
        
        logger.info(f"✅ Configurados {len(self.evaluators)} evaluadores médicos")
    
    async def evaluate_medical_response(self,
                                       query: str,
                                       response: str,
                                       context_nodes: List[Any] = None,
                                       expected_response: str = None) -> Dict[str, Any]:
        """
        Evaluar respuesta médica usando múltiples criterios
        
        Args:
            query: Consulta médica original
            response: Respuesta generada por el sistema
            context_nodes: Nodos de contexto utilizados
            expected_response: Respuesta esperada (opcional, para evaluación con ground truth)
        """
        
        try:
            evaluation_results = {}
            
            # Preparar contexto para evaluación
            context_text = ""
            if context_nodes:
                context_text = "\n\n".join([node.text for node in context_nodes])
            
            # 1. Evaluar corrección médica
            if "correctness" in self.evaluators:
                correctness_result = await self.evaluators["correctness"].aevaluate(
                    query=query,
                    response=response,
                    context=context_text
                )
                evaluation_results["correctness"] = {
                    "score": correctness_result.score,
                    "feedback": correctness_result.feedback,
                    "criteria": correctness_result.criteria
                }
            
            # 2. Evaluar relevancia
            if "relevancy" in self.evaluators:
                relevancy_result = await self.evaluators["relevancy"].aevaluate(
                    query=query,
                    response=response,
                    context=context_text
                )
                evaluation_results["relevancy"] = {
                    "score": relevancy_result.score,
                    "feedback": relevancy_result.feedback,
                    "criteria": relevancy_result.criteria
                }
            
            # 3. Evaluar fidelidad
            if "faithfulness" in self.evaluators:
                faithfulness_result = await self.evaluators["faithfulness"].aevaluate(
                    query=query,
                    response=response,
                    context=context_text
                )
                evaluation_results["faithfulness"] = {
                    "score": faithfulness_result.score,
                    "feedback": faithfulness_result.feedback,
                    "criteria": faithfulness_result.criteria
                }
            
            # 4. Evaluar relevancia de respuesta
            if "answer_relevancy" in self.evaluators:
                answer_relevancy_result = await self.evaluators["answer_relevancy"].aevaluate(
                    query=query,
                    response=response,
                    context=context_text
                )
                evaluation_results["answer_relevancy"] = {
                    "score": answer_relevancy_result.score,
                    "feedback": answer_relevancy_result.feedback,
                    "criteria": answer_relevancy_result.criteria
                }
            
            # 5. Evaluar relevancia de contexto
            if "context_relevancy" in self.evaluators and context_nodes:
                context_relevancy_result = await self.evaluators["context_relevancy"].aevaluate(
                    query=query,
                    response=response,
                    context=context_text
                )
                evaluation_results["context_relevancy"] = {
                    "score": context_relevancy_result.score,
                    "feedback": context_relevancy_result.feedback,
                    "criteria": context_relevancy_result.criteria
                }
            
            # Calcular score promedio
            scores = [result["score"] for result in evaluation_results.values()]
            average_score = sum(scores) / len(scores) if scores else 0.0
            
            # Determinar calidad general
            quality_level = self._determine_quality_level(average_score)
            
            evaluation_summary = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "context_used": bool(context_nodes),
                "evaluation_results": evaluation_results,
                "average_score": average_score,
                "quality_level": quality_level,
                "overall_feedback": self._generate_overall_feedback(evaluation_results)
            }
            
            # Guardar en historial
            self.evaluation_history.append(evaluation_summary)
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"❌ Error evaluando respuesta médica: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response
            }
    
    def _determine_quality_level(self, score: float) -> str:
        """Determinar nivel de calidad basado en el score"""
        
        if score >= 0.9:
            return "EXCELENTE"
        elif score >= 0.8:
            return "MUY BUENO"
        elif score >= 0.7:
            return "BUENO"
        elif score >= 0.6:
            return "ACEPTABLE"
        else:
            return "INSUFICIENTE"
    
    def _generate_overall_feedback(self, evaluation_results: Dict[str, Any]) -> str:
        """Generar feedback general basado en todos los evaluadores"""
        
        feedback_parts = []
        
        for evaluator_name, result in evaluation_results.items():
            score = result["score"]
            if score < 0.7:
                feedback_parts.append(f"Necesita mejora en {evaluator_name}: {result['feedback']}")
            elif score >= 0.9:
                feedback_parts.append(f"Excelente en {evaluator_name}")
        
        if not feedback_parts:
            return "Respuesta médica de buena calidad general"
        
        return "; ".join(feedback_parts)
    
    async def evaluate_batch_responses(self,
                                      queries: List[str],
                                      responses: List[str],
                                      contexts: List[List[Any]] = None) -> Dict[str, Any]:
        """
        Evaluar un lote de respuestas médicas
        
        Args:
            queries: Lista de consultas
            responses: Lista de respuestas correspondientes
            contexts: Lista de contextos utilizados
        """
        
        try:
            batch_results = []
            total_scores = []
            
            for i, (query, response) in enumerate(zip(queries, responses)):
                context = contexts[i] if contexts else None
                
                evaluation = await self.evaluate_medical_response(
                    query=query,
                    response=response,
                    context_nodes=context
                )
                
                batch_results.append(evaluation)
                if "average_score" in evaluation:
                    total_scores.append(evaluation["average_score"])
            
            # Estadísticas del lote
            batch_stats = {
                "total_queries": len(queries),
                "average_score": sum(total_scores) / len(total_scores) if total_scores else 0.0,
                "min_score": min(total_scores) if total_scores else 0.0,
                "max_score": max(total_scores) if total_scores else 0.0,
                "quality_distribution": self._calculate_quality_distribution(total_scores)
            }
            
            return {
                "batch_evaluation": batch_results,
                "batch_statistics": batch_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluando lote de respuestas: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calcular distribución de calidad"""
        
        distribution = {
            "EXCELENTE": 0,
            "MUY BUENO": 0,
            "BUENO": 0,
            "ACEPTABLE": 0,
            "INSUFICIENTE": 0
        }
        
        for score in scores:
            quality = self._determine_quality_level(score)
            distribution[quality] += 1
        
        return distribution
    
    async def evaluate_system_performance(self,
                                         test_queries: List[str],
                                         expected_responses: List[str] = None) -> Dict[str, Any]:
        """
        Evaluar rendimiento general del sistema
        
        Args:
            test_queries: Consultas de prueba
            expected_responses: Respuestas esperadas (opcional)
        """
        
        try:
            # Generar respuestas del sistema
            system_responses = []
            contexts_used = []
            
            for query in test_queries:
                result = await self.llama_rag.query_medical_system(query)
                system_responses.append(result["response"])
                contexts_used.append(result.get("retrieved_nodes", []))
            
            # Evaluar respuestas
            evaluation_results = await self.evaluate_batch_responses(
                queries=test_queries,
                responses=system_responses,
                contexts=contexts_used
            )
            
            # Métricas adicionales de rendimiento
            performance_metrics = {
                "response_time_avg": 0.0,  # Calcular tiempo real
                "context_retrieval_success": sum(1 for ctx in contexts_used if ctx) / len(contexts_used),
                "system_availability": 1.0,  # Asumir disponible
                "error_rate": 0.0  # Calcular tasa de errores
            }
            
            return {
                "system_evaluation": evaluation_results,
                "performance_metrics": performance_metrics,
                "test_queries_count": len(test_queries),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluando rendimiento del sistema: {e}")
            return {"error": str(e)}
    
    def get_evaluation_history(self,
                              limit: int = 100,
                              quality_filter: str = None) -> List[Dict[str, Any]]:
        """
        Obtener historial de evaluaciones
        
        Args:
            limit: Número máximo de evaluaciones a retornar
            quality_filter: Filtrar por nivel de calidad
        """
        
        history = self.evaluation_history[-limit:] if limit else self.evaluation_history
        
        if quality_filter:
            history = [
                eval_result for eval_result in history
                if eval_result.get("quality_level") == quality_filter
            ]
        
        return history
    
    def export_evaluation_report(self,
                                output_path: str = None,
                                format: str = "json") -> str:
        """
        Exportar reporte de evaluación
        
        Args:
            output_path: Ruta de salida
            format: Formato de salida (json, csv)
        """
        
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"medical_evaluation_report_{timestamp}.{format}"
            
            output_file = Path(output_path)
            
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "evaluation_history": self.evaluation_history,
                        "summary": {
                            "total_evaluations": len(self.evaluation_history),
                            "average_score": self._calculate_overall_average(),
                            "quality_distribution": self._calculate_overall_quality_distribution()
                        }
                    }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Reporte exportado a: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error exportando reporte: {e}")
            return ""
    
    def _calculate_overall_average(self) -> float:
        """Calcular promedio general de todas las evaluaciones"""
        
        scores = [
            eval_result["average_score"]
            for eval_result in self.evaluation_history
            if "average_score" in eval_result
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_overall_quality_distribution(self) -> Dict[str, int]:
        """Calcular distribución general de calidad"""
        
        return self._calculate_quality_distribution([
            eval_result["average_score"]
            for eval_result in self.evaluation_history
            if "average_score" in eval_result
        ])
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de evaluación"""
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "evaluators_available": list(self.evaluators.keys()),
            "overall_average_score": self._calculate_overall_average(),
            "quality_distribution": self._calculate_overall_quality_distribution(),
            "last_evaluation": self.evaluation_history[-1] if self.evaluation_history else None
        } 