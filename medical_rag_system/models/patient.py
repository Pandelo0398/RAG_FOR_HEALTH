"""
Modelos de datos del paciente
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class PatientData:
    """Datos del paciente"""
    age: int
    sex: str
    service: str
    admission_date: Optional[str] = None
    patient_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if self.age < 0 or self.age > 150:
            raise ValueError("Edad debe estar entre 0 y 150 años")
        
        if self.sex.upper() not in ['M', 'F', 'MALE', 'FEMALE', 'MASCULINO', 'FEMENINO']:
            raise ValueError("Sexo debe ser M/F o equivalente")
        
        # Normalizar sexo
        self.sex = self.sex.upper()
        if self.sex in ['MALE', 'MASCULINO']:
            self.sex = 'M'
        elif self.sex in ['FEMALE', 'FEMENINO']:
            self.sex = 'F'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return asdict(self)
    
    def get_risk_factors(self) -> list[str]:
        """Obtener factores de riesgo basados en edad y datos adicionales"""
        risk_factors = []
        
        if self.age >= 65:
            risk_factors.append("edad_avanzada")
        
        if self.age >= 75:
            risk_factors.append("edad_muy_avanzada")
        
        if self.additional_info:
            if self.additional_info.get("smoker"):
                risk_factors.append("tabaquismo")
            if self.additional_info.get("diabetes"):
                risk_factors.append("diabetes_mellitus")
            if self.additional_info.get("hypertension"):
                risk_factors.append("hipertension_arterial")
        
        return risk_factors
    
    def __str__(self) -> str:
        return f"Paciente({self.age} años, {self.sex}, {self.service})"
