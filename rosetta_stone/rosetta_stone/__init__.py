"""
AI Rosetta Stone: Neuro-Symbolic Engine for AI Regulatory Compliance

A novel approach to automated compliance and explainability for the EU AI Act.
"""

__version__ = "0.1.0"
__author__ = "AI Rosetta Stone Team"
__description__ = "Neuro-Symbolic Engine for AI Regulatory Compliance"

from .knowledge_base import SymbolicKnowledgeBase
from .bridge import NeuroSymbolicBridge
from .mapping import MappingReasoningEngine
from .reporting import ComplianceReportGenerator

__all__ = [
    "SymbolicKnowledgeBase",
    "NeuroSymbolicBridge", 
    "MappingReasoningEngine",
    "ComplianceReportGenerator"
]