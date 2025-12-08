"""
Tensor-Based Axiomatic Reasoning Architecture
GPU-Scalable Neural-Symbolic Hybrid System

Author: Adrian Neill (digitaltoaster) + Claude
Date: 2024-12-08
"""

from .axiom_embeddings import AxiomEmbedding, AxiomGraph
from .axiom_attention import AxiomAttention, PriorityModulatedAttention
from .axiom_gnn import AxiomGraphNN, MessagePassingLayer
from .axiom_executor import AxiomModule, ChainExecutor
from .hybrid_model import HybridModel, HybridRouter
from .axiom_discovery import AxiomDiscovery

__all__ = [
    'AxiomEmbedding',
    'AxiomGraph',
    'AxiomAttention',
    'PriorityModulatedAttention',
    'AxiomGraphNN',
    'MessagePassingLayer',
    'AxiomModule',
    'ChainExecutor',
    'HybridModel',
    'HybridRouter',
    'AxiomDiscovery',
]

__version__ = '0.1.0'
