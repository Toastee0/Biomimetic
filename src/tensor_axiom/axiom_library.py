"""
Axiom Library Management System

Handles loading, storing, and managing axioms with their test scenarios
and performance metrics.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from .axiom_embeddings import AxiomEmbedding, AxiomGraph


class AxiomLibrary:
    """Manages axiom definitions and their metadata"""
    
    def __init__(self, library_path: str = "data/axioms/base_axioms.json"):
        """
        Initialize the axiom library.
        
        Args:
            library_path: Path to the axiom JSON file
        """
        self.library_path = Path(library_path)
        self.axioms: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, AxiomEmbedding] = {}
        self.graph: Optional[AxiomGraph] = None
        self.meta: Dict[str, Any] = {}
        
        self.load()
    
    def load(self):
        """Load axioms from JSON file"""
        if not self.library_path.exists():
            raise FileNotFoundError(f"Axiom library not found: {self.library_path}")
        
        with open(self.library_path, 'r') as f:
            data = json.load(f)
        
        self.meta = data.get('meta', {})
        self.axioms = data.get('axioms', {})
        
        print(f"Loaded {len(self.axioms)} axioms from {self.library_path}")
    
    def save(self):
        """Save axioms and metrics back to JSON file"""
        data = {
            'meta': self.meta,
            'axioms': self.axioms
        }
        
        # Update timestamp
        self.meta['last_updated'] = datetime.now().isoformat()
        
        with open(self.library_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.axioms)} axioms to {self.library_path}")
    
    def build_embeddings(self, d_semantic: int = 256, d_logic: int = 192, d_dependency: int = 64):
        """
        Build PyTorch embeddings from axiom definitions.
        
        Args:
            d_semantic: Dimension for semantic embeddings
            d_logic: Dimension for logic embeddings
            d_dependency: Dimension for dependency embeddings
        """
        self.embeddings = {}
        
        for axiom_id, axiom_data in self.axioms.items():
            embedding = AxiomEmbedding(
                axiom_id=axiom_id,
                semantic_dim=d_semantic,
                logic_dim=d_logic,
                dependency_dim=d_dependency,
                priority=axiom_data.get('priority', 0.5),
                confidence=axiom_data.get('confidence', 0.5),
                description=axiom_data.get('name', axiom_id)
            )
            self.embeddings[axiom_id] = embedding
        
        print(f"Created {len(self.embeddings)} axiom embeddings")
    
    def build_graph(self):
        """
        Build axiom graph with relationships from definitions.
        """
        if not self.embeddings:
            self.build_embeddings()
        
        num_axioms = len(self.axioms)
        d_axiom = 256 + 192 + 64  # semantic + logic + dependency
        
        self.graph = AxiomGraph(num_axioms, d_axiom)
        
        # Add axioms to graph
        for idx, (axiom_id, embedding) in enumerate(self.embeddings.items()):
            self.graph.add_axiom(embedding, idx)
        
        # Add edges based on relationships
        for axiom_id, axiom_data in self.axioms.items():
            relationships = axiom_data.get('edge_relationships', {})
            
            # Add 'implies' edges
            for target_id in relationships.get('implies', []):
                if target_id in self.axioms:
                    try:
                        self.graph.add_edge(axiom_id, target_id, 'implies', weight=0.8)
                    except ValueError:
                        pass  # Skip if edge type not supported
            
            # Add 'requires' edges
            for target_id in relationships.get('requires', []):
                if target_id in self.axioms:
                    try:
                        self.graph.add_edge(axiom_id, target_id, 'requires', weight=0.9)
                    except ValueError:
                        pass
            
            # Add 'contradicts' edges
            for target_id in relationships.get('contradicts', []):
                if target_id in self.axioms:
                    try:
                        self.graph.add_edge(axiom_id, target_id, 'contradicts', weight=0.7)
                    except ValueError:
                        pass
            
            # Add 'composes_with' as 'composes' edges
            for target_id in relationships.get('composes_with', []):
                if target_id in self.axioms:
                    try:
                        self.graph.add_edge(axiom_id, target_id, 'composes', weight=0.6)
                    except ValueError:
                        pass
        
        print(f"Built axiom graph with {num_axioms} nodes")
        return self.graph
    
    def get_axiom(self, axiom_id: str) -> Dict[str, Any]:
        """Get axiom definition by ID"""
        return self.axioms.get(axiom_id)
    
    def get_test_scenarios(self, axiom_id: str) -> List[Dict[str, str]]:
        """Get test scenarios for an axiom"""
        axiom = self.get_axiom(axiom_id)
        if axiom:
            return axiom.get('test_scenarios', [])
        return []
    
    def update_metrics(self, axiom_id: str, metrics: Dict[str, Any]):
        """
        Update performance metrics for an axiom.
        
        Args:
            axiom_id: ID of the axiom
            metrics: Dictionary of metrics to update
        """
        if axiom_id in self.axioms:
            perf = self.axioms[axiom_id].get('performance_metrics', {})
            perf.update(metrics)
            perf['last_tested'] = datetime.now().isoformat()
            self.axioms[axiom_id]['performance_metrics'] = perf
    
    def get_axioms_by_category(self, category: str) -> List[str]:
        """Get all axiom IDs in a category"""
        return [
            aid for aid, data in self.axioms.items()
            if data.get('category') == category
        ]
    
    def get_axioms_needing_review(self, threshold: float = 0.6) -> List[str]:
        """
        Get axioms that need human review based on poor performance.
        
        Args:
            threshold: Success rate below this triggers review
            
        Returns:
            List of axiom IDs needing review
        """
        needs_review = []
        
        for axiom_id, axiom_data in self.axioms.items():
            metrics = axiom_data.get('performance_metrics', {})
            test_count = metrics.get('test_count', 0)
            success_rate = metrics.get('success_rate', 1.0)
            
            # Only flag if we have enough data
            if test_count >= 5 and success_rate < threshold:
                needs_review.append(axiom_id)
        
        return needs_review
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall library statistics"""
        total = len(self.axioms)
        tested = sum(1 for a in self.axioms.values() 
                    if a.get('performance_metrics', {}).get('test_count', 0) > 0)
        
        categories = {}
        for axiom_data in self.axioms.values():
            cat = axiom_data.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        avg_success = 0
        if tested > 0:
            success_rates = [
                a.get('performance_metrics', {}).get('success_rate', 0)
                for a in self.axioms.values()
                if a.get('performance_metrics', {}).get('test_count', 0) > 0
            ]
            avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
        
        return {
            'total_axioms': total,
            'tested_axioms': tested,
            'untested_axioms': total - tested,
            'categories': categories,
            'average_success_rate': avg_success,
            'needs_review': len(self.get_axioms_needing_review())
        }
    
    def __len__(self):
        return len(self.axioms)
    
    def __iter__(self):
        return iter(self.axioms.items())
