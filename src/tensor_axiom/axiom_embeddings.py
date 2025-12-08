"""
Axiom Embedding and Graph Representation

Represents axioms as learnable tensors and manages the axiom dependency graph.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class AxiomEmbedding(nn.Module):
    """
    Represents a single axiom as a learnable tensor embedding.
    
    Structure:
        - Semantic embedding: what the axiom means (256-dim)
        - Logic embedding: how it operates (128-dim)
        - Dependency vector: what it requires (64-dim)
        - Priority scalar: importance level (meta=1.0, domain=0.4)
        - Confidence: reliability [0,1]
    """
    
    def __init__(
        self,
        axiom_id: str,
        semantic_dim: int = 256,
        logic_dim: int = 128,
        dependency_dim: int = 64,
        priority: float = 0.5,
        confidence: float = 1.0,
        description: Optional[str] = None
    ):
        super().__init__()
        
        self.axiom_id = axiom_id
        self.description = description
        self.d_axiom = semantic_dim + logic_dim + dependency_dim + 1  # +1 for priority
        
        # Learnable components
        self.semantic = nn.Parameter(torch.randn(semantic_dim))
        self.logic = nn.Parameter(torch.randn(logic_dim))
        self.dependency = nn.Parameter(torch.randn(dependency_dim))
        
        # Fixed metadata (can be updated but not learned)
        self.register_buffer('priority', torch.tensor(priority))
        self.register_buffer('confidence', torch.tensor(confidence))
        
        # Initialize with small random values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize embeddings with small random values"""
        nn.init.normal_(self.semantic.data, mean=0.0, std=0.02)
        nn.init.normal_(self.logic.data, mean=0.0, std=0.02)
        nn.init.normal_(self.dependency.data, mean=0.0, std=0.02)
    
    def forward(self) -> torch.Tensor:
        """
        Returns the full axiom embedding vector.
        
        Returns:
            torch.Tensor: [d_axiom] dimensional embedding
        """
        # Concatenate all components (priority is metadata, not part of embedding)
        embedding = torch.cat([
            self.semantic,
            self.logic,
            self.dependency
        ], dim=0)
        
        return embedding
    
    def get_semantic(self) -> torch.Tensor:
        """Get semantic component"""
        return self.semantic
    
    def get_logic(self) -> torch.Tensor:
        """Get logic component"""
        return self.logic
    
    def get_dependency(self) -> torch.Tensor:
        """Get dependency component"""
        return self.dependency
    
    def set_priority(self, priority: float):
        """Update priority level"""
        self.priority.fill_(priority)
    
    def set_confidence(self, confidence: float):
        """Update confidence level"""
        self.confidence.fill_(confidence)


class AxiomGraph(nn.Module):
    """
    Manages the complete axiom graph with multiple edge types.
    
    Edge Types:
        - implies: A → B (logical implication)
        - requires: A needs B (dependency)
        - contradicts: A ⊥ B (contradiction)
        - composes: A + B → C (composition)
    """
    
    def __init__(
        self,
        num_axioms: int,
        d_axiom: int = 449,  # 256 + 128 + 64 + 1
        edge_types: List[str] = None
    ):
        super().__init__()
        
        self.num_axioms = num_axioms
        self.d_axiom = d_axiom
        
        if edge_types is None:
            edge_types = ['implies', 'requires', 'contradicts', 'composes']
        self.edge_types = edge_types
        
        # Adjacency matrices for each edge type (binary)
        self.adjacency = nn.ParameterDict()
        self.weights = nn.ParameterDict()
        
        for edge_type in edge_types:
            # Binary adjacency (0/1)
            adj = torch.zeros(num_axioms, num_axioms)
            self.register_buffer(f'A_{edge_type}', adj)
            
            # Edge weights (learned)
            weights = nn.Parameter(torch.randn(num_axioms, num_axioms) * 0.01)
            self.weights[edge_type] = weights
        
        # Axiom embeddings storage
        self.axiom_embeddings = nn.ModuleDict()
        self.axiom_id_to_idx = {}
        self.idx_to_axiom_id = {}
    
    def add_axiom(self, axiom: AxiomEmbedding, idx: int):
        """
        Add an axiom to the graph.
        
        Args:
            axiom: AxiomEmbedding instance
            idx: Index position in graph
        """
        self.axiom_embeddings[axiom.axiom_id] = axiom
        self.axiom_id_to_idx[axiom.axiom_id] = idx
        self.idx_to_axiom_id[idx] = axiom.axiom_id
    
    def add_edge(
        self,
        from_axiom: str,
        to_axiom: str,
        edge_type: str,
        weight: float = 1.0
    ):
        """
        Add an edge between two axioms.
        
        Args:
            from_axiom: Source axiom ID
            to_axiom: Target axiom ID
            edge_type: Type of edge ('implies', 'requires', etc.)
            weight: Edge weight (strength of connection)
        """
        if edge_type not in self.edge_types:
            raise ValueError(f"Unknown edge type: {edge_type}")
        
        from_idx = self.axiom_id_to_idx[from_axiom]
        to_idx = self.axiom_id_to_idx[to_axiom]
        
        # Set adjacency
        adj_matrix = getattr(self, f'A_{edge_type}')
        adj_matrix[from_idx, to_idx] = 1.0
        
        # Set weight
        self.weights[edge_type].data[from_idx, to_idx] = weight
    
    def get_adjacency(self, edge_type: str) -> torch.Tensor:
        """Get adjacency matrix for edge type"""
        return getattr(self, f'A_{edge_type}')
    
    def get_neighbors(
        self,
        axiom_idx: int,
        edge_type: str,
        direction: str = 'outgoing'
    ) -> torch.Tensor:
        """
        Get neighboring axioms.
        
        Args:
            axiom_idx: Index of axiom
            edge_type: Type of edges to follow
            direction: 'outgoing' or 'incoming'
            
        Returns:
            torch.Tensor: Indices of neighboring axioms
        """
        adj = self.get_adjacency(edge_type)
        
        if direction == 'outgoing':
            neighbors = torch.nonzero(adj[axiom_idx]).squeeze(-1)
        else:  # incoming
            neighbors = torch.nonzero(adj[:, axiom_idx]).squeeze(-1)
        
        return neighbors
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all axiom embeddings as a tensor.
        
        Returns:
            torch.Tensor: [num_axioms, d_axiom]
        """
        embeddings = []
        for idx in range(self.num_axioms):
            axiom_id = self.idx_to_axiom_id.get(idx)
            if axiom_id:
                axiom = self.axiom_embeddings[axiom_id]
                embeddings.append(axiom())
            else:
                # Empty slot
                embeddings.append(torch.zeros(self.d_axiom, device=self.device))
        
        return torch.stack(embeddings)
    
    def get_priority_weights(self) -> torch.Tensor:
        """
        Get priority weights for all axioms.
        
        Returns:
            torch.Tensor: [num_axioms] priority values
        """
        priorities = []
        for idx in range(self.num_axioms):
            axiom_id = self.idx_to_axiom_id.get(idx)
            if axiom_id:
                axiom = self.axiom_embeddings[axiom_id]
                priorities.append(axiom.priority)
            else:
                priorities.append(torch.tensor(0.0))
        
        return torch.stack(priorities)
    
    @property
    def device(self):
        """Get device of graph"""
        return next(self.parameters()).device
    
    def to_sparse(self, edge_type: str) -> torch.sparse.FloatTensor:
        """
        Convert adjacency matrix to sparse format for efficiency.
        
        Args:
            edge_type: Type of edge
            
        Returns:
            torch.sparse.FloatTensor: Sparse adjacency matrix
        """
        adj = self.get_adjacency(edge_type)
        weights = self.weights[edge_type]
        
        # Get non-zero indices
        indices = torch.nonzero(adj, as_tuple=False).t()
        
        # Get corresponding weights
        values = weights[indices[0], indices[1]]
        
        # Create sparse tensor
        sparse_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (self.num_axioms, self.num_axioms),
            device=self.device
        )
        
        return sparse_adj
    
    def visualize_subgraph(
        self,
        axiom_indices: List[int],
        edge_type: str = 'implies'
    ) -> Dict:
        """
        Extract subgraph for visualization.
        
        Args:
            axiom_indices: Indices of axioms to include
            edge_type: Type of edges to show
            
        Returns:
            dict: Subgraph data for visualization
        """
        adj = self.get_adjacency(edge_type)
        weights = self.weights[edge_type]
        
        edges = []
        for i in axiom_indices:
            for j in axiom_indices:
                if adj[i, j] > 0:
                    edges.append({
                        'from': self.idx_to_axiom_id[i],
                        'to': self.idx_to_axiom_id[j],
                        'weight': float(weights[i, j]),
                        'type': edge_type
                    })
        
        nodes = []
        for idx in axiom_indices:
            axiom_id = self.idx_to_axiom_id[idx]
            axiom = self.axiom_embeddings[axiom_id]
            nodes.append({
                'id': axiom_id,
                'priority': float(axiom.priority),
                'confidence': float(axiom.confidence),
                'description': axiom.description
            })
        
        return {'nodes': nodes, 'edges': edges}
