"""
Axiom Attention Mechanisms

Multi-head attention for axiom selection and priority-modulated attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AxiomAttention(nn.Module):
    """
    Multi-head attention for selecting relevant axioms given a situation.
    
    Given situation embedding S, computes attention scores over all axioms
    to determine which are most relevant for the current context.
    """
    
    def __init__(
        self,
        d_situation: int,
        d_axiom: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_situation = d_situation
        self.d_axiom = d_axiom
        self.num_heads = num_heads
        self.d_head = d_axiom // num_heads
        
        assert d_axiom % num_heads == 0, "d_axiom must be divisible by num_heads"
        
        # Query projection (from situation)
        self.W_q = nn.Linear(d_situation, d_axiom)
        
        # Key projection (from axioms)
        self.W_k = nn.Linear(d_axiom, d_axiom)
        
        # Value projection (axiom embeddings)
        self.W_v = nn.Linear(d_axiom, d_axiom)
        
        # Output projection
        self.W_o = nn.Linear(d_axiom, d_axiom)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
    
    def forward(
        self,
        situation: torch.Tensor,
        axiom_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over axioms for given situation.
        
        Args:
            situation: [batch_size, d_situation] situation embedding
            axiom_embeddings: [num_axioms, d_axiom] all axiom embeddings
            mask: [batch_size, num_axioms] optional mask for invalid axioms
            
        Returns:
            context: [batch_size, d_axiom] weighted axiom context
            scores: [batch_size, num_axioms] attention scores
        """
        batch_size = situation.size(0)
        num_axioms = axiom_embeddings.size(0)
        
        # Project to Q, K, V
        Q = self.W_q(situation)  # [batch_size, d_axiom]
        K = self.W_k(axiom_embeddings)  # [num_axioms, d_axiom]
        V = self.W_v(axiom_embeddings)  # [num_axioms, d_axiom]
        
        # Reshape for multi-head attention
        # Q: [batch_size, num_heads, d_head]
        Q = Q.view(batch_size, self.num_heads, self.d_head)
        
        # K, V: [num_axioms, num_heads, d_head]
        K = K.view(num_axioms, self.num_heads, self.d_head)
        V = V.view(num_axioms, self.num_heads, self.d_head)
        
        # Compute attention scores
        # [batch_size, num_heads, num_axioms]
        scores = torch.einsum('bhd,nhd->bhn', Q, K) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax over axioms
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [batch_size, num_heads, d_head]
        context = torch.einsum('bhn,nhd->bhd', attn_weights, V)
        
        # Reshape and project
        context = context.reshape(batch_size, self.d_axiom)
        context = self.W_o(context)
        
        # Average attention scores across heads for interpretability
        scores_avg = attn_weights.mean(dim=1)  # [batch_size, num_axioms]
        
        return context, scores_avg
    
    def select_top_k(
        self,
        situation: torch.Tensor,
        axiom_embeddings: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k most relevant axioms.
        
        Args:
            situation: [batch_size, d_situation]
            axiom_embeddings: [num_axioms, d_axiom]
            k: Number of axioms to select
            mask: Optional mask
            
        Returns:
            selected_indices: [batch_size, k] indices of selected axioms
            selected_embeddings: [batch_size, k, d_axiom]
            selected_scores: [batch_size, k] attention scores
        """
        _, scores = self.forward(situation, axiom_embeddings, mask)
        
        # Get top-k indices and scores
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        # Gather selected embeddings
        batch_size = situation.size(0)
        selected_embeddings = axiom_embeddings[top_indices]  # [batch_size, k, d_axiom]
        
        return top_indices, selected_embeddings, top_scores


class PriorityModulatedAttention(AxiomAttention):
    """
    Axiom attention with priority modulation.
    
    Boosts attention scores for high-priority axioms (e.g., meta-axioms)
    and dampens attention for low-priority axioms.
    """
    
    def __init__(
        self,
        d_situation: int,
        d_axiom: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        priority_alpha: float = 2.0
    ):
        super().__init__(d_situation, d_axiom, num_heads, dropout)
        
        self.priority_alpha = priority_alpha
    
    def forward(
        self,
        situation: torch.Tensor,
        axiom_embeddings: torch.Tensor,
        priority_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute priority-modulated attention.
        
        Args:
            situation: [batch_size, d_situation]
            axiom_embeddings: [num_axioms, d_axiom]
            priority_weights: [num_axioms] priority values (0-1)
            mask: Optional mask
            
        Returns:
            context: [batch_size, d_axiom]
            scores: [batch_size, num_axioms] modulated attention scores
        """
        # Get base attention scores
        context, base_scores = super().forward(situation, axiom_embeddings, mask)
        
        # Modulate by priority
        # priority_weights: [num_axioms]
        # Boost: priority^alpha
        priority_boost = priority_weights.pow(self.priority_alpha)
        
        # Apply boost to scores (before softmax, we multiply after)
        # We need to recompute with priority in the loop
        # For efficiency, we'll multiply the final scores
        modulated_scores = base_scores * priority_boost.unsqueeze(0)
        
        # Renormalize
        modulated_scores = F.softmax(modulated_scores / modulated_scores.sum(dim=-1, keepdim=True), dim=-1)
        
        return context, modulated_scores
    
    def select_top_k(
        self,
        situation: torch.Tensor,
        axiom_embeddings: torch.Tensor,
        priority_weights: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k axioms with priority modulation.
        
        Args:
            situation: [batch_size, d_situation]
            axiom_embeddings: [num_axioms, d_axiom]
            priority_weights: [num_axioms]
            k: Number to select
            mask: Optional mask
            
        Returns:
            selected_indices: [batch_size, k]
            selected_embeddings: [batch_size, k, d_axiom]
            selected_scores: [batch_size, k]
        """
        _, scores = self.forward(situation, axiom_embeddings, priority_weights, mask)
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        # Gather embeddings
        selected_embeddings = axiom_embeddings[top_indices]
        
        return top_indices, selected_embeddings, top_scores


class AdaptiveAxiomSelector(nn.Module):
    """
    Adaptive selector that learns to pick between different attention strategies.
    """
    
    def __init__(
        self,
        d_situation: int,
        d_axiom: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Standard attention
        self.standard_attn = AxiomAttention(d_situation, d_axiom, num_heads, dropout)
        
        # Priority-modulated attention
        self.priority_attn = PriorityModulatedAttention(
            d_situation, d_axiom, num_heads, dropout
        )
        
        # Learned mixing weight
        self.mixing_net = nn.Sequential(
            nn.Linear(d_situation, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        situation: torch.Tensor,
        axiom_embeddings: torch.Tensor,
        priority_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively mix standard and priority-modulated attention.
        
        Args:
            situation: [batch_size, d_situation]
            axiom_embeddings: [num_axioms, d_axiom]
            priority_weights: [num_axioms]
            mask: Optional mask
            
        Returns:
            context: [batch_size, d_axiom]
            scores: [batch_size, num_axioms]
        """
        # Get both attention types
        context_std, scores_std = self.standard_attn(
            situation, axiom_embeddings, mask
        )
        context_pri, scores_pri = self.priority_attn(
            situation, axiom_embeddings, priority_weights, mask
        )
        
        # Compute mixing weight (situation-dependent)
        mix_weight = self.mixing_net(situation)  # [batch_size, 1]
        
        # Mix contexts and scores
        context = mix_weight * context_pri + (1 - mix_weight) * context_std
        scores = mix_weight * scores_pri + (1 - mix_weight) * scores_std
        
        return context, scores
