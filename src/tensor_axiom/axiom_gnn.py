"""
Axiom Graph Neural Network

Message passing network for constructing axiom reasoning chains.
Uses Graph Attention Networks (GAT) to propagate information through the axiom graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class MessagePassingLayer(nn.Module):
    """
    Single layer of message passing for axiom graph.
    
    Aggregates messages from neighboring axioms along different edge types
    and updates axiom states.
    """
    
    def __init__(
        self,
        d_axiom: int,
        d_hidden: int,
        edge_types: List[str],
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_axiom = d_axiom
        self.d_hidden = d_hidden
        self.edge_types = edge_types
        self.num_heads = num_heads
        self.d_head = d_hidden // num_heads
        
        # Message networks for each edge type
        self.message_nets = nn.ModuleDict()
        self.attention_nets = nn.ModuleDict()
        
        for edge_type in edge_types:
            # Transform messages
            self.message_nets[edge_type] = nn.Linear(d_axiom, d_hidden)
            
            # Attention for message importance
            self.attention_nets[edge_type] = nn.Sequential(
                nn.Linear(d_axiom * 2, num_heads),
                nn.LeakyReLU(0.2)
            )
        
        # Update GRU
        self.gru = nn.GRUCell(d_hidden * len(edge_types), d_axiom)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_axiom)
    
    def forward(
        self,
        axiom_states: torch.Tensor,
        adjacency_dict: Dict[str, torch.Tensor],
        weight_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform one step of message passing.
        
        Args:
            axiom_states: [num_axioms, d_axiom] current states
            adjacency_dict: Dict of adjacency matrices for each edge type
            weight_dict: Dict of edge weight matrices
            
        Returns:
            torch.Tensor: [num_axioms, d_axiom] updated states
        """
        num_axioms = axiom_states.size(0)
        device = axiom_states.device
        
        # Collect messages from each edge type
        all_messages = []
        
        for edge_type in self.edge_types:
            adj = adjacency_dict[edge_type]  # [num_axioms, num_axioms]
            weights = weight_dict[edge_type]  # [num_axioms, num_axioms]
            
            # For each axiom, gather messages from neighbors
            messages = []
            
            for i in range(num_axioms):
                # Get incoming neighbors (axioms that connect to i)
                neighbors = torch.nonzero(adj[:, i]).squeeze(-1)
                
                if len(neighbors) == 0:
                    # No neighbors, zero message
                    messages.append(torch.zeros(self.d_hidden, device=device))
                    continue
                
                # Get neighbor states
                neighbor_states = axiom_states[neighbors]  # [num_neighbors, d_axiom]
                
                # Compute attention scores
                # Concatenate current state with each neighbor
                current_expanded = axiom_states[i].unsqueeze(0).expand(len(neighbors), -1)
                combined = torch.cat([current_expanded, neighbor_states], dim=-1)
                attn_scores = self.attention_nets[edge_type](combined)  # [num_neighbors, num_heads]
                
                # Apply edge weights
                edge_weights = weights[neighbors, i].unsqueeze(-1)  # [num_neighbors, 1]
                attn_scores = attn_scores * edge_weights
                
                # Softmax over neighbors for each head
                attn_weights = F.softmax(attn_scores, dim=0)  # [num_neighbors, num_heads]
                
                # Transform neighbor states to messages
                neighbor_messages = self.message_nets[edge_type](neighbor_states)  # [num_neighbors, d_hidden]
                
                # Apply multi-head attention
                # Reshape for heads
                neighbor_messages = neighbor_messages.view(len(neighbors), self.num_heads, self.d_head)
                attn_weights = attn_weights.unsqueeze(-1)  # [num_neighbors, num_heads, 1]
                
                # Weighted sum
                aggregated = (neighbor_messages * attn_weights).sum(dim=0)  # [num_heads, d_head]
                aggregated = aggregated.view(self.d_hidden)
                
                messages.append(aggregated)
            
            # Stack messages for all axioms
            edge_messages = torch.stack(messages)  # [num_axioms, d_hidden]
            all_messages.append(edge_messages)
        
        # Concatenate messages from all edge types
        combined_messages = torch.cat(all_messages, dim=-1)  # [num_axioms, d_hidden * num_edge_types]
        combined_messages = self.dropout(combined_messages)
        
        # Update states with GRU
        new_states = self.gru(combined_messages, axiom_states)
        
        # Layer norm
        new_states = self.layer_norm(new_states)
        
        return new_states


class AxiomGraphNN(nn.Module):
    """
    Multi-layer Graph Neural Network for axiom chain construction.
    
    Takes selected axioms as input and propagates information through
    the graph to identify which axioms should be part of the reasoning chain.
    """
    
    def __init__(
        self,
        d_axiom: int,
        d_hidden: int,
        edge_types: List[str],
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_axiom = d_axiom
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.edge_types = edge_types
        
        # Input projection
        self.input_proj = nn.Linear(d_axiom, d_axiom)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(d_axiom, d_hidden, edge_types, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (axiom activation scores)
        self.output_proj = nn.Sequential(
            nn.Linear(d_axiom, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        axiom_states: torch.Tensor,
        adjacency_dict: Dict[str, torch.Tensor],
        weight_dict: Dict[str, torch.Tensor],
        initial_activations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run message passing to construct axiom chains.
        
        Args:
            axiom_states: [num_axioms, d_axiom] axiom embeddings
            adjacency_dict: Adjacency matrices for each edge type
            weight_dict: Edge weights for each edge type
            initial_activations: [num_axioms] optional initial activation scores
            
        Returns:
            final_states: [num_axioms, d_axiom] final axiom states
            activations: [num_axioms] axiom activation scores (higher = more relevant)
        """
        # Project input
        states = self.input_proj(axiom_states)
        states = self.dropout(states)
        
        # Initialize with external activations if provided
        if initial_activations is not None:
            states = states * initial_activations.unsqueeze(-1)
        
        # Message passing
        for layer in self.mp_layers:
            states = layer(states, adjacency_dict, weight_dict)
        
        # Compute final activation scores
        activations = self.output_proj(states).squeeze(-1)  # [num_axioms]
        
        return states, activations
    
    def extract_chain(
        self,
        activations: torch.Tensor,
        adjacency_dict: Dict[str, torch.Tensor],
        max_length: int = 10,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Extract axiom chain from activation scores.
        
        Follows the 'implies' edges from high-activation axioms to construct
        a logical reasoning chain.
        
        Args:
            activations: [num_axioms] activation scores
            adjacency_dict: Adjacency matrices
            max_length: Maximum chain length
            threshold: Minimum activation to include
            
        Returns:
            list: Ordered list of axiom indices in chain
        """
        device = activations.device
        
        # Get high-activation axioms
        high_activation = (activations > threshold).nonzero().squeeze(-1)
        
        if len(high_activation) == 0:
            return []
        
        # Start with highest activation axiom
        chain = []
        current_idx = high_activation[torch.argmax(activations[high_activation])]
        visited = set()
        
        # Follow 'implies' edges
        implies_adj = adjacency_dict.get('implies', None)
        
        if implies_adj is None:
            # No implication graph, return top-k by activation
            top_k = min(max_length, len(high_activation))
            _, top_indices = torch.topk(activations, top_k)
            return top_indices.tolist()
        
        # Build chain by following implications
        for _ in range(max_length):
            if current_idx.item() in visited:
                break
            
            chain.append(current_idx.item())
            visited.add(current_idx.item())
            
            # Get next axioms (axioms implied by current)
            next_axioms = torch.nonzero(implies_adj[current_idx]).squeeze(-1)
            
            if len(next_axioms) == 0:
                break
            
            # Filter by activation
            next_axioms = next_axioms[activations[next_axioms] > threshold]
            
            if len(next_axioms) == 0:
                break
            
            # Pick highest activation
            current_idx = next_axioms[torch.argmax(activations[next_axioms])]
        
        return chain
    
    def extract_chain_differentiable(
        self,
        activations: torch.Tensor,
        max_length: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Extract chain in differentiable way using Gumbel-Softmax.
        
        Args:
            activations: [num_axioms] activation scores
            max_length: Maximum chain length
            temperature: Gumbel-Softmax temperature
            
        Returns:
            torch.Tensor: [max_length, num_axioms] soft chain selection
        """
        num_axioms = activations.size(0)
        
        # Gumbel-Softmax sampling
        chain_selections = []
        
        for i in range(max_length):
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(activations) + 1e-10
            ))
            
            # Softmax with temperature
            logits = (activations + gumbel_noise) / temperature
            soft_selection = F.softmax(logits, dim=-1)
            
            chain_selections.append(soft_selection)
            
            # Reduce activation of selected (approximate)
            activations = activations * (1 - soft_selection)
        
        # Stack to [max_length, num_axioms]
        chain_matrix = torch.stack(chain_selections)
        
        return chain_matrix
