"""
Hybrid Transformer-Axiom Model

Combines fast transformer inference with rigorous axiom-based reasoning.
Routes between paths based on novelty, confidence, and risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Tuple, Optional
from .axiom_attention import AdaptiveAxiomSelector
from .axiom_gnn import AxiomGraphNN
from .axiom_executor import ChainExecutor


class HybridRouter(nn.Module):
    """
    Decides whether to use fast transformer path or axiom reasoning path.
    
    Factors:
    - Novelty: Is this situation OOD?
    - Confidence: How certain is the transformer?
    - Risk: How risky is this situation?
    """
    
    def __init__(
        self,
        d_hidden: int,
        novelty_threshold: float = 0.7,
        confidence_threshold: float = 0.5,
        risk_threshold: float = 0.8
    ):
        super().__init__()
        
        self.novelty_threshold = novelty_threshold
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        
        # Learned routing policy
        self.router = nn.Sequential(
            nn.Linear(d_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # [fast_score, axiom_score]
        )
        
        # Novelty detector (VAE-style)
        self.novelty_encoder = nn.Sequential(
            nn.Linear(d_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Risk predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def compute_novelty(self, situation: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty score (how OOD is this situation).
        
        Uses reconstruction error as proxy for novelty.
        
        Args:
            situation: [batch_size, d_hidden]
            
        Returns:
            torch.Tensor: [batch_size] novelty scores (0-1)
        """
        # Encode and try to reconstruct
        encoded = self.novelty_encoder(situation)
        
        # Compute distance from mean (simplified OOD detection)
        mean_encoding = encoded.mean(dim=0, keepdim=True)
        distance = torch.norm(encoded - mean_encoding, dim=-1)
        
        # Normalize to [0, 1]
        novelty = torch.sigmoid(distance - 2.0)
        
        return novelty
    
    def compute_confidence(self, transformer_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute transformer confidence.
        
        Args:
            transformer_logits: [batch_size, num_classes] logits
            
        Returns:
            torch.Tensor: [batch_size] confidence scores (0-1)
        """
        # Use entropy of softmax as confidence measure
        probs = F.softmax(transformer_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Lower entropy = higher confidence
        max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float))
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence
    
    def compute_risk(self, situation: torch.Tensor) -> torch.Tensor:
        """
        Compute situation risk level.
        
        Args:
            situation: [batch_size, d_hidden]
            
        Returns:
            torch.Tensor: [batch_size] risk scores (0-1)
        """
        return self.risk_predictor(situation).squeeze(-1)
    
    def forward(
        self,
        situation: torch.Tensor,
        transformer_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decide routing for each situation in batch.
        
        Args:
            situation: [batch_size, d_hidden] situation embedding
            transformer_logits: Optional transformer output for confidence
            
        Returns:
            route: [batch_size] binary routing decision (0=fast, 1=axiom)
            scores: Dict of diagnostic scores
        """
        batch_size = situation.size(0)
        
        # Compute factors
        novelty = self.compute_novelty(situation)
        risk = self.compute_risk(situation)
        
        if transformer_logits is not None:
            confidence = self.compute_confidence(transformer_logits)
        else:
            confidence = torch.ones(batch_size, device=situation.device) * 0.5
        
        # Learned routing scores
        route_logits = self.router(situation)  # [batch_size, 2]
        route_probs = F.softmax(route_logits, dim=-1)
        
        # Rule-based override
        # Use axiom path if: high novelty OR low confidence OR high risk
        should_use_axioms = (
            (novelty > self.novelty_threshold) |
            (confidence < self.confidence_threshold) |
            (risk > self.risk_threshold)
        )
        
        # Combine learned and rule-based
        # If rules say axiom, use axiom; otherwise use learned routing
        route_decision = torch.where(
            should_use_axioms,
            torch.ones(batch_size, device=situation.device, dtype=torch.long),
            route_probs.argmax(dim=-1)
        )
        
        scores = {
            'novelty': novelty,
            'confidence': confidence,
            'risk': risk,
            'route_probs': route_probs
        }
        
        return route_decision, scores


class HybridModel(nn.Module):
    """
    Complete hybrid model combining transformer and axiom reasoning.
    
    Architecture:
    1. Situation encoder (shared)
    2. Router (decides path)
    3. Fast path (transformer)
    4. Axiom path (GNN + chain execution)
    5. Verifier (cross-check agreement)
    """
    
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        d_axiom: int,
        num_axioms: int,
        axiom_modules: Dict,
        axiom_graph: nn.Module,
        edge_types: List[str],
        num_transformer_layers: int = 6,
        num_gnn_layers: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        
        # Shared situation encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden)
        )
        
        # Router
        self.router = HybridRouter(d_hidden)
        
        # Fast path: Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=num_heads,
            dim_feedforward=d_hidden * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.transformer_head = nn.Linear(d_hidden, d_output)
        
        # Axiom path components
        self.axiom_graph = axiom_graph
        self.axiom_selector = AdaptiveAxiomSelector(
            d_situation=d_hidden,
            d_axiom=d_axiom,
            num_heads=num_heads
        )
        self.axiom_gnn = AxiomGraphNN(
            d_axiom=d_axiom,
            d_hidden=d_hidden,
            edge_types=edge_types,
            num_layers=num_gnn_layers,
            num_heads=num_heads
        )
        self.chain_executor = ChainExecutor(
            axiom_modules=axiom_modules,
            d_situation=d_hidden,
            d_output=d_output
        )
        
        # Verifier (cross-check)
        self.verifier = nn.Sequential(
            nn.Linear(d_output * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_explanation: bool = False,
        force_axiom_path: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: [batch_size, d_input] input
            return_explanation: Whether to return axiom chain explanations
            force_axiom_path: Force use of axiom reasoning (for testing)
            
        Returns:
            dict with keys:
                - output: [batch_size, d_output] final output
                - route: [batch_size] which path was used
                - axiom_chain: Optional list of axiom chains
                - agreement: Optional agreement scores
                - scores: Routing diagnostic scores
        """
        batch_size = x.size(0)
        
        # Encode situation
        situation = self.encoder(x)  # [batch_size, d_hidden]
        
        # Decide routing
        if not force_axiom_path:
            route, routing_scores = self.router(situation)
        else:
            route = torch.ones(batch_size, device=x.device, dtype=torch.long)
            routing_scores = {}
        
        # Execute both paths (for comparison/verification)
        # Fast path
        transformer_out = self.transformer(situation.unsqueeze(1)).squeeze(1)
        output_fast = self.transformer_head(transformer_out)
        
        # Axiom path
        axiom_embeddings = self.axiom_graph.get_all_embeddings()
        priority_weights = self.axiom_graph.get_priority_weights()
        
        # Select relevant axioms
        axiom_context, axiom_scores = self.axiom_selector(
            situation, axiom_embeddings, priority_weights
        )
        
        # Get top-k axioms (clamped to available axioms)
        k = min(10, axiom_embeddings.size(0))
        top_indices, selected_embeddings, selected_scores = \
            self.axiom_selector.priority_attn.select_top_k(
                situation, axiom_embeddings, priority_weights, k
            )
        
        # Construct chains with GNN
        adjacency_dict = {
            et: self.axiom_graph.get_adjacency(et)
            for et in self.axiom_graph.edge_types
        }
        weight_dict = {
            et: self.axiom_graph.weights[et]
            for et in self.axiom_graph.edge_types
        }
        
        # Process each batch item separately (GNN doesn't support batching)
        all_axiom_states = []
        all_activations = []
        for i in range(batch_size):
            axiom_states, activations = self.axiom_gnn(
                axiom_embeddings,
                adjacency_dict,
                weight_dict,
                initial_activations=axiom_scores[i]
            )
            all_axiom_states.append(axiom_states)
            all_activations.append(activations)
        
        # Stack results
        # axiom_states stays [num_axioms, d_axiom] - same for all batch items
        axiom_states = all_axiom_states[0]
        activations = torch.stack(all_activations)  # [batch_size, num_axioms]
        
        # Extract chains
        chains = []
        for i in range(batch_size):
            chain = self.axiom_gnn.extract_chain(
                activations[i] if len(activations.shape) > 1 else activations,
                adjacency_dict,
                max_length=10,
                threshold=0.5
            )
            chains.append(chain)
        
        # Execute chains - process each batch item
        outputs_axiom = []
        all_intermediates = []
        for i in range(batch_size):
            # Use activations as soft weights for chain steps
            chain_weights = activations[i].unsqueeze(0).repeat(10, 1)  # [num_steps, num_axioms]
            
            output_axiom, intermediates = self.chain_executor.execute_soft_chain(
                situation[i:i+1], chain_weights
            )
            outputs_axiom.append(output_axiom)
            all_intermediates.append(intermediates)
        
        output_axiom = torch.cat(outputs_axiom, dim=0)  # [batch_size, d_output]
        
        # Verify agreement
        agreement = self.verifier(
            torch.cat([output_axiom, output_fast], dim=-1)
        ).squeeze(-1)
        
        # Select output based on route
        output = torch.where(
            route.unsqueeze(-1) == 1,
            output_axiom,
            output_fast
        )
        
        result = {
            'output': output,
            'route': route,
            'agreement': agreement,
            'scores': routing_scores,
            'output_fast': output_fast,
            'output_axiom': output_axiom,
            'axiom_scores': axiom_scores
        }
        
        if return_explanation:
            result['axiom_chains'] = chains
            result['intermediates'] = intermediates
        
        return result
    
    def explain(self, x: torch.Tensor) -> Dict:
        """
        Get detailed explanation for input.
        
        Args:
            x: [1, d_input] single input
            
        Returns:
            dict: Comprehensive explanation
        """
        result = self.forward(x, return_explanation=True, force_axiom_path=True)
        
        # Convert to interpretable format
        explanation = {
            'input': x.detach().cpu(),
            'situation_embedding': self.encoder(x).detach().cpu(),
            'route_decision': 'axiom' if result['route'].item() == 1 else 'fast',
            'routing_scores': {
                k: v.detach().cpu().item() if torch.is_tensor(v) else v
                for k, v in result.get('scores', {}).items()
            },
            'axiom_chain': result.get('axiom_chains', [[]])[0],
            'output': result['output'].detach().cpu(),
            'agreement': result['agreement'].detach().cpu().item()
        }
        
        return explanation
