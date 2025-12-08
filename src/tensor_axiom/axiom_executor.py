"""
Axiom Execution and Chain Processing

Implements axiom modules as differentiable functions and chains their execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any, Tuple
import json


class AxiomModule(nn.Module):
    """
    Represents an axiom as a learnable differentiable function.
    
    Maintains both:
    - Neural approximation (learned, differentiable)
    - Symbolic formula (exact, for verification)
    """
    
    def __init__(
        self,
        axiom_id: str,
        d_input: int,
        d_output: int,
        d_hidden: int = 128,
        symbolic_formula: Optional[Callable] = None,
        description: str = ""
    ):
        super().__init__()
        
        self.axiom_id = axiom_id
        self.description = description
        self.symbolic_formula = symbolic_formula
        
        # Neural approximation
        self.neural_f = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_output)
        )
        
        # Confidence in neural vs symbolic
        self.register_buffer('neural_confidence', torch.tensor(0.5))
    
    def forward(
        self,
        x: torch.Tensor,
        use_symbolic: bool = False
    ) -> torch.Tensor:
        """
        Execute axiom function.
        
        Args:
            x: Input tensor [batch_size, d_input]
            use_symbolic: If True and available, use symbolic formula
            
        Returns:
            torch.Tensor: [batch_size, d_output]
        """
        if use_symbolic and self.symbolic_formula is not None:
            # Use exact symbolic formula
            return self.symbolic_formula(x)
        else:
            # Use neural approximation
            return self.neural_f(x)
    
    def symbolic_execute(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Execute symbolic formula if available"""
        if self.symbolic_formula is not None:
            return self.symbolic_formula(x)
        return None
    
    def neural_execute(self, x: torch.Tensor) -> torch.Tensor:
        """Execute neural approximation"""
        return self.neural_f(x)
    
    def compute_agreement(self, x: torch.Tensor) -> float:
        """
        Measure agreement between neural and symbolic.
        
        Args:
            x: Input tensor
            
        Returns:
            float: Agreement score (0-1)
        """
        if self.symbolic_formula is None:
            return 1.0
        
        with torch.no_grad():
            y_neural = self.neural_execute(x)
            y_symbolic = self.symbolic_execute(x)
            
            # Compute similarity (cosine or MSE)
            mse = F.mse_loss(y_neural, y_symbolic)
            agreement = torch.exp(-mse).item()
        
        return agreement
    
    def update_confidence(self, x: torch.Tensor):
        """Update neural confidence based on agreement"""
        agreement = self.compute_agreement(x)
        self.neural_confidence.fill_(agreement)


class ChainExecutor(nn.Module):
    """
    Executes a chain of axioms in sequence.
    
    Takes a situation and a chain of axiom modules, applies them in order,
    and produces the final output with full differentiability.
    """
    
    def __init__(
        self,
        axiom_modules: Dict[str, AxiomModule],
        d_situation: int,
        d_output: int,
        residual: bool = True
    ):
        super().__init__()
        
        self.axiom_modules = nn.ModuleDict(axiom_modules)
        self.d_situation = d_situation
        self.d_output = d_output
        self.residual = residual
        
        # Output projection
        self.output_proj = nn.Linear(d_situation, d_output)
        
        # Chain state tracker
        self.state_proj = nn.Linear(d_situation, d_situation)
    
    def execute_chain(
        self,
        situation: torch.Tensor,
        chain: List[str],
        use_symbolic: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute axiom chain sequentially.
        
        Args:
            situation: [batch_size, d_situation] initial situation
            chain: List of axiom IDs to execute in order
            use_symbolic: Whether to use symbolic formulas
            
        Returns:
            output: [batch_size, d_output] final output
            intermediates: List of intermediate states
        """
        state = situation
        intermediates = [state]
        
        for axiom_id in chain:
            if axiom_id not in self.axiom_modules:
                # Skip unknown axioms
                continue
            
            axiom = self.axiom_modules[axiom_id]
            
            # Apply axiom
            delta = axiom(state, use_symbolic=use_symbolic)
            
            # Update state (with optional residual connection)
            if self.residual:
                state = state + delta
            else:
                state = delta
            
            # Project to maintain dimensionality
            state = self.state_proj(state)
            
            intermediates.append(state)
        
        # Final output projection
        output = self.output_proj(state)
        
        return output, intermediates
    
    def execute_soft_chain(
        self,
        situation: torch.Tensor,
        chain_weights: torch.Tensor,
        use_symbolic: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute axiom chain with soft weights (differentiable).
        
        Args:
            situation: [batch_size, d_situation]
            chain_weights: [num_steps, num_axioms] soft selection of axioms
            use_symbolic: Whether to use symbolic formulas
            
        Returns:
            output: [batch_size, d_output]
            intermediates: List of intermediate states
        """
        batch_size = situation.size(0)
        num_steps, num_axioms = chain_weights.shape
        
        state = situation
        intermediates = [state]
        
        axiom_ids = list(self.axiom_modules.keys())
        
        for step in range(num_steps):
            # Soft selection of axioms for this step
            step_weights = chain_weights[step]  # [num_axioms]
            
            # Compute weighted combination of axiom outputs
            deltas = []
            for i, axiom_id in enumerate(axiom_ids):
                if i >= num_axioms:
                    break
                
                axiom = self.axiom_modules[axiom_id]
                delta = axiom(state, use_symbolic=use_symbolic)
                deltas.append(delta * step_weights[i])
            
            # Weighted sum
            if len(deltas) > 0:
                combined_delta = torch.stack(deltas).sum(dim=0)
            else:
                combined_delta = torch.zeros_like(state)
            
            # Update state
            if self.residual:
                state = state + combined_delta
            else:
                state = combined_delta
            
            state = self.state_proj(state)
            intermediates.append(state)
        
        # Final output
        output = self.output_proj(state)
        
        return output, intermediates
    
    def explain_chain(
        self,
        chain: List[str],
        situation: torch.Tensor,
        use_symbolic: bool = False
    ) -> Dict[str, Any]:
        """
        Execute chain and return detailed explanation.
        
        Args:
            chain: List of axiom IDs
            situation: Input situation
            use_symbolic: Whether to use symbolic formulas
            
        Returns:
            dict: Explanation with axiom descriptions and intermediate values
        """
        output, intermediates = self.execute_chain(situation, chain, use_symbolic)
        
        explanation = {
            'chain': chain,
            'steps': [],
            'final_output': output.detach().cpu()
        }
        
        for i, axiom_id in enumerate(chain):
            if axiom_id not in self.axiom_modules:
                continue
            
            axiom = self.axiom_modules[axiom_id]
            
            step_info = {
                'axiom_id': axiom_id,
                'description': axiom.description,
                'input_state': intermediates[i].detach().cpu(),
                'output_state': intermediates[i + 1].detach().cpu() if i + 1 < len(intermediates) else None,
            }
            
            # Add symbolic formula if available
            if axiom.symbolic_formula is not None:
                step_info['has_symbolic'] = True
                step_info['agreement'] = axiom.compute_agreement(intermediates[i])
            else:
                step_info['has_symbolic'] = False
            
            explanation['steps'].append(step_info)
        
        return explanation


class ParallelChainExecutor(nn.Module):
    """
    Executes multiple axiom chains in parallel (batched).
    
    Efficient GPU-parallel execution of different reasoning chains
    for different situations.
    """
    
    def __init__(
        self,
        axiom_modules: Dict[str, AxiomModule],
        d_situation: int,
        d_output: int,
        max_chain_length: int = 10
    ):
        super().__init__()
        
        self.executor = ChainExecutor(axiom_modules, d_situation, d_output)
        self.max_chain_length = max_chain_length
    
    def forward(
        self,
        situations: torch.Tensor,
        chains: List[List[str]],
        use_symbolic: bool = False
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Execute multiple chains in parallel.
        
        Args:
            situations: [batch_size, d_situation]
            chains: List of chains (one per batch element)
            use_symbolic: Whether to use symbolic formulas
            
        Returns:
            outputs: [batch_size, d_output]
            all_intermediates: List of intermediate states per chain
        """
        batch_size = situations.size(0)
        
        # Pad chains to same length
        padded_chains = []
        for chain in chains:
            padded = chain + [''] * (self.max_chain_length - len(chain))
            padded = padded[:self.max_chain_length]
            padded_chains.append(padded)
        
        # Execute each chain
        outputs = []
        all_intermediates = []
        
        for i in range(batch_size):
            situation = situations[i:i+1]
            chain = [ax for ax in padded_chains[i] if ax != '']
            
            output, intermediates = self.executor.execute_chain(
                situation, chain, use_symbolic
            )
            
            outputs.append(output)
            all_intermediates.append(intermediates)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=0)
        
        return outputs, all_intermediates
