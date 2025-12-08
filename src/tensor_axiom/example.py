"""
Example: Basic Tensor Axiom System

Demonstrates creating a simple axiom graph and running hybrid inference.
"""

import torch
import torch.nn as nn
from tensor_axiom import (
    AxiomEmbedding,
    AxiomGraph,
    AxiomModule,
    HybridModel
)


def create_simple_axiom_graph():
    """Create a simple axiom graph with a few test axioms"""
    
    # Create axiom graph
    num_axioms = 5
    d_axiom = 448  # 256 + 128 + 64 (divisible by 8)
    graph = AxiomGraph(num_axioms, d_axiom)
    
    # Define some example axioms
    axioms = {
        'A1': AxiomEmbedding(
            'A1',
            semantic_dim=256,
            logic_dim=128,
            dependency_dim=64,
            priority=1.0,  # Meta-axiom
            description="Kindness over correctness in social contexts"
        ),
        'A2': AxiomEmbedding(
            'A2',
            semantic_dim=256,
            logic_dim=128,
            dependency_dim=64,
            priority=0.7,
            description="Assess threat based on indicators"
        ),
        'A3': AxiomEmbedding(
            'A3',
            semantic_dim=256,
            logic_dim=128,
            dependency_dim=64,
            priority=0.6,
            description="Graduated response based on threat level"
        ),
        'A4': AxiomEmbedding(
            'A4',
            semantic_dim=256,
            logic_dim=128,
            dependency_dim=64,
            priority=0.8,
            description="Sensory accommodation for comfort"
        ),
        'A5': AxiomEmbedding(
            'A5',
            semantic_dim=256,
            logic_dim=128,
            dependency_dim=64,
            priority=0.5,
            description="Pattern recognition in behavior"
        ),
    }
    
    # Add axioms to graph
    for idx, (axiom_id, axiom) in enumerate(axioms.items()):
        graph.add_axiom(axiom, idx)
    
    # Add edges (relationships between axioms)
    # A1 (kindness) implies A4 (sensory accommodation)
    graph.add_edge('A1', 'A4', 'implies', weight=0.8)
    
    # A2 (threat assessment) implies A3 (graduated response)
    graph.add_edge('A2', 'A3', 'implies', weight=0.9)
    
    # A1 can contradict A2 in social contexts (meta-axiom dominance)
    graph.add_edge('A1', 'A2', 'contradicts', weight=0.7)
    
    # A5 requires A2 (pattern recognition needs threat assessment)
    graph.add_edge('A5', 'A2', 'requires', weight=0.6)
    
    return graph, axioms


def create_simple_axiom_modules():
    """Create executable axiom modules"""
    
    d_input = 512
    d_output = 512
    
    modules = {}
    
    # A1: Kindness boost
    def kindness_formula(x):
        # Boost positive sentiment, reduce negative
        return x * 1.2
    
    modules['A1'] = AxiomModule(
        'A1',
        d_input, d_output,
        symbolic_formula=kindness_formula,
        description="Apply kindness principle"
    )
    
    # A2: Threat assessment
    def threat_formula(x):
        # Simplified threat assessment
        return x * 0.8  # Reduce overreaction
    
    modules['A2'] = AxiomModule(
        'A2',
        d_input, d_output,
        symbolic_formula=threat_formula,
        description="Assess threat level"
    )
    
    # A3: Graduated response
    def response_formula(x):
        # Scale response to input
        return torch.clamp(x, min=0, max=1)
    
    modules['A3'] = AxiomModule(
        'A3',
        d_input, d_output,
        symbolic_formula=response_formula,
        description="Apply graduated response"
    )
    
    # A4: Sensory accommodation
    modules['A4'] = AxiomModule(
        'A4',
        d_input, d_output,
        description="Sensory accommodation"
    )
    
    # A5: Pattern recognition
    modules['A5'] = AxiomModule(
        'A5',
        d_input, d_output,
        description="Pattern recognition"
    )
    
    return modules


def example_inference():
    """Run example inference through hybrid model"""
    
    print("Creating axiom graph...")
    graph, axioms = create_simple_axiom_graph()
    
    print("Creating axiom modules...")
    axiom_modules = create_simple_axiom_modules()
    
    print("Building hybrid model...")
    model = HybridModel(
        d_input=512,
        d_hidden=512,
        d_output=256,
        d_axiom=448,  # Must be divisible by num_heads (8)
        num_axioms=5,
        axiom_modules=axiom_modules,
        axiom_graph=graph,
        edge_types=['implies', 'requires', 'contradicts', 'composes'],
        num_transformer_layers=4,
        num_gnn_layers=3
    )
    
    print("\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create a test situation
    print("\nRunning inference on test situation...")
    test_input = torch.randn(1, 512)
    
    with torch.no_grad():
        result = model(test_input, return_explanation=True)
    
    print(f"\nResults:")
    print(f"  Route taken: {'Axiom' if result['route'].item() == 1 else 'Fast'}")
    print(f"  Agreement score: {result['agreement'].item():.3f}")
    print(f"  Output shape: {result['output'].shape}")
    
    if 'axiom_chains' in result and len(result['axiom_chains'][0]) > 0:
        print(f"  Axiom chain: {result['axiom_chains'][0]}")
    
    # Get detailed explanation
    print("\nGetting detailed explanation...")
    explanation = model.explain(test_input)
    print(f"  Situation embedding norm: {explanation['situation_embedding'].norm().item():.3f}")
    print(f"  Route decision: {explanation['route_decision']}")
    
    return model, graph, axiom_modules


def visualize_graph():
    """Visualize the axiom graph structure"""
    
    graph, axioms = create_simple_axiom_graph()
    
    print("\nAxiom Graph Structure:")
    print("=" * 50)
    
    for idx in range(graph.num_axioms):
        axiom_id = graph.idx_to_axiom_id[idx]
        axiom = graph.axiom_embeddings[axiom_id]
        
        print(f"\n{axiom_id}: {axiom.description}")
        print(f"  Priority: {axiom.priority.item():.2f}")
        print(f"  Confidence: {axiom.confidence.item():.2f}")
        
        # Show connections
        for edge_type in graph.edge_types:
            neighbors = graph.get_neighbors(idx, edge_type, direction='outgoing')
            if len(neighbors) > 0:
                neighbor_ids = [graph.idx_to_axiom_id[n.item()] for n in neighbors]
                print(f"  {edge_type}: {neighbor_ids}")


if __name__ == '__main__':
    print("Tensor Axiom System - Basic Example")
    print("=" * 50)
    
    # Visualize graph
    visualize_graph()
    
    # Run inference
    print("\n" + "=" * 50)
    model, graph, modules = example_inference()
    
    print("\n✓ Example completed successfully!")
