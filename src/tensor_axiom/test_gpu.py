"""
Test GPU utilization with tensor axiom system
"""

import torch
from tensor_axiom import (
    AxiomEmbedding, AxiomGraph, HybridModel
)

def test_gpu():
    """Test GPU acceleration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create axiom graph
    num_axioms = 50  # Larger for GPU test
    d_axiom = 512  # Divisible by 8
    graph = AxiomGraph(num_axioms, d_axiom)
    
    # Add some axioms
    for i in range(num_axioms):
        axiom = AxiomEmbedding(
            f'A{i}',
            semantic_dim=256,
            logic_dim=192,
            dependency_dim=64,
            priority=1.0 - (i / num_axioms)
        )
        graph.add_axiom(axiom, i)
    
    # Create model on GPU
    model = HybridModel(
        d_input=512,
        d_hidden=1024,
        d_output=512,
        d_axiom=d_axiom,
        num_axioms=num_axioms,
        axiom_modules={},  # Empty for this test
        axiom_graph=graph,
        edge_types=['implies', 'requires', 'contradicts', 'composes'],
        num_transformer_layers=6,
        num_gnn_layers=4
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test batch inference
    batch_sizes = [1, 4, 16, 32]
    
    for bs in batch_sizes:
        test_input = torch.randn(bs, 512).to(device)
        
        # Warm up
        _ = model(test_input, return_explanation=False)
        
        # Timed inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                output = model(test_input, return_explanation=False)
            end.record()
            
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            
            print(f"Batch {bs:2d}: {elapsed:6.2f}ms ({elapsed/bs:.2f}ms/sample) - "
                  f"Output: {output['output'].shape}")
        else:
            import time
            start = time.time()
            with torch.no_grad():
                output = model(test_input, return_explanation=False)
            elapsed = (time.time() - start) * 1000
            
            print(f"Batch {bs:2d}: {elapsed:6.2f}ms ({elapsed/bs:.2f}ms/sample) - "
                  f"Output: {output['output'].shape}")
    
    if torch.cuda.is_available():
        print(f"\nPeak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == '__main__':
    test_gpu()
