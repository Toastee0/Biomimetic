"""
Axiom Discovery

Learns new axioms from experience through pattern detection and validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
import numpy as np


class AxiomDiscovery(nn.Module):
    """
    Discovers new axioms from experience patterns.
    
    Uses a VAE-like architecture to find invariant patterns across
    similar experiences, then validates them as candidate axioms.
    """
    
    def __init__(
        self,
        d_experience: int,
        d_pattern: int,
        d_axiom: int,
        num_clusters: int = 5,
        validation_threshold: float = 0.7
    ):
        super().__init__()
        
        self.d_experience = d_experience
        self.d_pattern = d_pattern
        self.d_axiom = d_axiom
        self.num_clusters = num_clusters
        self.validation_threshold = validation_threshold
        
        # Pattern encoder (finds invariant patterns)
        self.encoder = nn.Sequential(
            nn.Linear(d_experience, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, d_pattern)
        )
        
        # Pattern decoder (converts pattern to axiom embedding)
        self.decoder = nn.Sequential(
            nn.Linear(d_pattern, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, d_axiom)
        )
        
        # Axiom function generator
        self.function_generator = nn.Sequential(
            nn.Linear(d_axiom, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Pattern memory (stores seen patterns)
        self.register_buffer('pattern_memory', torch.zeros(1000, d_pattern))
        self.register_buffer('pattern_count', torch.tensor(0))
    
    def encode_experiences(self, experiences: torch.Tensor) -> torch.Tensor:
        """
        Encode experiences to pattern space.
        
        Args:
            experiences: [num_experiences, d_experience]
            
        Returns:
            torch.Tensor: [num_experiences, d_pattern]
        """
        return self.encoder(experiences)
    
    def cluster_patterns(
        self,
        patterns: torch.Tensor,
        num_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster patterns to find invariant groups.
        
        Args:
            patterns: [num_patterns, d_pattern]
            num_clusters: Number of clusters (default: self.num_clusters)
            
        Returns:
            labels: Cluster assignments
            centroids: Cluster centers
        """
        if num_clusters is None:
            num_clusters = self.num_clusters
        
        patterns_np = patterns.detach().cpu().numpy()
        
        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(patterns_np)
        centroids = kmeans.cluster_centers_
        
        return labels, centroids
    
    def pattern_to_axiom(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Convert pattern to axiom embedding.
        
        Args:
            pattern: [d_pattern] or [batch_size, d_pattern]
            
        Returns:
            torch.Tensor: Axiom embedding
        """
        return self.decoder(pattern)
    
    def validate_axiom(
        self,
        axiom_embedding: torch.Tensor,
        test_cases: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Validate axiom on held-out test cases.
        
        Args:
            axiom_embedding: [d_axiom] proposed axiom
            test_cases: List of (input, output) pairs
            
        Returns:
            float: Accuracy score (0-1)
        """
        if len(test_cases) == 0:
            return 0.0
        
        # Generate axiom function from embedding
        axiom_function = self.function_generator(axiom_embedding)
        
        correct = 0
        total = len(test_cases)
        
        with torch.no_grad():
            for input_x, target_y in test_cases:
                # Predict using axiom
                # This is simplified - in practice would be more complex
                pred = F.linear(input_x, axiom_function)
                
                # Check if close to target
                error = F.mse_loss(pred, target_y)
                if error < 0.1:  # Threshold for "correct"
                    correct += 1
        
        accuracy = correct / total
        return accuracy
    
    def discover(
        self,
        experiences: torch.Tensor,
        test_cases: List[Tuple[torch.Tensor, torch.Tensor]],
        min_cluster_size: int = 10
    ) -> List[Dict]:
        """
        Discover new axioms from experiences.
        
        Args:
            experiences: [num_experiences, d_experience] batch of similar experiences
            test_cases: Validation cases
            min_cluster_size: Minimum size for viable cluster
            
        Returns:
            list: Candidate axioms with metadata
        """
        # Encode to pattern space
        patterns = self.encode_experiences(experiences)
        
        # Cluster patterns
        labels, centroids = self.cluster_patterns(patterns)
        
        # Convert centroids to tensor
        centroids_tensor = torch.tensor(
            centroids,
            dtype=torch.float32,
            device=patterns.device
        )
        
        # Generate candidate axioms
        candidates = []
        
        for cluster_id in range(len(centroids)):
            # Get cluster members
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size < min_cluster_size:
                continue
            
            # Cluster experiences
            cluster_experiences = experiences[cluster_mask]
            
            # Pattern centroid
            pattern = centroids_tensor[cluster_id]
            
            # Generate axiom embedding
            axiom_embedding = self.pattern_to_axiom(pattern)
            
            # Validate on test cases
            accuracy = self.validate_axiom(axiom_embedding, test_cases)
            
            if accuracy >= self.validation_threshold:
                # Good axiom candidate!
                candidate = {
                    'embedding': axiom_embedding,
                    'pattern': pattern,
                    'confidence': accuracy,
                    'support': int(cluster_size),
                    'cluster_id': cluster_id,
                    'experiences': cluster_experiences
                }
                candidates.append(candidate)
        
        # Update pattern memory
        self._update_memory(patterns)
        
        return candidates
    
    def _update_memory(self, new_patterns: torch.Tensor):
        """Update pattern memory with new patterns"""
        num_new = new_patterns.size(0)
        current_count = self.pattern_count.item()
        memory_size = self.pattern_memory.size(0)
        
        if current_count + num_new <= memory_size:
            # Fits in memory
            self.pattern_memory[current_count:current_count + num_new] = new_patterns
            self.pattern_count.fill_(current_count + num_new)
        else:
            # Memory full, use reservoir sampling
            for pattern in new_patterns:
                if current_count < memory_size:
                    self.pattern_memory[current_count] = pattern
                    current_count += 1
                else:
                    # Random replacement
                    idx = torch.randint(0, current_count + 1, (1,)).item()
                    if idx < memory_size:
                        self.pattern_memory[idx] = pattern
                    current_count += 1
            
            self.pattern_count.fill_(min(current_count, memory_size))
    
    def compute_novelty(self, experience: torch.Tensor) -> float:
        """
        Compute how novel an experience is compared to memory.
        
        Args:
            experience: [d_experience] single experience
            
        Returns:
            float: Novelty score (0-1)
        """
        pattern = self.encode_experiences(experience.unsqueeze(0))
        
        # Compare to memory
        if self.pattern_count == 0:
            return 1.0  # Everything is novel if no memory
        
        memory = self.pattern_memory[:self.pattern_count]
        
        # Find closest pattern in memory
        distances = torch.norm(memory - pattern, dim=-1)
        min_distance = distances.min()
        
        # Convert distance to novelty score
        novelty = torch.sigmoid(min_distance - 1.0).item()
        
        return novelty
    
    def merge_similar_axioms(
        self,
        axiom_embeddings: List[torch.Tensor],
        similarity_threshold: float = 0.9
    ) -> List[torch.Tensor]:
        """
        Merge similar axioms to avoid redundancy.
        
        Args:
            axiom_embeddings: List of axiom embeddings
            similarity_threshold: Cosine similarity threshold for merging
            
        Returns:
            list: Merged axiom embeddings
        """
        if len(axiom_embeddings) <= 1:
            return axiom_embeddings
        
        # Compute pairwise similarities
        embeddings = torch.stack(axiom_embeddings)
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=-1
        )
        
        # Merge similar axioms
        merged = []
        merged_indices = set()
        
        for i in range(len(axiom_embeddings)):
            if i in merged_indices:
                continue
            
            # Find all similar axioms
            similar = (similarities[i] > similarity_threshold).nonzero().squeeze(-1)
            
            # Average embeddings
            averaged = embeddings[similar].mean(dim=0)
            merged.append(averaged)
            
            # Mark as merged
            for j in similar:
                merged_indices.add(j.item())
        
        return merged
