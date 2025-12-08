"""
Self-Testing Framework for Axioms

Tests axioms against scenarios and validates their behavior.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .axiom_library import AxiomLibrary
from .hybrid_model import HybridModel


@dataclass
class TestResult:
    """Result of a single test"""
    axiom_id: str
    scenario_index: int
    success: bool
    confidence: float
    output: Any
    explanation: str
    timestamp: str


class AxiomTester:
    """Tests axioms against their defined scenarios"""
    
    def __init__(
        self,
        library: AxiomLibrary,
        model: Optional[HybridModel] = None
    ):
        """
        Initialize the tester.
        
        Args:
            library: AxiomLibrary instance
            model: Optional HybridModel for neural evaluation
        """
        self.library = library
        self.model = model
        self.test_results: List[TestResult] = []
    
    def encode_scenario(self, scenario_text: str, d_input: int = 512) -> torch.Tensor:
        """
        Encode a test scenario as a tensor.
        For now, uses simple hashing. Could be replaced with proper embeddings.
        
        Args:
            scenario_text: The scenario description
            d_input: Input dimension
            
        Returns:
            Tensor representation of the scenario
        """
        # Simple hash-based encoding (replace with proper embeddings later)
        hash_val = hash(scenario_text)
        torch.manual_seed(abs(hash_val) % (2**32))
        return torch.randn(1, d_input)
    
    def evaluate_scenario(
        self,
        axiom_id: str,
        scenario: Dict[str, str]
    ) -> TestResult:
        """
        Evaluate a single test scenario for an axiom.
        
        Args:
            axiom_id: ID of the axiom being tested
            scenario: Test scenario dictionary
            
        Returns:
            TestResult with evaluation
        """
        scenario_input = scenario.get('input', '')
        expected_behavior = scenario.get('expected_behavior', '')
        success_criteria = scenario.get('success_criteria', '')
        
        # Encode scenario
        situation = self.encode_scenario(scenario_input)
        
        # Get axiom data
        axiom_data = self.library.get_axiom(axiom_id)
        if not axiom_data:
            return TestResult(
                axiom_id=axiom_id,
                scenario_index=-1,
                success=False,
                confidence=0.0,
                output=None,
                explanation="Axiom not found",
                timestamp=datetime.now().isoformat()
            )
        
        # Check if model inference is available
        if self.model:
            # Use the hybrid model to evaluate
            with torch.no_grad():
                result = self.model(situation, return_explanation=True)
                confidence = result['agreement'].item()
                output = result['output']
        else:
            # Fallback: simple rule-based evaluation
            confidence = axiom_data.get('confidence', 0.5)
            output = f"Would apply {axiom_data['name']}"
        
        # Evaluate success based on criteria
        # For now, use confidence threshold
        success = confidence > 0.6
        
        explanation = f"Scenario: {scenario_input}\nExpected: {expected_behavior}\nConfidence: {confidence:.3f}"
        
        return TestResult(
            axiom_id=axiom_id,
            scenario_index=self.library.get_test_scenarios(axiom_id).index(scenario),
            success=success,
            confidence=confidence,
            output=output,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
    
    def test_axiom(self, axiom_id: str) -> List[TestResult]:
        """
        Test all scenarios for a specific axiom.
        
        Args:
            axiom_id: ID of axiom to test
            
        Returns:
            List of TestResults
        """
        scenarios = self.library.get_test_scenarios(axiom_id)
        results = []
        
        print(f"\nTesting axiom: {axiom_id}")
        print(f"Scenarios: {len(scenarios)}")
        
        for scenario in scenarios:
            result = self.evaluate_scenario(axiom_id, scenario)
            results.append(result)
            self.test_results.append(result)
            
            status = "✓" if result.success else "✗"
            print(f"  {status} Scenario {result.scenario_index + 1}: {result.confidence:.3f}")
        
        # Update metrics
        successes = sum(1 for r in results if r.success)
        success_rate = successes / len(results) if results else 0
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        
        axiom_data = self.library.get_axiom(axiom_id)
        current_test_count = axiom_data.get('performance_metrics', {}).get('test_count', 0)
        
        self.library.update_metrics(axiom_id, {
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'test_count': current_test_count + len(results)
        })
        
        return results
    
    def test_all(self) -> Dict[str, List[TestResult]]:
        """
        Test all axioms in the library.
        
        Returns:
            Dictionary mapping axiom IDs to their test results
        """
        print("=" * 60)
        print("AXIOM SELF-TEST")
        print("=" * 60)
        
        all_results = {}
        
        for axiom_id in self.library.axioms.keys():
            results = self.test_axiom(axiom_id)
            all_results[axiom_id] = results
        
        # Save updated metrics
        self.library.save()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        stats = self.library.get_stats()
        print(f"Total axioms: {stats['total_axioms']}")
        print(f"Tested: {stats['tested_axioms']}")
        print(f"Average success rate: {stats['average_success_rate']:.1%}")
        print(f"Axioms needing review: {stats['needs_review']}")
        
        return all_results
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get all failed test results"""
        return [r for r in self.test_results if not r.success]
    
    def get_low_confidence_tests(self, threshold: float = 0.5) -> List[TestResult]:
        """Get tests with confidence below threshold"""
        return [r for r in self.test_results if r.confidence < threshold]
