"""
Axiom Refinement Loop

Automatically tests axioms, identifies issues, and queues for human review.
"""

import torch
from typing import Optional
from pathlib import Path

from .axiom_library import AxiomLibrary
from .self_test import AxiomTester
from .review_queue import ReviewQueue


class RefinementLoop:
    """Manages the continuous refinement process"""
    
    def __init__(
        self,
        library_path: str = "data/axioms/base_axioms.json",
        queue_path: str = "data/axioms/review_queue.json",
        model=None
    ):
        """
        Initialize the refinement loop.
        
        Args:
            library_path: Path to axiom library
            queue_path: Path to review queue
            model: Optional HybridModel for testing
        """
        self.library = AxiomLibrary(library_path)
        self.tester = AxiomTester(self.library, model)
        self.queue = ReviewQueue(queue_path, self.library)
    
    def run_test_cycle(self, save_results: bool = True):
        """
        Run a complete test cycle on all axioms.
        
        Args:
            save_results: Whether to save results to library
        """
        print("\n" + "=" * 70)
        print("STARTING REFINEMENT CYCLE")
        print("=" * 70)
        
        # Test all axioms
        results = self.tester.test_all()
        
        # Identify problems
        failed_tests = self.tester.get_failed_tests()
        low_confidence = self.tester.get_low_confidence_tests(threshold=0.5)
        
        print(f"\nFailed tests: {len(failed_tests)}")
        print(f"Low confidence tests: {len(low_confidence)}")
        
        # Group failures by axiom
        failures_by_axiom = {}
        for test in failed_tests:
            if test.axiom_id not in failures_by_axiom:
                failures_by_axiom[test.axiom_id] = []
            failures_by_axiom[test.axiom_id].append(test)
        
        # Add to review queue
        for axiom_id, failures in failures_by_axiom.items():
            reason = f"Failed {len(failures)} test scenarios"
            self.queue.add_for_review(axiom_id, reason, failures)
        
        # Scan for other issues
        self.queue.scan_library_for_issues()
        
        # Save everything
        if save_results:
            self.library.save()
            self.queue.save()
        
        # Generate report
        stats = self.library.get_stats()
        queue_report = self.queue.generate_report()
        
        print("\n" + "=" * 70)
        print("REFINEMENT CYCLE COMPLETE")
        print("=" * 70)
        print(f"\nLibrary Stats:")
        print(f"  Total axioms: {stats['total_axioms']}")
        print(f"  Average success: {stats['average_success_rate']:.1%}")
        print(f"\nReview Queue:")
        print(f"  Pending reviews: {queue_report['pending']}")
        print(f"  High priority: {queue_report['high_priority']}")
        
        return {
            'library_stats': stats,
            'queue_report': queue_report,
            'failed_tests': len(failed_tests),
            'low_confidence': len(low_confidence)
        }
    
    def display_review_queue(self, limit: int = 10):
        """Display the review queue"""
        self.queue.display_queue(limit)
    
    def review_axiom(self, axiom_id: str):
        """
        Display detailed information for reviewing an axiom.
        
        Args:
            axiom_id: ID of axiom to review
        """
        axiom = self.library.get_axiom(axiom_id)
        if not axiom:
            print(f"Axiom {axiom_id} not found")
            return
        
        print("\n" + "=" * 70)
        print(f"AXIOM REVIEW: {axiom_id}")
        print("=" * 70)
        print(f"\nName: {axiom['name']}")
        print(f"Description: {axiom['description']}")
        print(f"Category: {axiom['category']}")
        print(f"Priority: {axiom['priority']}")
        print(f"Confidence: {axiom['confidence']}")
        
        print(f"\nSemantic Tags:")
        for tag in axiom.get('semantic_tags', []):
            print(f"  - {tag}")
        
        print(f"\nRelationships:")
        relationships = axiom.get('edge_relationships', {})
        for rel_type, targets in relationships.items():
            if targets:
                print(f"  {rel_type}: {', '.join(targets)}")
        
        print(f"\nPerformance Metrics:")
        metrics = axiom.get('performance_metrics', {})
        print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
        print(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.2f}")
        print(f"  Test Count: {metrics.get('test_count', 0)}")
        print(f"  Human Approval: {metrics.get('human_approval_rate', 0):.1%}")
        print(f"  Last Tested: {metrics.get('last_tested', 'Never')}")
        
        print(f"\nTest Scenarios:")
        scenarios = axiom.get('test_scenarios', [])
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n  Scenario {i}:")
            print(f"    Input: {scenario.get('input', '')}")
            print(f"    Expected: {scenario.get('expected_behavior', '')}")
            print(f"    Success Criteria: {scenario.get('success_criteria', '')}")
    
    def approve_axiom(self, axiom_id: str, notes: str = ""):
        """Approve an axiom after review"""
        self.queue.update_status(axiom_id, "approved", notes)
        self.queue.save()
        print(f"✓ Approved: {axiom_id}")
    
    def reject_axiom(self, axiom_id: str, notes: str = ""):
        """Reject an axiom after review"""
        self.queue.update_status(axiom_id, "rejected", notes)
        self.queue.save()
        print(f"✗ Rejected: {axiom_id}")
