"""
Human Review Queue

Manages axioms that need human review and refinement.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .axiom_library import AxiomLibrary
from .self_test import TestResult


@dataclass
class ReviewItem:
    """Item in the review queue"""
    axiom_id: str
    axiom_name: str
    reason: str
    priority: float  # 0-1, higher = more urgent
    failed_tests: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    timestamp: str
    status: str = "pending"  # pending, approved, rejected, modified
    reviewer_notes: str = ""
    
    def to_dict(self):
        return asdict(self)


class ReviewQueue:
    """Manages human review queue for axioms"""
    
    def __init__(
        self,
        queue_path: str = "data/axioms/review_queue.json",
        library: Optional[AxiomLibrary] = None
    ):
        """
        Initialize review queue.
        
        Args:
            queue_path: Path to save queue data
            library: Optional AxiomLibrary instance
        """
        self.queue_path = Path(queue_path)
        self.library = library
        self.queue: List[ReviewItem] = []
        
        self.load()
    
    def load(self):
        """Load queue from JSON"""
        if self.queue_path.exists():
            with open(self.queue_path, 'r') as f:
                data = json.load(f)
                self.queue = [ReviewItem(**item) for item in data]
            print(f"Loaded {len(self.queue)} items from review queue")
        else:
            self.queue = []
    
    def save(self):
        """Save queue to JSON"""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.queue_path, 'w') as f:
            json.dump([item.to_dict() for item in self.queue], f, indent=2)
        
        print(f"Saved {len(self.queue)} items to review queue")
    
    def add_for_review(
        self,
        axiom_id: str,
        reason: str,
        failed_tests: Optional[List[TestResult]] = None,
        priority: Optional[float] = None
    ):
        """
        Add an axiom to the review queue.
        
        Args:
            axiom_id: ID of the axiom
            reason: Reason for review
            failed_tests: Optional list of failed test results
            priority: Optional priority override
        """
        if not self.library:
            raise ValueError("Library not set, cannot add to review queue")
        
        axiom_data = self.library.get_axiom(axiom_id)
        if not axiom_data:
            print(f"Warning: Axiom {axiom_id} not found")
            return
        
        # Check if already in queue
        if any(item.axiom_id == axiom_id and item.status == "pending" for item in self.queue):
            print(f"Axiom {axiom_id} already in review queue")
            return
        
        # Calculate priority if not provided
        if priority is None:
            metrics = axiom_data.get('performance_metrics', {})
            success_rate = metrics.get('success_rate', 1.0)
            axiom_priority = axiom_data.get('priority', 0.5)
            # Higher axiom priority + lower success rate = higher review priority
            priority = axiom_priority * (1 - success_rate)
        
        # Convert failed tests
        failed_test_dicts = []
        if failed_tests:
            for test in failed_tests:
                failed_test_dicts.append({
                    'scenario_index': test.scenario_index,
                    'confidence': test.confidence,
                    'explanation': test.explanation,
                    'timestamp': test.timestamp
                })
        
        item = ReviewItem(
            axiom_id=axiom_id,
            axiom_name=axiom_data.get('name', axiom_id),
            reason=reason,
            priority=priority,
            failed_tests=failed_test_dicts,
            metrics=axiom_data.get('performance_metrics', {}),
            timestamp=datetime.now().isoformat()
        )
        
        self.queue.append(item)
        print(f"Added {axiom_id} to review queue (priority: {priority:.2f})")
    
    def scan_library_for_issues(self):
        """
        Scan the library for axioms needing review.
        """
        if not self.library:
            raise ValueError("Library not set")
        
        print("\nScanning library for issues...")
        
        # Get axioms with poor performance
        needs_review = self.library.get_axioms_needing_review(threshold=0.6)
        
        for axiom_id in needs_review:
            axiom_data = self.library.get_axiom(axiom_id)
            metrics = axiom_data.get('performance_metrics', {})
            success_rate = metrics.get('success_rate', 0)
            
            reason = f"Low success rate: {success_rate:.1%} (threshold: 60%)"
            self.add_for_review(axiom_id, reason)
        
        print(f"Found {len(needs_review)} axioms needing review")
    
    def get_pending(self) -> List[ReviewItem]:
        """Get all pending review items, sorted by priority"""
        pending = [item for item in self.queue if item.status == "pending"]
        return sorted(pending, key=lambda x: x.priority, reverse=True)
    
    def get_by_status(self, status: str) -> List[ReviewItem]:
        """Get items by status"""
        return [item for item in self.queue if item.status == status]
    
    def update_status(
        self,
        axiom_id: str,
        status: str,
        notes: str = ""
    ):
        """
        Update the review status of an axiom.
        
        Args:
            axiom_id: ID of the axiom
            status: New status (pending, approved, rejected, modified)
            notes: Reviewer notes
        """
        for item in self.queue:
            if item.axiom_id == axiom_id and item.status == "pending":
                item.status = status
                item.reviewer_notes = notes
                print(f"Updated {axiom_id} status to: {status}")
                return
        
        print(f"No pending review found for {axiom_id}")
    
    def display_queue(self, limit: int = 10):
        """
        Display the review queue in a readable format.
        
        Args:
            limit: Maximum number of items to display
        """
        pending = self.get_pending()
        
        print("\n" + "=" * 70)
        print("HUMAN REVIEW QUEUE")
        print("=" * 70)
        
        if not pending:
            print("Queue is empty!")
            return
        
        for i, item in enumerate(pending[:limit]):
            print(f"\n[{i+1}] {item.axiom_name} (ID: {item.axiom_id})")
            print(f"    Priority: {item.priority:.2f}")
            print(f"    Reason: {item.reason}")
            print(f"    Success Rate: {item.metrics.get('success_rate', 0):.1%}")
            print(f"    Tests: {item.metrics.get('test_count', 0)}")
            print(f"    Failed Tests: {len(item.failed_tests)}")
            print(f"    Queued: {item.timestamp[:10]}")
        
        if len(pending) > limit:
            print(f"\n... and {len(pending) - limit} more items")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of review queue"""
        pending = self.get_pending()
        
        return {
            'total_items': len(self.queue),
            'pending': len(pending),
            'approved': len(self.get_by_status('approved')),
            'rejected': len(self.get_by_status('rejected')),
            'modified': len(self.get_by_status('modified')),
            'high_priority': len([i for i in pending if i.priority > 0.7]),
            'oldest_pending': pending[0].timestamp if pending else None,
            'newest_pending': pending[-1].timestamp if pending else None
        }
