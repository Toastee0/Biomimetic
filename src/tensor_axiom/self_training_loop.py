#!/usr/bin/env python3
"""
Self-Training Loop for Axiom Cortex

Autonomous training system that:
1. Tests axioms against scenarios using LLM inference
2. Evaluates axiom performance 
3. Generates improvement suggestions
4. Queues problematic axioms for human review
5. Discovers new axiom patterns from test results
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.daemon.textgen_client import TextGenClient
from src.tensor_axiom.axiom_library import AxiomLibrary
from src.tensor_axiom.review_queue import ReviewQueue
from src.core.prompts import (
    get_axiom_evaluation_prompt,
    get_axiom_refinement_prompt,
    get_axiom_improvement_prompt
)


class SelfTrainingLoop:
    """Autonomous training loop for axiom system"""
    
    def __init__(
        self,
        library_path: str = "data/axioms/base_axioms.json",
        queue_path: str = "data/axioms/review_queue.json",
        training_log_path: str = "logs/self_training.jsonl"
    ):
        """Initialize self-training system"""
        self.library = AxiomLibrary(library_path)
        self.queue = ReviewQueue(queue_path, self.library)
        self.textgen = TextGenClient()
        self.training_log_path = training_log_path
        
        # Ensure log directory exists
        Path(training_log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.iteration = 0
        self.discoveries = []
        
    def log_training_event(self, event_type: str, data: Dict):
        """Log training event to JSONL file"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration,
            "event_type": event_type,
            "data": data
        }
        
        with open(self.training_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def evaluate_axiom_with_llm(
        self,
        axiom_id: str,
        scenario: Dict
    ) -> Tuple[bool, float, str]:
        """
        Evaluate an axiom scenario using LLM inference.
        
        Args:
            axiom_id: ID of axiom being tested
            scenario: Test scenario with input, expected_behavior, success_criteria
            
        Returns:
            (success, confidence, reasoning)
        """
        axiom = self.library.axioms[axiom_id]
        
        # Build evaluation prompt
        prompt = f"""You are evaluating an axiom in a biomimetic AI reasoning system.

AXIOM: {axiom['name']}
DESCRIPTION: {axiom['description']}
FORMULA: {axiom.get('formula', 'N/A')}
PRIORITY: {axiom['priority']}

TEST SCENARIO:
Input: {scenario['input']}
Expected Behavior: {scenario['expected_behavior']}
Success Criteria: {scenario['success_criteria']}

TASK: Evaluate if this axiom would handle this scenario correctly.

1. Does the axiom apply to this situation? (yes/no)
2. Would it produce the expected behavior? (yes/no)
3. Would it meet the success criteria? (yes/no)
4. Confidence in evaluation (0.0-1.0)
5. Brief reasoning (1-2 sentences)

Respond in JSON format:
{{"applies": true/false, "correct_behavior": true/false, "meets_criteria": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""

        # Use centralized system prompt
        system_prompt = get_axiom_evaluation_prompt()
        
        try:
            response = self.textgen.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.3  # Lower temperature for more consistent evaluation
            )
            
            if not response:
                return False, 0.0, "LLM generation failed"
            
            # Parse JSON response
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                success = (
                    result.get('applies', False) and
                    result.get('correct_behavior', False) and
                    result.get('meets_criteria', False)
                )
                confidence = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                return success, confidence, reasoning
            else:
                return False, 0.3, f"Could not parse response: {response[:100]}"
                
        except Exception as e:
            return False, 0.0, f"Evaluation error: {str(e)}"
    
    def test_axiom(self, axiom_id: str) -> Dict:
        """
        Test all scenarios for an axiom.
        
        Returns:
            Dict with test results
        """
        axiom = self.library.axioms[axiom_id]
        scenarios = axiom.get('test_scenarios', [])
        
        if not scenarios:
            return {
                'axiom_id': axiom_id,
                'total_scenarios': 0,
                'passed': 0,
                'failed': 0,
                'avg_confidence': 0.0,
                'results': []
            }
        
        results = []
        total_confidence = 0.0
        
        print(f"\n  Testing {axiom_id}: {axiom['name']}")
        
        for idx, scenario in enumerate(scenarios):
            success, confidence, reasoning = self.evaluate_axiom_with_llm(
                axiom_id,
                scenario
            )
            
            result = {
                'scenario_index': idx,
                'success': success,
                'confidence': confidence,
                'reasoning': reasoning,
                'input': scenario['input']
            }
            
            results.append(result)
            total_confidence += confidence
            
            status = "✓" if success else "✗"
            print(f"    Scenario {idx+1}: {status} (conf={confidence:.2f})")
        
        passed = sum(1 for r in results if r['success'])
        failed = len(results) - passed
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        # Update axiom metrics
        metrics = {
            'success_rate': passed / len(results) if results else 0.0,
            'avg_confidence': avg_confidence,
            'test_count': axiom.get('performance_metrics', {}).get('test_count', 0) + len(results),
            'last_tested': datetime.now().isoformat()
        }
        
        self.library.update_metrics(axiom_id, metrics)
        
        return {
            'axiom_id': axiom_id,
            'total_scenarios': len(scenarios),
            'passed': passed,
            'failed': failed,
            'avg_confidence': avg_confidence,
            'results': results
        }
    
    def test_all_axioms(self) -> Dict:
        """Test all axioms in the library"""
        print("\n" + "=" * 70)
        print("SELF-TRAINING CYCLE - TESTING ALL AXIOMS")
        print("=" * 70)
        
        all_results = {}
        total_passed = 0
        total_failed = 0
        
        for axiom_id in self.library.axioms.keys():
            result = self.test_axiom(axiom_id)
            all_results[axiom_id] = result
            total_passed += result['passed']
            total_failed += result['failed']
        
        summary = {
            'total_axioms': len(all_results),
            'total_scenarios': total_passed + total_failed,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'pass_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0.0,
            'results': all_results
        }
        
        return summary
    
    def identify_problematic_axioms(self, test_summary: Dict) -> List[str]:
        """Identify axioms that need review"""
        problematic = []
        
        for axiom_id, result in test_summary['results'].items():
            # Flag if success rate < 70% or confidence < 0.6
            if result['total_scenarios'] > 0:
                success_rate = result['passed'] / result['total_scenarios']
                
                # Collect failed and low-confidence test details
                failed_tests = [r for r in result['results'] if not r['success']]
                low_confidence_tests = [r for r in result['results'] if r['confidence'] < 0.6]
                
                # Build detailed reason with LLM explanations
                reasons = []
                if success_rate < 0.5:
                    reasons.append(f"Critical: {result['failed']}/{result['total_scenarios']} scenarios failing")
                elif success_rate < 0.7:
                    reasons.append(f"Low success: {result['failed']}/{result['total_scenarios']} scenarios failing")
                
                if result['avg_confidence'] < 0.6:
                    reasons.append(f"Low confidence: {result['avg_confidence']:.2f}/1.0")
                
                # Add specific failure explanations from LLM
                if failed_tests:
                    example = failed_tests[0]
                    llm_reason = example['reasoning'][:150].replace('\n', ' ')
                    reasons.append(f"Example failure: {llm_reason}")
                elif low_confidence_tests:
                    example = low_confidence_tests[0]
                    llm_reason = example['reasoning'][:150].replace('\n', ' ')
                    reasons.append(f"Uncertainty: {llm_reason}")
                
                if reasons:
                    reason = "; ".join(reasons)
                    
                    # Generate clarification questions for problematic axioms
                    questions = self.generate_clarification_questions(
                        axiom_id, 
                        result, 
                        failed_tests, 
                        low_confidence_tests
                    )
                    
                    self.queue.add_for_review(
                        axiom_id, 
                        reason, 
                        priority=3 if success_rate < 0.5 else 2,
                        failed_tests=failed_tests,
                        clarification_questions=questions
                    )
                    problematic.append(axiom_id)
        
        return problematic
    
    def generate_clarification_questions(
        self, 
        axiom_id: str, 
        result: Dict, 
        failed_tests: List[Dict],
        low_confidence_tests: List[Dict]
    ) -> List[str]:
        """
        Generate specific questions to ask human for clarification.
        
        Args:
            axiom_id: ID of the axiom
            result: Test result summary
            failed_tests: List of failed test results
            low_confidence_tests: List of low confidence results
            
        Returns:
            List of clarification questions
        """
        axiom = self.library.axioms.get(axiom_id)
        if not axiom:
            return []
        
        # Build context for question generation
        context = f"""Axiom: {axiom['name']}
Description: {axiom['description']}
Formula: {axiom.get('formula', 'N/A')}

Test Performance:
- Success Rate: {result['passed']}/{result['total_scenarios']}
- Confidence: {result['avg_confidence']:.2f}

"""
        
        if failed_tests:
            context += "\nFailed Scenarios:\n"
            for i, test in enumerate(failed_tests[:2], 1):
                context += f"{i}. Input: {test['input']}\n"
                context += f"   LLM Reasoning: {test['reasoning'][:200]}\n\n"
        
        if low_confidence_tests:
            context += "\nUncertain Scenarios:\n"
            for i, test in enumerate(low_confidence_tests[:2], 1):
                context += f"{i}. Input: {test['input']}\n"
                context += f"   LLM Reasoning: {test['reasoning'][:200]}\n\n"
        
        prompt = f"""{context}

Generate 2-3 specific questions to ask the human developer to clarify this axiom's behavior.
Focus on the ambiguities or edge cases that caused test failures or low confidence.
Make questions concise and actionable."""
        
        try:
            response = self.textgen.generate(
                prompt,
                system_prompt=get_axiom_refinement_prompt(),
                max_tokens=300,
                temperature=0.7
            )
            
            if response:
                # Parse questions (assuming they're numbered or bulleted)
                questions = [
                    q.strip() 
                    for q in response.split('\n') 
                    if q.strip() and any(c in q for c in ['?', '1.', '2.', '3.', '-', '•'])
                ]
                return questions[:3]
        except Exception as e:
            print(f"Error generating clarification questions: {e}")
        
        return []
    
    def generate_improvement_suggestions(self, axiom_id: str, test_result: Dict) -> str:
        """
        Use LLM to suggest improvements for problematic axiom.
        
        Args:
            axiom_id: ID of axiom
            test_result: Test results for the axiom
            
        Returns:
            Improvement suggestions as string
        """
        axiom = self.library.axioms[axiom_id]
        
        # Build failed scenarios context
        failed_scenarios = [
            r for r in test_result['results'] 
            if not r['success']
        ]
        
        if not failed_scenarios:
            return "No failures - axiom performing well"
        
        failures_text = "\n".join([
            f"- Scenario: {f['input']}\n  Reasoning: {f['reasoning']}"
            for f in failed_scenarios
        ])
        
        prompt = f"""You are improving an axiom in a biomimetic AI reasoning system.

CURRENT AXIOM:
ID: {axiom_id}
Name: {axiom['name']}
Description: {axiom['description']}
Formula: {axiom.get('formula', 'N/A')}
Priority: {axiom['priority']}

FAILED SCENARIOS:
{failures_text}

TASK: Suggest improvements to make this axiom handle these scenarios correctly.

Consider:
1. Is the description clear enough?
2. Does the formula need adjustment?
3. Are there missing preconditions or constraints?
4. Should the priority be different?
5. Are there edge cases not covered?

Provide 2-3 specific, actionable suggestions."""

        # Use centralized system prompt
        system_prompt = get_axiom_improvement_prompt()
        
        try:
            suggestions = self.textgen.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=400,
                temperature=0.7
            )
            return suggestions or "Could not generate suggestions"
        except Exception as e:
            return f"Error generating suggestions: {e}"
    
    def analyze_axiom_relationships(self) -> Dict:
        """Analyze patterns in axiom performance to discover relationships"""
        print("\n  Analyzing axiom relationships...")
        
        # Get axioms by layer
        layers = {}
        for axiom_id, axiom in self.library.axioms.items():
            layer = axiom.get('layer', 'unknown')
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(axiom_id)
        
        # Analyze performance by layer
        layer_performance = {}
        for layer, axiom_ids in layers.items():
            metrics = []
            for axiom_id in axiom_ids:
                perf = self.library.axioms[axiom_id].get('performance_metrics', {})
                if perf.get('test_count', 0) > 0:
                    metrics.append(perf.get('success_rate', 0.0))
            
            if metrics:
                layer_performance[layer] = {
                    'count': len(axiom_ids),
                    'avg_success': sum(metrics) / len(metrics),
                    'axiom_ids': axiom_ids
                }
        
        return layer_performance
    
    def run_training_iteration(self) -> Dict:
        """Run one complete training iteration"""
        self.iteration += 1
        
        print(f"\n{'='*70}")
        print(f"SELF-TRAINING ITERATION {self.iteration}")
        print(f"{'='*70}")
        
        # Step 1: Test all axioms
        test_summary = self.test_all_axioms()
        
        # Step 2: Identify problems
        problematic = self.identify_problematic_axioms(test_summary)
        
        print(f"\n  Identified {len(problematic)} problematic axioms")
        
        # Step 3: Generate improvement suggestions for problematic axioms
        suggestions = {}
        for axiom_id in problematic[:3]:  # Limit to top 3 for efficiency
            print(f"\n  Generating suggestions for {axiom_id}...")
            suggestions[axiom_id] = self.generate_improvement_suggestions(
                axiom_id,
                test_summary['results'][axiom_id]
            )
        
        # Step 4: Analyze relationships
        relationships = self.analyze_axiom_relationships()
        
        # Step 5: Save results
        self.library.save()
        self.queue.save()
        
        # Step 6: Log iteration
        iteration_data = {
            'test_summary': {
                'total_axioms': test_summary['total_axioms'],
                'total_scenarios': test_summary['total_scenarios'],
                'pass_rate': test_summary['pass_rate']
            },
            'problematic_axioms': problematic,
            'suggestions': suggestions,
            'layer_performance': relationships
        }
        
        self.log_training_event('training_iteration', iteration_data)
        
        # Step 7: Print summary
        print(f"\n{'='*70}")
        print(f"ITERATION {self.iteration} COMPLETE")
        print(f"{'='*70}")
        print(f"  Total Scenarios: {test_summary['total_scenarios']}")
        print(f"  Pass Rate: {test_summary['pass_rate']:.1%}")
        print(f"  Problematic Axioms: {len(problematic)}")
        print(f"  Review Queue Size: {len(self.queue.queue)}")
        
        return iteration_data
    
    def run_continuous(self, max_iterations: int = 100, interval_seconds: int = 300):
        """
        Run continuous training loop.
        
        Args:
            max_iterations: Maximum iterations to run (0 for infinite)
            interval_seconds: Seconds between iterations
        """
        print(f"\n{'='*70}")
        print("STARTING CONTINUOUS SELF-TRAINING")
        print(f"{'='*70}")
        print(f"  Max Iterations: {'Infinite' if max_iterations == 0 else max_iterations}")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Library: {self.library.library_path}")
        print(f"  Queue: {self.queue.queue_path}")
        print(f"  Training Log: {self.training_log_path}")
        
        # Check LLM connection
        if not self.textgen.test_connection():
            print("\n  ERROR: Cannot connect to LLM inference server")
            return
        
        iteration_count = 0
        
        try:
            while max_iterations == 0 or iteration_count < max_iterations:
                # Run iteration
                self.run_training_iteration()
                
                iteration_count += 1
                
                # Wait before next iteration (unless this is the last one)
                if max_iterations == 0 or iteration_count < max_iterations:
                    print(f"\n  Waiting {interval_seconds}s before next iteration...")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\n  Training interrupted by user")
        
        print(f"\n{'='*70}")
        print(f"SELF-TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Total Iterations: {iteration_count}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Training Loop for Axiom Cortex")
    parser.add_argument('--iterations', type=int, default=1, 
                       help='Number of iterations (0 for continuous)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Seconds between iterations (continuous mode)')
    parser.add_argument('--library', type=str, default='data/axioms/base_axioms.json',
                       help='Path to axiom library')
    parser.add_argument('--queue', type=str, default='data/axioms/review_queue.json',
                       help='Path to review queue')
    
    args = parser.parse_args()
    
    # Initialize training loop
    trainer = SelfTrainingLoop(
        library_path=args.library,
        queue_path=args.queue
    )
    
    # Run
    if args.iterations == 1:
        trainer.run_training_iteration()
    else:
        trainer.run_continuous(
            max_iterations=args.iterations,
            interval_seconds=args.interval
        )


if __name__ == '__main__':
    main()
