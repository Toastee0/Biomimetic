#!/usr/bin/env python3
"""
Performance test for TaskClassifier
"""

import time
from task_classifiser import TaskClassifier


def test_performance():
    """Test that classification runs in < 5ms"""
    classifier = TaskClassifier()
    
    # Test messages of various types
    test_messages = [
        "Write a Python function to calculate factorial",
        "What's in this image?",
        "Hello there!",
        "I need a complex solution for a multi-threaded application that handles real-time data processing with fault tolerance and automatic recovery mechanisms. The system should scale horizontally and maintain data consistency across distributed nodes while providing low-latency responses."
    ]
    
    print("Performance Test Results:")
    print("-" * 50)
    
    for i, message in enumerate(test_messages, 1):
        start_time = time.perf_counter()
        
        # Run classification 100 times to get average
        iterations = 100
        for _ in range(iterations):
            result = classifier.classify(message)
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        print(f"Test {i}:")
        print(f"  Message: {message[:40]}...")
        print(f"  Result: {result}")
        print(f"  Avg time per classification: {avg_time_ms:.3f} ms")
        
        # Verify performance requirement
        if avg_time_ms < 5:
            print(f"  ✓ Meets requirement (< 5ms)")
        else:
            print(f"  ✗ Exceeds requirement ({avg_time_ms:.3f} ms)")
        print()


def test_classification_logic():
    """Test the classification logic thoroughly"""
    classifier = TaskClassifier()
    
    test_cases = [
        # (message, expected_type, description)
        ("implement a sorting algorithm", "coding", "Coding keyword 'implement'"),
        ("how to code a website", "coding", "Coding keyword 'code'"),
        ("what do you see in this picture", "vision", "Vision keyword 'what do you see'"),
        ("describe this scene", "vision", "Vision keyword 'describe scene'"),
        ("hi", "conversation", "Short message"),
        ("hello how are you doing today", "conversation", "Simple conversation"),
        ("This is a very long message " * 40, "reasoning", "Long message > 100 words"),
        ("debug my python script", "coding", "Coding keyword 'debug'"),
        ("can you see this image", "vision", "Vision keyword 'can you see'"),
    ]
    
    print("Classification Logic Test:")
    print("-" * 50)
    
    all_passed = True
    for message, expected, description in test_cases:
        result = classifier.classify(message)
        passed = result == expected
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        print(f"{status} {description}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  Message: {message[:60]}...")
        print()
    
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")


if __name__ == "__main__":
    test_classification_logic()
    print("\n" + "="*50 + "\n")
    test_performance()