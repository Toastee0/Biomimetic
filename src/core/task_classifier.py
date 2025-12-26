#!/usr/bin/env python3
"""
TaskClassifier for BioMimeticAi system
Classifies user messages to determine which model should handle the request
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path


class TaskClassifier:
    """
    Classifies user messages to determine required model type

    Methods:
    - classify(message: str) -> str  # Returns mode name
    - _check_keywords(message: str, keyword_list: list) -> bool
    """
    
    def __init__(self, config_path: str = "/home/toastee/BioMimeticAi/config/models.json"):
        """
        Initialize TaskClassifier with configuration from models.json
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.keyword_patterns = self._load_keyword_patterns()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from models.json file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default structure if file not found
            print(f"Warning: Config file not found at {self.config_path}. Using default configuration.")
            return {
                "task_classification": {
                    "coding_keywords": ["implement", "code", "function", "debug", "algorithm", "write a program",
                              "how to code", "programming", "syntax", "compile", "script", "python",
                              "javascript", "java", "c++", "html", "css", "fix this code", "error in code"],
                    "vision_keywords": ["what do you see", "describe scene", "describe image", "what's in this picture",
                              "analyze this image", "image analysis", "visual description", "picture of",
                              "photo", "screenshot", "can you see", "look at this", "describe this image"],
                    "reasoning_keywords": ["analyze", "compare", "evaluate", "explain why", "trade-offs"],
                    "min_message_length_for_reasoning": 100  # words
                }
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
    
    def _load_keyword_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load and compile keyword patterns for each task type"""
        task_config = self.config.get("task_classification", {})
        patterns = {}

        # Compile patterns for coding keywords
        coding_keywords = task_config.get("coding_keywords", [])
        patterns["coding"] = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                             for keyword in coding_keywords]

        # Compile patterns for vision keywords
        vision_keywords = task_config.get("vision_keywords", [])
        patterns["vision"] = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                             for keyword in vision_keywords]

        # Compile patterns for reasoning keywords
        reasoning_keywords = task_config.get("reasoning_keywords", [])
        patterns["reasoning"] = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                                for keyword in reasoning_keywords]

        # Get reasoning threshold
        self.reasoning_threshold = task_config.get("min_message_length_for_reasoning", 100)

        return patterns
    
    def _check_keywords(self, message: str, keyword_list: List[re.Pattern]) -> bool:
        """
        Check if message contains any of the specified keywords
        
        Args:
            message: User message to check
            keyword_list: List of compiled regex patterns
            
        Returns:
            bool: True if any keyword matches
        """
        if not message or not keyword_list:
            return False
            
        # Clean the message for better matching
        message_lower = message.lower().strip()
        
        # Check each pattern
        for pattern in keyword_list:
            if pattern.search(message_lower):
                return True
                
        return False
    
    def _count_words(self, message: str) -> int:
        """
        Count words in a message
        
        Args:
            message: Text to count words in
            
        Returns:
            int: Number of words
        """
        if not message or not message.strip():
            return 0
            
        # Split by whitespace and filter out empty strings
        words = [word for word in message.strip().split() if word]
        return len(words)
    
    def classify(self, message: str) -> str:
        """
        Classify user message to determine required model type

        Classification Logic:
        1. Check coding keywords: "implement", "code", "function", "debug", etc. → `coding`
        2. Check vision keywords: "what do you see", "describe scene", etc. → `vision`
        3. Check reasoning keywords: "analyze", "compare", "evaluate", etc. → `reasoning`
        4. Check message length: > 100 words → `reasoning`
        5. Default → `conversation`

        Args:
            message: User message to classify

        Returns:
            str: Task type - one of: 'conversation', 'coding', 'reasoning', 'vision'
        """
        if not message or not message.strip():
            return "conversation"

        # Step 1: Check for coding keywords
        if self._check_keywords(message, self.keyword_patterns.get("coding", [])):
            return "coding"

        # Step 2: Check for vision keywords
        if self._check_keywords(message, self.keyword_patterns.get("vision", [])):
            return "vision"

        # Step 3: Check for reasoning keywords
        if self._check_keywords(message, self.keyword_patterns.get("reasoning", [])):
            return "reasoning"

        # Step 4: Check message length for reasoning tasks
        word_count = self._count_words(message)
        if word_count > self.reasoning_threshold:
            return "reasoning"

        # Step 5: Default to conversation
        return "conversation"
    
    def classify_with_details(self, message: str) -> Dict[str, Any]:
        """
        Classify message and return detailed information about the classification
        
        Args:
            message: User message to classify
            
        Returns:
            Dict containing classification details
        """
        if not message or not message.strip():
            return {
                "task_type": "conversation",
                "word_count": 0,
                "reasoning_threshold": self.reasoning_threshold,
                "matched_keywords": [],
                "is_long_message": False
            }
        
        word_count = self._count_words(message)
        is_long_message = word_count > self.reasoning_threshold
        
        # Check for coding keywords
        coding_matches = []
        for pattern in self.keyword_patterns.get("coding", []):
            matches = pattern.findall(message.lower())
            coding_matches.extend(matches)
        
        # Check for vision keywords
        vision_matches = []
        for pattern in self.keyword_patterns.get("vision", []):
            matches = pattern.findall(message.lower())
            vision_matches.extend(matches)
        
        # Determine task type
        if coding_matches:
            task_type = "coding"
            matched_keywords = list(set(coding_matches))
        elif vision_matches:
            task_type = "vision"
            matched_keywords = list(set(vision_matches))
        elif is_long_message:
            task_type = "reasoning"
            matched_keywords = []
        else:
            task_type = "conversation"
            matched_keywords = []
        
        return {
            "task_type": task_type,
            "word_count": word_count,
            "reasoning_threshold": self.reasoning_threshold,
            "matched_keywords": matched_keywords,
            "is_long_message": is_long_message
        }


# Example usage and testing
if __name__ == "__main__":
    # Quick test of the classifier
    classifier = TaskClassifier()
    
    test_messages = [
        "Can you help me implement a function that sorts a list?",
        "What do you see in this image?",
        "Hello, how are you today?",
        "I need to write a Python script that analyzes data from a CSV file and generates a report with visualizations. The script should handle missing values, perform statistical analysis, and create both bar charts and scatter plots. Additionally, it should export the results to an Excel file with formatted cells.",
        "Debug this code: for i in range(10): print(i",
        "Describe the scene in this picture of a sunset.",
        "Hi"
    ]
    
    print("Task Classifier Test Results:")
    print("-" * 50)
    
    for msg in test_messages:
        result = classifier.classify(msg)
        details = classifier.classify_with_details(msg)
        print(f"Message: {msg[:50]}...")
        print(f"Classification: {result}")
        print(f"Details: {details}")
        print("-" * 30)