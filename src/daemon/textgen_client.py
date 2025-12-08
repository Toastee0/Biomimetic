#!/usr/bin/env python3
"""Client for llama-server (OpenAI-compatible API)"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/home/toastee/BioMimeticAi/config/.env")

class TextGenClient:
    def __init__(self):
        api_base = os.getenv("OPENAI_API_BASE", "http://localhost:53307/v1")
        self.client = OpenAI(
            base_url=api_base,
            api_key="not-needed"
        )
        self.model_name = None

    def list_models(self):
        """List available models"""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def generate(self, prompt, system_prompt=None, max_tokens=512, temperature=0.7):
        """Generate text using OpenAI chat completion API"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    def test_connection(self):
        """Test if API is accessible"""
        try:
            models = self.list_models()
            if models:
                print("Connected to llama-server")
                model_shortname = models[0].split("/")[-1]
                print(f"Available model: {model_shortname}")
                self.model_name = models[0]
                return True
            else:
                print("No models available")
                return False
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

if __name__ == "__main__":
    client = TextGenClient()

    print("Testing llama-server connection...")
    if client.test_connection():
        print("\nTesting generation with PopTartee personality...")

        system_prompt = "You are PopTartee, a friendly and helpful AI assistant with a warm personality."
        user_message = "Hi! Can you introduce yourself briefly?"

        response = client.generate(user_message, system_prompt=system_prompt, max_tokens=150)
        if response:
            print(f"\nUser: {user_message}")
            print(f"PopTartee: {response}")
    else:
        print("Failed to connect to llama-server")
