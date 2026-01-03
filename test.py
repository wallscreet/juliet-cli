from src.clients import LLMClient,XAIClient, OllamaClient
from typing import List


class ModuleXAIClient:
    client: LLMClient = XAIClient()
    model: str
    options: List[tuple[str, str, callable]]
    
    def __init__(self):
        self.test_message = "This is a test message."
        self.model = "grok-4-1-fast-non-reasoning"
        
        self.options = [
            ("1", "Get Response", self.get_response),
        ]
    
    def option_select(self):
        print("\nSelect an Option:")
        for key, desc, _ in self.options:
            print(f"{key}: {desc}")
        choice = input("> ").strip()
        for key, _, func in self.options:
            if choice == key:
                func()
                return
        print("Invalid option")
    
    def get_response(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "This is a test message. Please confirm receipt and introduce yourself."
            }
        ]        
        print(f"Get Response Results:\n{self.client.get_response(model=self.model, messages=messages)}")


class ModuleOllamaClient:
    client: LLMClient = OllamaClient()
    model: str
    options: List[tuple[str, str, callable]]
    
    def __init__(self):
        self.test_message = "This is a test message."
        self.model = "granite4:3b-h"
        
        self.options = [
            ("1", "Get Response", self.get_response),
        ]
    
    def option_select(self):
        print("\nSelect an Option:")
        for key, desc, _ in self.options:
            print(f"{key}: {desc}")
        choice = input("> ").strip()
        for key, _, func in self.options:
            if choice == key:
                func()
                return
        print("Invalid option")
    
    def get_response(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "This is a test message. Please confirm receipt and introduce yourself."
            }
        ]        
        print(f"Get Response Results:\n{self.client.get_response(model=self.model, messages=messages)}")


if __name__ == "__main__":
    print("Welcome to the Juliet CLI testing module. Please select a module to test.")
    
    modules = [
        ("1", "XAI Client", ModuleXAIClient),
        ("2", "Ollama Client", ModuleOllamaClient),
        # Add more: ("n", "Next Module", NextModuleClass),
    ]

    for key, desc, _ in modules:
        print(f"{key}: {desc}")

    choice = input("> ").strip()

    for key, _, module_class in modules:
        if choice == key:
            module = module_class()
            module.option_select()
