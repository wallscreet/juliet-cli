from src.clients import LLMClient,XAIClient, OllamaClient
from typing import List
from src.instructions import ModelInstructions


class ModuleXAIClient:
    client: LLMClient = XAIClient()
    model: str
    options: List[tuple[str, str, callable]]
    
    def __init__(self):
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


class ModuleInstructions:
    options: List[tuple[str, str, callable]]
    iso_name: str
    method: str
    
    def __init__(self):
        self.iso_name = input("Please enter name: ").lower().strip()
        self.method = input("Method: ").lower().strip()
        
        self.options = [
            ("1", "Print Instructions", self.print_instructions),
            ("2", "To Prompt Script", self.to_prompt_script),
            ("3", "To MD Prompt Script", self.to_prompt_script_md),
            ("4", "Create New Iso Instructions", self.create_new_iso_instructions),
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
    
    def print_instructions(self):
        instructions = ModelInstructions(method=self.method, assistant_name=self.iso_name)
        print(instructions.print_model_instructions())
    
    def to_prompt_script(self):
        instructions = ModelInstructions(method=self.method, assistant_name=self.iso_name)
        script = instructions.to_prompt_script(user_request="This is a test message.")
        print(f"\nGenerated Prompt Script:\n{script}")
    
    def to_prompt_script_md(self):
        instructions = ModelInstructions(method=self.method, assistant_name=self.iso_name)
        script = instructions.to_prompt_script_md(user_request="This is a test message.")
        print(f"\nGenerated Prompt Script in MD:\n{script}")
    
    def create_new_iso_instructions(self):
        instructions = ModelInstructions(method="create", assistant_name=self.iso_name)
        print(f"\nCreated new Instructions for {self.iso_name}:")


if __name__ == "__main__":
    print("Welcome to the Juliet CLI testing module. Please select a module to test.")
    
    modules = [
        ("1", "XAI Client", ModuleXAIClient),
        ("2", "Ollama Client", ModuleOllamaClient),
        ("3", "Iso Instructions", ModuleInstructions),
        # Add more: ("n", "Next Module", NextModuleClass),
    ]

    for key, desc, _ in modules:
        print(f"{key}: {desc}")

    choice = input("> ").strip()

    for key, _, module_class in modules:
        if choice == key:
            module = module_class()
            module.option_select()
