from src.clients import LLMClient,XAIClient, OllamaClient
from typing import List
from src.instructions import ModelInstructions
from src.todos import TodoStore
from src.fact_store import FactStore


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
        #self.method = input("Method: ").lower().strip()
        self.method = "load"
        
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
        print(f"\nCreated new Instructions for {instructions.name}:")


class ModuleTodos:
    options: List[tuple[str, str, callable]]
    store: TodoStore
    
    def __init__(self):
        self.options = [
            ("1", "Create Todo", self.create_todo),
            ("2", "Get All Todos", self.get_all_todos),
            ("3", "Mark Todo Completed", self.mark_todo_completed),
            ("4", "Filter Todos Completed", self.filter_todos_completed),
        ]
        
        self.store = TodoStore("todos.yaml")
    
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

    def create_todo(self):
        new_todo = input("Please enter new todo: ")
        item = self.store.append_todo(new_todo)
        print(f"Appended todo:\n    {item}")
        print("")

    def get_all_todos(self):
        print("Getting Todos...")
        todos = self.store.get_all_todos()
        print("\nTodos List..")
        for t in todos:
            print(t)
        print("")
    
    def mark_todo_completed(self):
        todo_id = input("Enter todo ID: ")
        self.store.mark_completed(todo_id=todo_id)
        print(f"\nTodo, {todo_id}, marked completed")
        print("")
    
    def filter_todos_completed(self):
        user_input = input("Filter Completed: ").strip().lower()
        if user_input in ['true', 't', 'yes', 'y', '1']:
            completed = True
        elif user_input in ['false', 'f', 'no', 'n', '0']:
            completed = False
        filtered_todos = self.store.filter_todos(completed=completed)
        print(f"\nFiltered Todos List Completed == {completed}..")
        for t in filtered_todos:
            print(t)
        print("")


class ModuleFacts:
    options: List[tuple[str, str, callable]]
    store: FactStore
    
    def __init__(self):
        self.options = [
            ("1", "Create Fact", self.append_fact),
            ("2", "Get All Facts", self.get_all_facts),
        ]
        
        self.store = FactStore(fact_store_path="facts.yaml")
    
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

    def append_fact(self):
        print("Appending new fact...")
        new_fact = input("Please enter a new fact: ")
        fact = self.store.append_fact(fact_str=new_fact)
        if fact:
            print(f"New fact appended to facts.yaml:\n{fact}")
        
    def get_all_facts(self):
        print("Listing facts...")
        facts = self.store.get_all_facts()
        for f in facts:
            print(f"   {f}")


if __name__ == "__main__":
    print("Welcome to the Juliet CLI testing module. Please select a module to test.")
    
    modules = [
        ("1", "XAI Client", ModuleXAIClient),
        ("2", "Ollama Client", ModuleOllamaClient),
        ("3", "Iso Instructions", ModuleInstructions),
        ("4", "Todo Store", ModuleTodos),
        ("5", "Fact Store", ModuleFacts),
        # Add more: ("n", "Next Module", NextModuleClass),
    ]

    for key, desc, _ in modules:
        print(f"{key}: {desc}")

    choice = input("> ").strip()

    for key, _, module_class in modules:
        if choice == key:
            module = module_class()
            module.option_select()
