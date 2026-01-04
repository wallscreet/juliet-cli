from src.clients import LLMClient, XAIClient, OllamaClient
from src.messages import MessageCache
from src.instructions import ModelInstructions
from src.fact_store import FactStore, Fact
from src.todos import TodoStore
import json


class IsoClient:
    iso_name: str
    user_name: str
    llm_client: LLMClient
    instructions: ModelInstructions
    fact_store: FactStore
    todo_store: TodoStore
    _tools: list
    
    def __init__(self, iso_name: str, user_name: str, cache_capacity: int = 20):
        self.iso_name = iso_name
        self.user_name = user_name
        self._tools = []
        
        # File paths
        fact_store_path = f"isos/{self.iso_name}/users/{self.user_name}/facts.yaml"
        todo_store_path = f"isos/{self.iso_name}/users/{self.user_name}/todos.yaml"
        
        # Modules
        self.message_cache = MessageCache(capacity=cache_capacity)
        self.llm_client = XAIClient()
        self.instructions = ModelInstructions(method="load", assistant_name=self.iso_name)
        self.fact_store = FactStore(fact_store_path)
        self.todo_store = TodoStore(todo_store_path=todo_store_path)
        
        # Tool Registration
        self.register_tools()

    # Toolbox Methods
    def _register_tool(self, name: str, description: str, parameters: dict):
        """Register a new tool that the LLM can call."""
        new_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self._tools.append(new_tool)
    
    # Toolbox Methods
    def register_tools(self):
        """Initialize the toolbox with all available tools being registered."""

        self._register_tool(
            name="add_fact",
            description="Add a new fact to remember.",
            parameters=Fact.model_json_schema()
        )
        
        self._register_tool(
            name="create_todo",
            description="Create a new todo item.",
            parameters=TodoStore.model_json_schema()
        )
        
        self._register_tool(
            name="list_active_todos",
            description="List all active todo items.",
            parameters=TodoStore.model_json_schema()
        )

    def get_tools(self):
        return self._tools
    
    def build_prompt(self, user_input: str):
        
        messages = self.instructions.to_prompt_script(
            user_request=user_input,
            facts_context=None,
            mem_context=None,
            knowledge_context=None,
            chat_history=None,
            workspace_contents=None
        )
        
        return messages
    
    def generate_response_with_tools(self, model: str, user_input: str):
        # Instructions - Entry - str
        messages = self.build_prompt(user_input)
        # Instructions - Output - List[dict]
        # Toolbox - Entry
        tools = self.get_tools()
        # Toolbox - Output
        
        while True:
            # LLM Client - Entry
            response, usage = self.llm_client.get_response_with_tools(model, messages, tools)
            
            if not response.tool_calls:
                return response.content, messages, usage
            
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Add Fact Tool
                if tool_name == "add_fact":
                    fact = Fact(**args)
                    self.fact_store.append_fact(fact)
                    result = {"status": "fact_added", "fact": args}
                
                # Create Todo Tool
                elif tool_name == "create_todo":
                    todo = self.todo_store.append_todo(description=args['description'])
                    if todo:
                        result = {"status": "todo_created", "todo": todo.model_dump()}
                    else:
                        result = {"status": "todo_exists", "description": args['description']}
                
                # List Active Todos Tool
                elif tool_name == "list_active_todos":
                    todos = self.todo_store.filter_todos(completed=False)
                    result = {"status": "active_todos", "todos": [t.model_dump() for t in todos]}

                else:
                    result = {"status": "unknown_tool", "tool": tool_name}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })