from src.clients import LLMClient, XAIClient, OllamaClient
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
    
    def __init__(self, iso_name, user_name):
        self.iso_name = iso_name
        self.user_name = user_name
        self._tools = []
        
        fact_store_path = f"isos/{self.iso_name}/users/{self.user_name}/facts.yaml"
        todo_store_path = f"isos/{self.iso_name}/users/{self.user_name}/todos.yaml"
        
        self.llm_client = XAIClient()
        self.instructions = ModelInstructions(method="load", assistant_name=self.iso_name)
        self.fact_store = FactStore(fact_store_path)
        self.todo_store = TodoStore(todo_store_path=todo_store_path)
        
        self.register_tool(
            name="add_fact",
            description="Add a new fact to remember.",
            parameters=Fact.model_json_schema()
        )

    def register_tool(self, name: str, description: str, parameters: dict):
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
        messages = self.build_prompt(user_input)
        tools = self.get_tools()
        
        while True:
            response, usage = self.llm_client.get_response_with_tools(model, messages, tools)
            
            if not response.tool_calls:
                return response.content, messages, usage
            
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if tool_name == "add_fact":
                    fact = Fact(**args)
                    self.fact_store.append_fact(fact)
                    result = {"status": "fact_added", "fact": args}

                else:
                    result = {"status": "unknown_tool", "tool": tool_name}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })