from collections import OrderedDict, deque
from src.instructions import ModelInstructions
from typing import Dict, List, Optional
from datetime import datetime
from string import Template


class BaseContextAdapter:
    """Base class for feed-forward context adapters (no Pydantic)."""

    def __init__(self, collection_name: str, chroma_path: str):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.client = None
        self._init_client()

    def _init_client(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=self.chroma_path)

    def get_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name)

    def build_messages(self) -> list[dict[str, str]]:
        """Return list of standard messages for pipeline."""
        raise NotImplementedError

    def ingest_if_needed(self):
        """Optional: vectorize source data on first use."""
        pass


class ContextPipeline:
    def __init__(self, iso_name: str = "juliet"):
        self.instructions = ModelInstructions(method="load", assistant_name=iso_name)
        self.adapters = OrderedDict()

        # ================================= #
        # === Register Context Adapters === #
        # ================================= #        
        # Timestamp
        self.register_adapter("timestamp", TimestampAdapter())
        # Workspace
        # TODO: Workspace adapter
        # Facts
        # TODO: Fact Store adapter
        # Knowledge Base (chroma)
        # TODO: KB adapter
        # Semantic Memory (chroma)
        # TODO: Chroma Memory Adapter
        # Message Cache (chat history)
        self.register_adapter("message_cache", MessageCacheAdapter(chroma_path="data/chroma"))
        # User Request
        self.register_adapter("user_request", UserRequestAdapter(tag_name="user"))
        # Assistant Prefix
        self.register_adapter("assistant_prefix", AssistantPrefixAdapter(prefix="<assistant>\n"))
    
    def register_adapter(self, name: str, adapter: BaseContextAdapter):
        self.adapters[name] = adapter
    
    def build_messages(self, user_request: str) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": f"<system>{self.instructions.system_message}</system>"},
            {"role": "assistant", "content": f"<assistant_intro>{self.instructions.assistant_intro}</assistant_intro>"},
            {"role": "system", "content": f"<focus>{self.instructions.assistant_focus}</focus>"},
        ]
        
        for name, adapter in self.adapters.items():
            if name == "user_request":
                messages.extend(adapter.build_messages(user_request=user_request))
            elif name == "assistant_prefix":
                messages.append(adapter.build_messages()[0])
            else:
                messages.extend(adapter.build_messages())
        
        return messages


class TimestampAdapter(BaseContextAdapter):
    def build_messages(self) -> list[dict[str, str]]:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        content = f"<timestamp>The current date and time is:\n{current_time}</timestamp>"
        return [{"role": "system", "content": content}]


class UserRequestAdapter(BaseContextAdapter):
    """
    Adapter for the final user request with configurable tag.
    """
    def __init__(self, tag_name: str = "user"):
        self.tag_name = tag_name

    def build_messages(self, user_request: str) -> list[dict[str, str]]:
        content = f"<{self.tag_name}>{user_request}</{self.tag_name}>"
        return [{"role": "user", "content": content}]


class AssistantPrefixAdapter(BaseContextAdapter):
    """
    Adapter that adds the forced assistant prefix.
    """
    def __init__(self, prefix: str = "<assistant>\n"):
        self.prefix = prefix

    def build_messages(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.prefix}]


class MessageCacheAdapter(BaseContextAdapter):
    """
    Short-term memory adapter using bounded deque for recent turns.
    """
    capacity: int = 10

    def __init__(self, capacity: int = 20):
        super().__init__(collection_name="message_cache")
        self.capacity = capacity
        self.cache = deque(maxlen=capacity)

    def add_turn(self, turn):
        self.cache.append(turn)

    def build_messages(self) -> list[dict[str, str]]:
        if not self.cache:
            return []

        history_lines = []
        for turn in self.cache:
            history_lines.append(turn.request.to_memory_string())
            history_lines.append(turn.response.to_memory_string())

        content = "<history>\n" + "\n".join(history_lines) + "\n</history>"
        return [{"role": "system", "content": content}]


# ================================================= #
# ================================================= #
def to_prompt_script(user_request: str, 
                     facts_context: Optional[List[str]] = None, 
                     mem_context: Optional[List[str]] = None, 
                     knowledge_context: Optional[List[str]] = None, 
                     chat_history: Optional[List[str]] = None, 
                     workspace_contents: str = None,
                     system_message: str = None,
                     assistant_intro: str = None,
                     assistant_focus: str = None
) -> List[Dict[str, str]]:
    """
    Export instructions class as a prompt template.
    """

    facts_context_str = "\n".join(facts_context) if facts_context else "No related Facts found"
    mem_context_str = "\n".join(mem_context) if mem_context else "No related memories"
    knowledge_context_str = "\n".join(knowledge_context) if knowledge_context else "No related knowledge"
    chat_history_str = "\n".join(chat_history) if chat_history else "No chat history"
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")  # Thursday, December 18, 2025 at 02:30 PM

    messages = [
        {"role": "system", "content": f"<system>>{system_message}</system>\n"},
        {"role": "assistant", "content": f"\n<assistant>{assistant_intro}</assistant>\n"},
        {"role": "user", "content": f"\n<focus>Your current focus should be: {assistant_focus}</focus>\n"},
        {"role": "system", "content": f"\n<timestamp>The current date and time is:\n{current_time}</timestamp>\n"},
        {"role": "system", "content": f"\n<workspace>Workspace directory contents:\n{workspace_contents}</workspace>\n"},
        {"role": "system", "content": f"\n<facts>Facts from your Facts Table:\n{facts_context_str}</facts>\n"},
        {"role": "system", "content": f"\n<memory>Request context from your memory:\n{mem_context_str}</memory>\n"},
        {"role": "system", "content": f"\n<knowledge>Request context from your knowledge base:\n{knowledge_context_str}</knowledge>\n"},
        {"role": "system", "content": f"\n<history>Conversation chat history:\n{chat_history_str}</history>\n"},
        {"role": "user", "content": f"\n<user>User Request: {user_request}</user>\n"},
        {"role": "assistant", "content": "<assistant>\n"},
    ]

    return messages

def to_prompt_script_md(user_request: str,
                        facts_context: Optional[List[str]] = None,
                        mem_context: Optional[List[str]] = None,
                        knowledge_context: Optional[List[str]] = None,
                        chat_history: Optional[List[str]] = None,
                        workspace_contents: Optional[List[str]] = None,
                        todos: Optional[List[str]] = None,
)-> List[Dict[str, str]]:
    """
    Create Markdown formatted instructions for the model
    """

    facts_context_str = "\n".join(facts_context) if facts_context else "No related Facts found"
    mem_context_str = "\n".join(mem_context) if mem_context else "No related memories"
    knowledge_context_str = "\n".join(knowledge_context) if knowledge_context else "No related knowledge"
    chat_history_str = "\n".join(chat_history) if chat_history else "No chat history"
    todos_str = ",\n".join(todos) if todos else "No TODOs"
    workspace_contents_str = "\n".join(workspace_contents) if workspace_contents else "No content in your Workspace"
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")  # Thursday, December 18, 2025 at 02:30 PM
    
    with open("templates/prompt_template.md") as f:
        t = Template(f.read())

    content = t.substitute(
            system_message=self.system_message,
            assistant_intro=self.assistant_intro,
            assistant_focus=self.assistant_focus,
            current_time=current_time,
            todos=todos_str,
            workspace_contents=workspace_contents_str,
            facts_context=facts_context_str,
            mem_context=mem_context_str,
            knowledge_context=knowledge_context_str,
            chat_history=chat_history_str,
    )

    messages = [
            {"role": "system", "content": f"{content}"},
            {"role": "user", "content": f"{user_request}"},
            {"role": "assistant", "content": "<assistant>"},
    ]

    return messages