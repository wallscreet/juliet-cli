from collections import OrderedDict, deque
from uuid import uuid4
from instructions import ModelInstructions
from datetime import datetime


class BaseContextAdapter:
    """Base class for feed-forward context adapters (no Pydantic)."""

    def __init__(self):
        pass

    def build_messages(self) -> list[dict[str, str]]:
        """Return list of standard messages for pipeline."""
        raise NotImplementedError


class ChromaContextAdapter(BaseContextAdapter):
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

    def build_messages(self, user_request: str, top_k: int = 10, tag: str = None) -> list[dict[str, str]]:
        if not user_request.strip():
            return []
        
        results = self.get_collection().query(query_texts=[user_request], n_results=top_k)
        passages = "\n".join(doc for doc in results["documents"][0] if doc)
        if not passages:
            return []
        
        tag_name = tag or self.collection_name
        content = f"<{tag_name}>\n{passages}\n</{tag_name}>"
        return [{"role": "system", "content": content}]


class ContextPipeline:
    def __init__(self, iso_name: str = "juliet", user_name: str = "wallscreet"):
        self.instructions = ModelInstructions(method="load", assistant_name=iso_name)
        self.adapters = OrderedDict()
        self.chroma_path = f"/isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/chroma_db"

        # === Register Context Adapters === #       
        # Timestamp
        self.register_adapter("timestamp", TimestampAdapter())
        # Workspace
        # TODO: Workspace adapter
        # Facts
        self.register_adapter("facts", FactAdapter(self.chroma_path))
        # Procedural Memory (chroma procedural memory)
        #self.register_adapter("procedural", ProceduralMemoryAdapter(self.chroma_path))
        # Semantic Memory (chroma knowledge base)
        self.register_adapter("semantic", SemanticMemoryAdapter(self.chroma_path))
        # Episodic Memory (chroma experiencial memory)
        self.register_adapter("episodic", EpisodicMemoryAdapter(self.chroma_path))
        # Message Cache (chat history)
        self.register_adapter("message_cache", MessageCacheAdapter(capacity=20))
        # User Request
        self.register_adapter("user_request", UserRequestAdapter(tag_name="user"))
        # Assistant Prefix
        self.register_adapter("assistant_prefix", AssistantPrefixAdapter(prefix="<assistant>"))
    
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
            elif isinstance(adapter, ChromaContextAdapter):
                messages.extend(adapter.build_messages(user_request))
            else:
                messages.extend(adapter.build_messages())
        
        return messages

def context_pipeline_test():
    adapter = ContextPipeline()
    print(adapter.build_messages(user_request="This is the context pipeline test message."))

class TimestampAdapter(BaseContextAdapter):
    def build_messages(self) -> list[dict[str, str]]:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        content = f"<timestamp>The current date and time is: {current_time}</timestamp>"
        return [{"role": "system", "content": content}]

def timestamp_adapter_test():
    time_adapter = TimestampAdapter()
    timestamp = time_adapter.build_messages()
    print(f"Timestamp: {timestamp}")


class UserRequestAdapter(BaseContextAdapter):
    """
    Adapter for the final user request with configurable tag.
    """
    def __init__(self, tag_name: str = "user"):
        self.tag_name = tag_name

    def build_messages(self, user_request: str) -> list[dict[str, str]]:
        content = f"<{self.tag_name}>{user_request}</{self.tag_name}>"
        return [{"role": "user", "content": content}]

def user_request_adapter_test():
    adapter = UserRequestAdapter()
    print(adapter.build_messages(user_request="This is a test message."))


class AssistantPrefixAdapter(BaseContextAdapter):
    """
    Adapter that adds the forced assistant prefix.
    """
    def __init__(self, prefix: str = "<assistant>"):
        self.prefix = prefix

    def build_messages(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.prefix}]

def asst_prefix_adapter_test():
    adapter = AssistantPrefixAdapter()
    print(adapter.build_messages())


class MessageCacheAdapter(BaseContextAdapter):
    """
    Short-term memory adapter using bounded deque for recent turns.
    """
    capacity: int = 10

    def __init__(self, capacity: int = 20):
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

def message_cache_adapter_test():
    from messages import Message, Turn
    from uuid import uuid4
    adapter = MessageCacheAdapter()
    
    request_msg = Message(
        uuid=uuid4(),
        role="user",
        speaker="Wallscreet",
        content="Test user request message."
    )
    response_msg = Message(
        uuid=uuid4(),
        role="assistant",
        speaker="Juliet",
        content="This is the test response to the user request test message."
    )
    
    test_turn = Turn(
        uuid=uuid4(),
        conversation_id="12345",
        request=request_msg,
        response=response_msg
    )
    
    adapter.add_turn(turn=test_turn)
    
    print(adapter.build_messages())


class EpisodicMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_path: str):
        super().__init__(collection_name="episodic", chroma_path=chroma_path)


class SemanticMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_path: str):
        super().__init__(collection_name="semantic", chroma_path=chroma_path)


class ProceduralMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_path: str):
        super().__init__(collection_name="procedural", chroma_path=chroma_path)


class FactAdapter(ChromaContextAdapter):
    """
    Chroma-based fact retrieval adapter using SPO triples.
    Stores facts as "subject | predicate | object" for better semantic search.
    """
    def __init__(self, chroma_path: str, collection_name: str = "facts"):
        super().__init__(collection_name=collection_name, chroma_path=chroma_path)
        from chromadb.utils import embedding_functions
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def append_fact(self, subject: str, predicate: str, object: str):
        """Store a single SPO triple."""
        triple = f"{subject.strip()} | {predicate.strip()} | {object.strip()}"
        fact_id = str(uuid4())
        self.collection.add(
            documents=[triple],
            ids=[fact_id],
            metadatas=[{
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "created_at": datetime.now().isoformat()
            }]
        )

    def build_messages(self, user_request: str, top_k: int = 10) -> list[dict[str, str]]:
        if not user_request.strip():
            return []

        results = self.collection.query(query_texts=[user_request], n_results=top_k)
        facts = [doc for doc in results["documents"][0] if doc]

        if not facts:
            return []

        content = "<facts>\n" + "\n".join(facts) + "\n</facts>"
        return [{"role": "system", "content": content}]


if __name__ == "__main__":
    #timestamp_adapter_test()
    #user_request_adapter_test()
    #asst_prefix_adapter_test()
    #message_cache_adapter_test()
    context_pipeline_test()