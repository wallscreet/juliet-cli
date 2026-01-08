from collections import OrderedDict, deque
from uuid import uuid4
from src.instructions import ModelInstructions
from datetime import datetime
import re
from typing import List, Dict
from src.context import ChromaMemoryStore
from src.messages import Message, Turn


class BaseContextAdapter:
    """Base class for feed-forward context adapters (no Pydantic)."""
    requires_user_request: bool = False

    def build_messages(self) -> list[dict[str, str]]:
        """Return list of standard messages for pipeline."""
        raise NotImplementedError


class ChromaContextAdapter(BaseContextAdapter):
    """Base class for feed-forward context adapters (no Pydantic)."""

    def __init__(self, 
                 collection_name: str, 
                 chroma_store: ChromaMemoryStore, 
                 tagging_mode: str = "source_file"
    ):
        
        self.requires_user_request = True
        self.collection_name = collection_name
        self.client = chroma_store
        self.tagging_mode = tagging_mode

    def get_collection(self):
        return self.client.client.get_or_create_collection(name=self.collection_name)

    def build_messages(self, 
                       user_request: str, 
                       top_k: int = 5, 
                       tag: str = None, 
                       max_overfetch: int = 30, 
                       min_similarity: float = 0.15, 
                       dynamic_multiplier: float = 4.0
    ) -> list[dict[str, str]]:
        if not user_request.strip():
            return []

        fetch_k = min(top_k * 4, max_overfetch)

        results = self.get_collection().query(
            query_texts=[user_request],
            n_results=fetch_k,
            include=["documents", "distances", "metadatas"]
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        if not docs:
            return []

        similarities = [1.0 - dist for dist in distances]
        best_similarity = max(similarities)
        dynamic_threshold = best_similarity / dynamic_multiplier
        threshold = max(dynamic_threshold, min_similarity)

        tagged_chunks = []
        for doc, sim, meta in zip(docs, similarities, metadatas):
            if not doc or not doc.strip():
                continue
            if sim < threshold:
                continue

            text = doc.strip()

            # inner tagging based on mode
            if self.tagging_mode == "source_file" and "source_file" in meta:
                source_file = meta["source_file"]
                clean_tag = re.sub(r'\.[^.]+$', '', source_file)
                clean_tag = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_tag)
                clean_tag = re.sub(r'_+', '_', clean_tag).strip('_')
                if not clean_tag:
                    clean_tag = "unknown"
                tagged_text = f"<{clean_tag}>{text}</{clean_tag}>"
            elif self.tagging_mode == "simple" and "role" in meta:
                role = meta["role"]
                tagged_text = f"<{role}>{text}</{role}>"
            else:
                tagged_text = text

            tagged_chunks.append((sim, tagged_text))

        # Sort by relevance and return top_k
        tagged_chunks.sort(key=lambda x: x[0], reverse=True)
        selected_chunks = [chunk for _, chunk in tagged_chunks[:top_k]]

        if not selected_chunks:
            return []

        outer_tag = tag or self.collection_name
        content = f"<{outer_tag}>\n" + "\n\n".join(selected_chunks) + f"\n</{outer_tag}>"

        return [{"role": "system", "content": content}]


class EpisodicMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_store: ChromaMemoryStore):
        super().__init__(collection_name="episodic", tagging_mode="simple", chroma_store=chroma_store)


class SemanticMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_store: ChromaMemoryStore):
        super().__init__(collection_name="semantic", tagging_mode="source_file", chroma_store=chroma_store)


class ProceduralMemoryAdapter(ChromaContextAdapter):
    def __init__(self, chroma_store: ChromaMemoryStore):
        super().__init__(collection_name="procedural", tagging_mode="none", chroma_store=chroma_store)


class TimestampAdapter(BaseContextAdapter):
    def build_messages(self) -> list[dict[str, str]]:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        content = f"<timestamp>The current date and time is: {current_time}</timestamp>"
        return [{"role": "system", "content": content}]


class UserRequestAdapter(BaseContextAdapter):
    """
    Adapter for the final user request with configurable tag.
    """
    def __init__(self, tag_name: str = "user"):
        self.tag_name = tag_name
        self.requires_user_request = True

    def build_messages(self, user_request: str) -> list[dict[str, str]]:
        content = f"<{self.tag_name}>{user_request}</{self.tag_name}>"
        return [{"role": "user", "content": content}]


class AssistantPrefixAdapter(BaseContextAdapter):
    """
    Adapter that adds the forced assistant prefix.
    """
    def __init__(self, prefix: str = "<assistant>"):
        self.prefix = prefix

    def build_messages(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.prefix}]


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

        content = "<chat_history>\n" + "\n".join(history_lines) + "\n</chat_history>"
        return [{"role": "system", "content": content}]


class ContextPipeline:
    def __init__(self, chroma_store: ChromaMemoryStore, iso_name: str = "juliet", user_name: str = "wallscreet"):
        
        self.instructions = ModelInstructions(method="load", assistant_name=iso_name)
        self.adapters = OrderedDict()
        self.chroma_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/chroma_store"
        self.semantic_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/semantic_memory.json"
        self.episodic_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/episodic_memory.json"
        self.facts_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/facts_memory.json"

        # === Register Context Adapters === #
        self.register_adapter("timestamp", TimestampAdapter())
        self.register_adapter("semantic", SemanticMemoryAdapter(chroma_store=chroma_store))
        self.register_adapter("episodic", EpisodicMemoryAdapter(chroma_store=chroma_store))
        # self.register_adapter("procedural", ProceduralMemoryAdapter(chroma_path=self.chroma_path))  # when ready
        self.register_adapter("message_cache", MessageCacheAdapter(capacity=20))
        self.register_adapter("user_request", UserRequestAdapter(tag_name="user"))
        self.register_adapter("assistant_prefix", AssistantPrefixAdapter(prefix="<assistant>"))

    def register_adapter(self, name: str, adapter: BaseContextAdapter):
        self.adapters[name] = adapter

    def build_messages(self, user_request: str) -> List[Dict[str, str]]:
        """
        Build the full message list by combining fixed instructions and all adapters.
        Adapters that require the current user query receive it; others do not.
        """
        messages = [
            {"role": "system", "content": f"<system>{self.instructions.system_message}</system>"},
            {"role": "assistant", "content": f"<assistant_intro>{self.instructions.assistant_intro}</assistant_intro>"},
            {"role": "system", "content": f"<focus>{self.instructions.assistant_focus}</focus>"},
        ]

        for adapter in self.adapters.values():
            if adapter.requires_user_request:
                msgs = adapter.build_messages(user_request=user_request)
            else:
                msgs = adapter.build_messages()
            if msgs:
                messages.extend(msgs)

        return messages


def chroma_store_test(chroma_store: ChromaMemoryStore):
    conversation_id = "12345"
    
    request_msg = Message(
        uuid=uuid4(),
        role="user",
        speaker="Wallscreet",
        content="Morgan is playing my guitar right now."
    )
    
    response_msg = Message(
        uuid=uuid4(),
        role="assistant",
        speaker="Juliet",
        content="That sounds like fun, what is she playing?"
    )
    
    test_turn = Turn(
        uuid=uuid4(),
        conversation_id=conversation_id,
        request=request_msg,
        response=response_msg
    )
    
    chroma_store.store_turn(conversation_id=conversation_id, 
                     turn=test_turn, 
                     collection_name="episodic",
                     json_export_path="isos/juliet/users/wallscreet/episodic_memory.json"
    )
    
    print("Stored test turn in Chroma collection.")


if __name__ == "__main__":
    #== init
    chroma_dir = "isos/juliet/users/wallscreet/chroma_store"
    iso_name = "juliet"
    user_name = "wallscreet"
    chroma_store = ChromaMemoryStore(persist_dir=chroma_dir)
    pipeline = ContextPipeline(chroma_store=chroma_store, iso_name=iso_name, user_name=user_name)
    
    #== Get User Request
    #user_request = input("Enter User Request: ")
    
    #== Build Messages
    #messages = pipeline.build_messages(user_request=user_request)
    #print(messages)
    
    #== Get Response
    
    #== Store Messages->Turn => chromadb + episodic json
    chroma_store_test(chroma_store=chroma_store)