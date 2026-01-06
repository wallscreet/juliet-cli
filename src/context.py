from datetime import datetime
from typing import List, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
import yaml
from messages import Conversation, Message, Turn


def format_chat_history(chat_history: list):
    """
    Formats the chat history for display. This didn't really solve anything..
    """
    formatted_history = ''.join(chat_history)
    formatted_history = formatted_history.replace('\n\n', '\n')

    return formatted_history


def message_cache_format_to_prompt(message_history):
    chat_history = []
    for turn in message_history:
        turn_request = f"{turn.request.speaker} ({turn.request.timestamp}):\n{turn.request.content}\n"
        turn_response = f"{turn.response.speaker} ({turn.response.timestamp}):\n{turn.response.content}\n"
        chat_history.append(turn_request)
        chat_history.append(turn_response)
    chat_history = format_chat_history(chat_history)
    #print(f"\n{chat_history}")
    return chat_history


class MemoryStore:
    """
    Abstract interface for long-term memory backends. 
    """

    def store_turn(self, conversation_id: str, turn: Turn):
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> List[Message]:
        raise NotImplementedError


class ChromaMemoryStore(MemoryStore):
    """
    ChromaDB manager class. Supports multiple collections for different memory types and purposes.
    """

    def __init__(self, 
                 persist_dir: str, 
                 embedding_model: str = "all-MiniLM-L6-v2"
    ):
    
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collections = {}

    def _get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_fn
            )
        return self.collections[name]
    
    def _delete_collection(self, name: str):
        self.client.delete_collection(name=name)
        print(f"Collection {name} has been deleted.")
        
    def store_knowledge(self):
        pass

    def store_turn(self, conversation_id: str, turn: Turn, collection_name: str = "memory"):
        """
        Store a single turn (request + response as separate docs).
        """
        self.store_batch(conversation_id, [turn], collection_name=collection_name)

    def store_batch(self, conversation_id: str, turns: List[Turn], collection_name: str = "history"):
        """
        """
        collection = self._get_collection(collection_name)

        docs, ids, metas = [], [], []
        for turn in turns:
            for suffix, msg in [("req", turn.request), ("res", turn.response)]:
                ids.append(f"{turn.uuid}_{suffix}")
                docs.append(msg.to_memory_string())
                metas.append({
                    "conversation_id": conversation_id,
                    "role": msg.role,
                    "speaker": msg.speaker,
                    "timestamp": msg.timestamp,
                    #"tags": json.dumps(msg.tags) if msg.tags else None,
                })

        collection.add(documents=docs, ids=ids, metadatas=metas)

    def retrieve(self, collection_name: str, query: str, top_k: int = 10) -> List[Message]:
        """
        Retrieve semantically relevant messages (normalized to Message schema).
        """
        collection = self._get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)

        messages = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            messages.append(
                Message(
                    uuid=str(uuid4()),
                    role=meta.get("role", "unknown"),
                    speaker=meta.get("speaker", "unknown"),
                    content=text,
                    timestamp=meta.get("timestamp", datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    tags=meta.get("tags", []),
                )
            )
        return messages


class YamlMemoryAdapter(MemoryStore):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def _load_all(self) -> List[dict]:
        try:
            with open(self.filepath, "r") as f:
                return yaml.safe_load(f) or []
        except FileNotFoundError:
            return []

    def _save_all(self, data: List[dict]):
        with open(self.filepath, "w") as f:
            yaml.safe_dump(data, f)

    def save_conversation(self, convo: Conversation):
        data = self._load_all()
        data = [c for c in data if c['uuid'] != convo.uuid]
        data.append(convo.to_dict())
        self._save_all(data)

    def load_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        data = self._load_all()
        conv_data = next((c for c in data if c['uuid'] == conversation_id), None)
        return Conversation.from_dict(conv_data) if conv_data else None

    def store_turn(self, conversation_id: str, turn: Turn):
        convo = self.load_conversation_by_id(conversation_id)
        if not convo:
            raise ValueError(f"No conversation {conversation_id}")
        convo.turns.append(turn)
        convo.last_active = turn.response.timestamp
        self.save_conversation(convo)

    def retrieve(self, query: str, top_k: int = 10) -> List[Turn]:
        data = self._load_all()
        turns = []
        for convo in data:
            for t in convo.get("turns", []):
                turns.append(
                    Turn(
                        uuid=t["uuid"],
                        conversation_id=convo["uuid"],
                        request=Message(**t["request"]),
                        response=Message(**t["response"]),
                        turn_number=t.get("turn_number", len(turns) + 1),
                    )
                )
        return turns[-top_k:]


class ConversationManager:
    def __init__(self, adapter: YamlMemoryAdapter):
        self.adapter = adapter

    def get_or_start(self, 
                     conversation_id: str, 
                     host: str, 
                     host_is_bot: bool, 
                     guest: str, 
                     guest_is_bot: bool
    ) -> Conversation:
    
        convo = self.adapter.load_conversation_by_id(conversation_id)
        if convo:
            print(f"Loaded existing conversation {convo.uuid}")
            return convo

        # start new one if not found
        convo = Conversation.start_new(
            host=host,
            host_is_bot=host_is_bot,
            guest=guest,
            guest_is_bot=guest_is_bot,
            uuid_override=conversation_id,
        )
        self.adapter.save_conversation(convo)
        return convo

    def add_turn(self, convo: Conversation, request: Message, response: Message) -> Turn:
        turn = convo.create_turn(request=request, response=response)
        self.adapter.store_turn(conversation_id=convo.uuid, turn=turn)
        return turn
