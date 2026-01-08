from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.batch_utils import create_batches
import yaml
from messages import Conversation, Message, Turn
from extract_docs import chunk_text, extract_text
import json





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
    
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collections = {}

    def _get_collection(self, name: str):
        if name not in self.collections:
            try:
                coll = self.client.get_collection(name=name)
                print(f"Loaded existing collection '{name}'")
            except Exception as e:
                # If collection doesn't exist create it with embedding function
                if "not found" in str(e).lower() or isinstance(e, chromadb.errors.NotFoundError):
                    print(f"Collection '{name}' not found. Creating new one...")
                    coll = self.client.create_collection(
                        name=name,
                        embedding_function=self.embedding_fn
                    )
                else:
                    raise e
            self.collections[name] = coll
        return self.collections[name]
    
    def _delete_collection(self, name: str):
        self.client.delete_collection(name=name)
        print(f"Collection {name} has been deleted.")
    
    def _get_collection_stats(self, collection_name: str) -> dict:
        from pathlib import Path
        
        coll = self._get_collection(collection_name)
        
        stats = {
            "name": collection_name,
            "count": coll.count(),
        }

        try:
            #collection_id = "uuid"
            collection_id = "uuid"
            
            coll_dir = Path(self.persist_dir) / str(collection_id)
            
            if coll_dir.exists():
                size_bytes = sum(p.stat().st_size for p in coll_dir.rglob('*') if p.is_file())
                stats["size_bytes"] = size_bytes
                stats["size_mb"] = round(size_bytes / (1024 * 1024), 2)
            else:
                stats["size_note"] = "Collection directory not found"
        except Exception as e:
            stats["size_note"] = f"Size estimation failed: {str(e)}"

        return stats

    def store_knowledge_from_file(
        self,
        file_path: str,
        author: Optional[str] = None,
        collection_name: str = "semantic",
        chunk_size: int = 1536,
        overlap: int = 256,
        json_export_path: Optional[str] = None
    ) -> dict:
        """
        Ingest a single supported file into Chroma with proper batching.
        Optionally appends each chunk as a JSON line to a file for inspection/debugging if json_export_path is provided.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        source_name = file_path.name
        file_type = file_path.suffix.lower()

        print(f"Extracting text from {source_name}...")
        try:
            raw_text = extract_text(str(file_path))
        except Exception as e:
            raise IOError(f"Failed to extract text from {source_name}: {e}")

        if not raw_text.strip():
            return {"status": "empty", "file": source_name, "chunks": 0}

        print(f"Chunking {len(raw_text):,} characters (size={chunk_size}, overlap={overlap})...")
        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            return {"status": "no_chunks", "file": source_name, "chunks": 0}

        docs = []
        ids = []
        metadatas = []
        json_records = []

        ingested_at = datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_id = str(uuid4())
            text = chunk.strip()

            metadata = {
                "source_file": source_name,
                "source_path": str(file_path),
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": ingested_at,
            }
            if author:
                metadata["author"] = author

            docs.append(text)
            ids.append(chunk_id)
            metadatas.append(metadata)

            record = {
                "id": chunk_id,
                "text": text,
                "metadata": metadata
            }
            json_records.append(record)

        collection = self._get_collection(collection_name)
        max_batch_size = self.client.get_max_batch_size()
        print(f"Storing {len(docs)} chunks in batches of ≤{max_batch_size}...")

        for batch_ids, _, batch_metadatas, batch_documents in create_batches(
            api=self.client,
            ids=ids,
            metadatas=metadatas,
            documents=docs,
        ):
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

        # Append to JSON Lines file
        if json_export_path:
            json_path = Path(json_export_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with json_path.open("a", encoding="utf-8") as f:
                for record in json_records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Appended {len(json_records)} records to {json_path}")

        print(f"Successfully stored {len(docs)} chunks from {source_name} → '{collection_name}'")

        return {
            "status": "success",
            "file": source_name,
            "chunks_stored": len(docs),
            "collection": collection_name,
            "preview": docs[0][:300] + "..." if docs else None,
            "json_exported": len(json_records) if json_export_path else 0
        }

    def store_turn(self, 
                   conversation_id: str, 
                   turn: Turn, 
                   collection_name: str, 
                   json_export_path: Optional[str] = None
    ) -> None:
        """
        Store a single turn (request + response as separate docs).
        Optionally exports to JSON Lines.
        """
        self.store_batch(conversation_id=conversation_id, 
                         turns=[turn], 
                         collection_name=collection_name, 
                         json_export_path=json_export_path
        )

    def store_batch(self, 
                    conversation_id: str, 
                    turns: List[Turn], 
                    collection_name: str, 
                    json_export_path: Optional[str] = None
    ) -> None:
        """
        Store multiple turns in batch.
        Each message (request/response) becomes a separate document in Chroma.
        Optionally appends full records to a JSON Lines file after storing.
        """
        collection = self._get_collection(collection_name)

        docs = []
        ids = []
        metadatas = []
        json_records = []

        for turn in turns:
            for suffix, msg in [("req", turn.request), ("res", turn.response)]:
                doc_id = f"{turn.uuid}_{suffix}"
                text = msg.to_memory_string()

                metadata = {
                    "conversation_id": conversation_id,
                    "turn_uuid": str(turn.uuid),
                    "message_type": suffix,  # "req" or "res"
                    "role": msg.role,
                    "speaker": msg.speaker,
                    "timestamp": msg.timestamp,
                    # "tags": json.dumps(msg.tags) if msg.tags else None,
                }

                docs.append(text)
                ids.append(doc_id)
                metadatas.append(metadata)

                json_records.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "conversation_id": conversation_id,
                    "turn_uuid": str(turn.uuid),
                    "message_type": suffix,
                    "role": msg.role,
                    "speaker": msg.speaker,
                    "timestamp": msg.timestamp
                })

        # Store in Chroma (with safe batching if needed)
        if docs:
            # Use create_batches for safety with large batches
            for batch_ids, _, batch_metadatas, batch_documents in create_batches(
                api=self.client,
                ids=ids,
                metadatas=metadatas,
                documents=docs,
            ):
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

        # Append to JSON Lines file
        if json_export_path and json_records:
            json_path = Path(json_export_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with json_path.open("a", encoding="utf-8") as f:
                for record in json_records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Appended {len(json_records)} episodic records to {json_path}")

        if docs:
            print(f"Stored {len(docs)} messages ({len(turns)} turns) in collection '{collection_name}'")


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
