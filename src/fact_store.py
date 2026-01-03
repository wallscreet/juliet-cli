from datetime import datetime
from uuid import uuid4
import chromadb
from pydantic import BaseModel
import yaml
from typing import List
import os
from context import ChromaMemoryAdapter
from messages import Message, Turn


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str

    def to_memory_string(self) -> str:
        """
        Serialize the fact to a string in the format 'subject predicate object'.
        """
        return f"{self.subject} {self.predicate} {self.object}"


class FactStore:
    def __init__(self,
                 fact_store_path: str, 
                 chroma_persist_dir: str,
                 chroma_adapter: ChromaMemoryAdapter = None, 
        ):
        
        self.file_path = fact_store_path
        self.chroma_adapter = chroma_adapter or ChromaMemoryAdapter(persist_dir=chroma_persist_dir)
        
        # Initialize YAML file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                yaml.dump({'facts': []}, f)

    def append_fact(self, fact: Fact):
        with open(self.file_path, 'r') as f:
            data = yaml.safe_load(f) or {'facts': []}
        
        # Check for duplicates
        existing_facts = data.get('facts', [])
        new_fact_dict = fact.model_dump()

        if new_fact_dict in existing_facts:
            print(f"Duplicate fact detected, skipping: {new_fact_dict}")
            return  # Don't append duplicates

        data['facts'].append(new_fact_dict)
        
        with open(self.file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def store_fact(self, fact: Fact, collection_name: str = "facts"):
        """
        Store a single fact as a subject-predicate-object triple in the specified collection.
        """
        collection = self.chroma_adapter._get_collection(name=collection_name)

        fact_id = str(uuid4())  # Unique ID for the fact
        fact_str = f"{fact.subject} {fact.predicate} {fact.object}"  # Serialize fact for embedding
        
        collection.add(
            documents=[fact_str],
            ids=[fact_id],
            metadatas=[{
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "timestamp": datetime.now().strftime('%Y-%m-%d @ %H:%M'),
                "type": "fact"
            }]
        )
        
        self.append_fact(fact=fact)

    def retrieve_facts(self, query: str, collection_name: str = "facts", top_k: int = 5) -> List[Fact]:
        """
        Retrieve semantically relevant facts from the specified collection.
        Returns a list of Fact objects constructed from metadata.
        """
        collection = self.chroma_adapter._get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)

        facts = []
        for meta in results["metadatas"][0]:
            if meta.get("type") == "fact":
                facts.append(
                    Fact(
                        subject=meta.get("subject", ""),
                        predicate=meta.get("predicate", ""),
                        object=meta.get("object", "")
                    )
                )
        return facts
