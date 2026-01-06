from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel
import yaml
from typing import List
import os


class Fact(BaseModel):
    id: str = str(uuid4())
    description: str
    created_at: datetime = datetime.now()


class FactStore:
    fact_store_path: str
    #chroma_persist_dir: str
    
    def __init__(self,
                 fact_store_path: str,
                 #chroma_persist_dir: str, 
    ):
        
        self.fact_store_path = fact_store_path
        
        # Initialize YAML file if it doesn't exist
        if not os.path.exists(self.fact_store_path):
            with open(self.fact_store_path, 'w') as f:
                yaml.dump({'facts': []}, f)

    def _load(self) -> List[dict]:
        with open(self.fact_store_path, 'r') as f:
            data = yaml.safe_load(f) or {'facts': []}
        return data['facts']

    def _save(self, facts: List[dict]):
        with open(self.fact_store_path, 'w') as f:
            yaml.dump({'facts': facts}, f, default_flow_style=False)

    def append_fact(self, fact_str: str):
        facts = self._load()
        
        # Check for duplicates
        if any(f['description'].strip().lower() == fact_str.strip().lower() for f in facts):
            print(f"Similar fact already exists, skipping: {fact_str}")
            return
        
        new_fact = Fact(description=fact_str)
        facts.append(new_fact.model_dump())
        self._save(facts)
        return new_fact

    def get_all_facts(self) -> List[Fact]:
        return [Fact(**f) for f in self._load()]
    
    def store_fact_in_chromadb(self):
        pass
