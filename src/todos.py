from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel
from typing import List, Optional
import yaml
import os


class TodoItem(BaseModel):
    id: str = str(uuid4())
    description: str
    completed: bool = False
    created_at: datetime = datetime.now()


class TodoStore:
    def __init__(self, todo_store_path: str):
        self.file_path = todo_store_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                yaml.dump({'todos': []}, f, default_flow_style=False)

    def _load(self) -> List[dict]:
        with open(self.file_path, 'r') as f:
            data = yaml.safe_load(f) or {'todos': []}
        return data['todos']

    def _save(self, todos: List[dict]):
        with open(self.file_path, 'w') as f:
            yaml.dump({'todos': todos}, f, default_flow_style=False)

    def append_todo(self, description: str):
        todos = self._load()
        # Loose duplicate check: same description and not completed
        if any(t['description'].strip().lower() == description.strip().lower() and not t['completed'] for t in todos):
            print(f"Similar active todo exists, skipping: {description}")
            return

        new_todo = TodoItem(description=description)
        todos.append(new_todo.model_dump())
        self._save(todos)
        return new_todo

    def get_all_todos(self) -> List[TodoItem]:
        return [TodoItem(**t) for t in self._load()]

    def filter_todos(
        self,
        completed: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TodoItem]:
        todos = self._load()
        filtered = []
        for t in todos:
            item = TodoItem(**t)
            if completed is not None and item.completed != completed:
                continue
            if start_date and item.created_at < start_date:
                continue
            if end_date and item.created_at > end_date:
                continue
            filtered.append(item)
        return filtered

    def mark_completed(self, todo_id: str):
        todos = self._load()
        for t in todos:
            if t['id'] == todo_id:
                t['completed'] = True
                break
        self._save(todos)

    def edit_description(self, todo_id: str, new_description: str):
        todos = self._load()
        for t in todos:
            if t['id'] == todo_id:
                t['description'] = new_description
                break
        self._save(todos)

    def delete_todo(self, todo_id: str):
        todos = [t for t in self._load() if t['id'] != todo_id]
        self._save(todos)