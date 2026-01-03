import os
from pathlib import Path
from pydantic import BaseModel


class FileCreateRequest(BaseModel):
    filename: str
    content: str


class FileEditRequest(BaseModel):
    filename: str
    new_content: str


class FileDeleteRequest(BaseModel):
    filename: str


class FileHandler:
    def __init__(self, base_dir: str = "./workspace"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _resolve_path(self, filename: str) -> str:
        """Ensure the file stays within the sandbox directory."""
        filepath = os.path.abspath(os.path.join(self.base_dir, filename))
        if not filepath.startswith(os.path.abspath(self.base_dir)):
            raise ValueError("File path escapes sandbox directory")
        return filepath

    def create_file(self, args: dict):
        req = FileCreateRequest(**args)
        filepath = self._resolve_path(req.filename)
        if os.path.exists(filepath):
            return {"status": "error", "message": f"File already exists: {filepath}"}
        with open(filepath, "w") as f:
            f.write(req.content)
        return {"status": "file_created", "path": filepath}

    def list_files(self, max_depth: int = 2) -> str:
        """Return a structured directory listing up to max_depth."""
        output = []

        def _recurse(path: Path, depth: int):
            if depth > max_depth:
                return
            for p in sorted(path.iterdir()):
                if p.is_dir():
                    output.append(f"DIR: {p.relative_to(self.base_dir)}/")
                    _recurse(p, depth + 1)
                else:
                    output.append(f"FILE: {p.relative_to(self.base_dir)}")

        _recurse(Path(self.base_dir), 0)
        return "\n".join(output)
    
    def read_file(self, args: dict) -> dict:
        filename = args["filename"]
        filepath = self._resolve_path(filename)
        if not os.path.exists(filepath):
            return {"status": "error", "message": f"File not found: {filepath}"}
        if os.path.isdir(filepath):
            return {"status": "error", "message": "Path is a directory"}
        try:
            with open(filepath, "r") as f:
                content = f.read()
            return {"status": "success", "content": content, "path": filepath}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def edit_file(self, args: dict):
        req = FileEditRequest(**args)
        filepath = self._resolve_path(req.filename)
        if not os.path.exists(filepath):
            return {"status": "error", "message": f"File not found: {filepath}"}
        with open(filepath, "w") as f:
            f.write(req.new_content)
        return {"status": "file_updated", "path": filepath}

    def delete_file(self, args: dict):
        req = FileDeleteRequest(**args)
        filepath = self._resolve_path(req.filename)
        if not os.path.exists(filepath):
            return {"status": "error", "message": f"File not found: {filepath}"}
        os.remove(filepath)
        return {"status": "file_deleted", "path": filepath}