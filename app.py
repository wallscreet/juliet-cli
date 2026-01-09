from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from datetime import datetime

from src.adapters import ContextPipeline, MessageCacheAdapter
from src.context import ChromaMemoryStore
from src.clients import XAIClient
from src.messages import Message, Turn

app = FastAPI(title="Juliet API")


# === Globals ===
iso_name = "juliet"
user_name = "wallscreet"
conversation_id = "12345"
base_path = f"isos/{iso_name}/users/{user_name}"
chroma_path = f"{base_path}/chroma_store"
json_export_path = f"{base_path}/episodic_memory.json"

chroma_store = ChromaMemoryStore(persist_dir=chroma_path)
message_cache = MessageCacheAdapter(capacity=20)

context_pipeline = ContextPipeline(
    chroma_store=chroma_store,
    message_cache=message_cache,
    iso_name=iso_name,
    user_name=user_name
)

llm_client = XAIClient()


class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str


def process_turn(user_message: str, conversation_id: str = conversation_id) -> str:
    """
    Core logic: build context → call LLM → store turn → return response
    This is synchronous and safe to call.
    """
    messages = context_pipeline.build_messages(user_request=user_message)

    response = llm_client.get_response(
        model="grok-4-1-fast-non-reasoning",
        messages=messages
    )

    request_msg = Message(
        uuid=uuid.uuid4(),
        role="user",
        speaker=user_name,
        content=user_message,
        timestamp=datetime.now().isoformat()
    )

    response_msg = Message(
        uuid=uuid.uuid4(),
        role="assistant",
        speaker="Juliet",
        content=response,
        timestamp=datetime.now().isoformat()
    )

    turn = Turn(
        uuid=uuid.uuid4(),
        conversation_id=conversation_id,
        request=request_msg,
        response=response_msg
    )

    chroma_store.store_turn(
        conversation_id=conversation_id,
        turn=turn,
        collection_name="episodic",
        json_export_path=json_export_path
    )

    message_cache.add_turn(turn)

    return messages, response


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    #conv_id = request.conversation_id or str(uuid.uuid4())
    conv_id = conversation_id
    
    # Non-streaming
    try:
        response_text = process_turn(
            user_message=request.user_message,
            conversation_id=conv_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}