from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db_session, require_api_key, get_optional_user
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import ChatService


router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
    user=Depends(get_optional_user),
):
    """
    OpenAI-compatible chat completion endpoint.
    Supports streaming and non-streaming responses.
    Auto-routes to the best available model based on message complexity.
    """
    chat_service = ChatService(db)

    # Simple model routing: pick based on total message length
    total_len = sum(len(m.content) for m in request.messages)
    if request.model:
        model_name = request.model
    elif total_len > 300:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        model_name = "google/gemma-7b-it"

    if request.stream:
        async def event_stream():
            async for chunk in chat_service.stream_response(
                request=request,
                model_name=model_name,
                user=user,
            ):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = await chat_service.generate_response(
        request=request,
        model_name=model_name,
        user=user,
    )
    return response