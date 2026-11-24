from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

app = FastAPI()

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b")

# Prompt template (history lets the bot remember the conversation)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and highly knowledgeable AI assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict] = []


def to_messages(history):
    """Convert [{role, content}, ...] from the browser into LangChain messages."""
    messages = []
    for item in history[-20:]:
        content = str(item.get("content", ""))
        if item.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "Please enter a prompt."}, status_code=400)

    history = to_messages(req.history)

    async def generate():
        try:
            async for chunk in chain.astream({"question": question, "history": history}):
                yield chunk
        except Exception as exc:
            print("Chat error:", exc)
            yield "\n\n⚠️ Something went wrong while contacting the model. Check your GROQ_API_KEY and try again."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serves static/index.html at "/" (keep this last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
