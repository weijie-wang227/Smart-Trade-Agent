from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel

from app.data import HISTORICAL_KB
from app.vectorstore import build_vectorstore, load_vectorstore
from app.rag_agent import HSCodeAgent
from app.benchmark import run_benchmark
from app.irindex import IRIndex
from app.llm import GeminiLLM
from dotenv import load_dotenv
load_dotenv()

agent: HSCodeAgent | None = None  # global agent instance
irindex: IRIndex | None = None  # global IR index instance
llm = GeminiLLM()  # global LLM instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent  # 🔑 REQUIRED

    print("🚀 Lifespan startup begin")

    vs = load_vectorstore()
    count = vs._collection.count()
    print(f"📦 Vector store count: {count}")

    if count == 0:
        print("🔧 Building vector store from HISTORICAL_KB")
        vs = build_vectorstore(HISTORICAL_KB)

    app.state.vectorstore = vs

    descriptions = [row["description"] for row in HISTORICAL_KB]
    ir_index = IRIndex(descriptions)

    agent = HSCodeAgent(vectorstore=vs, ir_index=ir_index, llm=llm)


    print("✅ Lifespan startup complete")
    yield


app = FastAPI(
    title="Smart Trade Agent - HS Code Suggester",
    lifespan=lifespan,
)


class SuggestRequest(BaseModel):
    description: str


@app.post("/suggest")
def suggest(req: SuggestRequest):
    assert agent is not None, "Agent not initialized"
    s = agent.suggest(req.description, use_hybrid=False)
    return {
        "suggested_hs_code": s.suggested_hs_code,
        "confidence": s.confidence,
        "manual_review": s.manual_review,
        "reason": s.reason,
        "retrieved": s.retrieved,
    }

@app.get("/benchmark")
def benchmark():
    assert agent is not None, "Agent not initialized"
    return {"results": run_benchmark(agent)}
