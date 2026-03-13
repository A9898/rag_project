"""
RAG Locale — Retrieval-Augmented Generation con Ollama + ChromaDB
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Zittisce ChromaDB telemetry

import json
import hashlib
import time
import traceback
from pathlib import Path

import httpx
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Config ──────────────────────────────────────────────────────────────
OLLAMA_BASE   = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL    = os.getenv("CHAT_MODEL", "mistral")
CHROMA_DIR    = os.getenv("CHROMA_DIR", "./chroma_data")
UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "./uploads")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K         = int(os.getenv("TOP_K", "5"))

Path(UPLOAD_DIR).mkdir(exist_ok=True)

# ─── Pydantic ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    model: str = ""

# ─── ChromaDB ────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# ─── FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Locale")

# ─── Helpers ─────────────────────────────────────────────────────────────

def get_collection():
    return chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )


async def ollama_embed(texts: list[str]) -> list[list[float]]:
    embeddings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text}
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return embeddings


async def ollama_chat_stream(messages: list[dict], context: str, model: str,
                             embed_ms: float = 0, retrieval_ms: float = 0):
    system_prompt = (
        "Sei un assistente AI utile e preciso. Rispondi SEMPRE nella stessa lingua della domanda.\n"
        "Usa ESCLUSIVAMENTE il contesto fornito per rispondere. Se il contesto non contiene "
        "informazioni sufficienti, dillo chiaramente. Non inventare informazioni.\n\n"
        f"CONTESTO:\n{context}"
    )

    all_messages = [{"role": "system", "content": system_prompt}] + messages
    use_model = model if model else CHAT_MODEL

    print(f"[OLLAMA] Invio richiesta → model='{use_model}'")

    t_start = time.time()
    first_token_time = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json={"model": use_model, "messages": all_messages, "stream": True}
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    err = body.decode(errors="ignore")
                    print(f"[OLLAMA] Errore HTTP {resp.status_code}: {err}")
                    yield f"data: {json.dumps({'token': f'Errore Ollama ({resp.status_code}): {err[:200]}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time()
                            token_count += 1
                            yield f"data: {json.dumps({'token': token})}\n\n"

                        if data.get("done"):
                            t_end = time.time()
                            total_s = t_end - t_start
                            ttft = (first_token_time - t_start) if first_token_time else total_s

                            # Ollama manda metriche nel messaggio finale (nanosecondi)
                            eval_count = data.get("eval_count", token_count)
                            eval_duration_ns = data.get("eval_duration", 0)
                            prompt_eval_count = data.get("prompt_eval_count", 0)
                            prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

                            # Calcola tok/s da Ollama se disponibile, altrimenti stima
                            if eval_duration_ns > 0:
                                tok_per_s = eval_count / (eval_duration_ns / 1e9)
                            elif total_s > 0:
                                tok_per_s = eval_count / total_s
                            else:
                                tok_per_s = 0

                            if prompt_eval_duration_ns > 0:
                                prompt_tok_per_s = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
                            else:
                                prompt_tok_per_s = 0

                            metrics = {
                                "model": use_model,
                                "total_s": round(total_s, 2),
                                "ttft_s": round(ttft, 2),
                                "tokens_generated": eval_count,
                                "tok_per_s": round(tok_per_s, 1),
                                "prompt_tokens": prompt_eval_count,
                                "prompt_tok_per_s": round(prompt_tok_per_s, 1),
                                "embed_ms": round(embed_ms),
                                "retrieval_ms": round(retrieval_ms),
                            }

                            print(f"[OLLAMA] Completato: {eval_count} tok in {total_s:.1f}s ({tok_per_s:.1f} tok/s), TTFT={ttft:.2f}s")

                            yield f"data: {json.dumps({'metrics': metrics})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError:
        print("[OLLAMA] Connessione rifiutata")
        yield f"data: {json.dumps({'token': 'Errore: Ollama non raggiungibile. Verifica che sia attivo.'})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"[OLLAMA] Eccezione: {traceback.format_exc()}")
        yield f"data: {json.dumps({'token': f'Errore: {e}'})}\n\n"
        yield "data: [DONE]\n\n"


def extract_text(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)
    elif ext == ".docx":
        from docx import Document
        return "\n\n".join(p.text for p in Document(file_path).paragraphs if p.text.strip())
    elif ext in (".txt", ".md", ".csv", ".json", ".html", ".xml"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raise HTTPException(400, f"Formato non supportato: {ext}")


def chunk_text(text: str, filename: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [
        {"text": c, "metadata": {"source": filename, "chunk_index": i, "total_chunks": len(chunks)}}
        for i, c in enumerate(chunks)
    ]


# ─── Routes ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "ok", "ollama": True, "models": models,
                    "embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL}
    except Exception as e:
        return {"status": "error", "ollama": False, "error": str(e)}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(file_path, file.filename)
    if not text.strip():
        raise HTTPException(400, "Nessun testo estratto.")

    chunks = chunk_text(text, file.filename)
    embeddings = await ollama_embed([c["text"] for c in chunks])

    collection = get_collection()
    ids = [hashlib.md5(f"{file.filename}_{i}_{c['text'][:50]}".encode()).hexdigest()
           for i, c in enumerate(chunks)]
    collection.add(ids=ids, embeddings=embeddings,
                   documents=[c["text"] for c in chunks],
                   metadatas=[c["metadata"] for c in chunks])

    return {"status": "ok", "filename": file.filename,
            "chunks": len(chunks), "characters": len(text)}


@app.get("/api/documents")
async def list_documents():
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    sources = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "?")
        sources.setdefault(src, {"name": src, "chunks": 0})
        sources[src]["chunks"] += 1
    return {"documents": list(sources.values()), "total_chunks": len(results["ids"])}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    collection = get_collection()
    results = collection.get(include=["metadatas"], where={"source": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    fp = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(fp):
        os.remove(fp)
    return {"status": "ok", "deleted_chunks": len(results["ids"])}


@app.post("/api/chat")
async def chat(request: Request):
    # Parse body manualmente per evitare errori Pydantic
    try:
        body = await request.json()
    except Exception as e:
        print(f"[CHAT] Errore parsing JSON: {e}")
        async def err():
            yield f"data: {json.dumps({'token': f'Errore: body non valido'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    query = body.get("query", "")
    model = body.get("model", "") or ""
    history = body.get("history", []) or []

    print(f"\n{'='*50}")
    print(f"[CHAT] query  = '{query[:80]}'")
    print(f"[CHAT] model  = '{model}' (default: '{CHAT_MODEL}')")
    print(f"[CHAT] history = {len(history)} messaggi")

    # 1. Embedding
    try:
        t0 = time.time()
        query_emb = (await ollama_embed([query]))[0]
        embed_ms = (time.time() - t0) * 1000
        print(f"[CHAT] Embedding OK ({len(query_emb)} dims) in {embed_ms:.0f}ms")
    except Exception as e:
        print(f"[CHAT] Embedding FALLITO: {e}")
        async def err():
            yield f"data: {json.dumps({'token': f'Errore embedding: {e}'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # 2. Retrieval
    t0 = time.time()
    collection = get_collection()
    count = collection.count()
    n = min(TOP_K, count) if count > 0 else 0
    print(f"[CHAT] ChromaDB: {count} chunks totali, cerco top {n}")

    context_parts = []
    sources = set()
    if n > 0:
        results = collection.query(query_embeddings=[query_emb], n_results=n,
                                   include=["documents", "metadatas", "distances"])
        for doc, meta, dist in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0]):
            sim = 1 - dist
            if sim > 0.3:
                src = meta.get("source", "?")
                sources.add(src)
                context_parts.append(f"[Da: {src}]\n{doc}")

    retrieval_ms = (time.time() - t0) * 1000
    context = "\n\n---\n\n".join(context_parts) if context_parts else "Nessun contesto trovato."
    print(f"[CHAT] Contesto: {len(context_parts)} chunk in {retrieval_ms:.0f}ms, fonti: {sources}")

    # 3. Stream
    messages = list(history) + [{"role": "user", "content": query}]

    return StreamingResponse(
        ollama_chat_stream(messages, context, model=model,
                           embed_ms=embed_ms, retrieval_ms=retrieval_ms),
        media_type="text/event-stream",
        headers={
            "X-Sources": json.dumps(list(sources)),
            "Cache-Control": "no-cache",
            "Access-Control-Expose-Headers": "X-Sources"
        }
    )


@app.delete("/api/reset")
async def reset():
    try:
        chroma_client.delete_collection("documents")
    except Exception:
        pass
    for f in Path(UPLOAD_DIR).iterdir():
        f.unlink()
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"""
╔══════════════════════════════════════════════╗
║         RAG Locale — Avviato!               ║
║  UI:     http://localhost:8000               ║
║  Ollama: {OLLAMA_BASE:<36s}║
║  LLM:    {CHAT_MODEL:<36s}║
║  Embed:  {EMBED_MODEL:<36s}║
╚══════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
