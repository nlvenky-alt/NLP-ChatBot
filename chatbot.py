# ============================================================================
# 25th ITMC CHATBOT - PRODUCTION VERSION 1.0
# GPU Support: CUDA (NVIDIA) + DirectML (Intel/AMD integrated)
# ============================================================================

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import subprocess
import time
import requests
import sys
import json
import pickle
import faiss
import re
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise RuntimeError("sentence-transformers package required. Run: pip install sentence-transformers")

# ============================================================================
# GPU DETECTION - Supports CUDA (NVIDIA) and DirectML (Intel/AMD)
# ============================================================================

DEVICE = "cpu"
GPU_NAME = None
GPU_MEM_GB = 0
GPU_TYPE = "None"

try:
    import torch
    
    # First, check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        GPU_TYPE = "NVIDIA CUDA"
    """
    else:
        # Check for DirectML (Intel/AMD GPU on Windows)
        try:
            import torch_directml
            DEVICE = torch_directml.device()
            GPU_NAME = "Intel/AMD GPU (DirectML)"
            GPU_TYPE = "DirectML"
            # Estimate shared memory (integrated GPUs typically use 25-50% of system RAM)
            if PSUTIL_AVAILABLE:
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                GPU_MEM_GB = round(total_ram_gb * 0.3, 1)  # Assume 30% shared
            else:
                GPU_MEM_GB = 4.0  # Default assumption
        except ImportError:
            # No GPU acceleration available
            pass
    """
except ImportError:
    pass

# Paths
BASEDIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(os.path.dirname(BASEDIR), "knowledge_base")
PERSIST_DIR = os.path.join(os.path.dirname(BASEDIR), "persistent_kb")
EMBED_DIR = os.path.join(os.path.dirname(BASEDIR), "embedding_models")
os.makedirs(PERSIST_DIR, exist_ok=True)

KB_MODE = "full"

print("=" * 70)
print("💬 25th ITMC CHATBOT - PRODUCTION v1.0")
print("=" * 70)

# Model registry (unchanged from your original)
MODEL_REGISTRY = {
    "llama3.2:1b": {
        "size_gb": 1.3, "min_ram_gb": 3.5, "speed": "very_fast", "quality": "good",
        "temperature": 0.1, "num_ctx": 2048, "num_predict": 600, "top_p": 0.85, "top_k": 30,
    },
    "llama3.2:3b": {
        "size_gb": 2.0, "min_ram_gb": 5.0, "speed": "fast", "quality": "very_good",
        "temperature": 0.2, "num_ctx": 3072, "num_predict": 800, "top_p": 0.85, "top_k": 30,
    },
    "llama3.1:8b": {
        "size_gb": 4.7, "min_ram_gb": 999.0, "speed": "slow", "quality": "exceptional",
        "temperature": 0.3, "num_ctx": 8192, "num_predict": 1500, "top_p": 0.9, "top_k": 40,
    },
}

MODEL_ALIASES = {
    "phi3.5": "phi3.5:3.8b-mini-instruct-q4_K_M",
    "phi": "phi3.5:3.8b-mini-instruct-q4_K_M",
}

# All your existing utility functions remain unchanged
def get_system_resources():
    if not PSUTIL_AVAILABLE:
        return {"total_ram_gb": 8.0, "available_ram_gb": 4.0, "used_ram_percent": 50.0, "cpu_count": 4, "cpu_usage_percent": 50.0}
    try:
        mem = psutil.virtual_memory()
        return {
            "total_ram_gb": round(mem.total / (1024**3), 1),
            "available_ram_gb": round(mem.available / (1024**3), 1),
            "used_ram_percent": round(mem.percent, 1),
            "cpu_count": psutil.cpu_count(),
            "cpu_usage_percent": round(psutil.cpu_percent(interval=0.5), 1),
        }
    except:
        return {"total_ram_gb": 8.0, "available_ram_gb": 4.0, "used_ram_percent": 50.0, "cpu_count": 4, "cpu_usage_percent": 50.0}

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

def start_ollama():
    if check_ollama():
        return True
    print("Starting Ollama server...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        if check_ollama():
            print("✓ Ollama started.")
            return True
        print("⚠ Ollama did not respond on port 11434.")
        return False
    except Exception as e:
        print(f"❌ Could not start Ollama: {e}")
        return False

def get_available_models():
    try:
        if not check_ollama():
            return []
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
        return []
    except:
        return []

def classify_query_complexity(query: str) -> str:
    q = query.lower()
    complex_kw = ["analyze", "analyse", "compare", "evaluate", "summarize", "summarise",
                  "explain in detail", "comprehensive", "step by step", "reasoning", "discuss"]
    numerical_kw = ["how many", "number", "count", "percentage", "percent", "%",
                    "statistics", "total", "amount", "budget", "cost", "figure", "share", "growth", "lakh", "crore"]
    simple_kw = ["what is", "who is", "when", "where", "define", "meaning of", "full form", "stands for"]
    score = 0
    if any(k in q for k in complex_kw): score += 3
    if any(k in q for k in numerical_kw): score += 2
    if any(k in q for k in simple_kw): score -= 1
    words = len(query.split())
    if words > 15: score += 2
    elif words < 8: score += 1
    if score >= 4: return "complex"
    elif score >= 2 or any(k in q for k in numerical_kw): return "moderate"
    else: return "simple"

def resolve_model_name(name: str) -> str:
    name = name.strip()
    if name in MODEL_REGISTRY: return name
    if name in MODEL_ALIASES: return MODEL_ALIASES[name]
    return name

def select_optimal_model(query, detailed=False, force_model=None):
    available = get_available_models()
    resources = get_system_resources()
    avail_ram = resources["available_ram_gb"]
    if force_model:
        fm = resolve_model_name(force_model)
        if fm in MODEL_REGISTRY and fm in available:
            if MODEL_REGISTRY[fm]["min_ram_gb"] > 100:
                print(f"⚠ Model {force_model} is disabled (known issues). Using auto-selection.")
            else:
                print(f"✓ Forced model: {fm}")
                return fm, MODEL_REGISTRY[fm]
        else:
            print(f"⚠ Forced model {force_model} not available; falling back to auto.")
    installed = {m: cfg for m, cfg in MODEL_REGISTRY.items() if m in available and cfg["min_ram_gb"] < 100}
    if not installed:
        if available: return available[0], MODEL_REGISTRY.get(available[0], MODEL_REGISTRY["llama3.2:1b"])
        return "llama3.2:1b", MODEL_REGISTRY["llama3.2:1b"]
    complexity = classify_query_complexity(query)
    candidates = []
    for name, cfg in installed.items():
        if avail_ram < cfg["min_ram_gb"]: continue
        score = 0
        if detailed or complexity == "complex":
            score = 10 if cfg["quality"] in ("exceptional", "excellent") else 7
        elif complexity == "moderate":
            score = 8 if cfg["quality"] in ("excellent", "very_good") else 5
        else:
            score = 8 if cfg["speed"] in ("very_fast", "fast") else 4
        candidates.append((name, cfg, score))
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]
    smallest = min(installed.items(), key=lambda x: x[1]["size_gb"])
    return smallest[0], smallest[1]

# ============================================================================
# LOAD KNOWLEDGE BASE WITH GPU SUPPORT
# ============================================================================

print("\n📚 Loading knowledge base...")
try:
    faiss_index = faiss.read_index(os.path.join(KB_DIR, "faiss_index.bin"))
    with open(os.path.join(KB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    for c in chunks:
        c["is_persistent"] = False
        if "databank" not in c:
            c["databank"] = "original"
    
    # Load embedding model with GPU if available
    emb_path = os.path.join(EMBED_DIR, "all-MiniLM-L6-v2")
    
    # For DirectML, we need to handle device differently
    if GPU_TYPE == "DirectML":
        # Load on CPU first, then move to DirectML device
        if os.path.exists(emb_path):
            embedding_model = SentenceTransformer(emb_path, device="cpu")
            print(f"✓ Loaded embedding model from {emb_path}")
        else:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            print("✓ Loaded embedding model from cache")
        
        # Move to DirectML device
        try:
            import torch
            embedding_model = embedding_model.to(DEVICE)
            print(f"🚀 Using DirectML GPU: {GPU_NAME} (~{GPU_MEM_GB} GB shared memory)")
        except Exception as e:
            print(f"⚠ Could not move to DirectML device: {e}")
            print("💻 Falling back to CPU")
            DEVICE = "cpu"
    else:
        # Standard loading for CUDA or CPU
        device_str = str(DEVICE) if DEVICE != "cpu" else "cpu"
        if os.path.exists(emb_path):
            embedding_model = SentenceTransformer(emb_path, device=device_str)
            print(f"✓ Loaded embedding model from {emb_path}")
        else:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device_str)
            print("✓ Loaded embedding model from cache")
        
        if GPU_TYPE == "NVIDIA CUDA":
            print(f"🚀 Using NVIDIA GPU: {GPU_NAME} ({GPU_MEM_GB} GB VRAM)")
        else:
            print(f"💻 Using CPU for embeddings")
    
    print(f"✓ Loaded {len(chunks)} chunks from knowledge base")
    
    # Load persistent KB if exists
    persist_chunks = []
    persist_chunks_path = os.path.join(PERSIST_DIR, "persistent_chunks.pkl")
    persist_embeddings_path = os.path.join(PERSIST_DIR, "persistent_embeddings.npy")
    
    if os.path.exists(persist_chunks_path) and os.path.exists(persist_embeddings_path):
        import numpy as np
        with open(persist_chunks_path, "rb") as f:
            persist_chunks = pickle.load(f)
        for c in persist_chunks:
            c["is_persistent"] = True
        persist_embeddings = np.load(persist_embeddings_path)
        faiss_index.add(persist_embeddings.astype("float32"))
        chunks.extend(persist_chunks)
        print(f"✓ Loaded {len(persist_chunks)} chunks from uploaded documents")
    
    if any("Python-Programming.pdf" in c.get("source_file", "") for c in chunks):
        print("⚠ Note: Python-Programming.pdf has known retrieval issues")
    
    print(f"✓ Total chunks: {len(chunks)}")

except Exception as e:
    print(f"❌ Error loading KB: {e}")
    raise RuntimeError(f"Failed to load knowledge base: {e}")

start_ollama()

print("\n🤖 Checking available models...")
available_models = get_available_models()
if available_models:
    print(f"✓ Found {len(available_models)} models:")
    for name in available_models:
        if name in MODEL_REGISTRY and MODEL_REGISTRY[name]["min_ram_gb"] < 100:
            cfg = MODEL_REGISTRY[name]
            print(f"  • {name} - {cfg['quality']} quality, {cfg['speed']} speed")
else:
    print("⚠ No models detected via Ollama.")

res = get_system_resources()
print("\n💾 Resources:")
print(f"  • RAM: {res['available_ram_gb']:.1f} GB available / {res['total_ram_gb']:.1f} GB total")
print(f"  • CPU: {res['cpu_count']} cores, {res['cpu_usage_percent']}% usage")
print(f"  • GPU: {GPU_TYPE} - {GPU_NAME or 'Not available'}")
if GPU_MEM_GB > 0:
    print(f"    Memory: {GPU_MEM_GB} GB {'(shared)' if GPU_TYPE == 'DirectML' else '(dedicated)'}")

# ============================================================================
# All remaining functions unchanged (extract_numbers, validate, save, query, etc.)
# Copy the rest of your existing chatbot.py code here...
# ============================================================================

def extract_numbers(text: str):
    return re.findall(r"\d+\.?\d*", text)

def validate_numerical_answer(question: str, answer: str, context: str):
    ans_nums = set(extract_numbers(answer))
    ctx_nums = set(extract_numbers(context))
    if not ans_nums:
        return {"verified": True, "warning": ""}
    unverified = [n for n in ans_nums if n not in ctx_nums]
    if unverified:
        return {"verified": False, "warning": f"Numbers not in source: {', '.join(unverified)}"}
    return {"verified": True, "warning": ""}

def save_persistent_kb():
    try:
        if persist_chunks:
            import numpy as np
            print("💾 Saving persistent KB...")
            emb = embedding_model.encode([c["text"] for c in persist_chunks], batch_size=32, convert_to_numpy=True)
            with open(os.path.join(PERSIST_DIR, "persistent_chunks.pkl"), "wb") as f:
                pickle.dump(persist_chunks, f)
            np.save(os.path.join(PERSIST_DIR, "persistent_embeddings.npy"), emb.astype("float32"))
            print("✓ Persistent KB saved.")
    except Exception as e:
        print(f"❌ Error saving: {e}")

def clear_persistent_kb():
    try:
        for f in [os.path.join(PERSIST_DIR, "persistent_chunks.pkl"), os.path.join(PERSIST_DIR, "persistent_embeddings.npy")]:
            if os.path.exists(f):
                os.remove(f)
        global chunks, persist_chunks, faiss_index
        chunks = [c for c in chunks if not c.get("is_persistent")]
        persist_chunks.clear()
        import numpy as np
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        if chunks:
            texts = [c["text"] for c in chunks]
            emb = embedding_model.encode(texts, batch_size=32, convert_to_numpy=True)
            faiss_index.add(emb.astype("float32"))
        print("✓ Cleared persistent KB.")
    except Exception as e:
        print(f"❌ Error: {e}")

def clear_file_chunks(filename: str):
    global chunks, persist_chunks, faiss_index
    print(f"🗑 Removing {filename}")
    try:
        remaining = [c for c in chunks if c.get("source_file") != filename]
        persist_remaining = [c for c in persist_chunks if c.get("source_file") != filename]
        removed = len(chunks) - len(remaining)
        chunks = remaining
        persist_chunks = persist_remaining
        import numpy as np
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        if chunks:
            emb = embedding_model.encode([c["text"] for c in chunks], batch_size=32, convert_to_numpy=True)
            faiss_index.add(emb.astype("float32"))
        save_persistent_kb()
        print(f"✓ Removed {removed} chunks")
    except Exception as e:
        print(f"❌ Error: {e}")

def simple_text_splitter(text, chunk_size=1500, overlap=500):
    out, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        start += chunk_size - overlap
    return out

def process_file(filepath, bank_name=None, persist=True):
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    text = ""
    print("=" * 70)
    print(f"📄 Processing: {filename}")
    print("=" * 70)
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            print(f"📄 {len(reader.pages)} pages")
            for i, page in enumerate(reader.pages):
                if i and i % 50 == 0:
                    print(f"   {i}/{len(reader.pages)} pages...")
                try:
                    pt = page.extract_text()
                    if pt and len(pt.strip()) > 10:
                        text += pt
                except:
                    continue
        elif ext == ".docx":
            from docx import Document
            for p in Document(filepath).paragraphs:
                if p.text.strip():
                    text += p.text + "\n"
        elif ext in (".xlsx", ".xls"):
            import openpyxl
            for sheet in openpyxl.load_workbook(filepath, read_only=True, data_only=True).worksheets:
                for row in sheet.iter_rows(values_only=True):
                    line = " ".join(str(c) for c in row if c)
                    if line.strip():
                        text += line + "\n"
        else:
            print(f"❌ Unsupported: {ext}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    if len(text) < 50:
        print("⚠ File too short")
        return None
    print("✂ Chunking...")
    chunk_texts = simple_text_splitter(text)
    print(f"✓ {len(chunk_texts)} chunks")
    new_chunks = [
        {"text": ct, "source_file": filename, "source_folder": bank_name or "uploaded",
         "file_type": ext, "chunk_id": i, "total_chunks": len(chunk_texts),
         "databank": bank_name or "uploaded", "timestamp": datetime.now().isoformat(),
         "is_persistent": bool(persist)}
        for i, ct in enumerate(chunk_texts)
    ]
    print("🔢 Encoding...")
    emb = embedding_model.encode([c["text"] for c in new_chunks], batch_size=32, convert_to_numpy=True)
    faiss_index.add(emb.astype("float32"))
    chunks.extend(new_chunks)
    if persist:
        persist_chunks.extend(new_chunks)
        save_persistent_kb()
    print(f"✓ Added {len(new_chunks)} chunks")
    print("=" * 70)
    return new_chunks

def load_databank(folder_path, persist=True):
    folder_path = folder_path.strip('"').strip("'").strip()
    if not os.path.exists(folder_path):
        print(f"❌ Not found: {folder_path}")
        return
    print(f"📁 Loading: {os.path.basename(folder_path)}")
    files = []
    for root, dirs, fs in os.walk(folder_path):
        for f in fs:
            if f.lower().endswith((".pdf", ".docx", ".txt", ".xlsx", ".xls")):
                files.append(os.path.join(root, f))
    if not files:
        print("⚠ No supported files")
        return
    print(f"Found {len(files)} files")
    count = sum(1 for i, p in enumerate(files, 1) if process_file(p, bank_name=os.path.basename(folder_path), persist=persist))
    print(f"✓ Loaded {count}/{len(files)} files")

def query_rag(question, top_k=25, detailed=False, target_file=None, force_model=None):
    if not check_ollama():
        print("❌ Ollama not responding")
        return None
    global KB_MODE
    complexity = classify_query_complexity(question)
    is_numerical = any(k in question.lower() for k in ["how many", "number", "count", "percentage", "percent", "%", "statistics", "total", "amount", "budget", "cost", "figure", "share", "growth"])
    model_name, model_cfg = select_optimal_model(question, detailed=detailed, force_model=force_model)
    print(f"🔍 Searching: {question}")
    print(f"✓ Selected Model: {model_name} ({model_cfg['quality']} quality)")
    print(f"✓ Query Type: {complexity.upper()} {'[NUMERICAL]' if is_numerical else ''}")
    print(f"✓ KB Mode: {KB_MODE.upper()}")
    if detailed: top_k = 40
    search_k = min(len(chunks), top_k + 20 if target_file else top_k)
    q_emb = embedding_model.encode([question])
    distances, indices = faiss_index.search(q_emb.astype("float32"), search_k)
    context, sources, found = "", [], 0
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= len(chunks): continue
        ch = chunks[idx]
        if target_file:
            if target_file.lower() not in ch.get("source_file", "").lower(): continue
        else:
            if KB_MODE == "base" and ch.get("is_persistent"): continue
        context += f"{ch['source_file']}:\n{ch['text']}\n\n"
        sources.append(ch["source_file"])
        found += 1
        if found >= top_k: break
    if found == 0:
        print("❌ No relevant info found")
        return None
    unique_sources = sorted(set(sources))
    print(f"✓ Retrieved {found} chunks from {len(unique_sources)} files")
    if is_numerical:
        prompt = f"""You are a precise data analyst. Answer ONLY using the document excerpts.
RULES:
1. Extract numbers EXACTLY as written (e.g. "3.6%", "1.54 lakh")
2. Include units and context
3. If specific number NOT in excerpts, say: "The specific numerical value requested is not mentioned in the provided excerpts."
4. DO NOT infer, estimate, or guess
5. NO phrases like "we can infer", "likely", "probably"
DOCUMENT EXCERPTS:
{context}
QUESTION: {question}
PRECISE ANSWER:"""
    elif detailed or complexity == "complex":
        prompt = f"""Provide detailed answer using ONLY the excerpts.
RULES:
1. Use only explicit information
2. State numbers/dates exactly as written
3. If insufficient info, say so clearly
4. Structure with clear paragraphs/bullets
DOCUMENT EXCERPTS:
{context}
QUESTION: {question}
DETAILED ANSWER:"""
    else:
        prompt = f"""Answer concisely using ONLY the excerpts.
DOCUMENT EXCERPTS:
{context}
QUESTION: {question}
CONCISE ANSWER:"""
    try:
        use_stream = not is_numerical
        if use_stream:
            print("🤖 Generating answer (streaming)...")
            r = requests.post("http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": True,
                      "options": {"temperature": model_cfg["temperature"], "top_p": model_cfg["top_p"], "top_k": model_cfg["top_k"],
                                  "repeat_penalty": 1.3, "num_ctx": model_cfg["num_ctx"], "num_predict": model_cfg["num_predict"]}},
                stream=True, timeout=300)
            print("=" * 70)
            print(f"ANSWER (Model: {model_name})")
            print("=" * 70)
            full_response = ""
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            text = data["response"]
                            print(text, end="", flush=True)
                            full_response += text
                    except: continue
        else:
            print("🤖 Generating answer (numerical, non-streaming)...")
            r = requests.post("http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": model_cfg["temperature"], "top_p": model_cfg["top_p"], "top_k": model_cfg["top_k"],
                                  "repeat_penalty": 1.3, "num_ctx": model_cfg["num_ctx"], "num_predict": model_cfg["num_predict"]}},
                timeout=300)
            full_response = r.json().get("response", "")
            validation = validate_numerical_answer(question, full_response, context)
            print("=" * 70)
            print(f"ANSWER (Model: {model_name})")
            print("=" * 70)
            if validation["verified"]:
                print(full_response)
            else:
                print("The specific numerical value requested cannot be safely extracted from the provided excerpts without guessing.")
                print(validation["warning"])
        print("\n" + "=" * 70)
        print(f"📚 Sources ({len(unique_sources)}): {', '.join(unique_sources[:5])}")
        if len(unique_sources) > 5:
            print(f"   + {len(unique_sources) - 5} more")
        print(f"🤖 Model: {model_name} | Type: {complexity} {'[numerical]' if is_numerical else ''}")
        print("=" * 70)
        return full_response
    except requests.exceptions.Timeout:
        print("⏱ Timeout. Try simpler question.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def parse_command(cmd):
    params = {"question": cmd.strip(), "force_model": None, "target_file": None, "detailed": False}
    while True:
        text, lower = params["question"].strip(), params["question"].lower().strip()
        original = text
        if lower.startswith("model:"):
            raw = text[len("model:"):].strip()
            if " " in raw:
                first = raw.find(" ")
                second = raw.find(":", first + 1) if first != -1 else -1
                if second != -1:
                    params["force_model"] = raw[:second].strip()
                    params["question"] = raw[second + 1:].strip()
                else:
                    parts = raw.split(" ", 1)
                    if len(parts) == 2:
                        params["force_model"] = parts[0].strip()
                        params["question"] = parts[1].strip()
            continue
        if lower.startswith("file:"):
            raw = text[len("file:"):].strip()
            if " " in raw:
                fname, rest = raw.split(" ", 1)
                params["target_file"] = fname.strip()
                params["question"] = rest.strip()
            continue
        if lower.startswith("detailed:"):
            params["detailed"] = True
            params["question"] = text[len("detailed:"):].strip()
            continue
        if params["question"] == original:
            break
    return params

def main():
    global KB_MODE
    print("=" * 70)
    print("💬 ADAPTIVE CHATBOT READY")
    print("=" * 70)
    print("\nType 'help' for commands.\n")
    while True:
        try:
            cmd = input("❓ Your question: ").strip()
            if not cmd: continue
            low = cmd.lower()
            if low in ("exit", "quit", "q"):
                print("\n👋 Goodbye!\n")
                break
            if low == "help":
                print("""
📖 COMMANDS:
  <question>                   Ask any question
  detailed: <question>         Detailed answer
  file: <filename> <question>  Query specific file
  model: <name> <question>     Force specific model
  mode: base/full              Toggle KB mode
  models                       List available models
  resources                    Show system resources
  status                       Show KB status
  list                         List indexed files
  upload                       Upload single file
  databank                     Load folder of files
  clear                        Clear all uploaded documents
  clear_file: <filename>       Remove specific file
  exit                         Quit
""")
                continue
            if low == "models":
                print("\n🤖 AVAILABLE MODELS:\n")
                avail = get_available_models()
                if not avail:
                    print("  (None detected)\n")
                else:
                    for m in avail:
                        if m in MODEL_REGISTRY and MODEL_REGISTRY[m]["min_ram_gb"] < 100:
                            cfg = MODEL_REGISTRY[m]
                            print(f"  • {m} - Quality: {cfg['quality']}, Speed: {cfg['speed']}, Min RAM: {cfg['min_ram_gb']} GB")
                print()
                continue
            if low == "resources":
                r = get_system_resources()
                print("\n💾 SYSTEM RESOURCES:")
                print(f"  • RAM: {r['available_ram_gb']:.1f} GB free / {r['total_ram_gb']:.1f} GB total ({r['used_ram_percent']}% used)")
                print(f"  • CPU: {r['cpu_count']} cores, {r['cpu_usage_percent']}% usage")
                print(f"  • GPU: {GPU_TYPE} - {GPU_NAME or 'Not available'}")
                if GPU_MEM_GB > 0:
                    print(f"    Memory: {GPU_MEM_GB} GB {'(shared)' if GPU_TYPE == 'DirectML' else '(dedicated)'}")
                print()
                continue
            if low == "status":
                files = set(c["source_file"] for c in chunks)
                print("\n📊 STATUS:")
                print(f"  • Files indexed: {len(files)}")
                print(f"  • Total chunks: {len(chunks)}")
                print(f"  • Persistent chunks: {len(persist_chunks)}")
                print(f"  • KB mode: {KB_MODE.upper()}")
                print(f"  • Device: {GPU_TYPE if GPU_TYPE != 'None' else 'CPU'}")
                print(f"  • Ollama: {'✓' if check_ollama() else '✗'}")
                print()
                continue
            if low == "list":
                files = sorted(set(c["source_file"] for c in chunks))
                print(f"\n📄 INDEXED FILES ({len(files)}):")
                for i, f in enumerate(files, 1):
                    count = sum(1 for c in chunks if c["source_file"] == f)
                    print(f"  {i:3d}. {f} ({count} chunks)")
                print()
                continue
            if low == "upload":
                path = input("📄 File path: ").strip('"').strip("'").strip()
                if path and os.path.exists(path):
                    process_file(path, persist=True)
                else:
                    print("❌ File not found")
                continue
            if low == "databank":
                path = input("📁 Folder path: ").strip('"').strip("'").strip()
                if path:
                    load_databank(path, persist=True)
                continue
            if low == "clear":
                ans = input("Clear all uploaded documents? (yes/no): ").strip().lower()
                if ans == "yes":
                    clear_persistent_kb()
                continue
            if low.startswith("clear_file:"):
                fname = cmd.split(":", 1)[1].strip()
                if fname:
                    clear_file_chunks(fname)
                else:
                    print("Usage: clear_file: <filename>")
                continue
            if low.startswith("mode:"):
                val = cmd.split(":", 1)[1].strip().lower()
                if val in ("base", "full"):
                    KB_MODE = val
                    print(f"✓ KB mode set to {KB_MODE.upper()}")
                else:
                    print("Usage: mode: base OR mode: full")
                continue
            parsed = parse_command(cmd)
            if parsed["force_model"] or parsed["target_file"] or parsed["detailed"]:
                print(f"📝 Parsed: Model={parsed['force_model']}, File={parsed['target_file']}, Detailed={parsed['detailed']}")
            query_rag(parsed["question"], detailed=parsed["detailed"], target_file=parsed["target_file"], force_model=parsed["force_model"])
        except KeyboardInterrupt:
            print("\n⚠ Use 'exit' to quit.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
