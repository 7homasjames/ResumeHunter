from fastapi import FastAPI
from dotenv import load_dotenv
import os
import hashlib
import json
import threading
from pydantic import BaseModel
from typing import List
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- Load Env & Keys ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "shlrag"

GENAI_MODEL_ID = "models/gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Initialize Models ----------
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel(GENAI_MODEL_ID)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Initialize Pinecone ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# ---------- Initialize FastAPI ----------
app = FastAPI()

# ---------- Utility Functions ----------
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def prepare_jsons_for_rag(json_paths):
    items = []
    global_index = 0
    for json_path in json_paths:
        filename = os.path.basename(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)
        for job in job_data:
            slug = job.get("slug", f"job-{global_index}")
            for rec in job.get("recommendations", []):
                text = json.dumps(rec)
                unique_id = f"{slug}-{global_index}"
                items.append({
                    "id": unique_id,
                    "line": text,
                    "filename": filename,
                    "page_number": "1"
                })
                global_index += 1
    return {"items": items}

def upsert_documents(documents, batch_size=50):
    vectors = []
    for doc in documents:
        embedding = embedding_model.encode(doc["line"]).tolist()
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {"text": doc["line"]}
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        print(f"Upserting batch {i//batch_size + 1} of {len(vectors)//batch_size + 1}")
        index.upsert(vectors=batch)

    return [doc["id"] for doc in documents]

# ---------- Pydantic Models ----------
class Item(BaseModel):
    id: str
    line: str
    filename: str
    page_number: str = "1"

class Docs(BaseModel):
    items: List[Item]

class ATSCheck(BaseModel):
    resume_text: str
    job_description: str

# ---------- API Endpoints ----------
@app.post("/push_docs/")
async def push_docs(item: Docs):
    try:
        docs = item.dict()["items"]
        ids = upsert_documents(docs)
        print("Inserted IDs:", ids)
        return {"status": "success", "inserted_ids": ids}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ats_check/")
async def ats_check(item: ATSCheck):
    try:
        prompt =  f"""
        You are an ATS (Applicant Tracking System) evaluation AI.

        Your behavior rules:
        - **ATS Match Score must be between 0 and 100 only.**
        - Focus primarily on matching **skills, experiences, and keywords** from the Job Description (JD).
        - If the JD **does NOT mention any experience requirement**, prefer candidates with **more overall experience** and award a slightly higher score.
        - Never return an ATS score above 100%.

        Compare the following Resume and Job Description:

        - Resume:
        {item.resume_text}

        - Job Description:
        {item.job_description}

        **Your Task:**
        1. Assign an **ATS Match Score (0-100)** based on the above rules.
        2. Create a detailed table:

        | Category | Matched Skills/Keywords | Missing Skills/Keywords | Comments |

        Focus points:
        - Skills and keywords alignment.
        - Relevant experiences mentioned.
        - If no experience is specified in JD, reward candidates with greater experience.

        **Output format (strictly):**
        1. ATS Match Score: __%
        2. Table:
        """


        response = generation_model.generate_content(prompt)
        return {"output": response.text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear_pinecone/")
async def clear_pinecone():
    try:
        index.delete(delete_all=True)
        return {"status": "success", "message": "Pinecone index cleared."}
    except Exception as e:
        return {"error": str(e)}

# ---------- Startup Loader ----------
def auto_push_job_data():
    try:
        file_path = "job_descriptions.json"
        if os.path.exists(file_path):
            print("📤 Indexing job_descriptions.json into Pinecone...")
            json_files = ["job_descriptions.json", "job_descriptions_1.json"]
            data = prepare_jsons_for_rag(json_files)
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.post("/push_docs/", json=data)
            print("✅ Job data indexed:", response.json())
        else:
            print("⚠️ job_descriptions.json not found.")
    except Exception as e:
        print("❌ Error pushing job data:", str(e))

threading.Thread(target=auto_push_job_data, daemon=True).start()
