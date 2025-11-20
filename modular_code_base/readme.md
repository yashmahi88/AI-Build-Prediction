# Yocto Build Analyzer 

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [How It Works](#how-it-works)
7. [Using the Application](#using-the-application)
8. [Vectorstore Management](#vectorstore-management)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

***

## Overview

### What is Yocto Build Analyzer?

The Yocto Build Analyzer is an AI-powered system that analyzes Yocto build configurations and predicts build success or failure. It uses:

- **RAG (Retrieval Augmented Generation)** to provide context-aware analysis
- **FAISS vectorstore** to store and retrieve relevant documentation
- **Ollama LLM** (qwen2.5:1.5b) for generating predictions and insights
- **Feedback learning** to improve predictions over time

### Key Features

- Analyze Yocto build scripts and predict outcomes
- Search knowledge base of Yocto files and Jenkins logs
- Learn from user feedback to improve accuracy
- Real-time vectorstore updates via file monitoring
- OpenAI-compatible API endpoints

***

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User/Application                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Application (Port 8000)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Routes: /v1/chat/completions, /api/rebuild │ │
│  └────────────────────────────────────────────────────────┘ │
└─────┬───────────────────┬─────────────────────┬────────────┘
      │                   │                     │
      ▼                   ▼                     ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ VectorStore  │  │   Ollama     │  │   PostgreSQL     │
│   Service    │  │   LLM API    │  │    Database      │
│  (FAISS)     │  │ (localhost:  │  │  (Predictions &  │
│              │  │   11434)     │  │   Feedback)      │
└──────┬───────┘  └──────────────┘  └──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│           Vectorstore (./vectorstore)                 │
│  ┌────────────────────────────────────────────────┐  │
│  │ index.faiss  - Vector embeddings                │  │
│  │ index.pkl    - Document metadata                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```
***

## Prerequisites

### Hardware Requirements

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **GPU**: Optional (CPU-only mode supported)

### Software Requirements

```bash
# Operating System
Ubuntu 20.04+ / Debian 11+ / RHEL 8+

# Python
Python 3.10+

# Ollama
Ollama v0.1.0+

# PostgreSQL
PostgreSQL 13+

# System packages
curl, git, build-essential
```

***

## Installation & Setup

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip \
    postgresql postgresql-contrib curl git build-essential

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Create Project Structure

```bash
# Create base directory
mkdir -p /home/azureuser/rag-system
cd /home/azureuser/rag-system

# Create virtual environment
python3.10 -m venv rag_env
source rag_env/bin/activate

```

### Step 3: Install Python Dependencies

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.0
langchain-community==0.0.10
langchain-ollama==0.0.1
faiss-cpu==1.7.4
psycopg2-binary==2.9.9
python-dotenv==1.0.0
pydantic==2.5.0
watchdog==3.0.0
requests==2.31.0
```

Install:

```bash
pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

```bash
# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull required models
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# Verify models
ollama list
```

**Configure Ollama for CPU optimization:**

```bash
# Edit Ollama service
sudo systemctl edit ollama

# Add these lines:
[Service]
Environment="OLLAMA_NUM_THREADS=8"
Environment="OLLAMA_NUM_PARALLEL=1"

# Save and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Step 5: Setup PostgreSQL Database

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE yocto_analyzer;
CREATE USER rag_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE yocto_analyzer TO rag_user;
\q
```

**Create database schema:**

```sql
-- Connect to database
psql -U rag_user -d yocto_analyzer

-- Create tables
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    pipeline_name VARCHAR(255),
    predicted_result VARCHAR(50),
    actual_result VARCHAR(50),
    confidence_score FLOAT,
    violated_rules INTEGER DEFAULT 0,
    satisfied_rules INTEGER DEFAULT 0,
    pipeline_script_hash VARCHAR(64),
    detected_stack TEXT[],
    rules_applied JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    feedback_received_at TIMESTAMP
);

CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES predictions(id),
    user_id VARCHAR(255),
    actual_build_result VARCHAR(50),
    correct_prediction BOOLEAN,
    corrected_confidence FLOAT,
    missed_issues TEXT[],
    false_positives TEXT[],
    user_comments TEXT,
    feedback_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE learned_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50),
    pattern_text TEXT,
    learned_from_feedback_id UUID REFERENCES feedback(id),
    confidence_boost FLOAT DEFAULT 0.0,
    occurrences INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(pattern_type, pattern_text)
);

CREATE TABLE rule_performance (
    id SERIAL PRIMARY KEY,
    rule_text TEXT UNIQUE,
    rule_type VARCHAR(50),
    total_applications INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX idx_learned_patterns_type ON learned_patterns(pattern_type);
```

***

## Configuration

### Environment Variables

Create `.env` file in `modular_code_base/`:

```bash
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=yocto_analyzer
DATABASE_USER=rag_user
DATABASE_PASSWORD=your_secure_password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:1.5b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Paths
VECTOR_STORE_PATH=./vectorstore
METADATA_PATH=./metadata

# API
API_PORT=8000
API_HOST=0.0.0.0

# File Watching
WATCH_ENABLED=false
MINIO_DATA_PATH=/path/to/minio/data
MINIO_BUCKET=yocto-builds
DEBOUNCE_SECONDS=5
```


## How It Works

### 1. Document Ingestion & Vectorization

**When documents are added to the system:**

```
Document (Yocto .bb file or Jenkins log)
    ↓
Text Splitter (chunks)
    ↓
Ollama Embedding Model (nomic-embed-text)
    ↓
FAISS Vectorstore (stores vector + metadata)
    ↓
Saved to disk (index.faiss + index.pkl)
```

### 2. Query Processing & RAG

**When a user asks a question:**

```
User Query: "Will this Yocto build succeed?"
    ↓
Query Embedding (via nomic-embed-text)
    ↓
FAISS Similarity Search (finds top-k relevant chunks)
    ↓
Context Builder (combines retrieved chunks)
    ↓
LLM Prompt (query + context)
    ↓
Ollama LLM (qwen2.5:1.5b generates answer)
    ↓
Response to User
```

### 3. Feedback Learning Loop

```
Prediction Made
    ↓
User Provides Feedback (actual result + corrections)
    ↓
Store in PostgreSQL
    ↓
Update Rule Performance
    ↓
Learn New Patterns (missed issues/false positives)
    ↓
Adjust Confidence Scores
    ↓
Improved Future Predictions
```

### 4. Vectorstore Auto-Refresh

**Mechanism keep vectorstore updated:**

#### API Endpoint
```
External system (GitHub Actions/Jenkins) updates vectorstore
    ↓
Calls POST /api/rebuild
    ↓
FastAPI triggers vectorstore.load()
    ↓
Application uses updated index
```

***

## Using the Application

### Starting the Application

```bash
# Activate virtual environment
cd /home/azureuser/rag-system/modular_code_base
source ../rag_env/bin/activate

# Start application
python -m app.main
```

**Verify startup:**
```
2025-11-20 14:27:00 - Starting Yocto Build Analyzer...
2025-11-20 14:27:01 - Database initialized
2025-11-20 14:27:02 - Vectorstore ready
2025-11-20 14:27:02 - Vectorstore file watcher active on ./vectorstore
2025-11-20 14:27:02 - Application ready on http://0.0.0.0:8000
```

### API Usage Examples

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "vectorstore_watcher_active": true
}
```


#### 2. Prediction & Suggestions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yocto-analyzer",
    "messages": [
      {"role": "user", "content": "xyz"}
    ]
  }'
```

#### 3. Submit Feedback

```bash
curl -X POST http://localhost:8000/api/feedback/submit \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "uuid-from-prediction",
    "actual_build_result": "FAILURE",
    "corrected_confidence": 95,
    "missed_issues": ["Missing dependency: libssl-dev"],
    "false_positives": [],
    "user_comments": "Build failed due to SSL library"
  }'
```

#### 4. Get Learning Statistics

```bash
curl http://localhost:8000/api/feedback/stats
```

**Response:**
```json
{
  "feedback_received": 45,
  "correct_predictions": 38,
  "accuracy_percentage": 84.44,
  "learned_patterns": 12,
  "is_learning": true
}
```

***

## Vectorstore Management

### Understanding the Vectorstore

The vectorstore consists of two files in `./vectorstore/`:

- **`index.faiss`**: Binary file containing vector embeddings (similarity search)
- **`index.pkl`**: Pickle file with document metadata and mappings

### Manual Vectorstore Update

**Add a single document:**

```python
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vectorstore_service import VectorStoreService

# Read document
with open("/path/to/document.txt", "r") as f:
    content = f.read()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(content)

# Create documents with metadata
documents = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": "manual_upload",
            "file_name": "document.txt",
            "chunk_index": i
        }
    )
    documents.append(doc)

# Add to vectorstore
service = VectorStoreService(vector_store_path="./vectorstore")
service.add_documents(documents)
print(f"Added {len(documents)} chunks")
```

### Automated Updates via GitHub Actions

**Confluence pages are automatically synced:**


### Automated Updates via Jenkins

**Jenkins logs are indexed automatically:**

```groovy
// Jenkins pipeline fetches build logs
// Processes with LangChain
// Updates vectorstore
// Calls /api/rebuild endpoint
```

### Vectorstore Refresh Mechanisms


#### Manual (API Endpoint)

```bash
curl -X POST http://localhost:8000/api/rebuild
```

Use this when:
- File watcher is disabled
- Forcing immediate reload
- Debugging vectorstore issues

***

## Troubleshooting

### Common Issues

#### 1. Ollama Model Not Loading

**Symptom:**
```
ERROR: Failed to initialize embeddings: connection refused
```

**Solution:**
```bash
# Check Ollama status
systemctl status ollama

# Restart if needed
sudo systemctl restart ollama

# Verify models
ollama list

# Pull models if missing
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

#### 2. Slow Response Times

**Symptom:** Queries take 30+ seconds

**Solutions:**

```bash
# Check CPU usage
htop

# Verify Ollama CPU configuration
sudo systemctl cat ollama | grep OLLAMA_NUM_THREADS

# Reduce context window
# Edit routes: num_ctx=2048 instead of 4096

# Reduce retrieved documents
# Edit services: k=5 instead of k=20
```

#### 3. Vectorstore Not Updating

**Symptom:** New documents don't appear in search results

**Check file watcher:**
```bash
# View logs
tail -f app.log | grep "Vectorstore"

# Expected output:
# "Vectorstore file watcher active on ./vectorstore"
# "File changed: ./vectorstore/index.faiss"
# "Vectorstore reloaded from disk"
```

**Manual verification:**
```bash
# Check file timestamps
ls -lht ./vectorstore/

# Force rebuild
curl -X POST http://localhost:8000/api/rebuild
```

#### 4. Database Connection Errors

**Symptom:**
```
ERROR: Database init warning: connection to server failed
```

**Solution:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U rag_user -d yocto_analyzer -h localhost

# Check credentials in .env file
cat .env | grep DATABASE
```

#### 5. Out of Memory

**Symptom:** Application crashes or becomes unresponsive

**Solutions:**
```bash
# Check memory usage
free -h

# Reduce vectorstore size
# Keep fewer documents or smaller chunks

# Use smaller model
ollama pull qwen2.5:0.5b  # Instead of 1.5b

# Enable swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debugging

**Enable detailed logging:**

```python
# In app/main.py
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

**Check application logs:**
```bash
python -m app.main 2>&1 | tee app.log
```

**Profile performance:**
```python
# Add timing in enhanced_analysis_service.py
import time

start = time.time()
# ... operation ...
logger.info(f"Operation took {time.time() - start:.2f}s")
```

***

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/analyze` | POST | Full pipeline analysis |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/completions` | POST | OpenAI-compatible completions |
| `/api/generate` | POST | Ollama-compatible generate |
| `/api/rebuild` | POST | Reload vectorstore |

### Feedback & Learning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/feedback/submit` | POST | Submit prediction feedback |
| `/api/feedback/stats` | GET | Learning statistics |
| `/api/feedback/list` | GET | List all feedback |
| `/api/learning/patterns` | GET | Learned patterns |
| `/api/learning/rules` | GET | Rule performance |
| `/api/learning/accuracy-trend` | GET | Accuracy over time |

### Request Examples

**Analyze Request:**
```json
{
  "pipeline_content": "string (Yocto build script)"
}
```

**Chat Request:**
```json
{
  "model": "yocto-analyzer",
  "messages": [
    {"role": "user", "content": "Your question"}
  ]
}
```

**Feedback Request:**
```json
{
  "prediction_id": "uuid",
  "actual_build_result": "SUCCESS|FAILURE",
  "corrected_confidence": 85,
  "missed_issues": ["issue1", "issue2"],
  "false_positives": ["false1"],
  "user_comments": "Additional notes"
}
```

***

## Performance Optimization

### CPU-Only Optimization

If you don't have GPU, optimize for CPU:

```bash
# 1. Set Ollama threads to match CPU cores
OLLAMA_NUM_THREADS=$(nproc)

# 2. Use quantized models
ollama pull qwen2.5:1.5b-q4_0  # 4-bit quantization

# 3. Reduce context in code
# Edit services to use num_ctx=2048

# 4. Limit retrieved documents
# Use k=3 to k=5 instead of k=20
```

### Memory Optimization

```python
# Reduce chunk size in text splitter
RecursiveCharacterTextSplitter(
    chunk_size=500,  # Instead of 1000
    chunk_overlap=100  # Instead of 200
)

# Limit vectorstore size
# Periodically rebuild with only recent documents
```

***

## Maintenance


- Monitor application logs for errors
- Check disk space (`df -h`)
- Verify Ollama service is running

- Review feedback statistics
- Check learned patterns accuracy
- Clean up old staging directories

- Rebuild vectorstore from scratch (optional)
- Backup PostgreSQL database
- Update dependencies

### Backup Strategy

```bash
# Backup vectorstore
tar -czf vectorstore-backup-$(date +%Y%m%d).tar.gz ./vectorstore/

# Backup database
pg_dump -U rag_user yocto_analyzer > backup-$(date +%Y%m%d).sql

# Backup configuration
cp .env .env.backup
```

***

## Summary

This system provides AI-powered Yocto build analysis through:
1. **RAG Architecture** - Retrieves relevant context before generating responses
2. **Automatic Learning** - Improves from user feedback
3. **Real-time Updates** - File watcher ensures vectorstore stays current
4. **OpenAI Compatibility** - Easy integration with existing tools

The key to understanding this system is the flow: **Documents → Vectors → Search → LLM → Response → Feedback → Improvement**.