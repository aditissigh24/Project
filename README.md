# Multi-Agent Long Document Intelligence

A multi-agent system for processing and reasoning over large scanned documents (~500+ pages) without retrieval-based approaches (no RAG, no vector databases).

Built on **LangGraph v1**, **ChatOpenAI**, and **MongoDB**.

---

## Architecture Overview

```
PDF Upload
    │
    ▼
[Ingestion Node]         PyMuPDF primary → Tesseract fallback per page
    │  writes pages → MongoDB
    ▼
[Cleaning Node]          Rule-based pass → LLM repair for low-confidence pages
    │  updates pages.cleaned_text → MongoDB
    ▼
[Segmentation Node]      Regex headings + LLM disambiguation (sliding window)
    │  writes segments → MongoDB
    ▼
[fan_out_local_analysis] Send(segment_id) × N  ← id-only payloads
    │
    ├── [analyze_segment_node] × N (parallel)
    │       reads segment.text from MongoDB
    │       writes segment_analyses → MongoDB
    ▼
[aggregate_sections_node]  map-reduce pass 1: segments → sections (≤20K tokens/call)
[aggregate_chapters_node]  map-reduce pass 2: sections → chapters (≤30K tokens/call)
[aggregate_document_node]  map-reduce pass 3: chapters → master summary (≤40K tokens/call)
[consistency_check_node]   embed claims → cluster → LLM contradiction check
[finalize_node]            status = READY, write run metrics

User Query
    │
    ▼
[route_query_node]       gpt-4o-mini structured output → query_type classification
    │
    ├── summarize_* → [direct_fetch_node]          reads pre-computed summary from MongoDB
    ├── extract     → [extract_data_node]          aggregates entities/risks/decisions
    ├── find_contradictions → [fetch_contradictions_node]
    └── compare / open_question → [global_reasoning_node]
                                     ReAct loop + ToolNode
                                     reads MongoDB via @tool functions
```

### State Philosophy

LangGraph state **never carries text payloads**. Every node writes its heavy outputs (page text, cleaned text, segment text, analyses) directly to MongoDB and returns only ids, counts, and flags to the state. This keeps in-memory state bounded regardless of document size.

### Context Management (no RAG)

The system builds a hierarchical summary tree during ingestion:

| Level | Content | Tokens |
|-------|---------|--------|
| 0 | Raw OCR text | ~200K (never in LLM context) |
| 1 | Cleaned page text | ~160K (in MongoDB) |
| 2 | Segment analyses | ~50K (in MongoDB) |
| 3 | Section summaries | ~15K (fetched by tools) |
| 4 | Chapter summaries | ~5K |
| 5 | Document master summary | ~2K |

At query time, the ReAct agent assembles context from levels 3–5 (~7–20K tokens), leaving 100K+ tokens for reasoning. **No vector search is used** — context is assembled from pre-built summaries.

---

## Project Structure

```
assignment/
├── main.py                       # CLI (ingest / query / status / reset)
├── config.py                     # pydantic-settings config
├── requirements.txt
├── .env.example
├── db/
│   ├── mongo_client.py           # MongoClient singleton + index setup
│   └── repositories.py           # all MongoDB CRUD
├── agent/
│   ├── graph.py                  # build_ingestion_graph(), build_query_graph()
│   ├── state.py                  # IngestionState, QueryState TypedDicts
│   ├── schemas.py                # Pydantic models for all data structures
│   ├── llm.py                    # get_llm("heavy"|"light"), get_embeddings()
│   ├── prompts.py                # all prompt templates
│   ├── nodes/
│   │   ├── ingestion.py          # ingest_node
│   │   ├── cleaning.py           # clean_node
│   │   ├── segmentation.py       # segment_node + fan_out_local_analysis
│   │   ├── local_analysis.py     # analyze_segment_node (runs via Send fan-out)
│   │   ├── aggregation.py        # aggregate_sections/chapters/document nodes
│   │   ├── consistency.py        # consistency_check_node
│   │   ├── query_router.py       # route_query_node + get_route edge
│   │   ├── global_reasoning.py   # global_reasoning_node + branch nodes
│   │   └── finalize.py           # finalize_node
│   └── tools/
│       ├── pdf_tools.py          # PyMuPDF extraction + rendering
│       ├── ocr_tools.py          # Tesseract OCR + preprocessing
│       ├── text_tools.py         # rule-based cleaning + token budget
│       ├── embedding_tools.py    # claim embedding + clustering
│       └── mongo_tools.py        # @tool functions for ToolNode
└── tests/
    ├── test_segmentation.py
    ├── test_aggregation.py
    ├── test_query_router.py
    └── test_repositories.py
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- MongoDB running locally (`mongodb://localhost:27017`) or a MongoDB Atlas URI
- Tesseract OCR binary installed:
  - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH
  - **Linux/Mac**: `sudo apt install tesseract-ocr` or `brew install tesseract`

### 2. Install dependencies

```bash
cd assignment
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
#   MONGO_URI=mongodb://localhost:27017
```

### 4. Run tests (no API key or MongoDB required)

```bash
pytest tests/ -v
```

---

## Usage

### Ingest a document

```bash
python -m main ingest "ICICI Bank Report.pdf"
# Output: Document ID (save this for querying)
```

With a custom doc ID:
```bash
python -m main ingest "ICICI Bank Report.pdf" --doc-id icici-2024
```

### Check status

```bash
python -m main status icici-2024
```

### Query the document

```bash
# Full document summary
python -m main query icici-2024 "Provide a concise summary of the entire report"

# Cross-section comparison
python -m main query icici-2024 "Compare the risks in the Risk Management section with those implied in the financial statements"

# Structured extraction
python -m main query icici-2024 "Extract key financial metrics, strategic initiatives, and risk categories"

# Contradiction detection
python -m main query icici-2024 "Identify any inconsistencies between management commentary and financial performance"

# Multi-hop reasoning
python -m main query icici-2024 "Trace how Net Interest Margin is discussed across the document"
```

### Reset (development)

```bash
python -m main reset icici-2024 --yes
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `document_intelligence` | Database name |
| `HEAVY_MODEL` | `gpt-4o` | Model for reasoning/aggregation |
| `LIGHT_MODEL` | `gpt-4o-mini` | Model for classification/cleaning |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for claim clustering |
| `LLM_SEMAPHORE` | `8` | Max concurrent LLM calls |
| `MAX_CONTEXT_TOKENS` | `120000` | Hard token limit (safety margin below 128K) |
| `MAX_SEGMENT_TOKENS` | `8000` | Max tokens per segment |
| `MAX_SECTION_AGGREGATE_TOKENS` | `20000` | Max tokens for section aggregation input |
| `MAX_CHAPTER_AGGREGATE_TOKENS` | `30000` | Max tokens for chapter aggregation input |
| `MAX_DOCUMENT_AGGREGATE_TOKENS` | `40000` | Max tokens for document aggregation input |
| `OCR_DPI` | `300` | DPI for page rendering before OCR |
| `OCR_MIN_ALPHA_RATIO` | `0.4` | Min alphabetic character ratio (below = OCR) |
| `OCR_MIN_CHAR_COUNT` | `50` | Min characters (below = OCR) |

---

## MongoDB Collections

| Collection | Content |
|------------|---------|
| `documents` | Document metadata, processing status |
| `pages` | Per-page raw and cleaned text, OCR flag |
| `segments` | Logical document units with heading, page range, text |
| `segment_analyses` | Structured analysis per segment (claims, risks, entities…) |
| `section_summaries` | Aggregated section-level summaries |
| `chapter_summaries` | Aggregated chapter-level summaries |
| `document_summary` | Master document summary (top entities, risks, decisions) |
| `contradictions` | Detected cross-section inconsistencies |
| `runs` | Processing run metadata and metrics |

---

## Model Constraint Trade-off

The assignment specifies **open-source models only**. This implementation uses **ChatOpenAI (GPT-4o / GPT-4o-mini)** per the user's requirement.

The LLM is fully abstracted behind `agent/llm.py`:

```python
# To swap to a vLLM-backed open-source model (e.g., Qwen2.5-72B), change only:
# agent/llm.py

from langchain_openai import ChatOpenAI

def get_llm(tier: Literal["heavy", "light"]) -> ChatOpenAI:
    # Replace with:
    # from langchain_openai import ChatOpenAI  # pointing at a vLLM endpoint
    # return ChatOpenAI(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-72B-Instruct")
    ...
```

Recommended open-source alternatives:
- **Heavy**: `Qwen2.5-72B-Instruct` (128K context, best open-source reasoning)
- **Light**: `Mistral-7B-Instruct` (fast classification and cleaning tasks)
- **Serving**: vLLM with `OPENAI_API_KEY=any` and `base_url=http://localhost:8000/v1`

---

## Agent Summary

| Agent / Node | Model | Role |
|---|---|---|
| `ingest_node` | — | PyMuPDF + Tesseract OCR |
| `clean_node` | gpt-4o-mini | Rule-based + LLM noise repair |
| `segment_node` | gpt-4o-mini | Heading detection + boundary disambiguation |
| `analyze_segment_node` | gpt-4o | Structured metadata extraction per segment |
| `aggregate_sections_node` | gpt-4o | Section-level map-reduce |
| `aggregate_chapters_node` | gpt-4o | Chapter-level map-reduce |
| `aggregate_document_node` | gpt-4o | Document master summary |
| `consistency_check_node` | gpt-4o + embeddings | Contradiction detection via clustering |
| `route_query_node` | gpt-4o-mini | Query classification |
| `global_reasoning_node` | gpt-4o + ToolNode | ReAct agent for complex queries |
| `direct_fetch_node` | — | Pre-computed summary retrieval (no LLM) |
| `extract_data_node` | gpt-4o | Structured data extraction + formatting |
| `fetch_contradictions_node` | gpt-4o-mini | Contradiction retrieval + topic filtering |
