# Multi-Agent System for Long Document Intelligence

## Full Industry-Standard Production Plan

---

## 1. Difficulty Assessment

| Dimension               | Rating       | Reason                                                                |
| ----------------------- | ------------ | --------------------------------------------------------------------- |
| **Overall Difficulty**  | **8.5 / 10** | Senior/Staff Engineer Level                                           |
| Architecture Design     | 9/10         | No RAG is a hard constraint; requires map-reduce reasoning            |
| OCR Pipeline            | 7/10         | Mature tools exist, but noise handling is non-trivial                 |
| Context Management      | 9/10         | The hardest part — 500+ pages must fit into 128K tokens progressively |
| Cross-Section Reasoning | 9/10         | Without retrieval, you must build a lossy but faithful summary tree   |
| Production Hardening    | 8/10         | Async agents, retries, observability, cost control                    |

### Why It's Hard

- **No RAG** is the killer constraint. Every other long-doc system uses vector search. Without it, you must solve long-context reasoning using **hierarchical map-reduce summarization** — a stateful, multi-pass pipeline.
- A 500-page scanned PDF at ~300 words/page ≈ **150,000–200,000 tokens raw** — already beyond the 128K limit before any processing.
- Consistency detection across distant sections requires the model to "remember" things it processed minutes ago, which is only possible through well-designed summary states.
- OCR noise corrupts reasoning — garbage in, garbage out applies doubly for LLMs.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (FastAPI REST + WebSocket)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │    QUERY ROUTER      │  ← Orchestrator Agent
                    │  (LangGraph Engine)  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼───────────────────┐
          │                    │                   │
  ┌───────▼──────┐  ┌──────────▼──────┐  ┌────────▼────────┐
  │  Ingestion   │  │   Aggregation   │  │ Global Reasoning │
  │  Pipeline    │  │   Agent         │  │ Agent            │
  └──────┬───────┘  └──────────┬──────┘  └────────┬────────┘
         │                     │                   │
  ┌──────▼───────┐      ┌──────▼──────┐    ┌──────▼────────┐
  │ Segmentation │      │  Local      │    │  Consistency  │
  │ Agent        │      │  Analysis   │    │  Checker      │
  └──────┬───────┘      │  Agents     │    │  Agent        │
         │              │  (parallel) │    └───────────────┘
  ┌──────▼───────┐      └─────────────┘
  │ OCR + Clean  │
  │ Agent        │
  └─────────────┘
```

### Document State Machine (Core Concept)

```
RAW PDF
  → OCR Pages (parallel)
  → Cleaned Text Chunks
  → Segmented Logical Units
  → Local Analysis Summaries (per segment)
  → Section Summaries (per chapter/section)
  → Document Meta-Summary (global)
  → Query-time retrieval from this summary tree
```

---

## 3. Tech Stack (Open Source Only)

### 3.1 OCR Layer

| Tool                         | Role                    | Why                                               |
| ---------------------------- | ----------------------- | ------------------------------------------------- |
| **Surya** (primary)          | Layout + line detection | Best open-source layout analysis, 90+ languages   |
| **PaddleOCR + PP-Structure** | Table/form extraction   | Best open-source for structured document elements |
| **Marker**                   | Full pipeline fallback  | Surya-backed end-to-end PDF→Markdown converter    |
| **PyMuPDF (fitz)**           | Page rendering          | Renders PDF pages to high-res images for OCR      |
| **OpenCV**                   | Preprocessing           | Deskew, denoise, binarize scanned images          |
| **img2table**                | Table post-processing   | Converts raw table OCR into structured data       |

### 3.2 LLM Layer

| Model                    | Role              | Why                                                    |
| ------------------------ | ----------------- | ------------------------------------------------------ |
| **Qwen2.5-72B-Instruct** | Primary reasoning | 128K context, best open-source reasoning at this scale |
| **Mistral-7B-Instruct**  | Fast local tasks  | Light tasks: classification, entity tagging            |
| **Llama-3.3-70B**        | Fallback          | Strong general reasoning, Apache 2.0 license           |

**Serving**: vLLM with tensor parallelism (4× A100 or H100 recommended for 72B)

### 3.3 Orchestration & Agent Framework

| Tool               | Role                                                    |
| ------------------ | ------------------------------------------------------- |
| **LangGraph**      | Agent state machine, conditional routing, checkpointing |
| **Celery + Redis** | Async task queue (parallel Local Analysis Agents)       |
| **Redis**          | Session state, segment cache, intermediate summaries    |
| **PostgreSQL**     | Persistent document store, audit trail                  |

### 3.4 API & Serving

| Tool                    | Role                                            |
| ----------------------- | ----------------------------------------------- |
| **FastAPI**             | REST API + WebSocket for streaming              |
| **Pydantic v2**         | Request/response schemas                        |
| **Docker + Kubernetes** | Containerized, horizontally scalable deployment |
| **NGINX**               | Load balancer, rate limiting                    |

### 3.5 Observability

| Tool                                        | Role                                          |
| ------------------------------------------- | --------------------------------------------- |
| **LangSmith** (or open-source **Langfuse**) | Agent trace observability                     |
| **Prometheus + Grafana**                    | Metrics: tokens/s, agent latency, error rates |
| **Sentry**                                  | Error tracking                                |
| **Structured Logging (structlog)**          | JSON logs for every agent event               |

---

## 4. Agent Specifications

### Agent 1 — Ingestion Agent

**Role**: PDF rendering and raw text extraction.

**Inputs**: Raw PDF file path  
**Outputs**: List of page images + raw OCR text per page + page metadata

```python
class IngestionAgent:
    """
    - Renders each PDF page to 300 DPI image via PyMuPDF
    - Runs OCR via Surya (primary) with PaddleOCR for tables
    - Outputs: List[PageResult(page_num, raw_text, confidence, layout_boxes)]
    - Parallelizes across pages using Celery workers
    - Handles: rotation detection, multi-column layout, embedded images
    """
    def process(pdf_path: str) -> List[PageResult]:
        pages = render_pdf_pages(pdf_path, dpi=300)
        return parallel_ocr(pages, workers=8)  # 500 pages / 8 workers ≈ 62 pages each
```

**Context budget**: Never passes more than 1 page to LLM at a time.  
**Error handling**: If OCR confidence < 0.6 on a page, flag for manual review and apply secondary model.

---

### Agent 2 — Cleaning Agent

**Role**: Fix OCR noise, reconstruct broken words, normalize text.

**Inputs**: Raw OCR text per page  
**Outputs**: Cleaned, normalized text per page

**Strategy**:

- Rule-based pass first (fix hyphenation, remove junk chars, fix whitespace)
- LLM-based pass (Mistral-7B) for sentences with OCR confidence < 0.75
- Preserve original alongside cleaned version for audit

**Prompt template (LLM cleaning pass)**:

```
The following text was extracted via OCR from a scanned document and may
contain recognition errors. Fix ONLY obvious OCR errors (broken words,
substituted characters). Do NOT paraphrase or change meaning.
Output only the corrected text.

OCR TEXT:
{noisy_text}
```

---

### Agent 3 — Segmentation Agent

**Role**: Split cleaned text into logical document units.

**Inputs**: Full cleaned text stream (all pages)  
**Outputs**: Structured segment tree: `[Document > Chapters > Sections > Paragraphs]`

**Strategy**:

1. Regex + heuristics for heading detection (ALL CAPS, numbered headers, font-size metadata from layout boxes)
2. LLM pass (Mistral-7B) to resolve ambiguous boundaries
3. Outputs a JSON segment manifest:

```json
{
  "doc_id": "abc123",
  "segments": [
    {
      "seg_id": "s_001",
      "type": "section",
      "heading": "Introduction",
      "pages": [1, 5],
      "token_count": 3200,
      "text": "..."
    }
  ]
}
```

**Context budget**: Processes max 20 pages at a time to detect headings, using sliding window overlap of 2 pages.

---

### Agent 4 — Local Analysis Agents (Fan-Out)

**Role**: Extract structured metadata from each segment.

**Inputs**: One segment (≤ 8,000 tokens)  
**Outputs**: `SegmentAnalysis` object

**Runs in parallel** via Celery for all segments simultaneously.

```python
class SegmentAnalysis(BaseModel):
    seg_id: str
    summary: str           # 200-300 word dense summary
    key_entities: List[Entity]   # people, orgs, dates, locations
    key_claims: List[str]        # important factual assertions
    decisions: List[str]         # decisions or conclusions
    risks: List[str]             # risks or warnings identified
    contradictions: List[str]    # internal contradictions in this segment
    sentiment: str               # neutral / positive / negative / mixed
    topics: List[str]            # topic tags
```

**Prompt template**:

```
You are a document intelligence agent. Analyze the following document
section and extract the structured information below. Be precise and
factual. Do not hallucinate.

SECTION HEADING: {heading}
SECTION TEXT:
{text}

Return a JSON object with these fields:
- summary (200-300 words, dense)
- key_entities (name, type, context)
- key_claims (list of factual assertions)
- decisions (conclusions or decisions made)
- risks (risks, warnings, caveats)
- contradictions (internal inconsistencies)
- topics (2-5 topic tags)
```

**Context budget**: Each call ≤ 10,000 tokens total. Max segment size is 8,000 tokens.

---

### Agent 5 — Aggregation Agent

**Role**: Combine Local Analysis outputs into section-level and document-level summaries.

**Inputs**: All `SegmentAnalysis` objects  
**Outputs**: `SectionSummary[]` + `DocumentMasterSummary`

**This is the core context management solution — hierarchical map-reduce.**

```
Pass 1 (Section Level):
  - Group segments by parent section
  - For each section: feed all SegmentAnalysis summaries (not raw text)
  - Produce SectionSummary (~500 words + entities + claims)
  - Token budget per call: ~20,000 tokens max

Pass 2 (Chapter Level):
  - Feed all SectionSummary objects for a chapter
  - Produce ChapterSummary (~800 words)
  - Token budget: ~30,000 tokens

Pass 3 (Document Level):
  - Feed all ChapterSummary objects
  - Produce DocumentMasterSummary (~2,000 words)
  - Token budget: ~40,000 tokens
```

**Why this works without RAG**: At query time, the agent has access to the **full structured summary tree** (all levels), not the raw text. The tree for a 500-page document fits well within 128K tokens.

---

### Agent 6 — Global Reasoning Agent

**Role**: Cross-document synthesis and complex query answering.

**Inputs**: Query + DocumentMasterSummary + relevant SectionSummaries  
**Outputs**: Answer with provenance citations

**Context assembly at query time**:

```python
def build_query_context(query: str, summary_tree: SummaryTree) -> str:
    context = [
        summary_tree.document_master_summary,  # Always included (~2K tokens)
    ]
    # Include section summaries most relevant to query (heuristic keyword match)
    relevant_sections = rank_sections_by_relevance(query, summary_tree.sections)
    for section in relevant_sections[:10]:  # top 10 sections
        context.append(section.summary)
    # Total budget: 2K + 10 * ~500 = ~7K tokens for context
    # Leaves ~120K tokens for reasoning and output
    return "\n\n".join(context)
```

**Supported query types and routing**:

- `summarize_section` → use SectionSummary directly
- `summarize_document` → use DocumentMasterSummary
- `compare_sections` → load 2+ SectionSummaries into context together
- `extract_entities` → aggregate from all SegmentAnalysis objects
- `find_contradictions` → Consistency Checker Agent
- `open_question` → Global Reasoning Agent with full summary tree

---

### Agent 7 — Consistency Checker Agent

**Role**: Detect contradictions and inconsistencies across the document.

**Inputs**: All `key_claims` from all segments (deduplicated)  
**Outputs**: `List[Contradiction]`

**Strategy**:

```
1. Group claims by entity/topic using embeddings (local sentence-transformers)
   Note: This is embedding for grouping only, NOT retrieval (not RAG)
2. For each topic cluster, pass all related claims to LLM
3. LLM checks: Are any of these claims mutually inconsistent?
4. Output structured contradictions with section references
```

```python
class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    section_a: str
    section_b: str
    explanation: str
    severity: Literal["low", "medium", "high"]
```

---

### Agent 8 — Query Router Agent

**Role**: Entry point. Interpret user query and orchestrate other agents.

**Inputs**: Natural language query  
**Outputs**: Routes to correct agent(s) + synthesized response

```python
QUERY_TYPES = {
    "summarize_section": SummarizationAgent,
    "summarize_document": SummarizationAgent,
    "compare": ComparativeAgent,
    "extract_entities": ExtractionAgent,
    "find_risks": ExtractionAgent,
    "find_contradictions": ConsistencyCheckerAgent,
    "open_question": GlobalReasoningAgent,
}

def route(query: str) -> QueryType:
    # LLM classifier (Mistral-7B, single call)
    # Returns one of the above types + extracted parameters
```

---

## 5. Context Management Strategy (Critical)

### The Problem

- 500 pages × ~400 tokens/page = **200,000 tokens** → exceeds 128K limit
- Even 500 pages × 200 tokens (compressed) = 100,000 tokens → barely fits, fragile

### The Solution: Progressive Compression Tree

```
Level 0: Raw OCR text           ~200,000 tokens  (never in LLM context)
Level 1: Cleaned chunks         ~160,000 tokens  (stored in DB)
Level 2: Segment analyses       ~50,000 tokens   (stored in DB, accessed by agents)
Level 3: Section summaries      ~15,000 tokens   (fits in context)
Level 4: Chapter summaries      ~5,000 tokens    (always in context)
Level 5: Document master        ~2,000 tokens    (always in context)
```

At query time, the system **always has Level 4+5 in context** (~7K tokens), and dynamically pulls Level 3 for relevant sections based on the query (~10-20K tokens total for context). This leaves 100K+ tokens for reasoning.

### Trade-offs

| Approach               | Information Retention  | Speed            | Accuracy                 |
| ---------------------- | ---------------------- | ---------------- | ------------------------ |
| Raw chunk per query    | 100% for that chunk    | Fast             | Misses cross-doc context |
| Map-reduce tree (ours) | ~80% semantic fidelity | Medium           | Good cross-doc reasoning |
| Full doc in context    | 100%                   | Slow + expensive | Best but rarely possible |

**Chosen approach**: Map-reduce tree, because it's the only viable strategy without RAG for 500+ page documents.

---

## 6. Data Flow Diagram

```
PDF Upload
    │
    ▼
[Job Queue (Celery)]
    │
    ├──▶ [Ingestion Agent × 8 workers] → Page images + raw OCR
    │
    ├──▶ [Cleaning Agent × 8 workers] → Cleaned text per page
    │
    ├──▶ [Segmentation Agent] → Segment manifest JSON
    │
    ├──▶ [Local Analysis Agents × N workers] → SegmentAnalysis[]
    │       (all segments processed in parallel)
    │
    ├──▶ [Aggregation Agent Pass 1] → SectionSummary[]
    ├──▶ [Aggregation Agent Pass 2] → ChapterSummary[]
    ├──▶ [Aggregation Agent Pass 3] → DocumentMasterSummary
    │
    └──▶ [Consistency Checker] → Contradiction[]

All results persisted to PostgreSQL.
Document status: READY

User Query
    │
    ▼
[Query Router] → identify query type
    │
    ▼
[Global Reasoning Agent]
  → load master summary + relevant section summaries
  → LLM call with assembled context
  → stream response back to user
```

---

## 7. API Design

### Endpoints

```
POST   /api/v1/documents/upload          # Upload PDF, returns job_id
GET    /api/v1/documents/{doc_id}/status # Processing status + progress
GET    /api/v1/documents/{doc_id}/structure # Document segment tree

POST   /api/v1/query                     # Run a query against a document

GET    /api/v1/documents/{doc_id}/summary            # Full-doc summary
GET    /api/v1/documents/{doc_id}/sections/{sec_id}/summary  # Section summary
GET    /api/v1/documents/{doc_id}/entities           # All extracted entities
GET    /api/v1/documents/{doc_id}/risks              # All identified risks
GET    /api/v1/documents/{doc_id}/contradictions     # All contradictions found
```

### Query Request Schema

```python
class QueryRequest(BaseModel):
    document_id: str
    query: str
    query_type: Optional[Literal[
        "summarize_section", "summarize_document",
        "compare", "extract_entities", "find_contradictions",
        "open_question"
    ]] = None  # Auto-detected if omitted
    section_ids: Optional[List[str]] = None  # For targeted queries
    stream: bool = True  # Stream response via SSE
```

---

## 8. Production Deployment Architecture

```
                        ┌──────────────┐
                        │  API Gateway  │
                        │  (NGINX)      │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │   FastAPI App        │
                    │   (3 replicas, K8s) │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
   ┌───────▼───────┐  ┌────────▼──────┐  ┌────────▼────────┐
   │  Redis Cache   │  │  PostgreSQL   │  │  Celery Workers  │
   │  (session +   │  │  (documents,  │  │  (16 workers)    │
   │   job state)  │  │   summaries)  │  │  OCR + LLM tasks │
   └───────────────┘  └───────────────┘  └────────┬────────┘
                                                   │
                                         ┌─────────▼────────┐
                                         │  vLLM Inference   │
                                         │  (Qwen2.5-72B)    │
                                         │  4× A100 GPUs     │
                                         └──────────────────┘
```

### Kubernetes Resource Estimates

| Component       | Replicas            | CPU     | Memory | GPU          |
| --------------- | ------------------- | ------- | ------ | ------------ |
| FastAPI App     | 3                   | 4 vCPU  | 8 GB   | —            |
| Celery Workers  | 4 pods              | 8 vCPU  | 16 GB  | —            |
| vLLM (Qwen 72B) | 1 pod               | 16 vCPU | 32 GB  | 4× A100 80GB |
| PostgreSQL      | 1 (primary+replica) | 4 vCPU  | 16 GB  | —            |
| Redis           | 1 cluster           | 2 vCPU  | 8 GB   | —            |

---

## 9. Implementation Phases

### Phase 1 — Foundation (Weeks 1–2)

- [ ] Set up project repo, Docker Compose, CI/CD pipeline
- [ ] Implement Ingestion Agent: PyMuPDF rendering + Surya OCR
- [ ] Implement Cleaning Agent: rule-based + Mistral-7B fallback
- [ ] Implement Segmentation Agent: heading detection + JSON manifest
- [ ] Unit tests for each agent with synthetic noisy PDFs
- [ ] Deliverable: Pipeline that converts PDF → structured segments

### Phase 2 — Intelligence Layer (Weeks 3–4)

- [ ] Implement Local Analysis Agents with Celery parallelization
- [ ] Implement Aggregation Agent: all 3 compression passes
- [ ] Implement Consistency Checker Agent
- [ ] Design and test the context budget at each level
- [ ] Deliverable: Full document processed → complete summary tree in DB

### Phase 3 — Query Interface (Week 5)

- [ ] Implement Query Router Agent
- [ ] Implement Global Reasoning Agent
- [ ] Implement all query types: summarize, compare, extract, check
- [ ] Build FastAPI endpoints + streaming via SSE
- [ ] Deliverable: Working end-to-end query system

### Phase 4 — Production Hardening (Week 6)

- [ ] Add LangSmith / Langfuse tracing to all agents
- [ ] Add Prometheus metrics + Grafana dashboards
- [ ] Implement retry logic with exponential backoff for all LLM calls
- [ ] Add token budget enforcement (hard cutoff with truncation strategy)
- [ ] Rate limiting, auth (JWT), input validation
- [ ] Load testing with k6 (simulate 10 concurrent document uploads)
- [ ] Deliverable: Production-ready deployment on Kubernetes

---

## 10. Critical Implementation Details

### OCR Noise Handling Strategy

```
Pass 1 (Deterministic):
  - Remove non-printable characters
  - Fix common OCR substitutions: rn→m, 0→O, 1→l etc. (via lookup table)
  - Fix broken hyphenated words across line breaks
  - Normalize Unicode (NFKC normalization)

Pass 2 (Statistical):
  - Flag sentences where > 15% tokens are not in a vocabulary dict
  - Those sentences go to LLM cleaning pass

Pass 3 (LLM - Mistral-7B):
  - Reconstruct flagged sentences
  - Max 2,000 tokens per call
  - Log all changes for audit
```

### Token Budget Enforcement

```python
MAX_TOKENS = 120_000  # Safety margin below 128K

def enforce_token_budget(text: str, budget: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > budget:
        # Truncate from the middle, preserve start and end
        half = budget // 2
        return tokenizer.decode(tokens[:half] + tokens[-half:])
    return text
```

### Retry Logic for Agent Calls

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(LLMTimeoutError),
)
async def call_llm(prompt: str, max_tokens: int) -> str:
    ...
```

### Provenance Tracking

Every answer must cite its source segments:

```python
class Answer(BaseModel):
    answer: str
    citations: List[Citation]  # [{seg_id, section_heading, page_range}]
    confidence: float
    token_usage: TokenUsage
```

---

## 11. Evaluation Metrics

| Metric                            | Target   | How to Measure                            |
| --------------------------------- | -------- | ----------------------------------------- |
| OCR accuracy (CER)                | < 3%     | Against ground-truth subset               |
| Segmentation F1                   | > 0.90   | Against hand-labeled boundaries           |
| Summary faithfulness              | > 0.85   | BERTScore vs. reference                   |
| Entity extraction F1              | > 0.88   | Against annotated entities                |
| Contradiction recall              | > 0.80   | Manual annotation of known contradictions |
| Query latency (P95)               | < 30 sec | Load test with k6                         |
| End-to-end processing (500 pages) | < 20 min | Integration test                          |
| Context budget violations         | 0        | Automated test suite                      |

---

## 12. Key Risks and Mitigations

| Risk                                | Likelihood | Mitigation                                                               |
| ----------------------------------- | ---------- | ------------------------------------------------------------------------ |
| OCR quality too low for dense text  | Medium     | Tune DPI (≥300), add image pre-processing (deskew, denoise)              |
| 128K budget exceeded mid-query      | Low        | Hard token enforcement + budget tests per agent                          |
| LLM hallucination in summaries      | High       | Factual grounding prompts + citation requirement                         |
| Slow processing for 500 pages       | Medium     | Parallelize all stateless steps via Celery                               |
| Summary tree loses critical details | Medium     | Tune compression ratios; allow "drill-down" queries to load raw segments |
| Cascading agent failures            | Low        | Circuit breakers, dead-letter queues, partial result saves               |

---

## 13. What This Is NOT (And Why That Matters)

- **Not RAG**: There is no vector DB, no embedding-based retrieval. Context is assembled from the pre-built summary tree.
- **Not a chatbot with memory**: State is explicit and stored in PostgreSQL/Redis, not in the LLM's context window between sessions.
- **Not a single prompt**: The system makes 500–2,000+ LLM calls to process a 500-page document. This is a processing pipeline, not a Q&A system.

---

## Summary

This is a **senior engineering challenge** that combines three hard problems simultaneously: scalable OCR, context-constrained LLM reasoning, and multi-agent orchestration. The key insight is that **hierarchical map-reduce summarization** solves the no-RAG + context-limit constraint together — you compress the document into a structured summary tree at processing time, so query-time context assembly is fast, cheap, and always within limits. Built correctly on LangGraph + Celery + vLLM + PostgreSQL, this system is genuinely production-shippable.
