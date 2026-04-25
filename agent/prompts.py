"""
All prompt templates as module-level constants.
"""

# ---------------------------------------------------------------------------
# Cleaning Agent
# ---------------------------------------------------------------------------

CLEAN_REPAIR_PROMPT = """\
The following text was extracted via OCR from a scanned document and may contain \
recognition errors. Fix ONLY obvious OCR errors (broken words, substituted characters, \
garbled numbers). Do NOT paraphrase, summarize, or change meaning. \
Output ONLY the corrected text with no commentary.

OCR TEXT:
{noisy_text}
"""

# ---------------------------------------------------------------------------
# Segmentation Agent
# ---------------------------------------------------------------------------

SEGMENT_BOUNDARY_PROMPT = """\
You are analyzing a section of a large document to identify logical boundaries between \
sections or chapters.

Below is a window of consecutive pages from the document. Identify where new logical \
sections begin. For each boundary, provide:
- The page number where the new section starts
- A short heading for that section (if discernible)
- The section type: "chapter", "section", "subsection"

Return a JSON array of objects with keys: page_num (int), heading (str), type (str).
If no new sections start in this window, return an empty array [].

DOCUMENT WINDOW (pages {start_page} to {end_page}):
{text}
"""

# ---------------------------------------------------------------------------
# Local Analysis Agent
# ---------------------------------------------------------------------------

LOCAL_ANALYSIS_PROMPT = """\
You are a document intelligence agent analyzing a section of a large financial document.

Analyze the section below and extract structured information. Be precise and factual. \
Do not hallucinate or infer information not present in the text.

SECTION HEADING: {heading}
SECTION TEXT:
{text}

Extract the following and return as a valid JSON object:
- summary: 200-300 word dense summary of this section
- key_entities: list of {{name, type (person/organization/location/date/metric/other), context}}
- key_claims: list of important factual assertions made in this section
- decisions: list of decisions or conclusions stated
- risks: list of risks, warnings, or caveats identified
- contradictions: list of internal inconsistencies within this section
- topics: 2-5 topic tags for this section
- sentiment: one of neutral/positive/negative/mixed
"""

# ---------------------------------------------------------------------------
# Aggregation Agent — Section level
# ---------------------------------------------------------------------------

SECTION_AGGREGATE_PROMPT = """\
You are synthesizing multiple sub-section analyses into a coherent section summary.

SECTION HEADING: {heading}

Below are summaries of the individual sub-sections that make up this section:
{segment_summaries}

Produce a JSON object with:
- summary: ~500 word synthesis of the entire section
- key_claims: consolidated list of the most important factual claims
- risks: consolidated list of risks from all sub-sections
- decisions: consolidated list of decisions from all sub-sections
- entities: top entities mentioned (deduplicated), each with {{name, type, context}}
"""

# ---------------------------------------------------------------------------
# Aggregation Agent — Chapter level
# ---------------------------------------------------------------------------

CHAPTER_AGGREGATE_PROMPT = """\
You are synthesizing multiple section summaries into a chapter-level summary.

CHAPTER HEADING: {heading}

Below are summaries of the sections that make up this chapter:
{section_summaries}

Produce a JSON object with:
- summary: ~800 word synthesis capturing the main themes, findings, and conclusions of this chapter
"""

# ---------------------------------------------------------------------------
# Aggregation Agent — Document level
# ---------------------------------------------------------------------------

DOCUMENT_AGGREGATE_PROMPT = """\
You are synthesizing chapter-level summaries into a master document summary for a \
large financial report.

Below are all chapter summaries:
{chapter_summaries}

Produce a JSON object with:
- summary: ~2000 word comprehensive document summary covering key financial performance, \
  strategic priorities, risk landscape, and major decisions
- top_entities: the most important entities across the entire document \
  (people, organizations, key metrics), each with {{name, type, context}}
- top_risks: the most critical risks identified across the document
- top_decisions: the most significant decisions or strategic conclusions
"""

# ---------------------------------------------------------------------------
# Consistency Checker
# ---------------------------------------------------------------------------

CONTRADICTION_CHECK_PROMPT = """\
You are a consistency checker reviewing a set of claims from a large financial document.

The following claims all relate to the same topic and come from different sections of the document. \
Identify any pairs of claims that are mutually inconsistent or contradictory.

CLAIMS:
{claims}

Return a JSON object with:
- contradictions: list of {{
    claim_a: str,
    claim_b: str,
    section_a: str (segment_id or heading of claim_a),
    section_b: str (segment_id or heading of claim_b),
    explanation: str (why these are inconsistent),
    severity: "low" | "medium" | "high"
  }}

If no contradictions are found, return {{"contradictions": []}}.
"""

# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------

QUERY_ROUTE_PROMPT = """\
Classify the following user query against a processed financial document into one of these types:

- summarize_section: user wants a summary of a specific section or topic
- summarize_document: user wants an overall document summary
- compare: user wants to compare two or more sections/topics
- extract: user wants structured data extracted (entities, risks, metrics, decisions)
- find_contradictions: user wants inconsistencies or contradictions identified
- open_question: any other analytical or reasoning question

Also identify any specific section names or topics mentioned in the query.

USER QUERY: {query}

Return a JSON object with:
- query_type: one of the types above
- target_section_ids: list of section names/ids mentioned (empty list if none)
- reasoning: brief explanation of your classification
"""

# ---------------------------------------------------------------------------
# Global Reasoning Agent — system prompt
# ---------------------------------------------------------------------------

GLOBAL_REASONING_SYSTEM = """\
You are a document intelligence assistant answering questions about a large financial report \
that has been pre-processed into a structured summary tree.

You have access to tools that let you retrieve:
- The master document summary
- Individual section summaries
- The list of all sections (with headings)
- Deep segment-level analysis for any section
- Extracted entities, risks, and contradictions

Strategy:
1. Always start by calling fetch_master_summary to get the big picture.
2. Call list_sections to see all available sections.
3. For questions about specific sections, call fetch_section_summary for those sections.
4. For very detailed questions, drill down with fetch_segment_analysis.
5. Synthesize across all retrieved context to produce a complete answer.

Your final answer MUST:
- Directly address the user's question
- Include specific citations (section headings and page ranges where possible)
- Be factual and grounded in the retrieved content
- Acknowledge if information is not available in the processed document

Document ID: {doc_id}
"""
