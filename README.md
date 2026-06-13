# Resume Gap Analyser

> **AI agent that tells job seekers exactly why their CV isn't getting interviews — and how to fix it.**

🔗 **[Live Demo](#)** _(coming soon)_ | Built by Vishnuventhira B R

---

## The Problem

Most resumes are filtered out by software before a human ever reads them. Applicant Tracking Systems scan for specific keywords and rank candidates on literal matches — so a qualified person using the wrong wording gets silently rejected, with no feedback on what went wrong.

The hard part isn't knowing this happens. It's knowing *which* keywords you're missing for *a specific role*, and what to actually change. Generic "ATS tips" don't help when you're staring at one job description trying to figure out the gap.

**This tool closes that loop.** Give it your CV and a target job description, and it reverse-engineers what the ATS is scanning for, scores your match the way the software would, pinpoints the exact missing keywords, and grounds each fix in established ATS best practices — then drafts a tailored cover letter to match.

---

## What It Does

**Input:** CV (PDF upload or text paste) + Job Description (text paste)

**Output:**
- 🎯 **Match score** (0-100) simulating real ATS keyword screening
- 🔍 **Missing keywords** — required keywords genuinely absent from the CV
- 💡 **Improvement suggestions** — grounded in a 50-document ATS best-practices knowledge base via semantic (vector) retrieval
- ✉️ **Tailored cover letter** — generated from the CV, JD, and match context

---

## Architecture

The agent orchestrates a sequence of structured tools, each returning a validated Pydantic schema:

```
User Input (CV + JD)
        │
        ▼
extract_job_requirements  ──►  Structured requirements
        │                      (skills, seniority, domain, OR conditions)
        ▼
parse_cv  ──────────────────►  Structured candidate profile
        │                      (skills, experience, education, projects)
        ▼
score_resume  ──────────────►  Match score + matched/missing keywords
        │                      (ATS-style keyword simulation, OR-aware)
        ▼
suggest_improvements  ──────►  FAISS RAG over 50 ATS best-practice docs
        │
        ▼
generate_cover_letter  ─────►  Tailored cover letter
        │
        ▼
   Final Summary
```

**Key engineering decisions:**
- **Structured output everywhere** — every tool enforces a Pydantic schema via LangChain's `with_structured_output()`, eliminating manual JSON parsing and guaranteeing consistent shape
- **Vector RAG over lexical** — uses FAISS semantic search instead of keyword matching, so advice retrieval works on meaning, not exact terms
- **OR conditions** — JD phrases like "Power BI or Tableau" are extracted as alternatives; satisfying either does not penalise the score
- **Literal ATS simulation** — scoring mimics how real ATS software keyword-matches, with light synonym/variant tolerance, so missing keywords are genuinely actionable
- **Decoupled embedding + index** — FAISS indexes vectors independent of source, so the embedding provider can be swapped without changing retrieval logic

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| RAG | FAISS vector search over 50 ATS best-practice documents |
| Framework | LangChain (`StructuredTool`, `with_structured_output`) |
| Schemas | Pydantic |
| Monitoring | Logfire — traces, token usage, match scores per session  |
| Testing | pytest — tool-call order, LLM-as-judge, out-of-scope  |
| UI | Streamlit  |
| Packaging | uv + pyproject.toml |

---

## Tools

| Tool | Input | Output | Purpose |
|---|---|---|---|
| `extract_job_requirements` | JD text | `JobRequirements` | Parses JD into structured skills, seniority, domain, OR conditions |
| `parse_cv` | CV text | `CandidateProfile` | Converts raw CV into structured profile (cross-profession) |
| `score_resume` | profile + requirements | `ResumeScore` | ATS-style match score, matched/missing keywords |
| `suggest_improvements` | CV + missing keywords | `Improvements` | FAISS RAG → actionable, grounded fixes |
| `generate_cover_letter` | CV + JD + score | cover letter | Tailored cover letter |

---

## Scoring Methodology

`score_resume` computes a weighted match score (0-100) simulating ATS keyword screening:

| Component | Weight | Logic |
|---|---|---|
| Skills | 50% | % of required skills present (literal, variant, or specific instance) |
| Experience | 30% | Years vs required, graduated penalty for shortfall |
| Seniority + Leadership | 20% | Seniority level match + leadership requirement match |

OR conditions are honoured throughout — satisfying either alternative counts as a full match.

---

## Project Structure

```
src/
└── agent/
    ├── knowledge.py    # Loads prebuilt FAISS index, exposes retriever
    ├── tools.py        # 5 structured tools with Pydantic schemas
    └── agent.py        # Agent loop (planned)
app.py                  # Streamlit UI (planned)
data/
    ├── ats_knowledge.json   # 50 ATS best-practice documents
    └── faiss_index/         # Prebuilt FAISS vector index
notebooks/
    └── rag-test.ipynb       # FAISS RAG prototype
tests/                       # pytest suite (planned)
pyproject.toml
```

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for packaging
- OpenAI API key

### Run Locally

```bash
# 1. Clone
git clone <your-repo-url>
cd Resume_gap_analyser

# 2. Install dependencies
uv sync

# 3. Add your key
# Create a .env file with:
#   OPENAI_API_KEY=sk-...

# 4. Build the knowledge base index (one-time)
#    Run the RAG notebook, or a build script, to create data/faiss_index

# 5. Test individual tools
uv run python src/agent/tools.py
uv run python src/agent/knowledge.py
```

---

## Known Limitations

- Match score quality depends on JD quality — keyword-rich JDs produce sharper scores than duties-focused ones
- For responsibility-heavy JDs, requirement extraction can produce abstract phrases rather than concrete keywords
- Knowledge base is static — expanding it with domain-specific or web-sourced content will improve suggestion quality
- ATS simulation approximates real systems; actual ATS behaviour varies by vendor (Workday, Greenhouse, Lever, iCIMS all differ)

---

## Roadmap

- [ ] `generate_cover_letter` tool
- [ ] Agent orchestration loop
- [ ] Logfire monitoring (traces, token usage, per-session match scores)
- [ ] pytest suite (tool-call order, LLM-as-judge, out-of-scope handling)
- [ ] Streamlit UI with PDF upload
- [ ] Phase 2: weeks-to-learn estimates on identified skill gaps
- [ ] Phase 2: generate a personalised study plan from the gap analysis on user request
- [ ] Phase 2: MCP integration with Gmail + Google Calendar — read availability and set study reminders around the plan
- [ ] Phase 2: hybrid BM25 + vector retrieval
- [ ] Phase 2: enriched knowledge base from web-sourced ATS guides

---

## About

Built as a capstone project for the AI Engineering course by Alexey Grigorev.

Demonstrates end-to-end AI engineering: structured tool design, vector RAG, schema-validated LLM outputs, agent orchestration, monitoring, systematic evaluation, and deployment.
