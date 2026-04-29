# PawPal+ (Module 2 Project)

> A smart daily pet care scheduler built with Python and Streamlit, extended with Retrieval-Augmented Generation (RAG) for answering pet care questions using a local knowledge base.

---

## Loom Video Walkthrough
[Video Link](https://www.loom.com/share/9ace3708e9314612b9348e74becf73c4) 

## Project Overview

**Original project:** PawPal+ is an original Module 2 project designed to help busy pet owners manage daily care routines.
It modeled pets, tasks, priorities, availability windows, and automatic schedule generation.
The original goals were to create a reliable planner that ranks tasks by urgency, handles recurring care, and detects conflicts before they affect the owner.

**Advanced AI feature:** This version adds a **Retrieval-Augmented Generation (RAG)** system that enables PawPal+ to answer user questions about pet care using a local knowledge base of custom documents. Instead of relying on generic LLM responses, the system retrieves relevant information from documents about feeding, grooming, behavior, health, and scheduling, then generates grounded answers with source attribution.

## Title and Summary

**PawPal+** builds a daily schedule for pet care tasks while also making the reasoning behind decisions easy to understand.
This portfolio version emphasizes clarity, explainability, and a retrieval-enabled architecture so the assistant can answer questions about why a task was scheduled at a particular time, and provide grounded answers to pet care questions from custom knowledge documents.

## Architecture Overview

The system is organized into four main layers:

- **Streamlit UI** (`app.py`): collects owner input, pet details, availability, and tasks; displays schedules, warnings, explanations, and RAG-powered Q&A.
- **Core scheduler** (`pawpal_system.py`): manages pets, due tasks, ranking by priority, placement into available windows, and conflict detection.
- **RAG system** (`rag.py`): loads and indexes pet-care documents from `knowledge_base/`, retrieves relevant chunks using TF-IDF similarity, and generates grounded answers with source tracking.
- **Knowledge base** (`knowledge_base/`): custom pet-care documents (feeding guidelines, grooming, behavior, health, scheduler help) that ground the RAG system.

This architecture keeps scheduling logic separate from user interaction and allows the AI explanation layer to stay grounded in actual task data and domain knowledge.

---

## What it does

PawPal+ is a smart pet-care scheduler and information assistant that helps owners plan daily tasks with clear reasoning and answer pet care questions.
It combines task prioritization, preferred time slots, availability windows, and retrieval-augmented generation to create a reliable daily plan and knowledge-grounded answers.

### Key features

**Scheduling:**
- Priority-based scheduling: tasks are ranked `high → medium → low`, and pets with medical conditions are prioritized.
- Preferred time slots: tasks aim for `morning`, `afternoon`, or `evening` windows to match owner preferences.
- Sorting by time: `Scheduler.sort_by_time()` returns the full day's plan in strict chronological order.
- Conflict warnings: overlapping tasks are detected and surfaced before they break the schedule.
- Daily & weekly recurrence: recurring tasks automatically create the next instance after completion.
- Filtering: view tasks by pet or completion status.
- Unscheduled task reporting: tasks that do not fit any window are shown separately so nothing is lost.

**RAG-Powered Q&A:**
- **Local knowledge base:** 5 custom documents covering feeding, grooming, behavior, health, and scheduling.
- **Grounded retrieval:** uses TF-IDF similarity to retrieve relevant document chunks for user questions.
- **Source attribution:** displays the source document and relevance score for each answer.
- **Medical safeguards:** automatically includes veterinarian disclaimers for health-related questions.
- **Query logging:** all questions and retrieved sources are logged to `logs/rag_queries.log`.
- **No hallucination:** if no relevant context is found, the system clearly states it cannot answer instead of guessing.

### Test coverage
- 12 original scheduler tests for scheduling, recurrence, sorting, and conflict handling.
- 8 new RAG tests covering knowledge base loading, retrieval, answer generation, guardrails, and edge cases.

## Data Flow

```
User Input (Streamlit UI)
    ↓
[1] Schedule Generation Path:
    Question about task scheduling
    → Scheduler.build_plan()
    → Rank tasks by priority/medical status
    → Fit into availability windows
    → Detect conflicts
    → Display sorted schedule with explanations

[2] Pet Care Q&A Path:
    Pet care question
    → Knowledge base retrieval (TF-IDF)
    → Retrieve top-3 relevant document chunks
    → Generate grounded answer from chunks
    → Detect if medical question (add disclaimer)
    → Display answer + sources + logging
    → Log to logs/rag_queries.log
```

## Scenario

A busy pet owner needs help staying consistent with pet care. They want an assistant that can:

- Track pet care tasks (walks, feeding, meds, enrichment, grooming, etc.)
- Consider constraints (time available, priority, owner preferences)
- Produce a daily plan and explain why it chose that plan
- Answer questions about pet care using reliable, grounded information

The system helps the owner add pets and tasks, set availability, build a schedule that balances urgency and care needs, and answer care questions using custom pet-care documents.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### 2. Installation

Open a terminal in the `noble-applied-ai-system-project` folder and run:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` (or similar).

### 4. Knowledge Base

The `knowledge_base/` folder already contains 5 custom pet-care documents:
- `feeding_guidelines.md` — daily feeding requirements, water intake, special diets
- `grooming_guidelines.md` — bathing, brushing, nail care, dental care
- `behavior_training.md` — training principles, common behavioral issues, professional help
- `health_safety.md` — preventive care, emergencies, toxins, parasites
- `scheduler_help.md` — guide to using PawPal+ scheduling features

To add more knowledge, simply create new `.md` files in the `knowledge_base/` folder and restart the app.

---

## Using the RAG Feature

### Ask a Pet Care Question

1. Scroll to the **"Ask PawPal: Pet Care Assistant"** section at the bottom of the app.
2. Type a question about pet care (e.g., "How often should I feed my cat?").
3. Click **"Get Answer"**.
4. PawPal will:
   - Search the knowledge base for relevant documents
   - Display a grounded answer based on retrieved content
   - Show the source documents and relevance scores
   - Include a veterinarian disclaimer if the question is medical-related
   - Log the query to `logs/rag_queries.log`

### Example Questions

- "How often should I feed my dog?"
- "What are signs of dental disease in cats?"
- "My dog has arthritis. What exercises should they do?"
- "Is chocolate toxic to pets?"
- "How do I train my puppy to sit?"
- "What should I do about my cat's scratching behavior?"

### Logging

All RAG queries are logged to `logs/rag_queries.log` with:
- User question (first 60 characters)
- Source documents retrieved
- Relevance scores
- Whether an answer was successfully generated

Example log entry:
```
2026-04-28 14:32:15 | Q: How often should I feed my dog? | Sources: feeding_guidelines.md | Scores: 0.85 | Success: True
```

---

## Sample Interactions

### Example 1: Generate a daily schedule

**Input:** owner availability set from 7:00 to 21:00, add a dog named Mochi with a high-priority medication task for afternoon and a medium-priority walk for morning.

**Output:** a sorted schedule table with task times, pet names, task names, priorities, and duration, plus a message explaining that the medication task was placed before the walk because it matched the preferred afternoon window and had higher urgency.

### Example 2: Ask about feeding requirements

**Input:** "How much should I feed my 5-year-old dog?"

**Output:** grounded answer from the feeding guidelines including portion size calculations, water intake recommendations, and source information showing the retrieval came from `feeding_guidelines.md` with 0.89 relevance score.

### Example 3: Medical question with disclaimer

**Input:** "My dog is limping. What should I do?"

**Output:** answer with relevant information from health guidelines, plus a prominent warning: "⚠️ This information is educational only and is NOT a substitute for professional veterinary care. Please consult your veterinarian."

### Example 4: See conflict detection

**Input:** two overlapping tasks with the same availability window.

**Output:** a warning card listing the overlapping tasks and a suggestion to adjust duration or availability so both tasks can fit safely.

---

## Design Decisions

**Scheduling:**
- **Modular classes:** `Owner`, `Pet`, `Task`, and `Scheduler` keep domain logic separate from the UI.
- **Priority-first scheduling:** tasks are sorted by priority and medical condition flags before placement.
- **Time preference awareness:** the scheduler attempts to place tasks near the owner's preferred morning/afternoon/evening time.
- **Explainability focus:** the system produces human-readable reasons for schedule placement.

**RAG System:**
- **TF-IDF retrieval:** lightweight, deterministic, requires no API keys or external dependencies.
- **Local documents:** all knowledge is stored in version-controlled markdown files, fully reproducible.
- **Top-3 retrieval:** balances relevance with computational efficiency; user sees best matches.
- **Medical safeguards:** automatic detection of health-related keywords to add veterinarian disclaimers.
- **Logging:** query logging enables monitoring and improvement without collecting user data.
- **Chunk overlap:** documents are chunked with overlap to preserve context across boundaries.

**Trade-offs:**
- A simple in-memory planner is easier to reason about and test, but not optimized for large task volumes.
- TF-IDF retrieval is fast and interpretable, but less sophisticated than neural semantic search (trade simplicity for reproducibility).
- Current approach favors schedule clarity and answer grounding over full optimization.

---

## Testing Summary

### Scheduler Tests (Original)
- `test_mark_complete_updates_last_completed()` — task completion tracking
- `test_add_task_increases_task_count()` — pet task association
- `test_pet_with_zero_tasks()` — empty pet handling
- `test_multiple_pets_independent_tasks()` — pet isolation
- `test_completing_one_task_independent_from_others()` — task independence
- `test_tasks_at_same_time()` — same-time task handling
- `test_sort_by_time_chronological_order()` — schedule sorting
- `test_daily_task_recurrence_spawns_next_day()` — recurring task generation
- `test_detect_conflicts_flags_overlapping_tasks()` — conflict detection
- Additional tests for edge cases and boundary conditions

### RAG Tests (New)
- `test_knowledge_base_loads()` — knowledge base initialization
- `test_rag_feeding_question_retrieves_feeding_docs()` — feeding question retrieval
- `test_rag_unrelated_question_no_crash()` — graceful handling of off-topic questions
- `test_rag_medical_question_includes_disclaimer()` — veterinarian disclaimers
- `test_rag_empty_question_shows_warning()` — empty input handling
- `test_rag_feeding_answer_has_sources()` — source attribution
- `test_rag_answer_uses_retrieved_context()` — answer grounding
- `test_initialize_knowledge_base_singleton()` — knowledge base caching

### Running Tests

```bash
python -m pytest tests/test_pawpal.py -v
```

All tests should pass successfully.

---

## File Structure

```
noble-applied-ai-system-project/
├── app.py                          # Streamlit UI for scheduling + RAG Q&A
├── pawpal_system.py                # Core scheduler logic (Owner, Pet, Task, Scheduler)
├── rag.py                          # RAG system (knowledge base, retrieval, answer generation)
├── requirements.txt                # Dependencies (streamlit, pytest, scikit-learn)
├── README.md                       # This file
├── reflection.md                   # Design reflection and decisions
├── logs/                           # Query logs (created automatically)
│   └── rag_queries.log            # RAG query history
├── knowledge_base/                 # Pet-care documents for RAG
│   ├── feeding_guidelines.md
│   ├── grooming_guidelines.md
│   ├── behavior_training.md
│   ├── health_safety.md
│   └── scheduler_help.md
└── tests/
    └── test_pawpal.py             # Unit tests for scheduler and RAG
```

---

## Reflection

This project taught me how important it is to structure AI systems around clear, grounded data.
A good pet-care assistant is not only about generating a schedule — it is also about explaining why that schedule makes sense for the owner and the pets, and reliably answering questions using trusted information sources.

The RAG enhancement deepens this principle: instead of relying on a generic language model to improvise answers, the system retrieves relevant information from domain documents, ensuring answers are grounded, traceable, and maintainable.

Working on PawPal+ reinforced the value of:
- **Modular design:** keeping scheduling, retrieval, and generation separate
- **Human-centered reasoning:** designing explanations and disclaimers for real users
- **Reproducibility:** using local documents and deterministic retrieval
- **Honest limitations:** clearly stating when the system cannot help instead of hallucinating

---

## Future Enhancements

Possible improvements to explore:
- **Multi-day scheduling:** extend planning to a full week or month
- **Personalization:** learn from past scheduling patterns
- **Advanced retrieval:** add semantic embedding-based retrieval (e.g., sentence transformers)
- **Document embedding:** generate embeddings at build time for faster queries
- **User feedback:** thumbs-up/down on answers to refine retrieval
- **Integration:** connect to vet appointment APIs or pet health apps
- **Fine-tuning:** train a small LLM on domain documents for better answers

---

## Questions or Issues?

If you have questions about the code or the RAG system, check:
1. The inline docstrings in `rag.py` and `pawpal_system.py`
2. The test cases in `tests/test_pawpal.py`
3. The knowledge base documents in `knowledge_base/`
4. The logs in `logs/rag_queries.log` for retrieval debugging

