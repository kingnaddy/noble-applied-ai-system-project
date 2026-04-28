# PawPal+ (Module 2 Project)

> A smart daily pet care scheduler built with Python and Streamlit.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Priority-based scheduling** | Tasks are ranked `high → medium → low`. Pets with medical conditions are always scheduled first. |
| **Preferred time slots** | Each task targets `morning` (8 AM), `afternoon` (1 PM), or `evening` (6 PM). The scheduler jumps to the ideal slot rather than stacking tasks back-to-back. |
| **Sorting by time** | `Scheduler.sort_by_time()` returns the full day's plan in strict chronological order using integer-minute arithmetic — no floating point drift. |
| **Conflict warnings** | `detect_conflicts()` compares every pair of scheduled items with `itertools.combinations`. Overlapping tasks surface as a red expandable alert in the UI. |
| **Daily & weekly recurrence** | Marking a task done via `mark_done()` automatically spawns the next occurrence: daily tasks reappear tomorrow, weekly tasks reappear in 7 days (calculated with `timedelta`). |
| **Filtering** | Slice the schedule by pet name (`filter_by_pet`) or completion status (`filter_by_status`). |
| **Unscheduled task reporting** | Tasks that don't fit any availability window are surfaced separately so nothing is silently dropped. |
| **12-test suite** | Covers task lifecycle, pet isolation, edge cases, sorting correctness, recurrence logic, and conflict detection. |

---

## 📸 Demo

<a href="/course_images/ai110/demo.png" target="_blank"><img src='/course_images/ai110/demo.png' title='PawPal App' width='' alt='PawPal App' class='center-block' /></a>

---

## Scenario

A busy pet owner needs help staying consistent with pet care. They want an assistant that can:

- Track pet care tasks (walks, feeding, meds, enrichment, grooming, etc.)
- Consider constraints (time available, priority, owner preferences)
- Produce a daily plan and explain why it chose that plan

Your job is to design the system first (UML), then implement the logic in Python, then connect it to the Streamlit UI.

## What you will build

Your final app should:

- Let a user enter basic owner + pet info
- Let a user add/edit tasks (duration + priority at minimum)
- Generate a daily schedule/plan based on constraints and priorities
- Display the plan clearly (and ideally explain the reasoning)
- Include tests for the most important scheduling behaviors

## Getting started

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Smarter Scheduling

PawPal+ now includes algorithmic features that make the scheduler more intelligent:

**Sorting by time** — `Scheduler.sort_by_time()` returns scheduled items in chronological order.

**Filtering** — Two filter methods let you slice the schedule by pet or status:
- `filter_by_pet(pet_name)` — returns only tasks for a named pet (case-insensitive)
- `filter_by_status(status)` — returns tasks matching `"pending"` or `"done"`

**Recurring tasks** — When `mark_done()` is called on a `daily` or `weekly` task, `_spawn_next_occurrence()` automatically creates the next instance using Python's `timedelta`:
- Daily → due tomorrow (`schedule_date + timedelta(days=1)`)
- Weekly → due in 7 days (`schedule_date + timedelta(days=7)`)

**Conflict detection** — `detect_conflicts()` checks every unique pair of scheduled items using `itertools.combinations` and returns warning strings (never crashes). Two tasks conflict when their time windows overlap: `a.start < b.end AND b.start < a.end`.

---

## Testing PawPal+

### Run the test suite

```bash
python -m pytest tests/test_pawpal.py -v
```

### What the tests cover

The suite contains **12 tests** across three layers of the system:

| Area | Tests |
|---|---|
| **Task behavior** | `mark_complete()` stamps `last_completed`; calling it with no timestamp raises a clear error |
| **Pet management** | Adding tasks increases count; back-reference (`task.pet`) is set correctly; pets don't share tasks |
| **Edge cases** | Pet with zero tasks, pet with empty name, deleting the last task, completing one task doesn't affect others, two tasks at the same time |
| **Sorting correctness** | `sort_by_time()` returns scheduled items in strict chronological order across morning, afternoon, and evening slots |
| **Recurrence logic** | Calling `mark_done()` on a daily task creates a new task on the pet with `next_due = today + 1 day` and `last_completed = None` |
| **Conflict detection** | `detect_conflicts()` flags overlapping scheduled items and returns a descriptive warning string |

### Confidence Level

⭐⭐⭐⭐⭐ **(5 / 5)**

All 12 tests pass. Core scheduling logic (sorting, recurrence, conflict detection), task lifecycle, and edge cases are all verified. The system behaves correctly and predictably across happy paths and boundary conditions.

---

### Suggested workflow

1. Read the scenario carefully and identify requirements and edge cases.
2. Draft a UML diagram (classes, attributes, methods, relationships).
3. Convert UML into Python class stubs (no logic yet).
4. Implement scheduling logic in small increments.
5. Add tests to verify key behaviors.
6. Connect your logic to the Streamlit UI in `app.py`.
7. Refine UML so it matches what you actually built.
