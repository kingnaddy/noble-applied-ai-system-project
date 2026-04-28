# PawPal+ Project Reflection

---

## 1. System Design

**a. Initial design**

I went with 4 classes: `Pet`, `Task`, `Owner`, and `Scheduler`. `Pet` holds the animal's info and owns a list of tasks. `Task` stores what needs to be done, more details about the task, and knows if it's due. `Owner` tracks the human's pets, time availability, and preferences. `Scheduler` takes all that info and builds a daily plan, ranks tasks, fits them into open time slots, and explains its choices.

**b. Design changes**

A few things shifted during the build. I added a `ScheduledItem` dataclass to replace an untyped `list[dict]` in `Scheduler` — untyped dicts are a misspelled key away from a silent bug. I gave `Task` a back-reference to its `Pet` so the scheduler doesn't have to manually carry that link through every method call. I also updated `rank_tasks()` to accept `(Task, Pet)` pairs so it can factor in medical conditions when prioritizing. I dropped `skip()` and `skipped_reason` entirely because they added complexity with no real payoff at this stage. Later in the build I added `sort_by_time()`, `detect_conflicts()`, `filter_by_pet()`, `filter_by_status()`, and `_spawn_next_occurrence()` as the scheduling logic matured.

---

## 2. Scheduling Logic and Tradeoffs

**a. Constraints and priorities**

The scheduler considers three main constraints: owner availability (time windows defined as hour ranges), task priority (`high → medium → low`), and time-of-day preference (`morning`, `afternoon`, `evening`). Medical conditions act as a tiebreaker — pets with conditions always get scheduled first regardless of priority level. I decided priority mattered most because a missed medication is more serious than a missed walk. Time preference came second because placing an evening task at 7 AM defeats the point of the preference. Availability windows are the hard outer boundary — nothing gets scheduled outside them.

**b. Tradeoffs**

The conflict detector uses an O(n²) pairwise comparison via `itertools.combinations`. A more performant approach would sort tasks by start time and sweep with a sliding window (O(n log n)), but that's significantly harder to read and debug. For a pet scheduler with fewer than 20 tasks per day, the pairwise approach runs in microseconds and the extra complexity isn't justified.

---

## 3. AI Collaboration

**a. How you used AI**

I used Claude Code for this project. I used it during design to think through the class structure and catch edge cases I hadn't thought of, like what happens when two tasks land at the exact same time. During implementation it helped me scaffold methods and catch bugs. For the UI it helped me connect the backend scheduler methods to Streamlit components like `st.expander` for the conflict warnings.

**b. Judgment and verification**

One example of a suggestion I modified: AI suggested sorting scheduled items with `sorted(..., key=lambda i: i.start_time.strftime("%H:%M"))`. I accepted it at first but then realized the real problem was that `fit_to_windows` was placing tasks at the wrong times due to a floating-point bug — so sorting a broken schedule correctly wasn't actually fixing anything. I went back and fixed the root cause instead. I verified it by running the app and checking that an evening task actually landed at 18:00 instead of right after the morning walk at 07:20.

---

## 4. Testing and Verification

**a. What you tested**

I tested things like `mark_complete()` stamping the right timestamp, adding a task increasing the pet's task count, and making sure two pets don't accidentally share tasks. I also tested the three main scheduler behaviors: that `sort_by_time()` returns tasks in order, that `mark_done()` on a daily task creates a new one for the next day, and that `detect_conflicts()` catches overlapping items. Those three mattered most because they were the core features of Phase 3 and I wanted to make sure the backend worked before trusting the UI output.

**b. Confidence**

All 12 tests pass so I'm pretty confident in the core logic. If I had more time I'd test a recurring task set for the 31st in a short month, what happens when the owner sets no availability windows, and a weekly task that has never been completed to confirm it shows up as due immediately.</p>

---

## 5. Reflection

**a. What went well**

Adding `ScheduledItem` as a proper dataclass early on was a good call. It meant every part of the code — sorting, filtering, conflict detection, the UI — could access task data cleanly without guessing dictionary key names. It also made writing tests easier because I could just build a `ScheduledItem` directly and inject it without running the whole scheduler pipeline.

**b. What you would improve**

The availability windows only support whole hours right now, so you can't say you're free from 7:30 to 9:15. I'd change `set_availability` to work with minutes instead of hours. I'd also add a button in the UI to mark a task as done live, so you can actually see the recurrence logic kick in during a real session instead of just in tests.

**c. Key takeaway**

AI is a fast, capable collaborator — but it doesn't know which tradeoffs matter to *your* system. It will suggest a working solution without knowing that you already decided to drop `skip()`, or that your window model uses hours not minutes, or that a cleaner architecture matters more than a marginally faster algorithm. The lead architect's job is to hold the design decisions in their head and use AI to execute within those constraints, not to let AI set the constraints. Every time I accepted a suggestion without checking it against the existing design, something broke or drifted. Every time I gave AI a specific constraint first ("here's the class, here's the invariant, here's the bug"), the output was immediately usable.

---

## 6. VS Code Copilot — AI Strategy

**Which Copilot features were most effective?**

Inline completions were most useful during method stubs — writing the signature and docstring and letting Copilot fill in a plausible implementation that I then reviewed and adjusted. Copilot Chat with `#codebase` was effective for questions like "does any method in this file already handle X?" because it searched across the whole file rather than just the current cursor context. The `/explain` command helped when revisiting `fit_to_windows` after the floating-point bug — it described what the existing code was doing step by step, which made the bug easier to spot.

**One AI suggestion I rejected or modified**

Copilot initially suggested storing the time preference comparison as `abs(w["cursor"] - ideal_hour)` where `ideal_hour` was in fractional hours. This looked fine but perpetuated the same floating-point representation that caused the 07:19 bug in the first place. I modified it to `abs(w["cursor"] - ideal_min)` where both values are integer minutes, keeping the fix consistent throughout the method rather than patching just the datetime construction line.

**How separate chat sessions helped**

Keeping design, implementation, testing, and UI as separate sessions prevented context bleed. When I was writing tests, the chat didn't try to refactor the scheduler logic at the same time. When I was fixing the UI, it wasn't suggesting changes to `pawpal_system.py`. Each session had a single responsibility, which mirrors good software design — and made it much easier to review what changed and why after each phase.

**Being the lead architect with powerful AI tools**

The biggest shift was realizing that AI fluency is not the same as design judgment. AI can generate a working `detect_conflicts()` method in seconds, but it can't decide whether conflicts should raise exceptions or return warning strings, whether the UI should show them as red text or modal dialogs, or whether O(n²) is acceptable given the expected task count. Those are architectural decisions that require understanding the full system — its scale, its users, and its tradeoffs. My job was to make those decisions first and then use AI to implement them quickly. When I got that order right, the build moved fast and the code stayed clean. When I skipped the decision and asked AI to "just build it," I got code that worked in isolation but didn't fit the rest of the system.
