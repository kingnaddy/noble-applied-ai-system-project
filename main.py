from datetime import date, timedelta
from pawpal_system import Owner, Pet, Task, Scheduler

# ── Setup Owner ────────────────────────────────────────────────────────────────
owner = Owner(
    name="Jordan",
    preferences={"walk_before": "09:00"}
)

# availability windows for today: (start_hour, end_hour)
today = date.today()
owner.set_availability(today, [(7, 10), (12, 14), (17, 21)])

# ── Setup Pets ─────────────────────────────────────────────────────────────────
mango = Pet(name="Mango", species="Dog", age=4, medical_conditions=["arthritis"])
luna  = Pet(name="Luna",  species="Cat", age=2)

owner.add_pet(mango)
owner.add_pet(luna)

# ── Add Tasks OUT OF ORDER to test sorting ─────────────────────────────────────
# Evening task added first
mango.add_task(Task(
    name="Evening Walk",
    category="walk",
    priority="medium",
    frequency="daily",
    duration_minutes=25,
    preferred_time="evening",
))

# Afternoon task added second
luna.add_task(Task(
    name="Playtime",
    category="enrichment",
    priority="medium",
    frequency="daily",
    duration_minutes=20,
    preferred_time="afternoon",
))

# Morning tasks added last
mango.add_task(Task(
    name="Morning Walk",
    category="walk",
    priority="high",
    frequency="daily",
    duration_minutes=30,
    preferred_time="morning",
))

mango.add_task(Task(
    name="Joint Supplement",
    category="meds",
    priority="high",
    frequency="daily",
    duration_minutes=5,
    preferred_time="morning",
))

mango.add_task(Task(
    name="Breakfast",
    category="feeding",
    priority="high",
    frequency="daily",
    duration_minutes=10,
    preferred_time="morning",
))

luna.add_task(Task(
    name="Breakfast",
    category="feeding",
    priority="high",
    frequency="daily",
    duration_minutes=5,
    preferred_time="morning",
))

luna.add_task(Task(
    name="Brush Coat",
    category="grooming",
    priority="low",
    frequency="weekly",
    duration_minutes=15,
    preferred_time="evening",
))

# ── Run Scheduler ──────────────────────────────────────────────────────────────
scheduler = Scheduler(owner=owner, schedule_date=today)
scheduler.build_plan()

# ── Print Full Schedule ────────────────────────────────────────────────────────
print(scheduler.explain())

# ── Demo: sort_by_time ─────────────────────────────────────────────────────────
print("=" * 55)
print("  SORTED BY TIME")
print("=" * 55)
for item in scheduler.sort_by_time():
    end_dt = item.start_time + timedelta(minutes=item.task.duration_minutes)
    print(f"  [{item.start_time.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}]"
          f"  {item.pet.name}: {item.task.name}  ({item.status})")

# ── Demo: filter_by_pet ───────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("  FILTERED: Mango's tasks only")
print("=" * 55)
for item in scheduler.filter_by_pet("Mango"):
    print(f"  {item.start_time.strftime('%H:%M')}  {item.task.name}  ({item.status})")

print(f"\n{'=' * 55}")
print("  FILTERED: Luna's tasks only")
print("=" * 55)
for item in scheduler.filter_by_pet("Luna"):
    print(f"  {item.start_time.strftime('%H:%M')}  {item.task.name}  ({item.status})")

# ── Demo: filter_by_status ────────────────────────────────────────────────────
# Mark one task done first
print(f"\n{'=' * 55}")
print("  Completing Mango's Morning Walk...")
print("=" * 55)
morning_walk = mango.tasks[1]  # Morning Walk (added second for Mango)
scheduler.mark_done(morning_walk)

print(f"\n  PENDING tasks:")
for item in scheduler.filter_by_status("pending"):
    print(f"    {item.pet.name}: {item.task.name}")

print(f"\n  DONE tasks:")
for item in scheduler.filter_by_status("done"):
    print(f"    {item.pet.name}: {item.task.name}")

# ── Demo: Recurring Tasks ─────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("  RECURRING TASK DEMO")
print("=" * 55)

# Mark a daily task done → should spawn next occurrence for tomorrow
print("\n  Completing Mango's Joint Supplement (daily)...")
scheduler.mark_done(mango.tasks[2])  # Joint Supplement

# Mark a weekly task done → should spawn next occurrence in 7 days
print("\n  Completing Luna's Brush Coat (weekly)...")
brush_coat = next(t for t in luna.tasks if t.name == "Brush Coat")
scheduler.mark_done(brush_coat)

# Show that new instances were added to the pet's task list
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(days=7)

print(f"\n  Mango's tasks due tomorrow ({tomorrow}):")
for t in mango.get_due_tasks(tomorrow):
    print(f"    - {t.name} (next_due={t.next_due or 'always'})")

print(f"\n  Luna's tasks due in 7 days ({next_week}):")
for t in luna.get_due_tasks(next_week):
    print(f"    - {t.name} (next_due={t.next_due or 'always'})")

# ── Demo: Conflict Detection ──────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("  CONFLICT DETECTION DEMO")
print("=" * 55)

# Build a second scheduler with a deliberately conflicting task:
# Force Mango's Vet Visit to start at 07:00 — same slot as Morning Walk.
conflict_owner = Owner(name="Jordan")
conflict_owner.set_availability(today, [(7, 10), (12, 14), (17, 21)])

conflict_pet = Pet(name="Mango", species="Dog", age=4, medical_conditions=["arthritis"])
conflict_owner.add_pet(conflict_pet)

conflict_pet.add_task(Task(
    name="Morning Walk",
    category="walk",
    priority="high",
    frequency="daily",
    duration_minutes=30,
    preferred_time="morning",
))
conflict_pet.add_task(Task(
    name="Vet Visit",
    category="meds",
    priority="high",
    frequency="as-needed",
    duration_minutes=45,
    preferred_time="morning",
))

conflict_scheduler = Scheduler(owner=conflict_owner, schedule_date=today)
conflict_scheduler.build_plan()

# Manually force Vet Visit to the same start time as Morning Walk to trigger conflict
from datetime import datetime as dt
conflict_scheduler.scheduled_items[1].start_time = conflict_scheduler.scheduled_items[0].start_time

print("\n  Schedule with forced overlap:")
for item in conflict_scheduler.sort_by_time():
    end_dt = item.start_time + timedelta(minutes=item.task.duration_minutes)
    print(f"    [{item.start_time.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}]"
          f"  {item.pet.name}: {item.task.name}")

print("\n  Running conflict check...")
warnings = conflict_scheduler.detect_conflicts()
if warnings:
    for w in warnings:
        print(w)
else:
    print("  No conflicts detected.")

# Also confirm the normal schedule has no conflicts
print("\n  Checking original schedule for conflicts...")
normal_warnings = scheduler.detect_conflicts()
if normal_warnings:
    for w in normal_warnings:
        print(w)
else:
    print("  No conflicts detected — schedule is clean.")
print()
