import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, date
from pawpal_system import Task, Pet, Owner, Scheduler, ScheduledItem


# ── Test 1: mark_complete() stamps last_completed ─────────────────────────────
def test_mark_complete_updates_last_completed():
    task = Task(
        name="Morning Walk",
        category="walk",
        priority="high",
        frequency="daily",
        duration_minutes=30,
        preferred_time="morning",
    )

    assert task.last_completed is None          # should start empty

    timestamp = datetime(2026, 3, 29, 8, 0)
    task.mark_complete(timestamp)

    assert task.last_completed == timestamp     # should now be set


# ── Test 2: add_task() increases pet's task count ─────────────────────────────
def test_add_task_increases_task_count():
    pet = Pet(name="Mango", species="Dog", age=4)

    assert len(pet.tasks) == 0                  # should start empty

    task = Task(
        name="Breakfast",
        category="feeding",
        priority="high",
        frequency="daily",
        duration_minutes=10,
        preferred_time="morning",
    )
    pet.add_task(task)

    assert len(pet.tasks) == 1                  # should now have one task
    assert pet.tasks[0].name == "Breakfast"     # and it should be the right one
    assert task.pet is pet                      # back-reference should be set too

# ── Test 3: mark_complete() with no timestamp uses current time ──────────────
def test_mark_complete_uses_current_time():
    task = Task(
        name="Evening Feed",
        category="feeding",
        priority="high",
        frequency="daily",
        duration_minutes=10,
        preferred_time="evening",
    )
    
    task.mark_complete(datetime.now())
    assert task.last_completed is not None
    assert isinstance(task.last_completed, datetime)


# ── Test 4: pet with zero tasks renders gracefully ──────────────────────────
def test_pet_with_zero_tasks():
    pet = Pet(name="Buddy", species="Cat", age=2)
    
    assert len(pet.tasks) == 0
    assert pet.name == "Buddy"
    assert pet.species == "Cat"


# ── Test 5: multiple pets don't cross-contaminate tasks ──────────────────────
def test_multiple_pets_independent_tasks():
    pet1 = Pet(name="Mango", species="Dog", age=4)
    pet2 = Pet(name="Whiskers", species="Cat", age=3)
    
    task1 = Task("Walk", "walk", "high", "daily", 30, "morning")
    task2 = Task("Play", "play", "medium", "daily", 20, "afternoon")
    
    pet1.add_task(task1)
    pet2.add_task(task2)
    
    assert len(pet1.tasks) == 1
    assert len(pet2.tasks) == 1
    assert pet1.tasks[0].name == "Walk"
    assert pet2.tasks[0].name == "Play"


# ── Test 6: completing one task doesn't affect other tasks ─────────────────
def test_completing_one_task_independent_from_others():
    pet = Pet(name="Mango", species="Dog", age=4)
    task1 = Task("Walk", "walk", "high", "daily", 30, "morning")
    task2 = Task("Feed", "feeding", "high", "daily", 10, "evening")
    
    pet.add_task(task1)
    pet.add_task(task2)
    
    timestamp = datetime(2026, 3, 29, 8, 0)
    task1.mark_complete(timestamp)
    
    assert task1.last_completed == timestamp
    assert task2.last_completed is None


# ── Test 7: two tasks at the same time ───────────────────────────────────────
def test_tasks_at_same_time():
    pet = Pet(name="Mango", species="Dog", age=4)
    task1 = Task("Walk A", "walk", "high", "daily", 30, "morning")
    task2 = Task("Walk B", "walk", "medium", "daily", 20, "morning")
    
    pet.add_task(task1)
    pet.add_task(task2)
    
    assert len(pet.tasks) == 2


# ── Test 8: pet with empty name ──────────────────────────────────────────────
def test_pet_with_empty_name():
    pet = Pet(name="", species="Dog", age=4)
    assert pet.name == ""
    assert len(pet.tasks) == 0


# ── Test 9: deleting the last task ───────────────────────────────────────────
def test_deleting_last_task():
    pet = Pet(name="Mango", species="Dog", age=4)
    task = Task("Walk", "walk", "high", "daily", 30, "morning")
    
    pet.add_task(task)
    assert len(pet.tasks) == 1
    
    pet.tasks.remove(task)
    assert len(pet.tasks) == 0


# ── Test 10: sort_by_time() returns tasks in chronological order ─────────────
def test_sort_by_time_chronological_order():
    schedule_date = date(2026, 3, 30)
    owner = Owner(name="Alex")
    owner.set_availability(schedule_date, [(7, 20)])

    pet = Pet(name="Mango", species="Dog", age=4)
    owner.add_pet(pet)

    # Add tasks in reverse order so sorting is non-trivial
    pet.add_task(Task("Evening Walk",   "walk",    "low",    "daily", 30, "evening"))
    pet.add_task(Task("Afternoon Feed", "feeding", "medium", "daily", 20, "afternoon"))
    pet.add_task(Task("Morning Med",    "meds",    "high",   "daily", 15, "morning"))

    scheduler = Scheduler(owner, schedule_date)
    scheduler.build_plan()

    sorted_items = scheduler.sort_by_time()

    assert len(sorted_items) == 3
    # Each item's start time must be <= the next one
    for i in range(len(sorted_items) - 1):
        assert sorted_items[i].start_time <= sorted_items[i + 1].start_time, (
            f"Out of order: {sorted_items[i].task.name} at "
            f"{sorted_items[i].start_time} is after "
            f"{sorted_items[i+1].task.name} at {sorted_items[i+1].start_time}"
        )


# ── Test 11: mark_done() on a daily task spawns a new task for the next day ──
def test_daily_task_recurrence_spawns_next_day():
    schedule_date = date(2026, 3, 30)
    owner = Owner(name="Alex")
    owner.set_availability(schedule_date, [(7, 20)])

    pet = Pet(name="Mango", species="Dog", age=4)
    owner.add_pet(pet)

    daily_task = Task("Morning Walk", "walk", "high", "daily", 30, "morning")
    pet.add_task(daily_task)

    scheduler = Scheduler(owner, schedule_date)
    scheduler.build_plan()

    task_count_before = len(pet.tasks)
    scheduler.mark_done(daily_task)
    task_count_after = len(pet.tasks)

    assert task_count_after == task_count_before + 1, "A new recurring task should be added"

    new_task = pet.tasks[-1]
    expected_next_due = date(2026, 3, 31)
    assert new_task.next_due == expected_next_due, (
        f"Expected next_due={expected_next_due}, got {new_task.next_due}"
    )
    assert new_task.name == daily_task.name
    assert new_task.last_completed is None


# ── Test 12: detect_conflicts() flags overlapping scheduled items ─────────────
def test_detect_conflicts_flags_overlapping_tasks():
    schedule_date = date(2026, 3, 30)
    owner = Owner(name="Alex")

    pet = Pet(name="Mango", species="Dog", age=4)
    owner.add_pet(pet)

    scheduler = Scheduler(owner, schedule_date)

    # Manually inject two overlapping ScheduledItems (08:00–08:30 and 08:15–08:45)
    task_a = Task("Walk A", "walk",    "high",   "daily", 30, "morning")
    task_b = Task("Walk B", "feeding", "medium", "daily", 30, "morning")
    pet.add_task(task_a)
    pet.add_task(task_b)

    start_a = datetime(2026, 3, 30, 8, 0)   # 08:00
    start_b = datetime(2026, 3, 30, 8, 15)  # 08:15 — overlaps with A (ends 08:30)

    scheduler.scheduled_items.append(ScheduledItem(task=task_a, pet=pet, start_time=start_a))
    scheduler.scheduled_items.append(ScheduledItem(task=task_b, pet=pet, start_time=start_b))

    conflicts = scheduler.detect_conflicts()

    assert len(conflicts) == 1, f"Expected 1 conflict, got {len(conflicts)}: {conflicts}"
    assert "CONFLICT" in conflicts[0]