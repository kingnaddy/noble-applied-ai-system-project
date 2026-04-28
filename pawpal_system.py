from dataclasses import dataclass, field
from datetime import datetime, date as Date, timedelta
from itertools import combinations
from typing import Optional


# ── ScheduledItem ─────────────────────────────────────────────────────────────
@dataclass
class ScheduledItem:
    task: "Task"
    pet: "Pet"
    start_time: datetime
    status: str = "pending"          # pending | done
    reason: str = ""                 # why it was placed in this slot


# ── Task ──────────────────────────────────────────────────────────────────────
@dataclass
class Task:
    name: str
    category: str                    # walk | feeding | meds | grooming | enrichment
    priority: str                    # high | medium | low
    frequency: str                   # daily | weekly | as-needed
    duration_minutes: int
    preferred_time: str              # morning | afternoon | evening
    last_completed: Optional[datetime] = None
    next_due: Optional[Date] = None  # set when a recurring task spawns a new instance
    pet: Optional["Pet"] = field(default=None, repr=False)  # back-ref to owning Pet

    def is_due(self, check_date: Date) -> bool:
        """Return True if this task should happen on the given date.

        If next_due is set (recurring instance), only becomes due on or after that date.
        """
        if self.next_due is not None:
            return check_date >= self.next_due
        if self.frequency == "daily":
            return True
        if self.frequency == "as-needed":
            return True
        if self.frequency == "weekly":
            if self.last_completed is None:
                return True
            days_since = (check_date - self.last_completed.date()).days
            return days_since >= 7
        return False

    def mark_complete(self, timestamp: datetime) -> None:
        """Log the task as completed at the given timestamp."""
        self.last_completed = timestamp
        print(f"  [done] '{self.name}' marked complete at {timestamp.strftime('%H:%M')}")


# ── Pet ───────────────────────────────────────────────────────────────────────
@dataclass
class Pet:
    name: str
    species: str
    age: int
    medical_conditions: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Associate a care task with this pet and set the back-reference."""
        task.pet = self
        self.tasks.append(task)

    def get_due_tasks(self, check_date: Date) -> list[Task]:
        """Return all tasks due on the given date."""
        return [t for t in self.tasks if t.is_due(check_date)]

    def get_profile(self) -> str:
        """Return a summary string of this pet's info."""
        conditions = ", ".join(self.medical_conditions) if self.medical_conditions else "none"
        return (
            f"{self.name} | {self.species} | Age: {self.age} | "
            f"Conditions: {conditions} | Tasks: {len(self.tasks)}"
        )


# ── Owner ─────────────────────────────────────────────────────────────────────
class Owner:
    def __init__(self, name: str, preferences: dict = None):
        self.name: str = name
        self.pets: list[Pet] = []
        self.availability: dict = {}     # { date: [(start_hour, end_hour), ...] }
        self.preferences: dict = preferences or {}

    def add_pet(self, pet: Pet) -> None:
        """Register a new pet under this owner."""
        self.pets.append(pet)

    def set_availability(self, check_date: Date, windows: list[tuple]) -> None:
        """Define available time windows for a given date.
        windows: list of (start_hour, end_hour) tuples, e.g. [(7, 9), (17, 20)]
        """
        self.availability[check_date] = windows

    def get_availability(self, check_date: Date) -> list[tuple]:
        """Return the list of available time windows for a given date."""
        return self.availability.get(check_date, [])

    def get_all_due_tasks(self, check_date: Date) -> list[tuple["Task", "Pet"]]:
        """Collect (task, pet) pairs for every due task across all pets."""
        pairs = []
        for pet in self.pets:
            for task in pet.get_due_tasks(check_date):
                pairs.append((task, pet))
        return pairs


# ── Scheduler ─────────────────────────────────────────────────────────────────
class Scheduler:
    # Maps priority label → sort key (lower = scheduled first)
    PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

    # Maps preferred_time → ideal start hour (used when picking a window)
    TIME_PREFERENCE = {"morning": 8, "afternoon": 13, "evening": 18}

    def __init__(self, owner: Owner, schedule_date: Date):
        self.owner: Owner = owner
        self.date: Date = schedule_date
        self.scheduled_items: list[ScheduledItem] = []
        self.unscheduled_tasks: list[Task] = []

    # ── core pipeline ──────────────────────────────────────────────────────────

    def build_plan(self) -> None:
        """Generate the full daily schedule."""
        self.scheduled_items.clear()
        self.unscheduled_tasks.clear()

        due_pairs = self.owner.get_all_due_tasks(self.date)
        ranked   = self.rank_tasks(due_pairs)
        windows  = self.owner.get_availability(self.date)
        self.fit_to_windows(ranked, windows)

    def rank_tasks(self, tasks: list[tuple[Task, Pet]]) -> list[tuple[Task, Pet]]:
        """Sort (task, pet) pairs by priority; pets with medical conditions go first."""
        def sort_key(pair: tuple[Task, Pet]):
            task, pet = pair
            priority_score  = self.PRIORITY_ORDER.get(task.priority, 99)
            medical_score   = 0 if pet.medical_conditions else 1
            return (priority_score, medical_score)

        return sorted(tasks, key=sort_key)

    def fit_to_windows(
        self,
        ranked_tasks: list[tuple[Task, Pet]],
        availability: list[tuple],
    ) -> None:
        """Assign ranked tasks to available time windows, respecting preferred_time."""
        # Store cursors in integer minutes to avoid floating-point drift.
        # e.g. 20 min task: 7h + 20/60 hrs = 7.3333... → int(0.3333*60) = 19 (wrong).
        # Integer minutes: 420 + 20 = 440 min → 440//60=7 h, 440%60=20 min (correct).
        windows = [{"start": s * 60, "end": e * 60, "cursor": s * 60} for s, e in availability]

        for task, pet in ranked_tasks:
            placed = False

            # Compare cursor (minutes) against ideal start minute for preferred_time
            ideal_min = self.TIME_PREFERENCE.get(task.preferred_time, 8) * 60
            windows_sorted = sorted(
                windows,
                key=lambda w: abs(w["cursor"] - ideal_min)
            )

            for window in windows_sorted:
                # Jump to ideal time if it's still ahead of the cursor;
                # otherwise fall back to the cursor (ideal slot already passed).
                slot_start_min = max(window["cursor"], ideal_min)
                slot_end_min   = slot_start_min + task.duration_minutes

                if slot_end_min <= window["end"]:
                    start_dt = datetime(
                        self.date.year, self.date.month, self.date.day,
                        slot_start_min // 60,
                        slot_start_min % 60,
                    )
                    reason = (
                        f"Scheduled at {start_dt.strftime('%H:%M')} | "
                        f"{task.priority} priority, fits {task.preferred_time} window"
                    )
                    if pet.medical_conditions:
                        reason += f" (medical flag: {', '.join(pet.medical_conditions)})"

                    self.scheduled_items.append(
                        ScheduledItem(task=task, pet=pet, start_time=start_dt, reason=reason)
                    )
                    window["cursor"] = slot_end_min   # advance cursor in minutes
                    placed = True
                    break

            if not placed:
                self.unscheduled_tasks.append(task)

    # ── sorting & filtering ───────────────────────────────────────────────────

    def sort_by_time(self) -> list[ScheduledItem]:
        """Return scheduled items sorted by start time.

        Uses a lambda key that converts each item's start_time to an
        "HH:MM" string, which sorts correctly in chronological order
        because the string comparison of zero-padded hours works the
        same as numeric comparison (e.g. "08:00" < "13:30" < "17:00").
        """
        return sorted(
            self.scheduled_items,
            key=lambda item: item.start_time.strftime("%H:%M"),
        )

    def filter_by_status(self, status: str) -> list[ScheduledItem]:
        """Return scheduled items matching the given status ('pending' or 'done')."""
        return [item for item in self.scheduled_items if item.status == status]

    def filter_by_pet(self, pet_name: str) -> list[ScheduledItem]:
        """Return scheduled items for a specific pet (case-insensitive match)."""
        return [
            item for item in self.scheduled_items
            if item.pet.name.lower() == pet_name.lower()
        ]

    def detect_conflicts(self) -> list[str]:
        """Check for overlapping tasks and return warning strings (never raises).

        Compares every unique pair of scheduled items once using combinations().
        Two items conflict when their time windows overlap:
            a.start < b.end  AND  b.start < a.end
        Returns an empty list when the schedule is conflict-free.
        """
        warnings = []
        for a, b in combinations(self.scheduled_items, 2):
            a_end = a.start_time + timedelta(minutes=a.task.duration_minutes)
            b_end = b.start_time + timedelta(minutes=b.task.duration_minutes)
            if a.start_time < b_end and b.start_time < a_end:
                warnings.append(
                    f"  [CONFLICT] '{a.pet.name}: {a.task.name}' "
                    f"({a.start_time.strftime('%H:%M')}-{a_end.strftime('%H:%M')}) "
                    f"overlaps with '{b.pet.name}: {b.task.name}' "
                    f"({b.start_time.strftime('%H:%M')}-{b_end.strftime('%H:%M')})"
                )
        return warnings

    # ── output & interaction ───────────────────────────────────────────────────

    def explain(self) -> str:
        """Return a human-readable explanation of the daily plan."""
        if not self.scheduled_items:
            return "No tasks scheduled. Check availability windows or add tasks."

        lines = [f"\n{'='*55}", f"  Daily Plan for {self.owner.name} | {self.date}", f"{'='*55}"]

        for item in sorted(self.scheduled_items, key=lambda i: i.start_time):
            end_dt = item.start_time + timedelta(minutes=item.task.duration_minutes)
            lines.append(
                f"\n  [{item.start_time.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}]"
                f"  {item.pet.name}: {item.task.name}"
            )
            lines.append(f"    >> {item.reason}")

        if self.unscheduled_tasks:
            lines.append(f"\n  [!] Could not fit {len(self.unscheduled_tasks)} task(s):")
            for t in self.unscheduled_tasks:
                lines.append(f"    - {t.name} ({t.priority} priority, {t.duration_minutes} min)")

        lines.append(f"\n{'='*55}\n")
        return "\n".join(lines)

    def mark_done(self, task: Task) -> None:
        """Mark a scheduled task as completed and spawn the next occurrence.

        For 'daily' tasks, the next instance is due tomorrow (today + 1 day).
        For 'weekly' tasks, the next instance is due in 7 days (today + 7 days).
        Uses timedelta to calculate the exact next_due date accurately.
        'as-needed' tasks are not recurring, so no new instance is created.
        """
        for item in self.scheduled_items:
            if item.task is task:
                item.status = "done"
                task.mark_complete(datetime.now())
                self._spawn_next_occurrence(task)
                return
        print(f"  [!] Task '{task.name}' not found in today's schedule.")

    def _spawn_next_occurrence(self, completed_task: Task) -> None:
        """Create and register the next instance of a recurring task.

        Called automatically by mark_done(). Computes next_due with timedelta
        so the calculation is always accurate regardless of month boundaries:
            daily  → schedule_date + 1 day
            weekly → schedule_date + 7 days
            as-needed → no new instance; returns immediately

        The new Task is a fresh copy (last_completed=None) with next_due set,
        added directly to the pet's task list so it appears in future build_plan() calls.
        """
        if completed_task.frequency == "daily":
            next_due = self.date + timedelta(days=1)
        elif completed_task.frequency == "weekly":
            next_due = self.date + timedelta(days=7)
        else:
            return  # as-needed tasks don't recur automatically

        next_task = Task(
            name=completed_task.name,
            category=completed_task.category,
            priority=completed_task.priority,
            frequency=completed_task.frequency,
            duration_minutes=completed_task.duration_minutes,
            preferred_time=completed_task.preferred_time,
            next_due=next_due,
        )

        if completed_task.pet is not None:
            completed_task.pet.add_task(next_task)

        print(
            f"  [recurring] Next '{next_task.name}' scheduled for {next_due} "
            f"({completed_task.frequency})"
        )
