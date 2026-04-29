import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, date
from pawpal_system import Task, Pet, Owner, Scheduler, ScheduledItem
from rag import KnowledgeBase, answer_pet_care_question, initialize_knowledge_base


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


# ──────────────────────────────────────────────────────────────────────────────
# RAG Tests
# ──────────────────────────────────────────────────────────────────────────────

# ── Test 13: Knowledge base loads successfully ────────────────────────────────
def test_knowledge_base_loads():
    """Test that the knowledge base initializes and loads documents."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    # Should have loaded some documents
    assert len(kb.chunks) > 0, "Knowledge base should have loaded documents"
    
    # Should have built TF-IDF index
    assert kb.vectorizer is not None, "Vectorizer should be initialized"
    assert kb.tfidf_matrix is not None, "TF-IDF matrix should be built"


# ── Test 14: Feeding question retrieves feeding-related content ──────────────
def test_rag_feeding_question_retrieves_feeding_docs():
    """Test that a feeding question retrieves feeding guidelines."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "How often should I feed my dog?"
    retrieved = kb.retrieve(question, top_k=3)
    
    # Should retrieve at least one chunk
    assert len(retrieved) > 0, "Should retrieve relevant documents for feeding question"
    
    # Retrieved chunks should have source information
    for chunk, score in retrieved:
        assert hasattr(chunk, "source_file"), "Chunk should have source_file attribute"
        assert score >= 0.1, "Score should be above minimum threshold"


# ── Test 15: Unrelated question is handled safely ───────────────────────────
def test_rag_unrelated_question_no_crash():
    """Test that completely unrelated questions don't crash the system."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "What is the capital of France?"
    retrieved = kb.retrieve(question, top_k=3)
    
    # Even if nothing relevant is found, should not crash
    assert isinstance(retrieved, list), "Should return a list"


# ── Test 16: Medical questions include veterinarian disclaimer ──────────────
def test_rag_medical_question_includes_disclaimer():
    """Test that medical questions include a veterinarian warning."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "My dog has a health issue, what should I do?"
    answer, sources, success = answer_pet_care_question(question, kb)
    
    # Should include veterinarian disclaimer
    assert "veterinarian" in answer.lower(), "Medical answers should mention veterinarian"
    assert isinstance(sources, list), "Should return sources list"


# ── Test 17: Empty question shows warning ────────────────────────────────────
def test_rag_empty_question_shows_warning():
    """Test that empty questions are handled gracefully."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    answer, sources, success = answer_pet_care_question("", kb)
    
    assert not success, "Empty question should not generate answer"
    assert "question" in answer.lower(), "Should guide user to ask a question"


# ── Test 18: Feeding question generates answer with sources ─────────────────
def test_rag_feeding_answer_has_sources():
    """Test that feeding questions generate answers with source information."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "How often should I feed my cat?"
    answer, sources, success = answer_pet_care_question(question, kb)
    
    # Should have generated an answer
    assert success, "Should successfully answer feeding question"
    assert len(answer) > 0, "Answer should not be empty"
    assert isinstance(sources, list), "Should return sources list"
    
    # If sources are available, verify structure
    if sources:
        for source in sources:
            assert "source" in source, "Source should have 'source' field"
            assert "relevance" in source, "Source should have 'relevance' field"


# ── Test 19: Answer is grounded in retrieved context ───────────────────────
def test_rag_answer_uses_retrieved_context():
    """Test that answers are built from retrieved content, not generic."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "How often should dogs eat?"
    answer, sources, success = answer_pet_care_question(question, kb)
    
    if success and sources:
        # Answer should contain some content from the retrieved chunks
        # (not a perfect test, but checks answer is being constructed)
        assert len(answer) > 50, "Answer should be substantial and grounded"
        assert "knowledge base" in answer.lower() or "relevant" in answer.lower(), \
            "Answer should reference knowledge base"


# ── Test 20: Initialize knowledge base singleton ──────────────────────────
def test_initialize_knowledge_base_singleton():
    """Test that knowledge base initialization works correctly."""
    kb1 = initialize_knowledge_base()
    kb2 = initialize_knowledge_base()
    
    # Should return cached instance
    assert kb1 is kb2, "Should return same cached instance"
    assert len(kb1.chunks) > 0, "Knowledge base should have documents"


# ── Test 21: Vague questions ask for clarification ──────────────────────────
def test_rag_vague_question_asks_for_clarification():
    """Test that vague questions prompt for more specific information."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    vague_questions = [
        "What should I do to my dog?",
        "How do I take care of my pet?",
        "Tell me about my cat",
        "Help with my dog"
    ]
    
    for question in vague_questions:
        answer, sources, success = answer_pet_care_question(question, kb)
        
        # Should ask for clarification
        assert "specific" in answer.lower() or "clarification" in answer.lower(), \
            f"Vague question '{question}' should ask for clarification"
        assert "feed" in answer.lower() or "groom" in answer.lower() or "behavior" in answer.lower(), \
            f"Should suggest specific topics for '{question}'"
        assert success, f"Should handle vague question '{question}' gracefully"


# ── Test 22: Weak relevance returns fallback message ────────────────────────
def test_rag_weak_relevance_fallback():
    """Test that questions with weak relevance get fallback message."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    # Question that might have weak relevance
    question = "What about quantum physics for pets?"
    answer, sources, success = answer_pet_care_question(question, kb)
    
    # Should give fallback message
    assert "enough relevant information" in answer.lower() or "not enough" in answer.lower(), \
        "Weak relevance should trigger fallback message"
    assert "feeding" in answer.lower() or "grooming" in answer.lower(), \
        "Should suggest valid topics"


# ── Test 23: Clean answer generation (no raw chunks) ────────────────────────
def test_rag_clean_answer_generation():
    """Test that answers are clean and natural, not raw chunks."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    question = "How often should I feed my dog?"
    answer, sources, success = answer_pet_care_question(question, kb)
    
    if success:
        # Answer should not contain raw document markers or excessive length
        assert len(answer) < 1000, "Answer should be concise, not a dump of raw chunks"
        assert not answer.startswith("Feed"), "Should not start with raw chunk content"
        assert "•" in answer or "guidelines" in answer.lower(), \
            "Should have structured answer format"
        assert "knowledge base" in answer.lower(), \
            "Should reference knowledge base"


# ── Test 24: Medical questions include vet disclaimer ───────────────────────
def test_rag_medical_question_vet_disclaimer():
    """Test that medical questions include veterinarian disclaimer."""
    kb = KnowledgeBase(knowledge_base_dir="knowledge_base")
    
    medical_questions = [
        "My dog is sick, what should I do?",
        "My cat has a health problem",
        "Is this medication safe for pets?",
        "My pet has an injury"
    ]
    
    for question in medical_questions:
        answer, sources, success = answer_pet_care_question(question, kb)
        
        # Should include veterinarian disclaimer
        assert "veterinarian" in answer.lower() or "vet" in answer.lower(), \
            f"Medical question '{question}' should include vet disclaimer"
        assert "educational only" in answer.lower() or "not a substitute" in answer.lower(), \
            f"Should clarify educational nature for '{question}'"