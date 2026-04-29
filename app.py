import streamlit as st
from datetime import date, timedelta
from pawpal_system import Owner, Pet, Task, Scheduler
from rag import initialize_knowledge_base, answer_pet_care_question

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")
st.title("🐾 PawPal+")

# Session State Vault
# Check before creating - Owner and Pet survive every rerun
if "owner" not in st.session_state:
    st.session_state["owner"] = Owner("Jordan")

owner: Owner = st.session_state["owner"]

# Owner Setup
st.subheader("Owner")
owner_name = st.text_input("Owner name", value=owner.name)
owner.name = owner_name

# Availability
st.markdown("#### Availability Windows (today)")
st.caption("Set the time blocks you're free to care for your pets.")

col1, col2 = st.columns(2)
with col1:
    start_hour = st.number_input("Window start (hour)", min_value=0, max_value=23, value=7)
with col2:
    end_hour = st.number_input("Window end (hour)", min_value=1, max_value=24, value=21)

if st.button("Set Availability"):
    owner.set_availability(date.today(), [(start_hour, end_hour)])
    st.success(f"Availability set: {start_hour}:00 - {end_hour}:00")

st.divider()

# Add a Pet
st.subheader("Pets")

with st.form("add_pet_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        pet_name = st.text_input("Pet name", value="Mochi")
    with col2:
        species = st.selectbox("Species", ["dog", "cat", "other"])
    with col3:
        age = st.number_input("Age", min_value=0, max_value=30, value=3)
    conditions = st.text_input("Medical conditions (comma-separated, or leave blank)")
    submitted = st.form_submit_button("Add Pet")

if submitted:
    condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
    new_pet = Pet(name=pet_name, species=species, age=age, medical_conditions=condition_list)
    owner.add_pet(new_pet)
    st.success(f"Added {pet_name}!")

# show current pets
if owner.pets:
    st.markdown("**Current pets:**")
    for pet in owner.pets:
        st.markdown(f"- {pet.get_profile()}")
else:
    st.info("No pets yet. Add one above.")

st.divider()

# Add a Task
st.subheader("Tasks")

if not owner.pets:
    st.warning("Add a pet first before adding tasks.")
else:
    with st.form("add_task_form", clear_on_submit=True):
        pet_names   = [p.name for p in owner.pets]
        target_pet  = st.selectbox("Assign to pet", pet_names)

        col1, col2 = st.columns(2)
        with col1:
            task_name = st.text_input("Task name", value="Morning walk")
            category  = st.selectbox("Category", ["walk", "feeding", "meds", "grooming", "enrichment"])
            priority  = st.selectbox("Priority", ["low", "medium", "high"], index=2)
        with col2:
            frequency      = st.selectbox("Frequency", ["daily", "weekly", "as-needed"])
            duration       = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
            preferred_time = st.selectbox("Preferred time", ["morning", "afternoon", "evening"])

        task_submitted = st.form_submit_button("Add Task")

    if task_submitted:
        new_task = Task(
            name=task_name,
            category=category,
            priority=priority,
            frequency=frequency,
            duration_minutes=int(duration),
            preferred_time=preferred_time,
        )
        pet_obj = next(p for p in owner.pets if p.name == target_pet)
        pet_obj.add_task(new_task)
        st.success(f"Added '{task_name}' to {target_pet}!")

    # show all tasks across pets
    all_tasks = [(t, p) for p in owner.pets for t in p.tasks]
    if all_tasks:
        st.markdown("**All tasks:**")
        for task, pet in all_tasks:
            st.markdown(
                f"- **{pet.name}** | {task.name} | {task.priority} priority | "
                f"{task.duration_minutes} min | {task.preferred_time}"
            )

st.divider()

# Generate Schedule
st.subheader("Build Schedule")

if st.button("Generate Schedule"):
    if not owner.pets or not any(p.tasks for p in owner.pets):
        st.warning("Add at least one pet and one task first.")
    elif not owner.get_availability(date.today()):
        st.warning("Set your availability windows before generating a schedule.")
    else:
        scheduler = Scheduler(owner=owner, schedule_date=date.today())
        scheduler.build_plan()

        # ── Conflict Detection (red dropdown) ──────────────────────────────
        conflicts = scheduler.detect_conflicts()
        if conflicts:
            with st.expander(
                f"⚠️ {len(conflicts)} scheduling conflict(s) detected — click to review",
                expanded=True,
            ):
                st.markdown(
                    "<span style='color:red; font-weight:bold;'>These tasks overlap in time. "
                    "Consider adjusting durations or availability windows.</span>",
                    unsafe_allow_html=True,
                )
                for warning in conflicts:
                    st.markdown(
                        f"<span style='color:red;'>🔴 {warning.strip()}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.success("Schedule generated — no conflicts detected!")

        # ── Sorted Schedule Table ──────────────────────────────────────────
        sorted_items = scheduler.sort_by_time()

        if sorted_items:
            st.markdown("#### Today's Plan")
            table_rows = []
            for item in sorted_items:
                end_dt = item.start_time + timedelta(minutes=item.task.duration_minutes)
                table_rows.append({
                    "Time": f"{item.start_time.strftime('%H:%M')} – {end_dt.strftime('%H:%M')}",
                    "Pet": item.pet.name,
                    "Task": item.task.name,
                    "Priority": item.task.priority.capitalize(),
                    "Duration": f"{item.task.duration_minutes} min",
                    "Status": item.status.capitalize(),
                })
            st.dataframe(table_rows, hide_index=True, use_container_width=True)

            # Detail cards beneath the table
            for item in sorted_items:
                end_dt = item.start_time + timedelta(minutes=item.task.duration_minutes)
                with st.container(border=True):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            f"**[{item.start_time.strftime('%H:%M')} – {end_dt.strftime('%H:%M')}]**  "
                            f"{item.pet.name}: {item.task.name}"
                        )
                        st.caption(f">> {item.reason}")
                    with col_b:
                        if item.status == "done":
                            st.success("Done")
                        else:
                            st.info("Pending")

        # ── Unscheduled Tasks ──────────────────────────────────────────────
        if scheduler.unscheduled_tasks:
            st.warning(f"Could not fit {len(scheduler.unscheduled_tasks)} task(s) — no available window:")
            for t in scheduler.unscheduled_tasks:
                st.markdown(f"- **{t.name}** | {t.priority} priority | {t.duration_minutes} min")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# PawPal+ Pet Care Assistant (RAG Feature)
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🤖 Ask PawPal: Pet Care Assistant")
st.caption("Ask any pet care question and get grounded answers from PawPal's knowledge base.")

# Initialize knowledge base on first load
if "knowledge_base" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state["knowledge_base"] = initialize_knowledge_base()

# User question input
question = st.text_input(
    "What would you like to know about pet care?",
    placeholder="e.g., How often should I feed my cat? What causes dental disease?",
    key="pet_care_question"
)

# Process question
if st.button("Get Answer", key="ask_button"):
    if not question or not question.strip():
        st.warning("Please ask a question about pet care.")
    else:
        with st.spinner("Searching PawPal's knowledge base..."):
            kb = st.session_state["knowledge_base"]
            answer, sources, success = answer_pet_care_question(question, kb)
        
        # Display answer
        st.markdown("#### Answer")
        st.write(answer)
        
        # Display sources if available
        # if sources:
        #     st.markdown("#### Sources")
        #     source_cols = st.columns(len(sources))
            
        #     for i, (source, col) in enumerate(zip(sources, source_cols)):
        #         with col:
        #             with st.container(border=True):
        #                 st.markdown(f"**{source['source']}**")
        #                 st.caption(f"Relevance: {source['relevance']}")
        #                 st.write(source['content_preview'])
        # else:
        #     st.info("No relevant documents found in the knowledge base for this question.")
        
        # Log interaction
        if success:
            st.success("✓ Answer generated from knowledge base.")
        else:
            st.warning(
                "⚠️ PawPal was unable to find relevant information. "
                "Consider consulting with a veterinarian."
            )