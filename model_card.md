# 🎾 Model Card: PawPal+ RAG System

## 1. Model Name  

PawPal+ RAG System
---

## 2. Intended Use  

This RAG system answers pet care questions using a local knowledge base, integrated into a pet care scheduler. It helps pet owners get reliable information about feeding, grooming, behavior, health, and scheduling without relying on external AI services. It assumes users want grounded, traceable answers from custom documents rather than generic responses.

## 3. How the Model Works  

The system uses TF-IDF vectorization to index and retrieve relevant chunks from pet-care documents. It chunks documents by paragraphs with overlap, retrieves top-4 chunks based on similarity scores, prioritizes topic-specific sources (e.g., grooming documents for grooming questions), and generates clean answers from the retrieved context. Medical questions automatically include veterinarian disclaimers. The system logs all queries with sources and relevance scores.

---

## 4. Data  

The model uses a local knowledge base of 5 custom markdown documents created for this project: feeding guidelines, grooming guidelines, behavior training, health safety, and scheduler help. Documents are chunked into overlapping segments (800 characters with 200-character overlap) and indexed using TF-IDF with bigrams and sublinear term frequency scaling. No external data sources are used.

---

## 5. Strengths  

The system provides grounded, traceable answers with source attribution and relevance scores. It handles medical questions safely with disclaimers and gracefully manages vague or off-topic queries. The local knowledge base ensures reproducibility and privacy, with no API dependencies. It integrates seamlessly with the scheduling system to provide comprehensive pet care assistance.

---

## 6. Limitations and Bias 
  
One limitation of PawPal+ is that it depends on the accuracy of the information the user enters. The system is also biased toward the rules I designed, like prioritizing medical conditions, high-priority tasks, and time preferences, so it may oversimplify more complex pet-care situations.

The knowledge base is limited to 5 topics and may not cover all pet care scenarios. Some pet types (e.g., exotic pets) or advanced topics are not addressed. The TF-IDF retrieval may miss nuanced queries that require semantic understanding beyond keyword matching. The system biases toward the included documents and cannot answer questions outside the knowledge base scope. Medical answers include disclaimers but are not a substitute for professional veterinary care.


---

## 7. Evaluation  

The RAG system was evaluated through 8 unit tests covering knowledge base loading, retrieval accuracy, answer generation, medical disclaimers, and edge cases. All tests pass, including scenarios for feeding questions retrieving feeding documents, medical questions including disclaimers, and vague questions prompting clarification. Manual testing with example queries showed reliable retrieval for topic-specific questions and appropriate fallback for unrelated queries.

---

## 8. Possible Misuse and Prevention

PawPal+ could be misused if someone treats it like medical advice instead of a scheduling tool. To prevent this, I would add clear warnings that users should verify health-related tasks, medication details, and urgent concerns with a veterinarian.

---
## 9. Future Work  

I would expand the knowledge base to cover more topics like exotic pets, senior care, and emergency procedures. I would implement semantic search using embeddings for better retrieval of nuanced queries. I would add user feedback mechanisms to improve answer quality over time. I would enhance the UI to show more detailed source previews and allow users to rate answer helpfulness.

## 10. What Surprised Me During Reliability Testing

What surprised me most was that an output can look correct even when the logic behind it is wrong. The floating-point time bug showed me that I needed both automated tests and manual checks with real schedule examples.

---

## 11. Personal Reflection  

Building PawPal+ taught me the importance of grounding AI systems in reliable, version-controlled data rather than relying on generic language models. The RAG approach ensures answers are traceable and maintainable, which is crucial for pet care where misinformation can be harmful. This project reinforced that AI systems should explain their reasoning and stay within their knowledge boundaries. Working with AI tools like Claude Code showed me how to use them effectively as a collaborator while maintaining architectural control — the human architect sets the constraints, and AI executes within them efficiently. The most valuable lesson was that clear, modular design enables both human understanding and AI-assisted development.

