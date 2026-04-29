"""
PawPal+ RAG (Retrieval-Augmented Generation) Module

This module provides a Retrieval-Augmented Generation system for answering
pet care questions using a local knowledge base. It loads documents from
the knowledge_base folder, chunks them, retrieves relevant content using
TF-IDF similarity, and generates grounded answers with source tracking.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("scikit-learn is required for RAG functionality. Install with: pip install scikit-learn")


# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging for RAG queries.
    
    Args:
        log_dir: Directory to store logs (created if it doesn't exist)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("pawpal_rag")
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(log_dir, "rag_queries.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Log format: timestamp | question | sources | scores | answer_found
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Document Chunk
# ──────────────────────────────────────────────────────────────────────────────

class DocumentChunk:
    """
    Represents a chunk of text from a document with metadata.
    
    Attributes:
        text: The chunk content
        source_file: Name of the source document
        section_title: Optional section title within the document
        chunk_id: Unique identifier for this chunk
    """
    
    def __init__(self, text: str, source_file: str, section_title: str = "", chunk_id: int = 0):
        """Initialize a document chunk."""
        self.text = text
        self.source_file = source_file
        self.section_title = section_title
        self.chunk_id = chunk_id
    
    def __repr__(self) -> str:
        """Return string representation."""
        title_part = f" | {self.section_title}" if self.section_title else ""
        return f"DocumentChunk(source={self.source_file}{title_part}, id={self.chunk_id})"


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Manages the pet care knowledge base.
    
    Loads and chunks documents from the knowledge_base folder, maintains
    TF-IDF index for retrieval, and tracks document sources.
    """
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_base_dir: Path to folder containing .md documents
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.logger = _setup_logging()
        
        self._load_documents()
        self._build_index()
    
    def _load_documents(self) -> None:
        """Load and chunk all markdown documents from knowledge_base folder."""
        kb_path = Path(self.knowledge_base_dir)
        
        if not kb_path.exists():
            self.logger.warning(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return
        
        # Find all markdown files
        md_files = sorted(kb_path.glob("*.md"))
        
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Chunk the document
                chunks = self._chunk_document(content, md_file.name)
                self.chunks.extend(chunks)
                self.logger.info(f"Loaded {md_file.name}: {len(chunks)} chunks")
            
            except Exception as e:
                self.logger.error(f"Error loading {md_file.name}: {e}")
    
    def _chunk_document(self, content: str, source_file: str, chunk_size: int = 800, overlap: int = 200) -> List[DocumentChunk]:
        """
        Split a document into overlapping chunks by paragraphs and sections.
        
        Args:
            content: Full document text
            source_file: Name of the source file
            chunk_size: Target characters per chunk
            overlap: Overlap characters between chunks
        
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_id = 0
        
        # Split by headers first to preserve section structure
        sections = re.split(r'(?=^#{1,6}\s+)', content, flags=re.MULTILINE)
        
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If adding this section would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(section) > chunk_size:
                if current_chunk.strip():
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        source_file=source_file,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                
                # Start new chunk with overlap from previous
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source_file=source_file,
                chunk_id=chunk_id
            ))
        
        return chunks if chunks else [DocumentChunk(content, source_file, chunk_id=0)]
    
    def _build_index(self) -> None:
        """Build TF-IDF index for all chunks."""
        if not self.chunks:
            return
        
        # Extract chunk texts
        chunk_texts = [chunk.text for chunk in self.chunks]
        
        # Create TF-IDF vectorizer with bigrams for better query-match quality
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            lowercase=True,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of (chunk, relevance_score) tuples, sorted by relevance
        """
        if not self.chunks or self.vectorizer is None:
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top-k indices
            top_indices = similarities.argsort()[::-1][:top_k]
            
            # Determine a quality threshold for this query
            query_lower = query.lower()
            topic_keywords = [
                "feed", "feeding", "food", "groom", "grooming", "brush", "bath",
                "nail", "teeth", "dental", "train", "behavior", "health", "medical",
                "vet", "veterinarian", "schedule", "pawpal", "exercise", "walk"
            ]
            min_score = 0.25
            if any(keyword in query_lower for keyword in topic_keywords):
                min_score = 0.12
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append((self.chunks[idx], score))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []


# ──────────────────────────────────────────────────────────────────────────────
# RAG Answer Generator
# ──────────────────────────────────────────────────────────────────────────────

def _is_medical_question(question: str) -> bool:
    """
    Detect if a question is about medical or emergency topics.
    
    Args:
        question: User question
    
    Returns:
        True if question appears to be medical-related
    """
    medical_keywords = {
        "health", "disease", "illness", "sick", "symptom", "disease",
        "pain", "injury", "wound", "infection", "fever", "vomit", "diarrhea",
        "seizure", "emergency", "urgent", "hospital", "vet", "veterinarian",
        "medication", "medicine", "drug", "prescription", "treatment",
        "diagnosis", "condition", "disorder", "syndrome", "allergy",
        "toxic", "poison", "toxic", "bleed", "blood", "fracture",
        "limp", "paralysis", "swelling", "rash", "lump", "tumor",
        "cancer", "diabetes", "arthritis", "kidney", "liver", "heart"
    }
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in medical_keywords)


def _is_vague_question(question: str) -> bool:
    """
    Detect if a question is too vague to answer specifically.
    
    Args:
        question: User question
    
    Returns:
        True if question is too vague
    """
    # First check if it's a medical question - those should not be considered vague
    if _is_medical_question(question):
        return False
    
    vague_patterns = [
        r"what should i do.*(?:dog|cat|pet)",
        r"how.*(?:do|should).*i.*(?:take care of|care for).*my.*(?:dog|cat|pet)",
        r"tell me about.*(?:dog|cat|pet)",
        r"what do i need.*(?:to know|to do).*for.*(?:dog|cat|pet)",
        r"help.*with.*(?:dog|cat|pet)"
    ]
    
    question_lower = question.lower().strip()
    
    topic_keywords = [
        "feed", "feeding", "food", "groom", "grooming", "brush", "bath",
        "nail", "teeth", "dental", "train", "training", "behavior", "health",
        "medical", "vet", "veterinarian", "schedule", "pawpal", "exercise", "walk"
    ]
    if any(keyword in question_lower for keyword in topic_keywords):
        return False
    
    # Check for vague patterns
    for pattern in vague_patterns:
        if re.search(pattern, question_lower):
            return True
    
    # Check for very short questions without specific topics
    words = question_lower.split()
    if len(words) <= 4 and not any(word in question_lower for word in [
        "feed", "feeding", "groom", "grooming", "train", "training", 
        "behavior", "health", "safety", "medical", "vaccine", "vaccination",
        "walk", "exercise", "bath", "bathing", "nail", "teeth", "dental"
    ]):
        return True
    
    return False


def _generate_answer(question: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> Tuple[str, List[Dict]]:
    """
    Generate a clean, natural answer from retrieved chunks.
    
    Args:
        question: User question
        retrieved_chunks: List of (chunk, score) tuples from retrieval
    
    Returns:
        Tuple of (answer_text, source_list)
    """
    # Check if any context was retrieved
    if not retrieved_chunks:
        answer = (
            "I do not have enough relevant information in PawPal's knowledge base to answer that clearly. "
            "Try asking about feeding, grooming, behavior, health safety, or scheduling."
        )
        return answer, []
    
    # Bias results toward topic-relevant knowledge sources when possible
    retrieved_chunks = _prioritize_topic_source(question, retrieved_chunks)
    
    # Check if best score is too low (weak relevance)
    best_score = retrieved_chunks[0][1]
    question_lower = question.lower()
    topic_keywords = [
        "feed", "feeding", "food", "groom", "grooming", "brush", "bath",
        "nail", "teeth", "dental", "train", "behavior", "health", "medical",
        "vet", "veterinarian", "schedule", "pawpal", "exercise", "walk"
    ]
    min_answer_score = 0.25
    if any(keyword in question_lower for keyword in topic_keywords):
        min_answer_score = 0.12
    if best_score < min_answer_score:
        answer = (
            "I do not have enough relevant information in PawPal's knowledge base to answer that clearly. "
            "Try asking about feeding, grooming, behavior, health safety, or scheduling."
        )
        return answer, []
    
    # Build source list with relevance info
    sources = []
    for chunk, score in retrieved_chunks:
        sources.append({
            "source": chunk.source_file,
            "relevance": f"{score:.2f}",
            "content_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
        })
    
    # Extract context from retrieved chunks
    context_parts = [chunk.text for chunk, _ in retrieved_chunks]
    full_context = "\n\n".join(context_parts)
    
    # Generate clean answer based on question type and context
    answer = _create_clean_answer(question, full_context, best_score)
    
    # Add medical disclaimer if needed
    if _is_medical_question(question):
        answer += "\n\n⚠️ **Important:** This information is educational only and is NOT a substitute for professional veterinary care. Please consult your veterinarian for medical advice."
    
    return answer, sources


def _prioritize_topic_source(question: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
    """
    Prefer topic-specific sources when retrieved chunks include them.
    """
    question_lower = question.lower()
    source_preferences = []
    if any(word in question_lower for word in ["feed", "feeding", "food", "eat", "meal"]):
        source_preferences = ["feeding_guidelines.md"]
    elif any(word in question_lower for word in ["groom", "grooming", "bath", "brush", "nail", "teeth", "dental"]):
        source_preferences = ["grooming_guidelines.md"]
    elif any(word in question_lower for word in ["train", "training", "behavior", "behave", "obedience"]):
        source_preferences = ["behavior_training.md"]
    elif any(word in question_lower for word in ["health", "medical", "sick", "illness", "disease", "safety", "vet", "veterinarian"]):
        source_preferences = ["health_safety.md"]
    elif any(word in question_lower for word in ["schedule", "time", "when", "pawpal"]):
        source_preferences = ["scheduler_help.md"]
    
    if source_preferences:
        preferred = [pair for pair in retrieved_chunks if pair[0].source_file in source_preferences]
        if preferred:
            return preferred
    return retrieved_chunks


def _create_clean_answer(question: str, context: str, relevance_score: float) -> str:
    """
    Create a clean, natural answer from context.
    
    Args:
        question: User question
        context: Retrieved context text
        relevance_score: Best relevance score
    
    Returns:
        Clean answer string
    """
    question_lower = question.lower()
    
    # Extract key information from context using patterns
    key_points = _extract_key_points(context)
    
    # Generate answer based on question topic
    if any(word in question_lower for word in ["feed", "feeding", "food", "eat", "meal"]):
        return _answer_feeding_question(key_points)
    elif any(word in question_lower for word in ["groom", "grooming", "bath", "brush", "nail", "teeth", "dental"]):
        return _answer_grooming_question(key_points)
    elif any(word in question_lower for word in ["train", "training", "behavior", "behave", "obedience"]):
        return _answer_behavior_question(key_points)
    elif any(word in question_lower for word in ["health", "medical", "sick", "illness", "disease", "safety"]):
        return _answer_health_question(key_points)
    elif any(word in question_lower for word in ["schedule", "time", "when", "pawpal"]):
        return _answer_scheduler_question(key_points)
    else:
        # General answer for other topics
        return _answer_general_question(key_points, relevance_score)


def _extract_key_points(context: str) -> List[str]:
    """
    Extract key points from context text.
    
    Args:
        context: Full context text
    
    Returns:
        List of key points
    """
    points = []
    
    # Split by bullet points, numbered lists, or sentences
    lines = context.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract bullet points
        if re.match(r'^[\-\*•–—]\s+', line):
            point = re.sub(r'^[\-\*•–—]\s+', '', line).strip()
            point = re.sub(r'^\*\*(.*?)\*\*', r'\1', point)
            points.append(point)
        # Extract numbered items
        elif re.match(r'^\d+\.\s', line):
            points.append(re.sub(r'^\d+\.\s', '', line).strip())
        # Extract sentences with key information
        elif len(line) > 20 and any(keyword in line.lower() for keyword in [
            "feed", "groom", "train", "exercise", "health", "schedule", "daily", "weekly"
        ]):
            # Split into sentences and take the first meaningful one
            sentences = re.split(r'[.!?]+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:
                    points.append(sentence)
                    break
    
    return points[:4]  # Limit to 4 key points


def _answer_feeding_question(key_points: List[str]) -> str:
    """Generate clean feeding answer."""
    base_answer = "For feeding your pet, here are the key guidelines from PawPal's knowledge base:"
    
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"{base_answer}\n\n{points_text}"
    else:
        return "Based on PawPal's knowledge base, maintain a consistent feeding schedule with appropriate portions for your pet's age, size, and activity level. Provide fresh water daily and consult feeding guidelines for specific recommendations."


def _answer_grooming_question(key_points: List[str]) -> str:
    """Generate clean grooming answer."""
    base_answer = "For grooming your pet, here are the key recommendations from PawPal's knowledge base:"
    
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"{base_answer}\n\n{points_text}"
    else:
        return "Based on PawPal's knowledge base, regular grooming includes brushing, nail care, dental hygiene, and bathing as needed. Frequency depends on your pet's coat type and lifestyle."


def _answer_behavior_question(key_points: List[str]) -> str:
    """Generate clean behavior answer."""
    base_answer = "For training and behavior, here are the key principles from PawPal's knowledge base:"
    
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"{base_answer}\n\n{points_text}"
    else:
        return "Based on PawPal's knowledge base, use positive reinforcement training methods, provide mental stimulation, ensure adequate exercise, and establish consistent routines. For behavior issues, consider professional training help."


def _answer_health_question(key_points: List[str]) -> str:
    """Generate clean health answer."""
    base_answer = "For health and safety, here are the key recommendations from PawPal's knowledge base:"
    
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"{base_answer}\n\n{points_text}"
    else:
        return "Based on PawPal's knowledge base, maintain preventive care schedules, watch for health changes, ensure parasite prevention, and provide a safe environment. Regular veterinary check-ups are essential."


def _answer_scheduler_question(key_points: List[str]) -> str:
    """Generate clean scheduler answer."""
    base_answer = "For using PawPal's scheduling features, here are the key guidelines:"
    
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"{base_answer}\n\n{points_text}"
    else:
        return "Based on PawPal's knowledge base, set up tasks with appropriate priorities, frequencies, and preferred times. The scheduler will optimize your daily plan while respecting availability windows and medical needs."


def _answer_general_question(key_points: List[str], relevance_score: float) -> str:
    """Generate clean general answer."""
    if key_points:
        points_text = "\n".join(f"• {point}" for point in key_points[:3])
        return f"Based on PawPal's knowledge base, here are the relevant guidelines:\n\n{points_text}"
    else:
        return f"Based on PawPal's knowledge base (relevance: {relevance_score:.2f}), please ask about specific topics like feeding, grooming, behavior, health, or scheduling for more detailed guidance."


def answer_pet_care_question(question: str, knowledge_base: Optional[KnowledgeBase] = None) -> Tuple[str, List[Dict], bool]:
    """
    Answer a pet care question using RAG.
    
    This is the main interface for the RAG system. It retrieves relevant
    knowledge base documents and generates a grounded answer.
    
    Args:
        question: User's pet care question
        knowledge_base: Optional pre-initialized KnowledgeBase (created if not provided)
    
    Returns:
        Tuple of:
            - answer_text: Generated answer
            - sources: List of source documents used
            - success: Whether an answer was generated
    """
    logger = _setup_logging()
    
    # Initialize knowledge base if not provided
    if knowledge_base is None:
        knowledge_base = KnowledgeBase()
    
    # Validate input
    if not question or not question.strip():
        return "Please ask a question about pet care.", [], False
    
    question = question.strip()
    
    # Check for vague questions
    if _is_vague_question(question):
        answer = (
            "I'd be happy to help with your pet care questions! To give you the most relevant advice, "
            "could you please be more specific? For example:\n\n"
            "• What should I feed my dog?\n"
            "• How often should I groom my cat?\n"
            "• What training tips do you have for behavior issues?\n"
            "• How can I keep my pet healthy?\n"
            "• How do I use PawPal's scheduling features?\n\n"
            "Feel free to ask about feeding, grooming, behavior, health, or scheduling!"
        )
        logger.info(f"Vague question: {question}")
        return answer, [], True  # Success=True because we handled it gracefully
    
    # Retrieve relevant chunks
    retrieved_chunks = knowledge_base.retrieve(question, top_k=4)
    
    # Generate answer
    answer, sources = _generate_answer(question, retrieved_chunks)
    
    # Special handling for medical questions - always include disclaimer
    if _is_medical_question(question):
        if not retrieved_chunks:  # No relevant chunks found
            answer = (
                "For health concerns, I recommend consulting your veterinarian immediately for proper diagnosis and treatment. "
                "PawPal's knowledge base focuses on general preventive care and wellness tips.\n\n"
                "⚠️ **Important:** This information is educational only and is NOT a substitute for professional veterinary care. "
                "Please consult your veterinarian for medical advice."
            )
        # If chunks were found, the disclaimer is already added in _generate_answer
    
    # Determine success (we generated an answer with sources)
    success = len(retrieved_chunks) > 0
    
    # Log the query
    source_names = [s["source"] for s in sources] if sources else ["no_sources"]
    scores = [s["relevance"] for s in sources] if sources else ["0.00"]
    log_message = f"Q: {question[:60]} | Sources: {','.join(source_names)} | Scores: {','.join(scores)} | Success: {success}"
    logger.info(log_message)
    
    return answer, sources, success


# ──────────────────────────────────────────────────────────────────────────────
# Standalone initialization (for testing)
# ──────────────────────────────────────────────────────────────────────────────

_kb_instance: Optional[KnowledgeBase] = None


def initialize_knowledge_base(knowledge_base_dir: str = "knowledge_base") -> KnowledgeBase:
    """
    Initialize or return the cached knowledge base.
    
    Args:
        knowledge_base_dir: Path to knowledge base folder
    
    Returns:
        Initialized KnowledgeBase instance
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(knowledge_base_dir)
    return _kb_instance


if __name__ == "__main__":
    # Example usage
    kb = initialize_knowledge_base()
    
    # Test questions
    test_questions = [
        "How often should I feed my dog?",
        "What should I do about my cat's dental health?",
        "My dog has arthritis, what exercises should they do?",
        "Is chocolate safe for pets?"
    ]
    
    print("=" * 70)
    print("PawPal+ RAG System Test")
    print("=" * 70)
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        answer, sources, success = answer_pet_care_question(q, kb)
        print(f"Answer:\n{answer}")
        if sources:
            print(f"Sources: {[s['source'] for s in sources]}")
        print("-" * 70)
