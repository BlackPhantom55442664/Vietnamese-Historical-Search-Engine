"""
PassageRetriever.py - Extract Relevant Passages from Documents

Module này tìm câu/đoạn văn liên quan nhất trong document dựa trên query.
Sử dụng sentence-level hoặc passage-level retrieval với embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class PassageRetriever:
    """
    Retrieve relevant passages/sentences from documents based on query similarity.

    Methods:
        - retrieve_relevant_passages: Sliding window passages (balanced)
        - retrieve_relevant_sentences: Individual sentences (precise)
    """

    def __init__(self, 
                 embedding_model: str = 'keepitreal/vietnamese-sbert',
                 window_size: int = 3,
                 overlap: int = 1):
        """
        Initialize PassageRetriever

        Args:
            embedding_model: SentenceTransformer model for Vietnamese
            window_size: Number of sentences per passage
            overlap: Number of overlapping sentences between passages
        """
        print(f"  → Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.window_size = window_size
        self.overlap = overlap
        print(f"  → Passage window: {window_size} sentences, overlap: {overlap}")

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (Vietnamese)

        Vietnamese sentence boundaries: . ! ? \n

        Args:
            text: Document text

        Returns:
            List of sentences
        """
        # Split by common sentence terminators
        sentences = re.split(r'[.!?\n]+', text)

        # Clean và filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def create_passages(self, sentences: List[str]) -> List[Dict]:
        """
        Create passages using sliding window

        Args:
            sentences: List of sentences

        Returns:
            List of passages with metadata:
            [{
                'text': str,
                'start_idx': int,
                'end_idx': int,
                'sentences': List[str]
            }]
        """
        passages = []

        step = max(1, self.window_size - self.overlap)

        for i in range(0, len(sentences), step):
            end_idx = min(i + self.window_size, len(sentences))
            passage_sentences = sentences[i:end_idx]

            passages.append({
                'text': ' '.join(passage_sentences),
                'start_idx': i,
                'end_idx': end_idx,
                'sentences': passage_sentences
            })

            if end_idx >= len(sentences):
                break

        return passages

    def retrieve_relevant_passages(self,
                                   query: str,
                                   document: str,
                                   top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant passages from document

        Args:
            query: User query
            document: Full document text
            top_k: Number of passages to return

        Returns:
            List of top-k passages sorted by relevance:
            [{
                'text': str,           # Passage text
                'score': float,        # Similarity score [0, 1]
                'start_idx': int,      # Start sentence index
                'end_idx': int,        # End sentence index
                'sentences': List[str] # Sentences in passage
            }]
        """
        # Step 1: Split document into sentences
        sentences = self.split_sentences(document)

        if not sentences:
            return []

        # Step 2: Create sliding window passages
        passages = self.create_passages(sentences)

        if not passages:
            return []

        # Step 3: Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Step 4: Encode passages
        passage_texts = [p['text'] for p in passages]
        passage_embeddings = self.embedding_model.encode(
            passage_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Step 5: Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, passage_embeddings)[0]

        # Step 6: Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Step 7: Build results
        results = []
        for idx in top_indices:
            passage = passages[idx].copy()
            passage['score'] = float(similarities[idx])
            results.append(passage)

        return results

    def retrieve_relevant_sentences(self,
                                    query: str,
                                    document: str,
                                    top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant individual sentences

        More precise than passages but less context.

        Args:
            query: User query
            document: Full document text
            top_k: Number of sentences to return

        Returns:
            List of top-k sentences:
            [{
                'text': str,     # Sentence text
                'score': float,  # Similarity score [0, 1]
                'index': int     # Position in document
            }]
        """
        # Split into sentences
        sentences = self.split_sentences(document)

        if not sentences:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Encode sentences
        sentence_embeddings = self.embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Calculate similarity
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'text': sentences[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            })

        return results


# Test module
if __name__ == "__main__":
    print("="*70)
    print("TESTING PassageRetriever")
    print("="*70)

    # Sample Vietnamese text
    sample_doc = """
    Hồ Chí Minh sinh ngày 19 tháng 5 năm 1890 tại làng Kim Liên, Nam Đàn, Nghệ An.
    Ông tên thật là Nguyễn Sinh Cung.
    Năm 1911, Hồ Chí Minh rời Việt Nam sang Pháp.
    Ông làm nghề bếp phó trên tàu Amiral Latouche-Tréville.
    Ngày 2 tháng 9 năm 1945, Hồ Chí Minh đọc Tuyên ngôn độc lập.
    Việt Nam Dân chủ Cộng hòa được thành lập.
    """

    query = "Hồ Chí Minh sinh năm nào?"

    retriever = PassageRetriever()

    print(f"\nQuery: {query}")
    print("\nRetrieving passages...")
    passages = retriever.retrieve_relevant_passages(query, sample_doc, top_k=2)

    for i, passage in enumerate(passages, 1):
        print(f"\n[Passage {i}] Score: {passage['score']:.4f}")
        print(f"Text: {passage['text']}")
