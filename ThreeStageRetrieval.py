"""
ThreeStageRetrieval.py - Three-Stage Retrieval Architecture
Theo paper: "Text Retrieval with Multi-Stage Re-Ranking Models"

Stage 1: BM25 (a0 -> a1 documents)
Stage 2: Embedding/Language Model (a1 -> a2 documents)  
Stage 3: Cross-encoder Reranking (a2 -> final top-k)
"""

import numpy as np
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time

class ThreeStageRetrieval:
    """Three-stage retrieval system"""

    def __init__(self,
                 embedding_model: str,
                 reranker_model: str,
                 stage1_top_k: int = 100,
                 stage2_top_k: int = 20,
                 use_stage3: bool = True):
        """
        Args:
            embedding_model: Model cho Stage 2 (bi-encoder)
            reranker_model: Model cho Stage 3 (cross-encoder)
            stage1_top_k: Số documents sau Stage 1 (BM25)
            stage2_top_k: Số documents sau Stage 2 (Embedding)
            use_stage3: Có sử dụng Stage 3 reranking không
        """
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.use_stage3 = use_stage3

        # Stage 1: BM25
        self.bm25 = None
        self.tokenized_corpus = []

        # Stage 2: Embedding model
        print(f"[Stage 2] Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_embeddings = None
        self.raw_documents = []

        # Stage 3: Cross-encoder reranker
        if use_stage3:
            print(f"[Stage 3] Loading reranker model: {reranker_model}...")
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None

        # Statistics cho visualization
        self.retrieval_stats = {
            'stage1_time': 0,
            'stage2_time': 0,
            'stage3_time': 0,
            'stage1_results': [],
            'stage2_results': [],
            'stage3_results': []
        }

        print("✓ ThreeStageRetrieval initialized")

    def index_documents(self,
                       tokenized_docs: List[List[str]],
                       raw_docs: List[str]):
        """
        Index documents cho cả 3 stages

        Args:
            tokenized_docs: Documents đã tokenize (cho BM25)
            raw_docs: Documents gốc (cho embedding)
        """
        print("\n" + "="*60)
        print("INDEXING DOCUMENTS - THREE STAGES")
        print("="*60)

        # Stage 1: Index BM25
        print("\n[Stage 1/3] Creating BM25 index...")
        start = time.time()
        self.tokenized_corpus = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✓ BM25 indexed {len(tokenized_docs)} documents ({time.time()-start:.2f}s)")

        # Stage 2: Create embeddings
        print("\n[Stage 2/3] Creating document embeddings...")
        start = time.time()
        self.raw_documents = raw_docs
        self.document_embeddings = self.embedding_model.encode(
            raw_docs,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        print(f"✓ Created embeddings for {len(raw_docs)} documents ({time.time()-start:.2f}s)")

        # Stage 3: Reranker không cần index trước
        if self.use_stage3:
            print("\n[Stage 3/3] Reranker ready (no pre-indexing needed)")

        print("\n" + "="*60)
        print("✓ INDEXING COMPLETED")
        print("="*60)

    def stage1_bm25_retrieval(self,
                             query_tokens: List[str],
                             top_k: int) -> List[Tuple[int, float]]:
        """Stage 1: BM25 retrieval"""
        start = time.time()

        if self.bm25 is None:
            raise ValueError("BM25 chưa được index!")

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]

        self.retrieval_stats['stage1_time'] = time.time() - start
        self.retrieval_stats['stage1_results'] = results

        return results

    def stage2_embedding_retrieval(self,
                                   query: str,
                                   candidate_ids: List[int],
                                   top_k: int) -> List[Tuple[int, float]]:
        """Stage 2: Re-rank candidates bằng embedding similarity"""
        start = time.time()

        if self.document_embeddings is None:
            raise ValueError("Embeddings chưa được tạo!")

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        # Chỉ tính similarity với candidates
        candidate_embeddings = self.document_embeddings[candidate_ids]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Sắp xếp và lấy top_k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        # Map lại về doc IDs gốc
        results = [
            (candidate_ids[idx], float(similarities[idx]))
            for idx in sorted_indices
        ]

        self.retrieval_stats['stage2_time'] = time.time() - start
        self.retrieval_stats['stage2_results'] = results

        return results

    def stage3_rerank(self,
                     query: str,
                     candidate_ids: List[int]) -> List[Tuple[int, float]]:
        """Stage 3: Re-rank bằng cross-encoder"""
        start = time.time()

        if not self.use_stage3 or self.reranker is None:
            return [(doc_id, 0.0) for doc_id in candidate_ids]

        # Tạo pairs (query, document) cho cross-encoder
        pairs = []
        for doc_id in candidate_ids:
            doc_text = self.raw_documents[doc_id]
            # Truncate document nếu quá dài
            doc_text = doc_text[:512] if len(doc_text) > 512 else doc_text
            pairs.append([query, doc_text])

        # Score bằng cross-encoder
        scores = self.reranker.predict(pairs)

        # Sắp xếp theo score
        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = [(int(doc_id), float(score)) for doc_id, score in ranked]

        self.retrieval_stats['stage3_time'] = time.time() - start
        self.retrieval_stats['stage3_results'] = results

        return results

    def retrieve(self,
                query: str,
                query_tokens: List[str],
                top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Main retrieval - chạy qua cả 3 stages

        Returns:
            List[(doc_id, score)] - final top_k results
        """
        print(f"\n{'='*60}")
        print(f"THREE-STAGE RETRIEVAL: '{query}'")
        print(f"{'='*60}")

        # STAGE 1: BM25
        print(f"\n[Stage 1] BM25 retrieval (top {self.stage1_top_k})...")
        stage1_results = self.stage1_bm25_retrieval(
            query_tokens,
            top_k=self.stage1_top_k
        )
        stage1_ids = [doc_id for doc_id, _ in stage1_results]
        print(f"✓ Stage 1 completed: {len(stage1_ids)} candidates ({self.retrieval_stats['stage1_time']:.3f}s)")

        # STAGE 2: Embedding
        print(f"\n[Stage 2] Embedding re-ranking (top {self.stage2_top_k})...")
        stage2_results = self.stage2_embedding_retrieval(
            query,
            candidate_ids=stage1_ids,
            top_k=self.stage2_top_k
        )
        stage2_ids = [doc_id for doc_id, _ in stage2_results]
        print(f"✓ Stage 2 completed: {len(stage2_ids)} candidates ({self.retrieval_stats['stage2_time']:.3f}s)")

        # STAGE 3: Cross-encoder reranking
        if self.use_stage3:
            print(f"\n[Stage 3] Cross-encoder re-ranking...")
            stage3_results = self.stage3_rerank(
                query,
                candidate_ids=stage2_ids
            )
            final_results = stage3_results[:top_k]
            print(f"✓ Stage 3 completed: {len(final_results)} final results ({self.retrieval_stats['stage3_time']:.3f}s)")
        else:
            print(f"\n[Stage 3] Skipped (disabled)")
            final_results = stage2_results[:top_k]

        print(f"\n{'='*60}")
        print(f"✓ RETRIEVAL COMPLETED")
        print(f"{'='*60}\n")

        return final_results

    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return self.retrieval_stats.copy()
