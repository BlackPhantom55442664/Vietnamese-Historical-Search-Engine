"""
SearchEngine.py - Main Search Engine Orchestrator
Kết hợp tất cả các modules theo loose coupling architecture
"""

from typing import List, Dict
from Config import Config
from DataHandler import DataHandler
from TextProcessor import TextProcessor
from ThreeStageRetrieval import ThreeStageRetrieval
from DataVisualize import DataVisualizer

import os

class SearchEngine:
    """
    Main Search Engine với Three-Stage Retrieval
    Architecture: Loose coupling, modular design
    """

    def __init__(self, config: Config = None):
        """
        Initialize search engine với config

        Args:
            config: Config object (nếu None sẽ dùng default)
        """
        self.config = config if config else Config()

        print("=" * 70)
        print("TMGSEG SEARCH ENGINE - THREE-STAGE RETRIEVAL")
        print("Based on: Text Retrieval with Multi-Stage Re-Ranking Models")
        print("=" * 70)

        # 1. DataHandler
        print("\n[1/4] Initializing DataHandler...")
        self.data_handler = DataHandler(self.config.DATA_PATH)

        # 2. TextProcessor
        print("\n[2/4] Initializing TextProcessor...")
        self.text_processor = TextProcessor(
            use_stopwords=self.config.USE_STOPWORDS,
            library=self.config.TOKENIZER_LIBRARY
        )

        # 3. ThreeStageRetrieval
        print("\n[3/4] Initializing Three-Stage Retrieval...")
        self.retrieval = ThreeStageRetrieval(
            embedding_model=self.config.EMBEDDING_MODEL,
            reranker_model=self.config.RERANKER_MODEL,
            stage1_top_k=self.config.STAGE1_TOP_K,
            stage2_top_k=self.config.STAGE2_TOP_K,
            use_stage3=self.config.USE_STAGE3
        )

        # 4. Visualizer
        if self.config.VISUALIZE_STEPS:
            print("\n[4/4] Initializing Visualizer...")
            self.visualizer = DataVisualizer(save_dir=self.config.VIS_OUTPUT_DIR)
        else:
            self.visualizer = None

        # Storage
        self.documents = []

        print("\n✓ Search Engine initialized successfully!")

    def build_index(self):
        """Build index cho search engine"""
        print("\n" + "=" * 70)
        print("BUILDING INDEX")
        print("=" * 70)

        # Step 1: Load documents
        print("\n[Step 1/3] Loading documents...")
        self.documents = self.data_handler.load_data()
        contents = [doc['content'] for doc in self.documents]

        # Step 2: Process text (Tokenizing + Stopping)
        print("\n[Step 2/3] Processing text...")
        print("  → Tokenizing documents...")
        print("  → Removing stopwords...")
        tokenized_docs = self.text_processor.process_documents(contents)

        # Get text processing stats
        if self.visualizer and len(contents) > 0:
            # Process one sample for stats
            self.text_processor.process(contents[0])
            text_stats = self.text_processor.get_stats()
            self.visualizer.visualize_text_processing(text_stats)

        print(f"✓ Processed {len(tokenized_docs)} documents")

        # Step 3: Index documents
        print("\n[Step 3/3] Indexing documents...")
        self.retrieval.index_documents(
            tokenized_docs=tokenized_docs,
            raw_docs=contents
        )

        print("\n" + "=" * 70)
        print("✓ INDEX BUILT SUCCESSFULLY")
        print("=" * 70)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search với three-stage retrieval

        Args:
            query: Search query
            top_k: Số kết quả trả về

        Returns:
            List[Dict]: Ranked results
        """
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS

        # Process query
        query_tokens = self.text_processor.process(query)

        # Three-stage retrieval
        results = self.retrieval.retrieve(
            query=query,
            query_tokens=query_tokens,
            top_k=top_k
        )

        # Visualize if enabled
        if self.visualizer:
            retrieval_stats = self.retrieval.get_stats()
            self.visualizer.visualize_retrieval_stages(retrieval_stats)

            # Visualize score distributions
            if retrieval_stats.get('stage1_results'):
                self.visualizer.visualize_score_distribution(
                    retrieval_stats['stage1_results'][:20],
                    'Stage 1 BM25'
                )
            if retrieval_stats.get('stage3_results'):
                self.visualizer.visualize_score_distribution(
                    retrieval_stats['stage3_results'],
                    'Stage 3 Final'
                )

        # Format results
        formatted_results = []
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = self.documents[doc_id]
            formatted_results.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': score,
                'file_name': doc.get('file_name', f'Document {doc_id}'),
                'content': doc['content'][:300] + '...' if len(doc['content']) > 300 else doc['content']
            })

        return formatted_results

    def print_results(self, query: str, results: List[Dict]):
        """Print search results"""
        print("\n" + "=" * 70)
        print(f"SEARCH RESULTS: '{query}'")
        print("=" * 70)

        if not results:
            print("\nNo results found.")
            return

        for result in results:
            print(f"\n[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"📄 File: {result['file_name']}")
            print(f"💡 Preview: {result['content'][:200]}...")
            print("-" * 70)

    def interactive_search(self):
        """Interactive search mode"""
        print("\n" + "=" * 70)
        print("INTERACTIVE SEARCH MODE")
        print("=" * 70)
        print("Commands:")
        print("  - Nhập query để search")
        print("  - 'config' để xem cấu hình")
        print("  - 'exit' để thoát")
        print("=" * 70)

        while True:
            query = input("\n🔍 Query: ").strip()

            if query.lower() == 'exit':
                print("Goodbye!")
                break

            if query.lower() == 'config':
                self.config.print_config()
                continue

            if not query:
                continue

            # Search
            os.system('cls')
            results = self.search(query)
            self.print_results(query, results)
