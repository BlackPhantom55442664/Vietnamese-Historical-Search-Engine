"""
Config.py - Configuration for TMGSEG Search Engine

Centralized configuration để dễ maintain và modify.
Theo loose coupling principle.
"""

import os


class Config:
    """Configuration class for search engine"""

    # ==================== DATA PATHS ====================
    DATA_PATH = os.path.join("dataset", "data_content.json")
    USER_QUERY_PATH = os.path.join("dataset", "userQuery.json")

    # ==================== MODELS ====================
    EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    # ==================== TEXT PROCESSING ====================
    TOKENIZER_MODE = 'word'           # 'word' or 'syllable'
    TOKENIZER_LIBRARY = 'underthesea' # 'underthesea' or 'pyvi'
    USE_STOPWORDS = True

    # ==================== THREE-STAGE RETRIEVAL ====================
    # Stage 1: BM25
    STAGE1_TOP_K = 100     # Number of candidates after BM25

    # Stage 2: Embedding
    STAGE2_TOP_K = 35      # Number of candidates after embedding (30-40 range)

    # Stage 3: Cross-encoder
    USE_STAGE3 = True
    TOP_K_RESULTS = 15     # Final top-k results

    # ==================== PASSAGE RETRIEVAL ====================
    USE_PASSAGE_RETRIEVAL = True    # Enable/disable passage retrieval
    PASSAGE_WINDOW_SIZE = 3         # Sentences per passage
    PASSAGE_OVERLAP = 1             # Overlap between passages
    TOP_K_PASSAGES = 3              # Number of passages to return per document

    # ==================== QUERY HANDLING ====================
    QUERY_COMBINATION_MODE = 'primary'  # 'primary' (Q1 only), 'combined' (Q1+Q2+...)

    # ==================== OUTPUT ====================
    SHOW_PASSAGES_IN_RESULTS = True  # Show relevant passages in search results
    MAX_CONTENT_PREVIEW = 300        # Max characters for content preview (fallback)

    # ==================== PERFORMANCE ====================
    BATCH_SIZE = 32                 # Batch size for embedding encoding
    USE_GPU = False                 # Use GPU if available
    QUERY_COMBINATION_MODE = 'primary'
    VISUALIZE_STEPS = False
    VIS_OUTPUT_DIR = "visualizations"
    
    # ==================== LOGGING ====================
    VERBOSE = True                  # Print detailed logs

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)

        print("\n[Data Paths]")
        print(f"  DATA_PATH: {cls.DATA_PATH}")
        print(f"  USER_QUERY_PATH: {cls.USER_QUERY_PATH}")

        print("\n[Models]")
        print(f"  EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}")
        print(f"  RERANKER_MODEL: {cls.RERANKER_MODEL}")

        print("\n[Text Processing]")
        print(f"  TOKENIZER_MODE: {cls.TOKENIZER_MODE}")
        print(f"  TOKENIZER_LIBRARY: {cls.TOKENIZER_LIBRARY}")
        print(f"  USE_STOPWORDS: {cls.USE_STOPWORDS}")

        print("\n[Three-Stage Retrieval]")
        print(f"  Stage 1 (BM25): Top-{cls.STAGE1_TOP_K}")
        print(f"  Stage 2 (Embedding): Top-{cls.STAGE2_TOP_K}")
        print(f"  Stage 3 (Cross-Encoder): {cls.USE_STAGE3}")
        print(f"  Final Results: Top-{cls.TOP_K_RESULTS}")

        print("\n[Passage Retrieval]")
        print(f"  USE_PASSAGE_RETRIEVAL: {cls.USE_PASSAGE_RETRIEVAL}")
        print(f"  PASSAGE_WINDOW_SIZE: {cls.PASSAGE_WINDOW_SIZE} sentences")
        print(f"  PASSAGE_OVERLAP: {cls.PASSAGE_OVERLAP} sentences")
        print(f"  TOP_K_PASSAGES: {cls.TOP_K_PASSAGES}")

        print("\n[Query Handling]")
        print(f"  QUERY_COMBINATION_MODE: {cls.QUERY_COMBINATION_MODE}")

        print("\n[Output]")
        print(f"  SHOW_PASSAGES_IN_RESULTS: {cls.SHOW_PASSAGES_IN_RESULTS}")
        print(f"  MAX_CONTENT_PREVIEW: {cls.MAX_CONTENT_PREVIEW} chars")


if __name__ == "__main__":
    Config.print_config()
