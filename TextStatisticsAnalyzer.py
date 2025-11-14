"""
TextStatisticsAnalyzer.py - Standalone Text Statistics Analyzer

Script này chạy độc lập để phân tích text statistics theo slide 4:
- Zipf's Law: Frequency vs Rank distribution
- Heap's Law: Vocabulary growth
- N-grams: Most frequent phrases
- Token statistics

Có thể sử dụng lại TextProcessor từ project chính.
"""

import os
import json
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np

# Import từ project chính (nếu có)
try:
    from TextProcessor import TextProcessor
    from DataHandler import DataHandler
    USE_PROJECT_MODULES = True
    print("✓ Imported modules from main project")
except ImportError:
    print("⚠ Cannot import from main project, using standalone mode")
    USE_PROJECT_MODULES = False


class SimpleTextProcessor:
    """Simple text processor nếu không import được từ project chính"""

    def __init__(self):
        self.stopwords = {
            'và', 'của', 'có', 'cho', 'với', 'được', 'từ', 'trong',
            'là', 'một', 'các', 'để', 'theo', 'này', 'đó', 'những',
            'nhưng', 'hoặc', 'nếu', 'thì', 'khi', 'vì', 'do', 'bởi'
        }

    def process(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords and len(t) >= 2]
        return tokens


class TextStatisticsAnalyzer:
    """Analyzer để tính toán text statistics theo slide 4"""

    def __init__(self, use_project_processor: bool = True):
        """
        Args:
            use_project_processor: Có dùng TextProcessor từ project chính không
        """
        if use_project_processor and USE_PROJECT_MODULES:
            self.processor = TextProcessor(use_stopwords=True)
            print("✓ Using TextProcessor from main project")
        else:
            self.processor = SimpleTextProcessor()
            print("✓ Using standalone SimpleTextProcessor")

        # Storage
        self.all_tokens = []
        self.documents = []
        self.vocabulary = set()

        # Statistics
        self.stats = {
            'total_tokens': 0,
            'unique_tokens': 0,
            'total_documents': 0,
            'avg_doc_length': 0,
            'zipf_data': [],
            'heaps_data': [],
            'ngrams': {}
        }

    def load_and_process_data(self, data_path: str):
        """
        Load và process documents từ JSON

        Args:
            data_path: Path đến file JSON data
        """
        print(f"\n{'='*70}")
        print("LOADING AND PROCESSING DATA")
        print(f"{'='*70}")

        # Load data
        if USE_PROJECT_MODULES:
            handler = DataHandler(data_path)
            raw_docs = handler.load_data()
            contents = [doc['content'] for doc in raw_docs]
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_docs = json.load(f)
                contents = [doc['content'] for doc in raw_docs]
            print(f"✓ Loaded {len(raw_docs)} documents from {data_path}")

        # Process all documents
        print(f"\nProcessing {len(contents)} documents...")
        self.documents = []
        self.all_tokens = []

        for i, content in enumerate(contents, 1):
            tokens = self.processor.process(content)
            self.documents.append(tokens)
            self.all_tokens.extend(tokens)

            if i % 100 == 0:
                print(f"  Processed {i}/{len(contents)} documents...")

        self.vocabulary = set(self.all_tokens)

        # Update basic statistics
        self.stats['total_tokens'] = len(self.all_tokens)
        self.stats['unique_tokens'] = len(self.vocabulary)
        self.stats['total_documents'] = len(self.documents)
        self.stats['avg_doc_length'] = len(self.all_tokens) / len(self.documents) if self.documents else 0

        print(f"\n✓ Processing completed!")
        print(f"  - Total documents: {self.stats['total_documents']}")
        print(f"  - Total tokens: {self.stats['total_tokens']:,}")
        print(f"  - Unique tokens: {self.stats['unique_tokens']:,}")
        print(f"  - Average doc length: {self.stats['avg_doc_length']:.2f} tokens")

    def calculate_zipf_law(self, top_n: int = 50):
        """
        Zipf's Law: frequency of r-th most common word is inversely proportional to r
        f(r) ≈ k / r  or  r × f(r) ≈ k

        Args:
            top_n: Number of top words to analyze
        """
        print(f"\n{'='*70}")
        print("ZIPF'S LAW ANALYSIS")
        print(f"{'='*70}")

        # Count word frequencies
        word_freq = Counter(self.all_tokens)
        most_common = word_freq.most_common(top_n)

        # Calculate Zipf's law
        zipf_data = []
        k_values = []

        print(f"\n{'Rank':<8} {'Word':<20} {'Frequency':<12} {'r×f':<10}")
        print("-" * 70)

        for rank, (word, freq) in enumerate(most_common, 1):
            k = rank * freq
            zipf_data.append({
                'rank': rank,
                'word': word,
                'frequency': freq,
                'k': k
            })
            k_values.append(k)

            if rank <= 20:  # Print top 20
                print(f"{rank:<8} {word:<20} {freq:<12} {k:<10}")

        # Calculate average k
        avg_k = np.mean(k_values)
        print(f"\nAverage k value: {avg_k:.2f}")
        print(f"Zipf's Law: r × f ≈ {avg_k:.2f}")

        self.stats['zipf_data'] = zipf_data
        self.stats['zipf_k'] = avg_k

        return zipf_data

    def calculate_heaps_law(self):
        """
        Heap's Law: Vocabulary growth as corpus size increases
        V = K × N^β
        where:
        - V: vocabulary size
        - N: total number of tokens
        - K: constant (10-100)
        - β: constant (0.4-0.6)

        We calculate this by gradually increasing corpus size
        """
        print(f"\n{'='*70}")
        print("HEAP'S LAW ANALYSIS")
        print(f"{'='*70}")

        # Sample points for vocabulary growth
        n_points = 20
        step = max(1, len(self.all_tokens) // n_points)

        heaps_data = []
        vocab = set()

        print(f"\n{'N (tokens)':<15} {'V (vocabulary)':<18} {'V/N^0.5':<15}")
        print("-" * 70)

        for i in range(step, len(self.all_tokens) + 1, step):
            tokens_subset = self.all_tokens[:i]
            vocab.update(tokens_subset)

            n = i
            v = len(vocab)
            ratio = v / (n ** 0.5)

            heaps_data.append({
                'n': n,
                'v': v,
                'ratio': ratio
            })

            if len(heaps_data) % 5 == 0:  # Print every 5th point
                print(f"{n:<15,} {v:<18,} {ratio:<15.2f}")

        # Fit Heap's law: V = K × N^β
        # Using log: log(V) = log(K) + β × log(N)
        if len(heaps_data) > 1:
            N_values = np.array([d['n'] for d in heaps_data])
            V_values = np.array([d['v'] for d in heaps_data])

            # Linear regression on log scale
            log_N = np.log(N_values)
            log_V = np.log(V_values)

            coeffs = np.polyfit(log_N, log_V, 1)
            beta = coeffs[0]
            log_K = coeffs[1]
            K = np.exp(log_K)

            print(f"\nHeap's Law fitted: V = {K:.2f} × N^{beta:.3f}")
            print(f"  K = {K:.2f} (typical: 10-100)")
            print(f"  β = {beta:.3f} (typical: 0.4-0.6)")

            self.stats['heaps_data'] = heaps_data
            self.stats['heaps_K'] = K
            self.stats['heaps_beta'] = beta

        return heaps_data

    def calculate_ngrams(self, n: int = 2, top_k: int = 20):
        """
        Calculate most frequent n-grams

        Args:
            n: N-gram size (2=bigram, 3=trigram)
            top_k: Number of top n-grams to show
        """
        print(f"\n{'='*70}")
        print(f"{n}-GRAM ANALYSIS")
        print(f"{'='*70}")

        # Generate n-grams
        ngrams = []
        for doc_tokens in self.documents:
            for i in range(len(doc_tokens) - n + 1):
                ngram = tuple(doc_tokens[i:i+n])
                ngrams.append(ngram)

        # Count n-grams
        ngram_freq = Counter(ngrams)
        most_common = ngram_freq.most_common(top_k)

        print(f"\nTop {top_k} most frequent {n}-grams:")
        print(f"\n{'Rank':<8} {f'{n}-gram':<40} {'Frequency':<12}")
        print("-" * 70)

        for rank, (ngram, freq) in enumerate(most_common, 1):
            ngram_str = ' '.join(ngram)
            print(f"{rank:<8} {ngram_str:<40} {freq:<12}")

        self.stats['ngrams'][n] = most_common

        return most_common

    def save_tokens_to_file(self, output_path: str = "tokens_output.txt"):
        """Save all tokens to a text file"""
        print(f"\n{'='*70}")
        print("SAVING TOKENS")
        print(f"{'='*70}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Total tokens: {len(self.all_tokens)}\n")
            f.write(f"Unique tokens: {len(self.vocabulary)}\n")
            f.write(f"{'='*70}\n\n")

            for token in self.all_tokens:
                f.write(token + '\n')

        print(f"✓ Saved {len(self.all_tokens):,} tokens to '{output_path}'")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")

    def save_vocabulary_to_file(self, output_path: str = "vocabulary_output.txt"):
        """Save unique vocabulary to a text file"""
        vocab_sorted = sorted(self.vocabulary)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Total unique tokens: {len(vocab_sorted)}\n")
            f.write(f"{'='*70}\n\n")

            for token in vocab_sorted:
                f.write(token + '\n')

        print(f"✓ Saved {len(vocab_sorted):,} unique tokens to '{output_path}'")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")

    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")

        print(f"\n📊 BASIC STATISTICS")
        print(f"  Total documents: {self.stats['total_documents']:,}")
        print(f"  Total tokens: {self.stats['total_tokens']:,}")
        print(f"  Unique tokens: {self.stats['unique_tokens']:,}")
        print(f"  Average document length: {self.stats['avg_doc_length']:.2f} tokens")
        print(f"  Vocabulary richness: {self.stats['unique_tokens'] / self.stats['total_tokens']:.4f}")

        print(f"\n📈 ZIPF'S LAW")
        if 'zipf_k' in self.stats:
            print(f"  r × f ≈ {self.stats['zipf_k']:.2f}")
            print(f"  (The product of rank and frequency is approximately constant)")

        print(f"\n📈 HEAP'S LAW")
        if 'heaps_K' in self.stats and 'heaps_beta' in self.stats:
            print(f"  V = {self.stats['heaps_K']:.2f} × N^{self.stats['heaps_beta']:.3f}")
            print(f"  K = {self.stats['heaps_K']:.2f} (typical: 10-100)")
            print(f"  β = {self.stats['heaps_beta']:.3f} (typical: 0.4-0.6)")

        print(f"\n📝 N-GRAMS")
        for n, ngrams in self.stats.get('ngrams', {}).items():
            if ngrams:
                top_ngram = ' '.join(ngrams[0][0])
                print(f"  Most frequent {n}-gram: '{top_ngram}' ({ngrams[0][1]} occurrences)")


def main():
    """Main function"""
    print("="*70)
    print("TEXT STATISTICS ANALYZER")
    print("Based on Slide 4: Processing Text")
    print("="*70)

    # Configuration
    DATA_PATH = "data_content.json"

    # Initialize analyzer
    analyzer = TextStatisticsAnalyzer(use_project_processor=True)

    # Load and process data
    analyzer.load_and_process_data(DATA_PATH)

    # Calculate Zipf's Law
    analyzer.calculate_zipf_law(top_n=50)

    # Calculate Heap's Law
    analyzer.calculate_heaps_law()

    # Calculate N-grams
    analyzer.calculate_ngrams(n=2, top_k=20)  # Bigrams
    analyzer.calculate_ngrams(n=3, top_k=20)  # Trigrams

    # Save tokens
    analyzer.save_tokens_to_file("tokens_output.txt")
    analyzer.save_vocabulary_to_file("vocabulary_output.txt")

    # Print summary
    analyzer.print_summary()

    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
