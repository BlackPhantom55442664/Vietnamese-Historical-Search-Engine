
import json
import os
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import từ project chính
try:
    from SearchEngine import SearchEngine
    from Config import Config
    from TextProcessor import TextProcessor
    from QueryHandler import QueryHandler
    USE_PROJECT_MODULES = True
    print("✓ Imported modules from main project")
except ImportError as e:
    print(f"⚠ Cannot import from main project: {e}")
    USE_PROJECT_MODULES = False


class NDCGCalculator:
    """Calculate NDCG metrics"""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int = 15) -> float:
        """
        Calculate DCG@k
        DCG@k = Σ(i=1 to k) (rel_i / log₂(i + 1))
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg

    @staticmethod
    def idcg_at_k(relevances: List[float], k: int = 15) -> float:
        """
        Calculate IDCG@k (Ideal DCG)
        """
        sorted_rels = sorted(relevances, reverse=True)
        return NDCGCalculator.dcg_at_k(sorted_rels, k)

    @staticmethod
    def ndcg_at_k(retrieved_relevances: List[float],
                  all_relevances: List[float],
                  k: int = 15) -> float:
        """
        Calculate NDCG@k = DCG@k / IDCG@k
        """
        dcg = NDCGCalculator.dcg_at_k(retrieved_relevances, k)
        idcg = NDCGCalculator.idcg_at_k(all_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg


class TestSystemEnhanced:
    """
    Enhanced test system with userQuery.json support
    """

    def __init__(self,
                 test_file: str = "dataset/test.json",
                 user_query_file: str = "dataset/userQuery.json",
                 data_file: str = "dataset/data_content.json",
                 k: int = 15,
                 stage2_top_k: int = 35,
                 stage3_top_k: int = 15,
                 query_mode: str = 'primary'):
        """
        Args:
            test_file: Path to test.json (ground truth)
            user_query_file: Path to userQuery.json (NEW!)
            data_file: Path to data_content.json
            k: NDCG@k value
            stage2_top_k: Number of docs after Stage 2
            stage3_top_k: Number of docs after Stage 3
            query_mode: 'primary' (Q1 only) or 'combined' (Q1+Q2+...)
        """
        self.test_file = test_file
        self.user_query_file = user_query_file
        self.data_file = data_file
        self.k = k
        self.stage2_top_k = stage2_top_k
        self.stage3_top_k = stage3_top_k
        self.query_mode = query_mode

        # Data storage
        self.ground_truth = {}  # test.json
        self.user_queries = {}  # userQuery.json
        self.data = {}  # data_content.json

        # Results
        self.results = {
            'per_query_ndcg': {},
            'mean_ndcg': 0.0,
            'median_ndcg': 0.0,
            'min_ndcg': 0.0,
            'max_ndcg': 0.0,
            'detailed_results': []
        }

        # Components
        self.search_engine = None
        self.query_handler = None

        print("="*70)
        print(f"TMGSEG SYSTEM - NDCG@{self.k} EVALUATION (ENHANCED)")
        print("="*70)
        print(f"Configuration:")
        print(f"  - NDCG@{self.k}")
        print(f"  - Stage 2: Top {self.stage2_top_k} docs")
        print(f"  - Stage 3: Top {self.stage3_top_k} docs")
        print(f"  - Query mode: {self.query_mode}")

    def load_files(self):
        """Load test.json, userQuery.json, data_content.json"""
        print("\n[1/6] Loading files...")

        # Load ground truth
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"test.json not found: {self.test_file}")

        with open(self.test_file, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)

        print(f"  ✓ Loaded {len(self.ground_truth)} test queries from test.json")

        # Load user queries (NEW!)
        if not os.path.exists(self.user_query_file):
            raise FileNotFoundError(f"userQuery.json not found: {self.user_query_file}")

        self.query_handler = QueryHandler(self.user_query_file)
        self.query_handler.load_queries()

        print(f"  ✓ Loaded {self.query_handler.get_query_count()} user queries")

        # Load data
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"data_content.json not found: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        for i, doc in enumerate(data_list):
            self.data[i] = doc

        print(f"  ✓ Loaded {len(self.data)} documents")

    def initialize_search_engine(self):
        """Initialize search engine"""
        print("\n[2/6] Initializing search engine...")

        if USE_PROJECT_MODULES:
            config = Config()
            config.STAGE1_TOP_K = 100
            config.STAGE2_TOP_K = self.stage2_top_k
            config.TOP_K_RESULTS = self.stage3_top_k

            self.search_engine = SearchEngine(config)
            self.search_engine.build_index()

            print(f"  ✓ Search engine initialized")
            print(f"    - Stage 1 (BM25): Top {config.STAGE1_TOP_K}")
            print(f"    - Stage 2 (Embedding): Top {config.STAGE2_TOP_K}")
            print(f"    - Stage 3 (Reranking): Top {config.TOP_K_RESULTS}")
        else:
            print("  ⚠ Could not initialize search engine")

    def get_query_text(self, query_id: str) -> str:
        """
        Get query text from userQuery.json

        Args:
            query_id: Query ID (e.g., "query_201")

        Returns:
            Query text (Q1 only or combined Q1+Q2+...)
        """
        if self.query_mode == 'primary':
            # Use Q1 only
            return self.query_handler.get_primary_query(query_id)
        else:
            # Combine Q1 + Q2 + ...
            return self.query_handler.get_combined_query(query_id, separator=' ')

    def run_tests(self):
        """Run NDCG tests"""
        print(f"\n[3/6] Running NDCG@{self.k} tests...")

        # Get test query IDs (from test.json)
        test_query_ids = list(self.ground_truth.keys())
        total_queries = len(test_query_ids)

        print(f"  Testing {total_queries} queries...")
        print()

        # Progress bar
        with tqdm(total=total_queries, desc="  Processing queries",
                  unit="query", ncols=100) as pbar:

            for query_id in test_query_ids:
                # Get query text from userQuery.json
                query_text = self.get_query_text(query_id)

                if not query_text:
                    pbar.write(f"  ⚠ Query {query_id} not found in userQuery.json")
                    pbar.update(1)
                    continue

                # Get ground truth
                gt_docs = self.ground_truth[query_id]

                # Search
                try:
                    retrieved_results = self.search_engine.search(
                        query_text,
                        top_k=self.k
                    )
                except Exception as e:
                    pbar.write(f"  ✗ Error searching {query_id}: {e}")
                    pbar.update(1)
                    continue

                # Calculate NDCG
                ndcg = self._calculate_ndcg_for_query(
                    query_id,
                    query_text,
                    retrieved_results,
                    gt_docs
                )

                self.results['per_query_ndcg'][query_id] = ndcg

                # Update progress
                pbar.set_postfix({'Current NDCG': f'{ndcg:.4f}'})
                pbar.update(1)

        print(f"\n  ✓ Testing completed for {len(self.results['per_query_ndcg'])} queries")

    def _calculate_ndcg_for_query(self,
                                  query_id: str,
                                  query_text: str,
                                  retrieved_results: List[Dict],
                                  ground_truth_docs: Dict) -> float:
        """Calculate NDCG for a single query"""

        # Map retrieved results to relevances
        retrieved_relevances = []

        for result in retrieved_results[:self.k]:
            # Get doc_id from result
            doc_id = result.get('doc_id')

            # Look up in ground truth
            found = False
            for gt_doc_id_str, gt_relevance in ground_truth_docs.items():
                # Extract number from "doc_X"
                gt_num = int(gt_doc_id_str.split('_')[1])

                if doc_id == gt_num:
                    retrieved_relevances.append(gt_relevance)
                    found = True
                    break

            if not found:
                retrieved_relevances.append(0.0)

        # All relevances for IDCG
        all_relevances = list(ground_truth_docs.values())

        # Calculate NDCG
        ndcg = NDCGCalculator.ndcg_at_k(
            retrieved_relevances,
            all_relevances,
            k=self.k
        )

        # Store detailed results
        self.results['detailed_results'].append({
            'query_id': query_id,
            'query_text': query_text,
            'ndcg': ndcg,
            'num_relevant': len([r for r in all_relevances if r > 0]),
            'num_retrieved': len([r for r in retrieved_relevances if r > 0])
        })

        return ndcg

    def calculate_statistics(self):
        """Calculate overall statistics"""
        print("\n[4/6] Calculating statistics...")

        ndcg_scores = list(self.results['per_query_ndcg'].values())

        if ndcg_scores:
            self.results['mean_ndcg'] = np.mean(ndcg_scores)
            self.results['median_ndcg'] = np.median(ndcg_scores)
            self.results['min_ndcg'] = np.min(ndcg_scores)
            self.results['max_ndcg'] = np.max(ndcg_scores)
            self.results['std_ndcg'] = np.std(ndcg_scores)

        print(f"  ✓ Statistics calculated")

    def print_summary(self):
        """Print summary results"""
        print("\n[5/6] Evaluation Summary")
        print("="*70)
        print(f"\nNDCG@{self.k} RESULTS")
        print("-"*70)
        print(f"Mean NDCG:   {self.results['mean_ndcg']:.4f}")
        print(f"Median NDCG: {self.results['median_ndcg']:.4f}")
        print(f"Std NDCG:    {self.results.get('std_ndcg', 0):.4f}")
        print(f"Min NDCG:    {self.results['min_ndcg']:.4f}")
        print(f"Max NDCG:    {self.results['max_ndcg']:.4f}")
        print(f"\nTotal queries tested: {len(self.results['per_query_ndcg'])}")

        # Distribution
        ndcg_scores = list(self.results['per_query_ndcg'].values())
        excellent = len([s for s in ndcg_scores if s >= 0.9])
        good = len([s for s in ndcg_scores if 0.7 <= s < 0.9])
        fair = len([s for s in ndcg_scores if 0.5 <= s < 0.7])
        poor = len([s for s in ndcg_scores if s < 0.5])

        total = len(ndcg_scores)
        print(f"\nNDCG Distribution:")
        print(f"  Excellent (≥0.9):  {excellent:4d} queries ({excellent/total*100:.1f}%)")
        print(f"  Good (0.7-0.9):    {good:4d} queries ({good/total*100:.1f}%)")
        print(f"  Fair (0.5-0.7):    {fair:4d} queries ({fair/total*100:.1f}%)")
        print(f"  Poor (<0.5):       {poor:4d} queries ({poor/total*100:.1f}%)")

    def print_detailed_results(self, top_n: int = 10):
        """Print detailed results"""
        print("\n" + "="*70)
        print(f"DETAILED RESULTS (Top {top_n} and Bottom {top_n})")
        print("="*70)

        # Sort by NDCG
        sorted_results = sorted(
            self.results['detailed_results'],
            key=lambda x: x['ndcg'],
            reverse=True
        )

        # Top queries
        print(f"\n🌟 TOP {top_n} QUERIES (Best NDCG)")
        print("-"*70)
        print(f"{'#':<4} {'Query ID':<15} {'NDCG':<10} {'Retrieved':<12} {'Relevant':<10}")
        print("-"*70)

        for i, result in enumerate(sorted_results[:top_n], 1):
            qid = result['query_id']
            print(f"{i:<4} {qid:<15} {result['ndcg']:.4f}     "
                  f"{result['num_retrieved']:<12} {result['num_relevant']:<10}")

        # Bottom queries
        print(f"\n📉 BOTTOM {top_n} QUERIES (Worst NDCG)")
        print("-"*70)
        print(f"{'#':<4} {'Query ID':<15} {'NDCG':<10} {'Retrieved':<12} {'Relevant':<10}")
        print("-"*70)

        for i, result in enumerate(sorted_results[-top_n:], 1):
            qid = result['query_id']
            print(f"{i:<4} {qid:<15} {result['ndcg']:.4f}     "
                  f"{result['num_retrieved']:<12} {result['num_relevant']:<10}")

    def save_results(self, output_file: str = "ndcg_results_enhanced.json"):
        """Save results to JSON"""
        print(f"\n[6/6] Saving results...")

        results_to_save = {
            'summary': {
                'mean_ndcg': float(self.results['mean_ndcg']),
                'median_ndcg': float(self.results['median_ndcg']),
                'min_ndcg': float(self.results['min_ndcg']),
                'max_ndcg': float(self.results['max_ndcg']),
                'std_ndcg': float(self.results.get('std_ndcg', 0)),
                'total_queries': len(self.results['per_query_ndcg']),
                'k': self.k,
                'query_mode': self.query_mode
            },
            'per_query': self.results['per_query_ndcg'],
            'detailed': [
                {
                    'query_id': r['query_id'],
                    'query_text': r['query_text'],
                    'ndcg': float(r['ndcg']),
                    'num_relevant': r['num_relevant'],
                    'num_retrieved': r['num_retrieved']
                }
                for r in self.results['detailed_results']
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Results saved to '{output_file}'")

    def run_evaluation(self):
        """Run full evaluation pipeline"""
        try:
            self.load_files()
            self.initialize_search_engine()
            self.run_tests()
            self.calculate_statistics()
            self.print_summary()
            self.print_detailed_results(top_n=10)
            self.save_results("ndcg_results_enhanced.json")

            print("\n" + "="*70)
            print("✓ EVALUATION COMPLETED")
            print("="*70)

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    # Configuration
    test_file = "dataset/test.json"  # Same as before
    user_query_file = "dataset/userQuery.json"  # NEW!
    data_file = "dataset/data_content.json"

    k = 15  # NDCG@15
    stage2_top_k = 35
    stage3_top_k = 15

    query_mode = 'primary'  # 'primary' or 'combined'

    # Run evaluation
    tester = TestSystemEnhanced(
        test_file=test_file,
        user_query_file=user_query_file,
        data_file=data_file,
        k=k,
        stage2_top_k=stage2_top_k,
        stage3_top_k=stage3_top_k,
        query_mode=query_mode
    )

    tester.run_evaluation()


if __name__ == "__main__":
    main()
