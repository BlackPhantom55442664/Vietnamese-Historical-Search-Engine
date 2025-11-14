
import json
import os
from typing import List, Dict, Tuple, Set
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


class MetricsCalculator:
    """Calculate classification metrics for IR systems"""

    @staticmethod
    def calculate_metrics(retrieved: Set[int],
                         relevant: Set[int],
                         total_docs: int) -> Dict[str, float]:
        """
        Calculate Accuracy, Precision, Recall, F1-Score

        Args:
            retrieved: Set of retrieved document IDs
            relevant: Set of relevant document IDs
            total_docs: Total number of documents in collection

        Returns:
            Dict with metrics: accuracy, precision, recall, f1_score
        """
        # True Positives: Retrieved AND Relevant
        tp = len(retrieved & relevant)

        # False Positives: Retrieved but NOT Relevant
        fp = len(retrieved - relevant)

        # False Negatives: NOT Retrieved but Relevant
        fn = len(relevant - retrieved)

        # True Negatives: NOT Retrieved AND NOT Relevant
        tn = total_docs - tp - fp - fn

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / total_docs if total_docs > 0 else 0.0

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) \
                   if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }


class TestSystemMetricsEnhanced:
    """
    Enhanced test system with userQuery.json support
    """

    def __init__(self,
                 test_file: str = "dataset/test.json",
                 user_query_file: str = "dataset/userQuery.json",
                 data_file: str = "dataset/data_content.json",
                 k: int = 15,
                 query_mode: str = 'primary'):
        """
        Args:
            test_file: Path to test.json (ground truth)
            user_query_file: Path to userQuery.json (NEW!)
            data_file: Path to data_content.json
            k: Top-k results to retrieve
            query_mode: 'primary' (Q1 only) or 'combined' (Q1+Q2+...)
        """
        self.test_file = test_file
        self.user_query_file = user_query_file
        self.data_file = data_file
        self.k = k
        self.query_mode = query_mode

        # Data storage
        self.ground_truth = {}
        self.user_queries = {}
        self.data = {}
        self.total_docs = 0

        # Results
        self.results = {
            'per_query_metrics': {},
            'mean_accuracy': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1_score': 0.0,
            'detailed_results': []
        }

        # Components
        self.search_engine = None
        self.query_handler = None

        print("="*70)
        print("TMGSEG SYSTEM - CLASSIFICATION METRICS (ENHANCED)")
        print("="*70)
        print(f"Metrics: Accuracy, Precision, Recall, F1-Score")
        print(f"Top-K: {self.k}")
        print(f"Query mode: {self.query_mode}")

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

        self.total_docs = len(data_list)
        for i, doc in enumerate(data_list):
            self.data[i] = doc

        print(f"  ✓ Loaded {self.total_docs} documents")

    def initialize_search_engine(self):
        """Initialize search engine"""
        print("\n[2/6] Initializing search engine...")

        if USE_PROJECT_MODULES:
            config = Config()
            config.TOP_K_RESULTS = self.k

            self.search_engine = SearchEngine(config)
            self.search_engine.build_index()

            print(f"  ✓ Search engine initialized (Top-{self.k})")
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
            return self.query_handler.get_primary_query(query_id)
        else:
            return self.query_handler.get_combined_query(query_id, separator=' ')

    def run_tests(self):
        """Run classification metrics tests"""
        print(f"\n[3/6] Running Classification Metrics tests...")

        test_queries = list(self.ground_truth.keys())
        total_queries = len(test_queries)

        print(f"  Testing {total_queries} queries...")
        print()

        # Progress bar
        with tqdm(total=total_queries, desc="  Processing queries",
                  unit="query", ncols=100) as pbar:

            for query_id in test_queries:
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

                # Calculate metrics
                metrics = self._calculate_metrics_for_query(
                    query_id,
                    query_text,
                    retrieved_results,
                    gt_docs
                )

                self.results['per_query_metrics'][query_id] = metrics

                # Update progress
                pbar.set_postfix({
                    'P': f'{metrics["precision"]:.3f}',
                    'R': f'{metrics["recall"]:.3f}',
                    'F1': f'{metrics["f1_score"]:.3f}'
                })
                pbar.update(1)

        print(f"\n  ✓ Testing completed for {len(self.results['per_query_metrics'])} queries")

    def _calculate_metrics_for_query(self,
                                    query_id: str,
                                    query_text: str,
                                    retrieved_results: List[Dict],
                                    ground_truth_docs: Dict) -> Dict:
        """Calculate metrics for a single query"""

        # Get retrieved doc IDs
        retrieved_ids = set()
        for result in retrieved_results[:self.k]:
            doc_id = result.get('doc_id')
            if doc_id is not None:
                retrieved_ids.add(doc_id)

        # Get relevant doc IDs (relevance > 0)
        relevant_ids = set()
        for gt_doc_id_str, gt_relevance in ground_truth_docs.items():
            if gt_relevance > 0:
                doc_num = int(gt_doc_id_str.split('_')[1])
                relevant_ids.add(doc_num)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(
            retrieved=retrieved_ids,
            relevant=relevant_ids,
            total_docs=self.total_docs
        )

        # Store detailed results
        self.results['detailed_results'].append({
            'query_id': query_id,
            'query_text': query_text,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'num_retrieved': len(retrieved_ids),
            'num_relevant': len(relevant_ids)
        })

        return metrics

    def calculate_statistics(self):
        """Calculate overall statistics"""
        print("\n[4/6] Calculating statistics...")

        # Collect all metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for metrics in self.results['per_query_metrics'].values():
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])

        # Calculate means
        if accuracies:
            self.results['mean_accuracy'] = np.mean(accuracies)
            self.results['mean_precision'] = np.mean(precisions)
            self.results['mean_recall'] = np.mean(recalls)
            self.results['mean_f1_score'] = np.mean(f1_scores)

            # Additional stats
            self.results['std_accuracy'] = np.std(accuracies)
            self.results['std_precision'] = np.std(precisions)
            self.results['std_recall'] = np.std(recalls)
            self.results['std_f1_score'] = np.std(f1_scores)

            self.results['median_accuracy'] = np.median(accuracies)
            self.results['median_precision'] = np.median(precisions)
            self.results['median_recall'] = np.median(recalls)
            self.results['median_f1_score'] = np.median(f1_scores)

        print(f"  ✓ Statistics calculated")

    def print_summary(self):
        """Print summary results in table format"""
        print("\n[5/6] Evaluation Summary")
        print("="*70)
        print(f"\nCLASSIFICATION METRICS RESULTS (Top-{self.k})")
        print("-"*70)

        # Table format
        print(f"\n{'Statistic':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)
        print(f"{'Mean':<15} {self.results['mean_accuracy']:<12.4f} "
              f"{self.results['mean_precision']:<12.4f} "
              f"{self.results['mean_recall']:<12.4f} "
              f"{self.results['mean_f1_score']:<12.4f}")
        print(f"{'Median':<15} {self.results['median_accuracy']:<12.4f} "
              f"{self.results['median_precision']:<12.4f} "
              f"{self.results['median_recall']:<12.4f} "
              f"{self.results['median_f1_score']:<12.4f}")
        print(f"{'Std Dev':<15} {self.results['std_accuracy']:<12.4f} "
              f"{self.results['std_precision']:<12.4f} "
              f"{self.results['std_recall']:<12.4f} "
              f"{self.results['std_f1_score']:<12.4f}")

        print(f"\nTotal queries tested: {len(self.results['per_query_metrics'])}")

        # Distribution analysis
        print(f"\nF1-Score Distribution:")
        f1_scores = [m['f1_score'] for m in self.results['per_query_metrics'].values()]
        excellent = len([s for s in f1_scores if s >= 0.8])
        good = len([s for s in f1_scores if 0.6 <= s < 0.8])
        fair = len([s for s in f1_scores if 0.4 <= s < 0.6])
        poor = len([s for s in f1_scores if s < 0.4])

        total = len(f1_scores)
        print(f"  Excellent (≥0.8):  {excellent:4d} queries ({excellent/total*100:.1f}%)")
        print(f"  Good (0.6-0.8):    {good:4d} queries ({good/total*100:.1f}%)")
        print(f"  Fair (0.4-0.6):    {fair:4d} queries ({fair/total*100:.1f}%)")
        print(f"  Poor (<0.4):       {poor:4d} queries ({poor/total*100:.1f}%)")

    def print_detailed_results(self, top_n: int = 10):
        """Print detailed results"""
        print("\n" + "="*70)
        print(f"DETAILED RESULTS (Top {top_n} and Bottom {top_n} by F1-Score)")
        print("="*70)

        # Sort by F1-Score
        sorted_results = sorted(
            self.results['detailed_results'],
            key=lambda x: x['f1_score'],
            reverse=True
        )

        # Top queries
        print(f"\n🌟 TOP {top_n} QUERIES (Best F1-Score)")
        print("-"*70)
        print(f"{'#':<4} {'Query ID':<15} {'P':<8} {'R':<8} {'F1':<8} {'TP':<5}")
        print("-"*70)

        for i, result in enumerate(sorted_results[:top_n], 1):
            qid = result['query_id']
            print(f"{i:<4} {qid:<15} {result['precision']:.4f}   "
                  f"{result['recall']:.4f}   {result['f1_score']:.4f}   "
                  f"{result['tp']:<5}")

        # Bottom queries
        print(f"\n📉 BOTTOM {top_n} QUERIES (Worst F1-Score)")
        print("-"*70)
        print(f"{'#':<4} {'Query ID':<15} {'P':<8} {'R':<8} {'F1':<8} {'TP':<5}")
        print("-"*70)

        for i, result in enumerate(sorted_results[-top_n:], 1):
            qid = result['query_id']
            print(f"{i:<4} {qid:<15} {result['precision']:.4f}   "
                  f"{result['recall']:.4f}   {result['f1_score']:.4f}   "
                  f"{result['tp']:<5}")

    def save_results(self, output_file: str = "metrics_results_enhanced.json"):
        """Save results to JSON"""
        print(f"\n[6/6] Saving results...")

        results_to_save = {
            'summary': {
                'mean_accuracy': float(self.results['mean_accuracy']),
                'mean_precision': float(self.results['mean_precision']),
                'mean_recall': float(self.results['mean_recall']),
                'mean_f1_score': float(self.results['mean_f1_score']),
                'median_accuracy': float(self.results['median_accuracy']),
                'median_precision': float(self.results['median_precision']),
                'median_recall': float(self.results['median_recall']),
                'median_f1_score': float(self.results['median_f1_score']),
                'total_queries': len(self.results['per_query_metrics']),
                'k': self.k,
                'query_mode': self.query_mode
            },
            'per_query': {
                qid: {
                    'accuracy': float(m['accuracy']),
                    'precision': float(m['precision']),
                    'recall': float(m['recall']),
                    'f1_score': float(m['f1_score'])
                }
                for qid, m in self.results['per_query_metrics'].items()
            },
            'detailed': [
                {
                    'query_id': r['query_id'],
                    'query_text': r['query_text'],
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1_score': float(r['f1_score']),
                    'tp': r['tp'],
                    'fp': r['fp'],
                    'fn': r['fn'],
                    'num_retrieved': r['num_retrieved'],
                    'num_relevant': r['num_relevant']
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
            self.save_results("metrics_results_enhanced.json")

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
    test_file = "dataset/test.json"
    user_query_file = "dataset/userQuery.json"  # NEW!
    data_file = "dataset/data_content.json"

    k = 15  # Top-15 results
    query_mode = 'primary'  # 'primary' or 'combined'

    # Run evaluation
    tester = TestSystemMetricsEnhanced(
        test_file=test_file,
        user_query_file=user_query_file,
        data_file=data_file,
        k=k,
        query_mode=query_mode
    )

    tester.run_evaluation()


if __name__ == "__main__":
    main()