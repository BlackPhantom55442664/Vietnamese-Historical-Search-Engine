"""
FinalTest.py - Final Testing with User Queries

Test search engine với user queries từ userQuery.json
Bao gồm:
- Multi-question queries (Q1, Q2)
- Passage retrieval
- Result visualization
"""

import os
import json
from typing import List, Dict
from datetime import datetime

from Config import Config
from SearchEngine import SearchEngine
from QueryHandler import QueryHandler


class FinalTest:
    """
    Final test suite cho search engine với user queries
    """

    def __init__(self, config: Config = None):
        """
        Initialize FinalTest

        Args:
            config: Config object (None = use default)
        """
        self.config = config if config else Config()

        print("="*70)
        print("FINAL TEST - USER QUERY TESTING")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize components
        self.query_handler = None
        self.search_engine = None

    def initialize(self):
        """Initialize QueryHandler và SearchEngine"""
        print("\n[1/3] Initializing QueryHandler...")
        self.query_handler = QueryHandler(self.config.USER_QUERY_PATH)
        self.query_handler.load_queries()

        print("\n[2/3] Initializing SearchEngine...")
        self.search_engine = SearchEngine(self.config)
        self.search_engine.build_index()

        print("\n[3/3] Initialization complete!")

    def test_single_query(self, query_id: str, verbose: bool = True):
        """
        Test một query cụ thể

        Args:
            query_id: Query ID (e.g., "query_201")
            verbose: Print detailed results
        """
        if verbose:
            print("\n" + "="*70)
            print(f"TESTING: {query_id}")
            print("="*70)

        # Get query
        query_data = self.query_handler.get_query(query_id)
        if not query_data:
            print(f"✗ Query {query_id} not found!")
            return None

        # Get sub-queries
        sub_queries = self.query_handler.get_all_sub_queries(query_id)

        if verbose:
            print(f"\nQuery has {len(sub_queries)} sub-questions:")
            for key, text in sub_queries:
                print(f"  {key}: {text}")

        # Decide query mode
        if self.config.QUERY_COMBINATION_MODE == 'combined':
            # Use combined query
            query_text = self.query_handler.get_combined_query(query_id)
            if verbose:
                print(f"\nUsing combined query: {query_text[:100]}...")
        else:
            # Use primary query (Q1)
            query_text = self.query_handler.get_primary_query(query_id)
            if verbose:
                print(f"\nUsing primary query (Q1): {query_text}")

        # Search
        if verbose:
            print(f"\nSearching...")

        results = self.search_engine.search(
            query=query_text,
            top_k=self.config.TOP_K_RESULTS
        )

        if verbose:
            self._print_results(query_id, query_text, results)

        return results

    def _print_results(self, query_id: str, query_text: str, results: List[Dict]):
        """Print search results"""
        print(f"\n{'='*70}")
        print(f"RESULTS FOR: {query_id}")
        print(f"Query: {query_text}")
        print(f"{'='*70}")

        if not results:
            print("\n✗ No results found!")
            return

        print(f"\nFound {len(results)} results:")

        for result in results:
            rank = result['rank']
            score = result['score']
            file_name = result['file_name']

            print(f"\n{'─'*70}")
            print(f"[Rank {rank}] Score: {score:.4f}")
            print(f"📄 File: {file_name}")

            # Show relevant passages if available
            if 'relevant_passages' in result and result['relevant_passages']:
                print(f"\n🔍 RELEVANT PASSAGES:")
                for i, passage in enumerate(result['relevant_passages'], 1):
                    print(f"\n  [Passage {i}] Relevance: {passage['score']:.4f}")
                    print(f"  {passage['text']}")
            else:
                # Fallback to content preview
                print(f"\n💡 Preview:")
                print(f"  {result['content']}")

    def test_all_queries(self, limit: int = None):
        """
        Test tất cả queries

        Args:
            limit: Limit số queries test (None = test all)
        """
        query_ids = self.query_handler.get_all_query_ids()

        if limit:
            query_ids = query_ids[:limit]

        print(f"\n{'='*70}")
        print(f"TESTING ALL QUERIES (Total: {len(query_ids)})")
        print(f"{'='*70}")

        results_summary = []

        for idx, query_id in enumerate(query_ids, 1):
            print(f"\n\n[{idx}/{len(query_ids)}] Testing {query_id}...")

            try:
                results = self.test_single_query(query_id, verbose=False)

                # Summary
                if results:
                    top_score = results[0]['score']
                    top_file = results[0]['file_name']
                    num_results = len(results)

                    results_summary.append({
                        'query_id': query_id,
                        'num_results': num_results,
                        'top_score': top_score,
                        'top_file': top_file
                    })

                    print(f"  ✓ Found {num_results} results | Top: {top_file} ({top_score:.4f})")
                else:
                    print(f"  ✗ No results")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        # Print summary
        self._print_test_summary(results_summary)

    def _print_test_summary(self, results_summary: List[Dict]):
        """Print overall test summary"""
        print(f"\n\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")

        if not results_summary:
            print("\n✗ No results to summarize")
            return

        # Statistics
        total_queries = len(results_summary)
        avg_results = sum(r['num_results'] for r in results_summary) / total_queries
        avg_score = sum(r['top_score'] for r in results_summary) / total_queries

        print(f"\nTotal queries tested: {total_queries}")
        print(f"Average results per query: {avg_results:.2f}")
        print(f"Average top score: {avg_score:.4f}")

        # Score distribution
        high_score = len([r for r in results_summary if r['top_score'] >= 0.8])
        med_score = len([r for r in results_summary if 0.5 <= r['top_score'] < 0.8])
        low_score = len([r for r in results_summary if r['top_score'] < 0.5])

        print(f"\nScore Distribution (Top-1 scores):")
        print(f"  High (≥0.8):  {high_score} queries ({high_score/total_queries*100:.1f}%)")
        print(f"  Med (0.5-0.8): {med_score} queries ({med_score/total_queries*100:.1f}%)")
        print(f"  Low (<0.5):   {low_score} queries ({low_score/total_queries*100:.1f}%)")

        # Top 5 best queries
        print(f"\n🌟 TOP 5 BEST RESULTS:")
        top_5 = sorted(results_summary, key=lambda x: x['top_score'], reverse=True)[:5]
        for i, result in enumerate(top_5, 1):
            print(f"  {i}. {result['query_id']}: {result['top_file']} ({result['top_score']:.4f})")

    def save_results(self, output_file: str = "final_test_results.json"):
        """
        Save test results to JSON

        Args:
            output_file: Output file path
        """
        print(f"\nSaving results to {output_file}...")

        # Run all tests and collect results
        query_ids = self.query_handler.get_all_query_ids()

        all_results = {}
        for query_id in query_ids:
            query_text = self.query_handler.get_primary_query(query_id)
            results = self.search_engine.search(query_text, top_k=self.config.TOP_K_RESULTS)

            all_results[query_id] = {
                'query_text': query_text,
                'num_results': len(results),
                'results': [
                    {
                        'rank': r['rank'],
                        'score': r['score'],
                        'file_name': r['file_name'],
                        'has_passages': 'relevant_passages' in r
                    }
                    for r in results
                ]
            }

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to {output_file}")

    def run_full_test(self, test_all: bool = False, limit: int = 5):
        """
        Run full test suite

        Args:
            test_all: Test all queries (True) or sample only (False)
            limit: Number of queries for sample test
        """
        try:
            self.initialize()

            if test_all:
                self.test_all_queries()
            else:
                # Test sample queries
                query_ids = self.query_handler.get_all_query_ids()[:limit]

                for query_id in query_ids:
                    self.test_single_query(query_id, verbose=True)

            # Save results
            self.save_results()

            print(f"\n{'='*70}")
            print("✓ FINAL TEST COMPLETED")
            print(f"{'='*70}")

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    print("\nSelect test mode:")
    print("1. Test sample queries (5 queries, detailed)")
    print("2. Test all queries (full test)")
    print("3. Test specific query")

    choice = input("\nEnter choice (1/2/3): ").strip()

    tester = FinalTest()

    if choice == '1':
        print("\n→ Testing sample queries...")
        tester.run_full_test(test_all=False, limit=5)

    elif choice == '2':
        print("\n→ Testing all queries...")
        tester.run_full_test(test_all=True)

    elif choice == '3':
        query_id = input("Enter query ID (e.g., query_201): ").strip()
        tester.initialize()
        tester.test_single_query(query_id, verbose=True)

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
