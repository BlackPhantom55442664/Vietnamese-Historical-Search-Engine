"""
UserSimulate.py - Interactive User Query Simulation

Enhanced version of main.py với:
- Passage Retrieval (tìm câu liên quan nhất)
- Better result visualization
- Query history
- Save results option
"""

from Config import Config
from SearchEngine import SearchEngine
from datetime import datetime
import json
import os


class UserSimulate:
    """
    Interactive query simulator với enhanced features
    """

    def __init__(self):
        """Initialize UserSimulate"""
        print("="*70)
        print("TMGSEG SEARCH ENGINE - USER SIMULATION")
        print("Enhanced with Passage Retrieval")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Config
        self.config = Config()

        # Search engine
        self.engine = None

        # Query history
        self.query_history = []

    def initialize(self):
        """Initialize search engine"""
        print("\n[1/2] Loading configuration...")
        self.config.print_config()

        print("\n[2/2] Initializing search engine...")
        self.engine = SearchEngine(self.config)
        self.engine.build_index()

        print("\n" + "="*70)
        print("✓ SYSTEM READY!")
        print("="*70)

    def print_results(self, query: str, results: list):
        """
        Print search results với enhanced format

        Args:
            query: User query
            results: Search results
        """
        print("\n" + "="*70)
        print(f"RESULTS FOR: {query}")
        print("="*70)

        if not results:
            print("\n✗ No results found!")
            return

        print(f"\nFound {len(results)} documents:")

        for result in results:
            rank = result['rank']
            score = result['score']
            file_name = result['file_name']

            print(f"\n{'─'*70}")
            print(f"[Rank {rank}] Score: {score:.4f}")
            print(f"📄 Document: {file_name}")

            # Show relevant passages if available
            if 'relevant_passages' in result and result['relevant_passages']:
                print(f"\n🔍 RELEVANT PASSAGES:")

                for i, passage in enumerate(result['relevant_passages'], 1):
                    print(f"\n  [{i}] Relevance: {passage['score']:.4f}")

                    # Wrap text nicely
                    text = passage['text']
                    if len(text) > 200:
                        text = text[:200] + "..."

                    print(f"  {text}")
            else:
                # Fallback: show content preview
                print(f"\n💡 Preview:")
                content = result.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"  {content}")

    def save_query_history(self, filename: str = "query_history.json"):
        """
        Save query history to file

        Args:
            filename: Output filename
        """
        if not self.query_history:
            print("\n⚠ No queries to save!")
            return

        # Prepare data
        history_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': len(self.query_history),
            'queries': self.query_history
        }

        # Save
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Query history saved to '{filename}'")

    def print_query_history(self):
        """Print query history summary"""
        if not self.query_history:
            print("\n⚠ No query history yet!")
            return

        print("\n" + "="*70)
        print("QUERY HISTORY")
        print("="*70)

        for i, record in enumerate(self.query_history, 1):
            query = record['query']
            num_results = record['num_results']
            top_score = record['top_score'] if num_results > 0 else 0.0

            print(f"\n[{i}] {query}")
            print(f"    Results: {num_results} | Top Score: {top_score:.4f}")

    def interactive_search(self):
        """
        Interactive search loop với enhanced features
        """
        print("\n" + "="*70)
        print("INTERACTIVE SEARCH MODE")
        print("="*70)
        print("\nCommands:")
        print("  - Enter query to search")
        print("  - 'history' - View query history")
        print("  - 'save' - Save query history")
        print("  - 'stats' - Show statistics")
        print("  - 'exit' or 'quit' - Exit program")
        print("="*70)

        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter query: ").strip()

                if not query:
                    continue

                # Check commands
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                elif query.lower() == 'history':
                    self.print_query_history()
                    continue

                elif query.lower() == 'save':
                    self.save_query_history()
                    continue

                elif query.lower() == 'stats':
                    self.print_statistics()
                    continue

                # Search
                print(f"\n⏳ Searching for: '{query}'...")

                results = self.engine.search(
                    query=query,
                    top_k=self.config.TOP_K_RESULTS
                )

                # Print results
                self.print_results(query, results)

                # Save to history
                query_record = {
                    'query': query,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'num_results': len(results),
                    'top_score': results[0]['score'] if results else 0.0
                }
                self.query_history.append(query_record)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Goodbye!")
                break

            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Please try again.")

    def print_statistics(self):
        """Print system statistics"""
        stats = self.engine.get_statistics()

        print("\n" + "="*70)
        print("SYSTEM STATISTICS")
        print("="*70)

        print(f"\nDocuments:")
        print(f"  Total: {stats['num_documents']}")
        print(f"  Indexed: {stats['indexed']}")

        print(f"\nRetrieval Pipeline:")
        print(f"  Stage 1 (BM25): Top-{stats['stage1_top_k']}")
        print(f"  Stage 2 (Embedding): Top-{stats['stage2_top_k']}")
        print(f"  Final Results: Top-{stats['final_top_k']}")

        print(f"\nPassage Retrieval:")
        print(f"  Enabled: {stats['passage_retrieval_enabled']}")

        print(f"\nQuery History:")
        print(f"  Total Queries: {len(self.query_history)}")

        if self.query_history:
            avg_results = sum(q['num_results'] for q in self.query_history) / len(self.query_history)
            avg_score = sum(q['top_score'] for q in self.query_history if q['top_score'] > 0)
            avg_score = avg_score / len([q for q in self.query_history if q['top_score'] > 0]) if avg_score > 0 else 0

            print(f"  Avg Results per Query: {avg_results:.2f}")
            print(f"  Avg Top Score: {avg_score:.4f}")

    def run(self):
        """Run the simulation"""
        try:
            self.initialize()
            self.interactive_search()

            # Ask to save history
            if self.query_history:
                save = input("\nSave query history? (y/n): ").strip().lower()
                if save == 'y':
                    self.save_query_history()

            print("\n" + "="*70)
            print("✓ SESSION ENDED")
            print("="*70)

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    simulator = UserSimulate()
    simulator.run()


if __name__ == "__main__":
    main()
