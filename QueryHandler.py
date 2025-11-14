"""
QueryHandler.py - Handle User Query Input

Module này xử lý input queries từ người dùng với format:
{
    "query_201": {
        "Q1": "...",
        "Q2": "..."
    }
}

Responsibilities:
- Load user queries từ userQuery.json
- Parse và validate query structure
- Combine multiple sub-queries (Q1, Q2) nếu cần
- Provide clean interface cho SearchEngine
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class QueryHandler:
    """
    Handle user query input với multi-question format

    Support formats:
    - Single query: {"query_1": {"Q1": "..."}}
    - Multiple sub-queries: {"query_1": {"Q1": "...", "Q2": "..."}}
    """

    def __init__(self, user_query_file: str = None):
        """
        Initialize QueryHandler

        Args:
            user_query_file: Path to userQuery.json
                           Default: dataset/userQuery.json
        """
        if user_query_file is None:
            user_query_file = os.path.join("dataset", "userQuery.json")

        self.user_query_file = Path(user_query_file)
        self.queries = {}

        print(f"  → User query file: {self.user_query_file}")

    def load_queries(self) -> Dict:
        """
        Load queries từ userQuery.json

        Returns:
            Dict of queries: {
                "query_201": {"Q1": "...", "Q2": "..."},
                ...
            }
        """
        if not self.user_query_file.exists():
            raise FileNotFoundError(
                f"User query file not found: {self.user_query_file}"
            )

        try:
            with open(self.user_query_file, 'r', encoding='utf-8') as f:
                self.queries = json.load(f)

            print(f"  ✓ Loaded {len(self.queries)} user queries")

            # Validate structure
            self._validate_queries()

            return self.queries

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {self.user_query_file}: {e}"
            )

    def _validate_queries(self):
        """
        Validate query structure

        Expected format:
        {
            "query_X": {
                "Q1": str,
                "Q2": str (optional),
                ...
            }
        }
        """
        for query_id, query_data in self.queries.items():
            if not isinstance(query_data, dict):
                raise ValueError(
                    f"Invalid query format for {query_id}: "
                    f"Expected dict, got {type(query_data)}"
                )

            # Check có ít nhất Q1
            if 'Q1' not in query_data:
                raise ValueError(
                    f"Query {query_id} missing 'Q1'"
                )

    def get_query(self, query_id: str) -> Optional[Dict]:
        """
        Get query by ID

        Args:
            query_id: Query identifier (e.g., "query_201")

        Returns:
            Query dict: {"Q1": "...", "Q2": "..."}
            None if not found
        """
        return self.queries.get(query_id)

    def get_all_sub_queries(self, query_id: str) -> List[Tuple[str, str]]:
        """
        Get all sub-queries (Q1, Q2, ...) cho một query

        Args:
            query_id: Query identifier

        Returns:
            List of (sub_query_key, sub_query_text):
            [("Q1", "..."), ("Q2", "...")]
        """
        query_data = self.get_query(query_id)

        if not query_data:
            return []

        sub_queries = []
        for key in sorted(query_data.keys()):  # Q1, Q2, Q3, ...
            if key.startswith('Q'):
                sub_queries.append((key, query_data[key]))

        return sub_queries

    def get_combined_query(self, query_id: str, separator: str = " ") -> str:
        """
        Combine tất cả sub-queries thành 1 query string

        Useful cho document-level retrieval.

        Args:
            query_id: Query identifier
            separator: Separator giữa các sub-queries

        Returns:
            Combined query string
        """
        sub_queries = self.get_all_sub_queries(query_id)

        if not sub_queries:
            return ""

        # Combine all sub-query texts
        texts = [text for _, text in sub_queries]
        return separator.join(texts)

    def get_primary_query(self, query_id: str) -> str:
        """
        Get Q1 (primary query) only

        Args:
            query_id: Query identifier

        Returns:
            Q1 text
        """
        query_data = self.get_query(query_id)

        if not query_data:
            return ""

        return query_data.get('Q1', '')

    def get_all_query_ids(self) -> List[str]:
        """
        Get all query IDs

        Returns:
            List of query IDs: ["query_201", "query_202", ...]
        """
        return list(self.queries.keys())

    def get_query_count(self) -> int:
        """
        Get total number of queries

        Returns:
            Number of queries
        """
        return len(self.queries)

    def get_sub_query_count(self, query_id: str) -> int:
        """
        Get number of sub-queries cho một query

        Args:
            query_id: Query identifier

        Returns:
            Number of sub-queries (Q1, Q2, ...)
        """
        return len(self.get_all_sub_queries(query_id))

    def print_query_summary(self):
        """
        Print summary của user queries
        """
        print("\n" + "="*70)
        print("USER QUERY SUMMARY")
        print("="*70)

        print(f"\nTotal queries: {self.get_query_count()}")

        # Count sub-queries
        total_sub_queries = 0
        for query_id in self.get_all_query_ids():
            total_sub_queries += self.get_sub_query_count(query_id)

        print(f"Total sub-queries: {total_sub_queries}")
        print(f"Average sub-queries per query: {total_sub_queries/self.get_query_count():.1f}")

        # Sample
        print(f"\nSample queries:")
        for query_id in list(self.get_all_query_ids())[:3]:
            print(f"\n[{query_id}]")
            sub_queries = self.get_all_sub_queries(query_id)
            for key, text in sub_queries:
                print(f"  {key}: {text[:60]}...")


# Test module
if __name__ == "__main__":
    print("="*70)
    print("TESTING QueryHandler")
    print("="*70)

    # Create sample userQuery.json for testing
    sample_queries = {
        "query_201": {
            "Q1": "Quốc hiệu Đại Cồ Việt tồn tại trong khoảng thời gian nào?",
            "Q2": "Ý nghĩa của ba chữ \"Đại Cồ Việt\" là gì?"
        },
        "query_202": {
            "Q1": "Quốc hiệu Đại Việt tồn tại trong những giai đoạn nào?"
        }
    }

    # Save to test file
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/userQuery_test.json", 'w', encoding='utf-8') as f:
        json.dump(sample_queries, f, indent=2, ensure_ascii=False)

    # Test QueryHandler
    handler = QueryHandler(user_query_file="dataset/userQuery_test.json")
    handler.load_queries()

    print("\n--- Testing get_query ---")
    query = handler.get_query("query_201")
    print(f"Query 201: {query}")

    print("\n--- Testing get_all_sub_queries ---")
    sub_queries = handler.get_all_sub_queries("query_201")
    for key, text in sub_queries:
        print(f"{key}: {text}")

    print("\n--- Testing get_combined_query ---")
    combined = handler.get_combined_query("query_201")
    print(f"Combined: {combined}")

    print("\n--- Testing get_primary_query ---")
    primary = handler.get_primary_query("query_201")
    print(f"Primary (Q1): {primary}")

    handler.print_query_summary()
