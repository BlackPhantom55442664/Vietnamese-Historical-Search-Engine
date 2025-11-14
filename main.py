"""
main.py - Entry Point
"""

from Config import Config
from SearchEngine import SearchEngine

def main():
    """Main function"""

    # Load config
    config = Config()

    # Print configuration
    config.print_config()

    # Initialize search engine
    engine = SearchEngine(config)

    # Build index
    engine.build_index()

    # Interactive search
    engine.interactive_search()

if __name__ == "__main__":
    main()
