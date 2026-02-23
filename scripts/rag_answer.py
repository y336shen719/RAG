import sys
from rag_core import answer_query

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/rag_CLI.py \"your question\"")
        sys.exit(1)

    query = sys.argv[1]

    answer, debug = answer_query(query)

    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80)

if __name__ == "__main__":
    main()
