import sys
from pathlib import Path
from scripts.rag_core import answer_query

# Project root detection for stable output path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = PROJECT_ROOT / "rag_answer.txt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.rag_answer \"your question\"")
        sys.exit(1)

    query = sys.argv[1]

    answer = answer_query(query)

    print("\n" + "=" * 80)
    print("Query:")
    print(query)
    print("\nAnswer:")
    print(answer)
    print("=" * 80)

    OUTPUT_FILE.write_text(
        f"Query:\n{query}\n\nAnswer:\n{answer}\n",
        encoding="utf-8"
    )

    print(f"\nSaved answer to {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
