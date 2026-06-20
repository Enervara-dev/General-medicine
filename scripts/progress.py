import os
import glob
from collections import defaultdict

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "logs", "processed_blocks")
    output_dir = os.path.join(base_dir, "chunking", "output", "v1")

    print("=" * 60)
    print(" CHUNKING PIPELINE PROGRESS SUMMARY ".center(60, "="))
    print("=" * 60)

    # 1. Count processed blocks
    if os.path.exists(processed_dir):
        done_files = glob.glob(os.path.join(processed_dir, "*.done"))
        num_blocks = len(done_files)
        print(f"Total Semantic Blocks Processed: {num_blocks}")
    else:
        print("Total Semantic Blocks Processed: 0")

    print("-" * 60)

    # 2. Count chunks per category/book
    if os.path.exists(output_dir):
        total_chunks = 0
        book_counts = defaultdict(int)

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".json"):
                    total_chunks += 1
                    # Get the relative path to identify the book category
                    relative_path = os.path.relpath(root, output_dir)
                    parts = relative_path.split(os.sep)
                    if parts and parts[0] != ".":
                        book_name = parts[0].replace("_", " ").title()
                        book_counts[book_name] += 1
        
        print(f"Total Micro-Chunks Generated: {total_chunks}")
        print("-" * 60)
        
        if total_chunks > 0:
            print(f"{'CATEGORY/BOOK':<45} | {'CHUNKS'}")
            print("-" * 60)
            for book, count in sorted(book_counts.items(), key=lambda x: x[1], reverse=True):
                # Truncate book name if too long for clean formatting
                display_name = book[:42] + "..." if len(book) > 42 else book
                print(f"{display_name:<45} | {count}")
    else:
        print("Total Micro-Chunks Generated: 0")

    print("=" * 60)

if __name__ == "__main__":
    main()
