"""Build the persistent FAISS index for the local insurance knowledge base."""

from __future__ import annotations

import argparse
import json

from rag import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_SOURCE_PATH,
    build_faiss_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建保险资料向量索引")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH), help="PDF/Excel/Markdown 文件或目录")
    parser.add_argument("--output", default=str(DEFAULT_INDEX_DIR), help="索引目录")
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING_PATH))
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=420)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = build_faiss_index(
        source_path=args.source,
        output_dir=args.output,
        embedding_path=args.embedding_model,
        embedding_device=args.embedding_device,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
