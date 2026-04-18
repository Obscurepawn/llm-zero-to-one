#!/usr/bin/env python3
"""统一运行第 5 周的 tokenizer、embedding 和语言模型任务实验。"""

from __future__ import annotations

import tokenizer_training_algorithms_demo
import tokenizer_and_embedding_demo


def main() -> None:
    print("Running tokenizer / embedding / next-token demo...\n")
    tokenizer_and_embedding_demo.main()

    print("\n" + "=" * 80 + "\n")

    print("Running tokenizer training algorithms demo...\n")
    tokenizer_training_algorithms_demo.main()


if __name__ == "__main__":
    main()
