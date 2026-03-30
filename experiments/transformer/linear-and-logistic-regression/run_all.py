#!/usr/bin/env python3
"""一次性运行第 2 周的两个教学示例。"""

from __future__ import annotations

import linear_regression_numpy
import logistic_regression_numpy


def main() -> None:
    print("Running linear regression demo...\n")
    linear_regression_numpy.main()

    print("\n" + "=" * 80 + "\n")

    print("Running logistic regression demo...\n")
    logistic_regression_numpy.main()


if __name__ == "__main__":
    main()
