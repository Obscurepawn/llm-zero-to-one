#!/usr/bin/env python3
"""
用 PyTorch FX + Graphviz 把第 4 周的 MLP 结构画成 SVG。

这份脚本不训练模型，只做结构可视化，目的是把
`pytorch_mlp_mnist.py` 里的 `MLPClassifier` 真正画出来。
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.fx import symbolic_trace
    from torch.fx.passes.graph_drawer import FxGraphDrawer
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "当前环境缺少 torch。请先安装 PyTorch 再运行本脚本。"
    ) from exc

try:
    import pydot  # noqa: F401  # pragma: no cover - runtime guard
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "当前环境缺少 pydot。请先执行 `python -m pip install pydot graphviz`。"
    ) from exc

from pytorch_mlp_mnist import MLPClassifier, OUTPUT_DIR, ensure_directories


def render_model_graph(
    name: str,
    hidden_dims: tuple[int, int],
    dropout: float,
    output_dir: Path,
) -> Path:
    """
    把一个 MLPClassifier 结构导出成 SVG。

    这里直接复用训练脚本里的模型定义，保证“图里画的结构”
    和“真正训练时用的结构”是一回事。
    """
    model = MLPClassifier(hidden_dims=hidden_dims, dropout=dropout)
    model.eval()

    traced_model = symbolic_trace(model)
    drawer = FxGraphDrawer(
        traced_model,
        name=f"mlp_{name}",
        ignore_getattr=True,
        ignore_parameters_and_buffers=True,
    )

    output_path = output_dir / f"pytorch_mlp_{name}_fx_graph.svg"
    drawer.get_dot_graph().write_svg(str(output_path))
    return output_path


def main() -> None:
    ensure_directories()

    rendered_paths = [
        render_model_graph(
            name="baseline",
            hidden_dims=(256, 128),
            dropout=0.0,
            output_dir=OUTPUT_DIR,
        ),
        render_model_graph(
            name="regularized",
            hidden_dims=(256, 128),
            dropout=0.2,
            output_dir=OUTPUT_DIR,
        ),
    ]

    print("Saved model graphs:")
    for path in rendered_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
