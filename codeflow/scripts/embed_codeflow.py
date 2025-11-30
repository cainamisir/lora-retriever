#!/usr/bin/env python
"""
Embed Codeflow Bench problems with OpenAI text-embedding-3-small and optionally visualize.

Examples
--------
Embed all problems and save artifacts under codeflow/output/embeddings:
    python codeflow/scripts/embed_codeflow.py embed \\
        --input codeflow/data/codeflowbench_full.jsonl \\
        --output-dir codeflow/output/embeddings

Embed only the first 10 problems (quick smoke test):
    python codeflow/scripts/embed_codeflow.py embed --limit 10

Make a PCA scatter plot from previously saved embeddings:
    python codeflow/scripts/embed_codeflow.py plot \\
        --embeddings-file codeflow/output/embeddings/embeddings.npy \\
        --metadata-file codeflow/output/embeddings/metadata.jsonl \\
        --output-figure codeflow/output/embeddings/embeddings_pca.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


def load_examples(path: pathlib.Path, limit: int | None = None) -> List[dict]:
    """Load JSONL records from file."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_problem_text(record: dict) -> str:
    """Combine problem fields into a single string for embedding."""
    parts = [
        record.get("title", ""),
        record.get("problem-description", ""),
        record.get("input", ""),
        record.get("output", ""),
    ]
    return "\n\n".join(p for p in parts if p)


def get_tokenizer(model: str):
    """Return a tiktoken encoding for the model, or None if unavailable."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to cl100k_base which matches text-embedding-3-* models.
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, encoder) -> int:
    """Exact token count when encoder is available, else fallback to len/2."""
    if encoder is None:
        return max(1, len(text) // 2)
    return len(encoder.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, encoder, max_tokens: int) -> str:
    """Truncate text to max_tokens using the encoder (or len/2 fallback)."""
    if encoder is None:
        # Fallback: cut by characters approximately aligned with len/2 heuristic.
        return text[: max_tokens * 2]
    ids = encoder.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return encoder.decode(ids[:max_tokens])


def batch_by_size(
    texts: List[str],
    max_items: int,
    max_tokens: int,
    encoder,
) -> Iterable[List[str]]:
    """Yield batches respecting both item count and total token budget."""
    current: List[str] = []
    token_sum = 0
    for text in texts:
        tok = count_tokens(text, encoder)
        if current and (len(current) >= max_items or token_sum + tok > max_tokens):
            yield current
            current = []
            token_sum = 0
        current.append(text)
        token_sum += tok
    if current:
        yield current


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int,
    max_request_tokens: int,
    encoder,
) -> List[List[float]]:
    """Call OpenAI embeddings API in batches to reduce HTTP overhead."""
    all_embeddings: List[List[float]] = []
    for chunk in batch_by_size(texts, batch_size, max_request_tokens, encoder):
        response = client.embeddings.create(model=model, input=chunk)
        # The API preserves order; append embeddings in sequence.
        for item in response.data:
            all_embeddings.append(item.embedding)
    return all_embeddings


def save_embeddings(
    embeddings: List[List[float]],
    metadata: List[dict],
    out_dir: pathlib.Path,
    model: str,
    prefix: str,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Persist embeddings matrix and lightweight metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{prefix}_embeddings.npy"
    meta_path = out_dir / f"{prefix}_metadata.jsonl"

    np.save(emb_path, np.asarray(embeddings, dtype=np.float32))
    with meta_path.open("w", encoding="utf-8") as f:
        for record in metadata:
            record_with_model = dict(record)
            record_with_model["model"] = model
            f.write(json.dumps(record_with_model) + "\n")
    return emb_path, meta_path


def run_embedding(args: argparse.Namespace) -> None:
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_examples(input_path, limit=args.limit)
    client = OpenAI()
    encoder = get_tokenizer(args.model)

    def maybe_truncate(text: str) -> str:
        tok = count_tokens(text, encoder)
        if tok <= args.max_item_tokens and len(text) <= args.max_item_chars:
            return text
        cap_tokens = min(args.max_item_tokens, args.max_item_chars)  # chars checked below
        truncated = truncate_to_tokens(text, encoder, cap_tokens)
        # Ensure character cap as a final guard.
        if len(truncated) > args.max_item_chars:
            truncated = truncated[: args.max_item_chars]
        return truncated

    # Problem statements
    problem_texts = [maybe_truncate(build_problem_text(r)) for r in records]
    problem_meta = []
    for r in records:
        problem_meta.append(
            {
                "kind": "problem",
                "problem_id": r.get("problem-id"),
                "title": r.get("title"),
                "rating": r.get("rating"),
                "tags": r.get("tags"),
            }
        )
    problem_embeddings = embed_texts(
        client=client,
        texts=problem_texts,
        model=args.model,
        batch_size=args.batch_size,
        max_request_tokens=args.max_request_tokens,
        encoder=encoder,
    )
    p_emb_path, p_meta_path = save_embeddings(
        embeddings=problem_embeddings,
        metadata=problem_meta,
        out_dir=output_dir,
        model=args.model,
        prefix="problems",
    )
    print(f"Wrote problem embeddings to {p_emb_path}")
    print(f"Wrote problem metadata to {p_meta_path}")

    # Solutions (text and code)
    solution_texts: List[str] = []
    solution_meta: List[dict] = []
    for r in records:
        for idx, sol in enumerate(r.get("solutions", []) or []):
            content = sol.get("content")
            if not content:
                continue
            solution_texts.append(
                maybe_truncate(
                    "\n\n".join(
                        [
                            r.get("title", ""),
                            f"Solution type: {sol.get('type', 'unknown')}",
                            content,
                        ]
                    )
                )
            )
            solution_meta.append(
                {
                    "kind": "solution",
                    "problem_id": r.get("problem-id"),
                    "title": r.get("title"),
                    "rating": r.get("rating"),
                    "solution_index": idx,
                    "solution_type": sol.get("type"),
                    "tags": r.get("tags"),
                }
            )

    if solution_texts:
        solution_embeddings = embed_texts(
            client=client,
            texts=solution_texts,
            model=args.model,
            batch_size=args.batch_size,
            max_request_tokens=args.max_request_tokens,
            encoder=encoder,
        )
        s_emb_path, s_meta_path = save_embeddings(
            embeddings=solution_embeddings,
            metadata=solution_meta,
            out_dir=output_dir,
            model=args.model,
            prefix="solutions",
        )
        print(f"Wrote solution embeddings to {s_emb_path}")
        print(f"Wrote solution metadata to {s_meta_path}")
    else:
        print("No solutions found; skipped solution embeddings.")


def compute_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Simple PCA via SVD; returns components with shape (n_samples, n_components)."""
    # Center data
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def load_metadata(path: pathlib.Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def run_plot(args: argparse.Namespace) -> None:
    emb_path = pathlib.Path(args.embeddings_file)
    meta_path = pathlib.Path(args.metadata_file)

    embeddings = np.load(emb_path)
    meta = load_metadata(meta_path)

    dims = 3 if args.d3 else 2
    coords = compute_pca(embeddings, n_components=dims)

    plt.figure(figsize=(10, 8))
    if dims == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = plt.axes(projection="3d")
    else:
        ax = plt.gca()

    def color_values():
        if args.color_by == "rating":
            vals: List[float] = []
            for m in meta:
                try:
                    vals.append(float(m.get("rating")))
                except Exception:
                    vals.append(math.nan)
            return vals, "viridis", "Rating (NaN shown in gray)", None
        if args.color_by == "tag":
            tag_to_idx = {}
            tag_labels = []
            colors = []
            for m in meta:
                tags = m.get("tags") or []
                tag = tags[0] if tags else "NA"
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)
                    tag_labels.append(tag)
                colors.append(tag_to_idx[tag])
            return colors, "tab20", tag_labels, None
        if args.color_by == "focus-tag":
            focus = args.focus_tag
            colors = []
            for m in meta:
                tags = m.get("tags") or []
                has_focus = focus in tags
                colors.append(1 if has_focus else 0)
            # cmap with two colors: gray (0) and highlight (1)
            cmap = plt.cm.get_cmap("coolwarm", 2)
            return colors, cmap, ["other", focus or "focus-tag"], focus
        return None, None, None, None

    cvals, cmap, legend_or_label, focus_tag = color_values()

    if dims == 2:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=cvals,
            cmap=cmap,
            s=12,
        )
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=cvals,
            cmap=cmap,
            s=8,
        )

    if args.color_by == "rating":
        plt.colorbar(scatter, label=legend_or_label)
    elif args.color_by == "tag":
        tag_labels = legend_or_label
        handles = []
        max_tags = min(len(tag_labels), 20)
        for i in range(max_tags):
            handles.append(
                plt.Line2D(
                    [], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)), label=tag_labels[i]
                )
            )
        plt.legend(handles=handles, title="Tag (first 20)", bbox_to_anchor=(1.05, 1), loc="upper left")
    elif args.color_by == "focus-tag":
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(0), label="other"),
            plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(1), label=focus_tag or "focus-tag"),
        ]
        plt.legend(handles=handles, title="Focus tag", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title("Codeflow Bench embeddings (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if dims == 3:
        ax.set_zlabel("PC3")
    plt.tight_layout()
    out_fig = pathlib.Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=200)
    print(f"Wrote plot to {out_fig}")


def run_plot_tags(args: argparse.Namespace) -> None:
    emb_path = pathlib.Path(args.embeddings_file)
    meta_path = pathlib.Path(args.metadata_file)

    embeddings = np.load(emb_path)
    meta = load_metadata(meta_path)

    dims = 3 if args.d3 else 2
    coords = compute_pca(embeddings, n_components=dims)

    # Collect unique tags
    tag_set = set()
    for m in meta:
        for t in m.get("tags") or []:
            if t:
                tag_set.add(t)

    if not tag_set:
        print("No tags found; nothing to plot.")
        return

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.get_cmap("coolwarm", 2)

    for tag in sorted(tag_set):
        colors = []
        for m in meta:
            tags = m.get("tags") or []
            colors.append(1 if tag in tags else 0)

        fig = plt.figure(figsize=(10, 8))
        if dims == 3:
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                cmap=cmap,
                s=8,
            )
            ax.set_zlabel("PC3")
        else:
            ax = plt.gca()
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                cmap=cmap,
                s=12,
            )

        ax.set_title(f"Codeflow Bench embeddings (tag={tag})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=cmap(0), label="other"),
            plt.Line2D([], [], marker="o", linestyle="", color=cmap(1), label=tag),
        ]
        plt.legend(handles=handles, title="Focus tag", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "NA"
        suffix = "_3d" if dims == 3 else ""
        out_fig = out_dir / f"tag_{safe_tag}{suffix}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=200)
        plt.close(fig)
        print(f"Wrote plot for tag '{tag}' to {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed CodeflowBench problems and visualize embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Compute embeddings and save.")
    embed_parser.add_argument(
        "--input",
        type=str,
        default="codeflow/data/codeflowbench_full.jsonl",
        help="Path to input JSONL of problems.",
    )
    embed_parser.add_argument(
        "--output-dir",
        type=str,
        default="codeflow/output/embeddings",
        help="Directory to write embeddings.npy and metadata.jsonl.",
    )
    embed_parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name.",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Embedding batch size per API call. Keep small unless you know token sizes.",
    )
    embed_parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=7900,
        help="Hard token budget per embedding request (sum across the batch).",
    )
    embed_parser.add_argument(
        "--max-item-tokens",
        type=int,
        default=7800,
        help="Per-item token cap; text above this is truncated to this many tokens.",
    )
    embed_parser.add_argument(
        "--max-item-chars",
        type=int,
        default=16000,
        help="Hard per-item character cap (backup guard alongside token cap).",
    )
    embed_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of problems (useful for smoke tests).",
    )
    embed_parser.set_defaults(func=run_embedding)

    plot_parser = subparsers.add_parser(
        "plot", help="Create a PCA scatter plot from saved embeddings."
    )
    plot_parser.add_argument(
        "--embeddings-file",
        type=str,
        default="codeflow/output/embeddings/embeddings.npy",
        help="Path to embeddings.npy.",
    )
    plot_parser.add_argument(
        "--metadata-file",
        type=str,
        default="codeflow/output/embeddings/metadata.jsonl",
        help="Path to metadata.jsonl produced by the embed command.",
    )
    plot_parser.add_argument(
        "--output-figure",
        type=str,
        default="codeflow/output/embeddings/embeddings_pca.png",
        help="Where to save the PCA scatter plot.",
    )
    plot_parser.add_argument(
        "--color-by",
        type=str,
        choices=["rating", "tag", "focus-tag", "none"],
        default="rating",
        help="Color points by rating (default), first tag, a specific tag, or leave uncolored.",
    )
    plot_parser.add_argument(
        "--focus-tag",
        type=str,
        default=None,
        help="When --color-by focus-tag, highlight only points containing this tag; others are gray.",
    )
    plot_parser.add_argument(
        "--d3",
        action="store_true",
        help="Render a 3D PCA scatter (PC1, PC2, PC3). Defaults to 2D.",
    )
    plot_parser.set_defaults(func=run_plot)

    plot_tags_parser = subparsers.add_parser(
        "plot-tags", help="Create a PCA scatter for every tag (highlighting one tag at a time)."
    )
    plot_tags_parser.add_argument(
        "--embeddings-file",
        type=str,
        default="codeflow/output/embeddings/embeddings.npy",
        help="Path to embeddings.npy.",
    )
    plot_tags_parser.add_argument(
        "--metadata-file",
        type=str,
        default="codeflow/output/embeddings/metadata.jsonl",
        help="Path to metadata.jsonl produced by the embed command.",
    )
    plot_tags_parser.add_argument(
        "--output-dir",
        type=str,
        default="codeflow/output/embeddings/tag_plots",
        help="Directory to write one PNG per tag.",
    )
    plot_tags_parser.add_argument(
        "--d3",
        action="store_true",
        help="Render 3D PCA plots (default 2D).",
    )
    plot_tags_parser.set_defaults(func=run_plot_tags)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
