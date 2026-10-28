from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def summarize(data_dir: Path, out_dir: Path):
    events = pd.read_csv(data_dir / "events.csv")
    users = pd.read_csv(data_dir / "user_features.csv") if (data_dir / "user_features.csv").exists() else None
    items = pd.read_csv(data_dir / "item_features.csv") if (data_dir / "item_features.csv").exists() else None
    for col in ("user_id", "item_id"):
        if col not in events.columns:
            raise ValueError(f"events.csv must contain {col!r}; found {events.columns.tolist()}")

    n_events = len(events)
    n_users, n_items = events["user_id"].nunique(), events["item_id"].nunique()
    density = n_events / (n_users * n_items) if n_users and n_items else 0.0
    lines = [
        "# EDA: movie recommendation dataset", "",
        "## Events", "",
        f"- Interaction rows: **{n_events:,}**",
        f"- Unique users: **{n_users:,}**",
        f"- Unique interacted items: **{n_items:,}**",
        f"- Matrix density (event rows / users × items): **{density:.6%}**",
        f"- Sparsity: **{1-density:.6%}**",
        f"- Columns: `{', '.join(events.columns.astype(str))}`", ""
    ]
    if "rating" in events:
        lines += ["### Rating distribution", "",
                  events["rating"].value_counts(dropna=False).sort_index().to_frame("count").to_markdown(), "",
                  f"- Mean rating: {pd.to_numeric(events['rating'], errors='coerce').mean():.4f}", ""]
    if "timestamp" in events:
        ts = pd.to_numeric(events["timestamp"], errors="coerce")
        lines += ["### Timestamp", "",
                  "- The supplied timestamp column is numeric and appears to represent event order, not a calendar date.",
                  f"- Minimum: {ts.min():g}; maximum: {ts.max():g}",
                  f"- Distinct timestamp values: {ts.nunique():,}", ""]
    user_counts = events.groupby("user_id")["item_id"].nunique()
    item_counts = events.groupby("item_id")["user_id"].nunique()
    lines += ["### Interaction activity", "",
              f"- Users with 1 distinct item: {(user_counts == 1).sum():,}",
              f"- Users with fewer than 5 distinct items: {(user_counts < 5).sum():,}",
              f"- Items interacted with by 1 user: {(item_counts == 1).sum():,}",
              f"- Items interacted with by fewer than 5 users: {(item_counts < 5).sum():,}", "",
              "### Top 20 items by interaction count", "",
              events.groupby("item_id").size().sort_values(ascending=False).head(20).to_frame("interactions").to_markdown(), ""]

    if users is not None:
        lines += ["## User features", "",
                  f"- Rows: {len(users):,}",
                  f"- Columns: `{', '.join(users.columns.astype(str))}`", "",
                  "### Missing values", "", users.isna().sum().to_frame("missing").to_markdown(), "",
                  "### Sample", "", users.head(5).to_markdown(index=False), ""]
        if "gender" in users:
            lines += ["### Gender distribution", "", users["gender"].value_counts(dropna=False).to_frame("count").to_markdown(), ""]
        if "age" in users:
            lines += ["### Age distribution", "", users["age"].value_counts(dropna=False).sort_index().to_frame("count").to_markdown(), ""]

    if items is not None:
        genre_cols = [c for c in items.columns if c.startswith("genre_")]
        lines += ["## Item features", "",
                  f"- Rows: {len(items):,}",
                  f"- Columns: `{', '.join(items.columns.astype(str))}`",
                  f"- Genre indicator columns: {len(genre_cols)}", "",
                  "### Missing values", "", items.isna().sum().to_frame("missing").to_markdown(), "",
                  "### Sample", "", items.head(5).to_markdown(index=False), ""]
        if genre_cols:
            genre_counts = items[genre_cols].sum().sort_values(ascending=False).rename("item_count").to_frame()
            lines += ["### Items per genre indicator", "", genre_counts.to_markdown(), ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "eda.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved EDA report: {report}")
    print(f"Events: {n_events:,}; users: {n_users:,}; items: {n_items:,}; density: {density:.6%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    summarize(args.data_dir, args.out_dir)
