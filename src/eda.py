from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def summarize(path: Path, out_dir: Path):
    events_path = path / "events.csv"
    users_path = path / "user_features.csv"
    items_path = path / "item_features.csv"
    events = pd.read_csv(events_path)
    users = pd.read_csv(users_path) if users_path.exists() else None
    items = pd.read_csv(items_path) if items_path.exists() else None

    user_col, item_col = "user_id", "item_id"
    for col in (user_col, item_col):
        if col not in events.columns:
            raise ValueError(f"{events_path} must contain column {col!r}; found {events.columns.tolist()}")

    n_events = len(events)
    n_users = events[user_col].nunique()
    n_items = events[item_col].nunique()
    possible = n_users * n_items
    density = n_events / possible if possible else 0.0
    lines = [
        "# EDA: movie recommendation dataset", "",
        "## Events", "",
        f"- Rows in events: **{n_events:,}**",
        f"- Unique users in events: **{n_users:,}**",
        f"- Unique items in events: **{n_items:,}**",
        f"- User-item matrix density (rows / users×items): **{density:.8%}**",
        f"- Sparsity (1 − density): **{1-density:.8%}**",
        f"- Event columns: `{', '.join(map(str, events.columns))}`", ""
    ]
    if "rating" in events.columns:
        lines += ["### Rating distribution", "", events["rating"].value_counts(dropna=False).sort_index().to_frame("count").to_markdown(), "",
                  f"Mean rating: {pd.to_numeric(events['rating'], errors='coerce').mean():.4f}", ""]
    time_candidates = [c for c in events.columns if c.lower() in {"timestamp", "datetime", "date", "event_time", "created_at"}]
    if time_candidates:
        col = time_candidates[0]
        dt = pd.to_datetime(events[col], errors="coerce", utc=True)
        lines += [f"### Time distribution (`{col}`)", "",
                  f"- Parseable timestamps: {dt.notna().sum():,} / {len(dt):,}",
                  f"- Earliest: {dt.min()}",
                  f"- Latest: {dt.max()}", ""]
    user_counts = events.groupby(user_col)[item_col].nunique()
    item_counts = events.groupby(item_col)[user_col].nunique()
    lines += ["### Activity", "",
              f"- Users with one distinct item: {(user_counts == 1).sum():,}",
              f"- Users with fewer than 5 distinct items: {(user_counts < 5).sum():,}",
              f"- Items with one user: {(item_counts == 1).sum():,}",
              f"- Items with fewer than 5 users: {(item_counts < 5).sum():,}", "",
              "### Top 20 items by interactions", "",
              events.groupby(item_col).size().sort_values(ascending=False).head(20).to_frame("interactions").to_markdown(), ""]
    for name, frame in (("user_features.csv", users), ("item_features.csv", items)):
        if frame is not None:
            lines += [f"## `{name}`", "",
                      f"- Rows: **{len(frame):,}**",
                      f"- Columns: `{', '.join(map(str, frame.columns))}`", "",
                      "### Missing values", "",
                      frame.isna().sum().to_frame("missing").to_markdown(), "",
                      "### Sample", "", frame.head(5).to_markdown(index=False), ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "eda.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved EDA report: {report}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    summarize(args.data_dir, args.out_dir)
