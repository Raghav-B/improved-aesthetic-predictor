import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_scores(scores_file: str):
   """Load scores TSV (Image Path \t Aesthetic Score)."""
   scores_path = Path(scores_file)
   if not scores_path.exists():
      raise FileNotFoundError(f"Scores file not found: {scores_path}")

   df = pd.read_csv(scores_path, sep="\t")
   # Normalize column names
   df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
   if "image_path" not in df.columns or "aesthetic_score" not in df.columns:
      raise ValueError("Scores file must have columns: Image Path, Aesthetic Score")

   df = df.rename(columns={"image_path": "filepath", "aesthetic_score": "prediction"})
   df["filepath"] = df["filepath"].astype(str)
   df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
   df = df.dropna(subset=["prediction"])

   # Keep only files that exist
   df["filepath"] = df["filepath"].apply(lambda p: str(Path(p).expanduser().resolve()))
   df = df[df["filepath"].apply(lambda p: Path(p).exists())]
   return df


def build_buckets(min_score: float, max_score: float, step: float):
   start = np.floor(min_score / step) * step
   end = np.ceil(max_score / step) * step
   edges = np.arange(start, end + step, step)
   buckets = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
   return buckets


def make_html(df: pd.DataFrame, output_file: str, bucket_step: float = 0.5, samples_per_bucket: int = 50):
   if df.empty:
      raise ValueError("No scores available to visualize.")

   min_score = df["prediction"].min()
   max_score = df["prediction"].max()
   buckets = build_buckets(min_score, max_score, bucket_step)

   html_parts = ["<h1>Aesthetic subsets from local data</h1>"]

   for a, b in buckets:
      total_part = df[(df["prediction"] >= a) & (df["prediction"] < b)]
      count_part = len(total_part)
      if count_part == 0:
         continue

      percent = count_part / len(df) * 100
      part = total_part.head(samples_per_bucket)

      html_parts.append(
         f"<h2>Bucket {a:.2f} - {b:.2f}: {percent:.2f}% ({count_part} samples)</h2><div>"
      )
      for filepath in part["filepath"]:
         src = Path(filepath).as_uri()
         html_parts.append(f'<img src="{src}" height="200" />')
      html_parts.append("</div>")

   html = "\n".join(html_parts)
   out_path = Path(output_file)
   out_path.write_text(html)
   return out_path


def main():
   parser = argparse.ArgumentParser(description="Visualize aesthetic score buckets for local data")
   parser.add_argument("--scores-file", default="aesthetic_scores/all_scores.txt", help="Path to TSV with Image Path and Aesthetic Score columns")
   parser.add_argument("--output-file", default="aesthetic_viz_local.html", help="Output HTML file")
   parser.add_argument("--bucket-step", type=float, default=0.5, help="Bucket width for scores")
   parser.add_argument("--samples-per-bucket", type=int, default=50, help="Max images to show per bucket")
   args = parser.parse_args()

   df = load_scores(args.scores_file)
   print(f"Loaded {len(df)} scored images from {args.scores_file}")

   out_path = make_html(df, args.output_file, bucket_step=args.bucket_step, samples_per_bucket=args.samples_per_bucket)
   print(f"Wrote visualization to {out_path}")
   print("Open the HTML in a browser to view the buckets.")


if __name__ == "__main__":
   main()