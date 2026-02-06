import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_scores(scores_file: str, label: str):
   """Load scores TSV (Image Path \t Aesthetic Score) and tag with label."""
   scores_path = Path(scores_file)
   if not scores_path.exists():
      raise FileNotFoundError(f"Scores file not found: {scores_path}")

   df = pd.read_csv(scores_path, sep="\t")
   df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
   if "image_path" not in df.columns or "aesthetic_score" not in df.columns:
      raise ValueError("Scores file must have columns: Image Path, Aesthetic Score")

   df = df.rename(columns={"image_path": "filepath", "aesthetic_score": f"score_{label}"})
   df["filepath"] = df["filepath"].astype(str)
   df[f"score_{label}"] = pd.to_numeric(df[f"score_{label}"], errors="coerce")
   df = df.dropna(subset=[f"score_{label}"])

   df["filepath"] = df["filepath"].apply(lambda p: str(Path(p).expanduser().resolve()))
   df = df[df["filepath"].apply(lambda p: Path(p).exists())]
   df["parent_dir"] = df["filepath"].apply(lambda p: str(Path(p).parent))
   return df


def build_buckets(min_score: float, max_score: float, step: float):
   start = np.floor(min_score / step) * step
   end = np.ceil(max_score / step) * step
   edges = np.arange(start, end + step, step)
   buckets = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
   return buckets


def make_html(df: pd.DataFrame, output_file: str, bucket_step: float = 0.5, samples_per_bucket: int = 50):
   if df.empty:
      raise ValueError("No paired scores available to visualize.")

   min_score = df["weighted_score"].min()
   max_score = df["weighted_score"].max()
   buckets = build_buckets(min_score, max_score, bucket_step)

   html_parts = ["<h1>Paired thumbnails & inputs by weighted score</h1>"]

   for a, b in buckets:
      total_part = df[(df["weighted_score"] >= a) & (df["weighted_score"] < b)]
      count_part = len(total_part)
      if count_part == 0:
         continue

      percent = count_part / len(df) * 100
      part = total_part.head(samples_per_bucket)

      html_parts.append(
         f"<h2>Bucket {a:.2f} - {b:.2f}: {percent:.2f}% ({count_part} samples)</h2><div style=\"display:flex;flex-wrap:wrap;gap:12px;\">"
      )
      for _, row in part.iterrows():
         thumb_src = Path(row["filepath_thumb"]).as_uri()
         input_src = Path(row["filepath_input"]).as_uri()
         html_parts.append(
            '<div style="border:1px solid #ddd;padding:6px;width:220px;">'
            f'<div style="text-align:center">'
            f'<img src="{thumb_src}" height="100" />'
            f'<div style="font-size:12px;color:#555">thumb: {row["score_thumb"]:.3f}</div>'
            '</div>'
            f'<div style="text-align:center;margin-top:6px">'
            f'<img src="{input_src}" height="100" />'
            f'<div style="font-size:12px;color:#555">input: {row["score_input"]:.3f}</div>'
            '</div>'
            f'<div style="font-size:12px;color:#222;margin-top:6px">weighted: {row["weighted_score"]:.3f}</div>'
            '</div>'
         )
      html_parts.append("</div>")

   html = "\n".join(html_parts)
   out_path = Path(output_file)
   out_path.write_text(html)
   return out_path


def main():
   parser = argparse.ArgumentParser(description="Visualize paired thumbnail/input scores with weighted buckets")
   parser.add_argument("--thumbnail-scores", default="aesthetic_scores/gen_mesh_thumbnail_scores.txt", help="TSV of thumbnail scores (Image Path, Aesthetic Score)")
   parser.add_argument("--input-scores", default="aesthetic_scores/input_image_scores.txt", help="TSV of input image scores (Image Path, Aesthetic Score)")
   parser.add_argument("--output-file", default="aesthetic_viz_weighted.html", help="Output HTML file")
   parser.add_argument("--bucket-step", type=float, default=0.5, help="Bucket width for weighted scores")
   parser.add_argument("--samples-per-bucket", type=int, default=50, help="Max pairs to show per bucket")
   parser.add_argument("--thumb-weight", type=float, default=0.3, help="Weight for thumbnail score")
   parser.add_argument("--input-weight", type=float, default=0.7, help="Weight for input image score")
   args = parser.parse_args()

   df_thumb = load_scores(args.thumbnail_scores, label="thumb")
   df_input = load_scores(args.input_scores, label="input")

   # Join on parent directory so we pair files in the same folder
   df = pd.merge(df_thumb, df_input, on="parent_dir", suffixes=("_thumb", "_input"))
   if df.empty:
      print("No matching thumbnail/input pairs found.")
      return

   df["weighted_score"] = df["score_thumb"] * args.thumb_weight + df["score_input"] * args.input_weight

   out_path = make_html(df, args.output_file, bucket_step=args.bucket_step, samples_per_bucket=args.samples_per_bucket)
   print(f"Paired rows: {len(df)}")
   print(f"Wrote visualization to {out_path}")
   print("Open the HTML in a browser to view the buckets.")


if __name__ == "__main__":
   main()