import os
import argparse
import json
import struct
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


# MLP model definition (same as in simple_inference.py)
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def find_all_images(root_dir, filename="gen_mesh_thumbnail.png"):
    """Recursively find all files with the given filename."""
    root_path = Path(root_dir).expanduser()
    image_paths = list(root_path.rglob(filename))
    return image_paths


def glb_has_texture(glb_path: Path) -> bool:
    """Return True if GLB has any textures/images; False otherwise."""
    try:
        with open(glb_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[0:4] != b"glTF":
                return False
            _, _, _ = struct.unpack("<III", header)

            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                return False
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_type != 0x4E4F534A:  # 'JSON'
                return False

            json_str = f.read(chunk_length).decode("utf-8", errors="replace")
            data = json.loads(json_str)

            images = data.get("images", [])
            textures = data.get("textures", [])
            materials = data.get("materials", [])

            has_tex = bool(images or textures)
            if not has_tex:
                for mat in materials:
                    pmr = mat.get("pbrMetallicRoughness", {})
                    if "baseColorTexture" in pmr:
                        has_tex = True
                        break
                    if mat.get("normalTexture") or mat.get("emissiveTexture"):
                        has_tex = True
                        break
            return has_tex
    except Exception as e:
        print(f"Texture check failed for {glb_path}: {e}")
        return False


def process_batch(image_paths, clip_model, preprocess, mlp_model, device, batch_size=32, glb_name="gen_mesh_decimated_5000.glb"):
    """Process a batch of images and return predictions with paths."""
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []
        
        # Load and preprocess images
        for img_path in batch_paths:
            # Skip if paired GLB is missing or lacks texture
            glb_path = img_path.parent / glb_name
            if not glb_path.exists() or not glb_has_texture(glb_path):
                continue

            try:
                pil_image = Image.open(img_path).convert('RGB')
                image = preprocess(pil_image)
                batch_images.append(image)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into a batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Encode images with CLIP
        with torch.no_grad():
            image_features = clip_model.encode_image(batch_tensor)
        
        # Normalize embeddings
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        
        # Get aesthetic predictions
        predictions = mlp_model(torch.from_numpy(im_emb_arr).to(device).float())
        
        # Store results
        for path, pred in zip(valid_paths, predictions):
            results.append((path, pred.item()))
    
    return results


def save_results(results, input_file_pattern:str, output_dir="aesthetic_scores"):
    """Save results as txt files maintaining directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create a summary file
    summary_file = output_path / f"{input_file_pattern}_scores.txt"
    with open(summary_file, 'w') as f:
        f.write("Image Path\tAesthetic Score\n")
        for img_path, score in results:
            f.write(f"{img_path}\t{score:.4f}\n")
    
    print(f"\nSaved summary to {summary_file}")
    
    # Also save individual txt files next to each image
    for img_path, score in results:
        txt_path = img_path.with_suffix('.aesthetic_score.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Aesthetic score predicted by the model:\n")
            f.write(f"{score:.4f}\n")
    
    print(f"Saved {len(results)} individual score files")


def main():
    parser = argparse.ArgumentParser(description="Batch aesthetic scoring")
    parser.add_argument("--root-dir", default="~/devlab/integral/degen_sim/s3_backups", help="Root directory to search recursively")
    parser.add_argument("--filename", default="gen_mesh_thumbnail.png", help="Target filename to score (searched recursively)")
    parser.add_argument("--output-dir", default="aesthetic_scores", help="Directory to store summary and outputs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP encoding")
    parser.add_argument("--glb-name", default="gen_mesh_decimated_5000.glb", help="Name of GLB file colocated with each image")
    args = parser.parse_args()

    root_dir = args.root_dir
    batch_size = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    mlp_model = MLP(768)
    s = torch.load("sac+logos+ava1-l14-linearMSE.pth")
    mlp_model.load_state_dict(s)
    mlp_model.to(device)
    mlp_model.eval()
    
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    
    # Find all images
    print(f"\nSearching for {args.filename} files in {root_dir}...")
    image_paths = sorted(find_all_images(root_dir, filename=args.filename))
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return
    
    # Process images
    print(f"\nProcessing images with batch size {batch_size}...")
    results = process_batch(image_paths, clip_model, preprocess, mlp_model, device, batch_size, glb_name=args.glb_name)
    
    # Save results
    print("\nSaving results...")
    input_file_pattern = os.path.splitext(args.filename)[0]
    save_results(results, input_file_pattern, output_dir=args.output_dir)
    
    # Print statistics
    scores = [score for _, score in results]
    print(f"\nStatistics:")
    print(f"Total images processed: {len(scores)}")
    print(f"Mean aesthetic score: {np.mean(scores):.4f}")
    print(f"Std aesthetic score: {np.std(scores):.4f}")
    print(f"Min aesthetic score: {np.min(scores):.4f}")
    print(f"Max aesthetic score: {np.max(scores):.4f}")


if __name__ == "__main__":
    main()
