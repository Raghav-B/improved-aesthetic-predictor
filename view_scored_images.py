import cv2
import numpy as np
from pathlib import Path
import sys


def load_scores(scores_file="aesthetic_scores/all_scores.txt"):
    """Load image paths and scores from the summary file."""
    scores_dict = {}
    
    with open(scores_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_path = Path(parts[0])
                score = float(parts[1])
                scores_dict[str(img_path)] = score
    
    return scores_dict


def put_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                             font_scale=0.8, font_thickness=2, 
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility."""
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    x, y = position
    # Draw background rectangle
    padding = 10
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  bg_color, 
                  -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)
    
    return text_height + baseline + padding


def display_image_with_score(img_path, score, current_idx, total_images):
    """Load and display an image with its score and navigation info."""
    # Read image
    img = cv2.imread(str(img_path))
    
    if img is None:
        # Create a placeholder for missing images
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        error_text = f"Error loading image"
        cv2.putText(img, error_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Create a larger canvas if image is small
    h, w = img.shape[:2]
    max_height = 900
    max_width = 1600
    
    # Resize if image is too large
    if h > max_height or w > max_width:
        scale = min(max_height / h, max_width / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = img.shape[:2]
    
    # Add padding at top for text
    padding_top = 100
    display_img = np.zeros((h + padding_top, w, 3), dtype=np.uint8)
    display_img[padding_top:, :] = img
    
    # Draw score
    score_text = f"Aesthetic Score: {score:.4f}"
    put_text_with_background(display_img, score_text, (10, 30), 
                            font_scale=1.0, font_thickness=2,
                            text_color=(0, 255, 255))
    
    # Draw image counter
    counter_text = f"Image {current_idx + 1}/{total_images}"
    put_text_with_background(display_img, counter_text, (10, 70), 
                            font_scale=0.8, font_thickness=2,
                            text_color=(0, 255, 0))
    
    # Draw filename (truncated if too long)
    filename = str(img_path)
    if len(filename) > 100:
        filename = "..." + filename[-97:]
    put_text_with_background(display_img, filename, (10, h + padding_top - 10), 
                            font_scale=0.5, font_thickness=1,
                            text_color=(200, 200, 200))
    
    return display_img


def main():
    # Load scores
    scores_file = "aesthetic_scores/all_scores.txt"
    
    if not Path(scores_file).exists():
        print(f"Error: {scores_file} not found. Please run batch_inference.py first.")
        return
    
    print("Loading scores...")
    scores_dict = load_scores(scores_file)
    
    # Get sorted list of image paths
    image_paths = sorted(list(scores_dict.keys()))
    total_images = len(image_paths)
    
    if total_images == 0:
        print("No images found in scores file.")
        return
    
    print(f"Loaded {total_images} images with scores")
    print("\nControls:")
    print("  Left Arrow  - Previous image")
    print("  Right Arrow - Next image")
    print("  'q' or ESC  - Quit")
    print("  Home        - First image")
    print("  End         - Last image")
    
    # Start viewer
    current_idx = 0
    window_name = "Aesthetic Score Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        # Get current image and score
        img_path = image_paths[current_idx]
        score = scores_dict[img_path]
        
        # Display image with score
        display_img = display_image_with_score(img_path, score, current_idx, total_images)
        cv2.imshow(window_name, display_img)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Handle navigation
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break
        elif key == 81 or key == 2:  # Left arrow (key code may vary by system)
            current_idx = max(0, current_idx - 1)
        elif key == 83 or key == 3:  # Right arrow
            current_idx = min(total_images - 1, current_idx + 1)
        elif key == 82:  # Up arrow (go to first)
            current_idx = 0
        elif key == 84:  # Down arrow (go to last)
            current_idx = total_images - 1
        elif key == ord('h'):  # Home
            current_idx = 0
        elif key == ord('e'):  # End
            current_idx = total_images - 1
        
        # Print current position for debugging
        print(f"Viewing image {current_idx + 1}/{total_images} - Score: {score:.4f}")
    
    cv2.destroyAllWindows()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
