import torch
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor
)

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

def display_image_with_text(image_path, generated_text):
    """Display image and generated text side by side"""
    try:
        img = Image.open(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Input Image: {os.path.basename(image_path)}", fontsize=12)
        
        # Display text
        ax2.text(0.1, 0.5, generated_text, fontsize=14, wrap=True, 
                ha='left', va='center', transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title("Generated Description", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Print image info
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
    except Exception as e:
        print(f"Error displaying image with text: {e}")
    """Load and return PIL Image from path"""
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def load_image_from_path(image_path):
    """Load and return PIL Image from path"""
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def generate_text_from_image(image_path, checkpoint_dir, device='cuda:0'):
    """Generate text description from image using trained model"""
    
    # Load image
    img = load_image_from_path(image_path)
    if img is None:
        return None
    
    # Initialize feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/swin-base-patch4-window7-224-in22k"
    )
    
    # Process image
    pixel_values = feature_extractor(
        images=img,
        return_tensors="pt",
    ).pixel_values.to(device)  # Move to device and keep batch dimension
    
    # Load model and tokenizer
    try:
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        print(f"Loaded model from: {checkpoint_dir}")
        print(f"Using device: {device}")
        
    except Exception as e:
        print(f"Error loading model from {checkpoint_dir}: {e}")
        return None
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values, 
            num_beams=4, 
            max_length=256, 
            early_stopping=True
        )
    
    # Decode generated text
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_texts[0] if generated_texts else None


def inference_image(image_path,checkpoint_dir):
    """Process a single image"""
    # Configuration
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Generate text
    result = generate_text_from_image(image_path, checkpoint_dir, device)
    
    if result:
        display_image_with_text(image_path,result)
    else:
        print("Failed to generate text")


if __name__ == "__main__":
    image_path = "data/infer/sysml1.JPG"
    checkpoint_dir = "checkpoints/checkpoint_epoch_60_step_180"
    # Check if specific image exists
    if os.path.exists(image_path):
        inference_image(image_path,checkpoint_dir)