import os
import json
import re


from PIL import Image

def pad_to_square(image):
    """
    Pad an image to make it square with white pixels.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image that is square
    """
    width, height = image.size
    
    # If already square, return original
    if width == height:
        return image
    
    # Determine the size of the square (use the larger dimension)
    new_size = max(width, height)
    
    # Create a black square image
    if image.mode == 'RGB':
        padded_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))

    else:
        # Convert to RGB if not in RGB
        image = image.convert('RGB')
        padded_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))
    
    # Calculate position to paste the original image (centered)
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    
    # Paste the original image onto the square canvas
    padded_image.paste(image, (paste_x, paste_y))
    
    return padded_image


def create_json_from_image_text_folders(image_dir, text_dir, output_json_path):
    """
    Creates a JSON metadata file compatible with HuggingFace's imagefolder dataset loader
    
    Args:
        image_dir (str): Directory containing image files
        text_dir (str): Directory containing text files with same base names as images
        output_json_path (str): Path where the output JSON file will be saved
    """
    data = []
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for image_file in image_files:
       # Load and pad the image
        image_path = os.path.join(image_dir, image_file)
        try:
            image = Image.open(image_path)
            original_format = image.format  # Preserve original format
            padded_image = pad_to_square(image)
            
            # Save the padded image back to the original file
            padded_image.save(image_path, format=original_format)
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
        # Get base name without extension
        base_name = os.path.splitext(image_file)[0]
        
        # Corresponding text file path
        text_file = f"{base_name}.txt"
        text_file_path = os.path.join(text_dir, text_file)
        
        # Check if text file exists
        if os.path.isfile(text_file_path):
            try:
                # Read text content
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                # Create entry for this image-text pair
                # Note: The structure needs to match what imagefolder expects
                entry = {
                    "file_name": image_file,
                    "text": text_content
                }
                
                data.append(entry)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
    
    # Write each JSON object on a separate line (JSON Lines format)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created JSON metadata file with {len(data)} entries at {output_json_path}")


if __name__ == "__main__":
    # Set your directories here
    image_directory = "data//im"   # Directory with image files
    text_directory = "data//lb"    # Directory with text files
    output_json = "data//im//metadata.jsonl"  # Output JSON file
    
    # Create the JSON file
    create_json_from_image_text_folders(image_directory, text_directory, output_json)
    


