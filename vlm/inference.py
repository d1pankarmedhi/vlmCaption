import torch
from PIL import Image
from .data import get_transforms

def generate_caption(model, image_path, device, max_length=50, temperature=0.7):
    """
    Generate a caption for a single image file.
    """
    model.eval()
    transform = get_transforms()
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Call model generation
    # Ensure dimensions
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    caption = model.generate_caption(
        image_tensor, 
        max_length=max_length, 
        temperature=temperature, 
    )
    
    return caption
