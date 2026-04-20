"""
Image processor module for extracting and processing images from PDFs.
Encodes images as base64 for LLM processing.
"""
import base64
from typing import List, Dict
from io import BytesIO
from PIL import Image


class ImageProcessor:
    """Process images extracted from PDFs."""

    def __init__(self):
        pass

    def process_images(self, images: List[Dict]) -> List[Dict]:
        """
        Process extracted images, encode to base64.
        
        Args:
            images: List of image dicts from PDFParser
            
        Returns:
            Processed image dictionaries ready for ChromaDB
        """
        processed = []
        for img_dict in images:
            try:
                # Encode bytes to base64
                b64_string = base64.b64encode(img_dict["bytes"]).decode("utf-8")
                
                # Get image info
                try:
                    img = Image.open(BytesIO(img_dict["bytes"]))
                    width, height = img.size
                except:
                    width, height = None, None
                
                processed_img = {
                    "page": img_dict["page"],
                    "image_index": img_dict["image_index"],
                    "xref": img_dict["xref"],
                    "base64": b64_string,
                    "width": width,
                    "height": height,
                    "description": None,  # Will be filled by Gemini
                }
                processed.append(processed_img)
            except Exception as e:
                print(f"Error processing image: {e}")
        
        return processed

    def get_image_info_text(self, image_dict: Dict) -> str:
        """
        Generate info text for an image (for metadata).
        
        Args:
            image_dict: Processed image dictionary
            
        Returns:
            Text describing the image
        """
        dimensions = f"{image_dict['width']}x{image_dict['height']}" if image_dict['width'] else "unknown"
        
        text = f"Image from Page {image_dict['page']}: {dimensions} pixels"
        if image_dict.get('description'):
            text += f"\nDescription: {image_dict['description']}"
        
        return text


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Create a simple test image
    from PIL import Image as PILImage
    
    test_img = PILImage.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    sample_image = {
        "page": 1,
        "image_index": 0,
        "xref": 10,
        "bytes": img_bytes.read(),
    }
    
    # Process
    processed = processor.process_images([sample_image])
    print(f"Processed image: page={processed[0]['page']}, size={processed[0]['width']}x{processed[0]['height']}")
