"""
Simple OCR Text Extractor
extracts raw text from images
"""

import easyocr


def extract_text(image_path):
    """
    Extract all text from an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        String containing all detected text
    """
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image_path)
    
    # Combine all text into one string
    text = ' '.join([text for (bbox, text, conf) in results])
    
    return text


# Usage
if __name__ == "__main__":
    # Extract text from image
    text = extract_text('data/images/i0015.png')
    
    # Print the text
    print(text)