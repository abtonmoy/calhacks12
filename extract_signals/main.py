import easyocr
import anthropic
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_text(image_path):
    """Extract all text from an image"""
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image_path)
    text = ' '.join([text for (bbox, text, conf) in results])
    return text

def get_signals(text):
    """Send text to Claude API and get marketing signals"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Analyze this text and return ONLY a JSON object (no markdown formatting):

Text: {text}

Instructions:
- brand_name_text: The brand/company name
- product_name: The specific product or service being advertised
- industry: The business category
- promo_present: true if there's a promotional offer, false otherwise
- promo_text: ONLY the core offer itself (e.g., "1¢ to join & get 1 month free", "Buy one get one free", "50% off"), NOT the full sentence or description
- promo_deadline: Time limit for the offer (e.g., "48 hours", "ends today", "limited time")
- price_value: Specific price mentioned (e.g., "$9.99/mo", "$0.01 down")
- cta_present: true if there's a call-to-action button/text, false otherwise
- cta_type: The action requested (e.g., "Sign up button", "Order now", "Learn more")
- text_density: "low" (minimal text), "medium" (moderate text), "high" (lots of text)
- brand_text_contrast: "low", "medium", or "high" - how much the brand name stands out visually
- is_nsfw: true if the content contains explicit sexual content, graphic violence, or other not-safe-for-work material, false otherwise

Return format:
{{
    "brand_name_text": "",
    "product_name": "",
    "industry": "",
    "promo_present": true/false,
    "promo_text": "",
    "promo_deadline": "",
    "price_value": "",
    "cta_present": true/false,
    "cta_type": "",
    "text_density": "low/medium/high",
    "brand_text_contrast": "low/medium/high",
    "is_nsfw": true/false
}}"""
        }]
    )
    
    response = message.content[0].text.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1])
    
    return json.loads(response)

if __name__ == "__main__":
    text = extract_text('data/images/i0015.png')
    response = get_signals(text=text)
    print(response)