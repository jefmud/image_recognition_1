"""
Folder image "description"

by Jeff + some LLM magic (Claude code)

This program utilizes the local LLM multimodal capabilities to describe
images in a folder.  What it does is provide the filename and immedately
following, a description as seen by a multimodal LLM.

It "works" based on the capabilites of the LLM
* devstral-small-2
* ministral-3:14b (decent capabilities, given standard GPU)
* qwen3-vl:8b

These were tested, perhaps a more "edge" model will work too

Dependencies Python 3.12, Ollama and enough VRAM/CPU to load weights.

Disclaimer: Not suitable for extremely large caches of pictures,
does not recurse directories.  Use at your own risk.

"""
import requests
import json
import argparse
import os
import base64
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_image(image_path, model="ministral-3:14b"):
    url = f"{OLLAMA_URL}/api/generate"
    
    image_base64 = encode_image(image_path)
    
    payload = {
        "model": model,
        "prompt": "Describe this image in detail. What do you see?",
        "images": [image_base64],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def scan_directory(input_dir, output_file, model="mistrallite"):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"No image files found in '{input_dir}'")
        return
    
    print(f"Found {len(image_files)} image(s) to analyze...")
    
    with open(output_file, 'w') as f:
        for idx, image_path in enumerate(image_files, 1):
            print(f"Analyzing {idx}/{len(image_files)}: {image_path.name}")
            
            f.write(f"{image_path.name}\n")
            
            description = analyze_image(image_path, model)
            f.write(f"{description}\n\n")
            f.flush()
    
    print(f"\nAnalysis complete! Results saved to '{output_file}'")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze images in a directory using Ollama vision model"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing images to analyze"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output text file for descriptions"
    )
    parser.add_argument(
        "--model",
        default="ministral-3:14b",
        help="Ollama model to use (default: ministral-3:14b)"
    )
    
    args = parser.parse_args()
    
    scan_directory(args.input_dir, args.output_file, args.model)

if __name__ == "__main__":
    main()
