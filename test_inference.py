"""
Run the Roboflow model on a single image and print the JSON predictions.
Also saves an annotated copy next to the original.

Usage:
    python test_inference.py path/to/image.jpg
    python test_inference.py https://example.com/drone.jpg

Requires ROBOFLOW_API_KEY in your environment (or in a .env file).
"""
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

import cv2
import detector


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <image_path_or_url>")
        sys.exit(1)

    image = sys.argv[1]
    print(f"Model: {detector.MODEL_ID}")
    print(f"Running inference on: {image}\n")

    predictions = detector.infer(image)
    print(json.dumps(predictions, indent=2))
    print(f"\nDetections: {len(predictions)}")

    # Save an annotated copy for local files
    if os.path.exists(image):
        img = cv2.imread(image)
        if img is not None:
            annotated = detector.draw(img, predictions)
            out = os.path.splitext(image)[0] + "_annotated.jpg"
            cv2.imwrite(out, annotated)
            print(f"Annotated image saved to: {out}")


if __name__ == "__main__":
    main()
