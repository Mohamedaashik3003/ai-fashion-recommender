import cv2
import os
import numpy as np
import pandas as pd


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image


def extract_features(image):
    height, width = image.shape[:2]

    average_color = np.mean(image, axis=(0, 1))
    brightness = np.mean(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )

    return {
        "width": width,
        "height": height,
        "blue": round(float(average_color[0]), 2),
        "green": round(float(average_color[1]), 2),
        "red": round(float(average_color[2]), 2),
        "brightness": round(float(brightness), 2)
    }


if __name__ == "__main__":
    image_dir = "data/images"
    feature_data = []

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            path = os.path.join(image_dir, filename)

            image = load_image(path)
            features = extract_features(image)

            product_id = filename.replace(".jpg", "")

            features["product_id"] = product_id

            feature_data.append(features)

    df = pd.DataFrame(feature_data)

    df = df[
        [
            "product_id",
            "width",
            "height",
            "blue",
            "green",
            "red",
            "brightness"
        ]
    ]

    df.to_csv("data/image_features.csv", index=False)

    print("Image features saved successfully.")
    print(df)