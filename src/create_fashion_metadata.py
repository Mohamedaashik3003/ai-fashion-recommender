import os
import pandas as pd


IMAGE_ROOT = "data/real_images/img"
OUTPUT_FILE = "data/fashion_metadata.csv"


records = []

for product_name in sorted(os.listdir(IMAGE_ROOT)):

    product_folder = os.path.join(
        IMAGE_ROOT,
        product_name
    )

    if not os.path.isdir(product_folder):
        continue

    for filename in os.listdir(product_folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                product_folder,
                filename
            )

            records.append({
                "product_name": product_name,
                "image_path": image_path
            })


df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Fashion metadata created successfully.")
print("Total images:", len(df))
print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))