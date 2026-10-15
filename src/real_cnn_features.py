import os
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms


INPUT_FILE = "data/fashion_subset.csv"
OUTPUT_FILE = "data/real_cnn_features.csv"

device = torch.device("cpu")


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


df = pd.read_csv(INPUT_FILE)

feature_data = []

for index, row in df.iterrows():

    image_path = row["image_path"]

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        with torch.no_grad():

            feature = model(
                image
            )

        feature = feature.squeeze().numpy()

        result = {
            "product_name": row["product_name"],
            "image_path": image_path,
            "category": row["category"]
        }

        for i, value in enumerate(feature):

            result[f"feature_{i}"] = float(value)

        feature_data.append(result)

        if (index + 1) % 100 == 0:

            print(
                f"Processed: {index + 1}/{len(df)}"
            )

    except Exception as error:

        print(
            f"Error processing {image_path}: {error}"
        )


features_df = pd.DataFrame(
    feature_data
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nReal CNN features saved successfully.")
print(
    "Feature shape:",
    features_df.shape
)