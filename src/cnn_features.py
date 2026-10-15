import os
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms


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


def extract_feature(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        feature = model(image)

    return feature.squeeze().numpy()


if __name__ == "__main__":

    image_dir = "data/images"

    feature_data = []

    for filename in sorted(os.listdir(image_dir)):

        if filename.endswith(".jpg"):

            image_path = os.path.join(
                image_dir,
                filename
            )

            feature = extract_feature(
                image_path
            )

            product_id = filename.replace(
                ".jpg",
                ""
            )

            row = {
                "product_id": product_id
            }

            for i, value in enumerate(feature):

                row[f"feature_{i}"] = float(value)

            feature_data.append(row)

            print(
                f"Processed: {filename}"
            )

    df = pd.DataFrame(feature_data)

    df.to_csv(
        "data/cnn_features.csv",
        index=False
    )

    print("\nCNN features saved successfully.")

    print(
        "Feature shape:",
        df.shape
    )