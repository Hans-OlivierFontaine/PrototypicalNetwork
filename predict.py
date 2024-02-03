from tqdm import tqdm
import pandas as pd
from PIL import Image
import argparse
from pathlib import Path

import torch.nn
from torchvision import transforms

from utils.Classifier import classifier_factory
from utils.PrototypicalNetwork import PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Training")

    # Dataset parameters
    parser.add_argument("--support_dir", type=Path, default=Path("./data/"), help="Root directory of the support dataset")
    parser.add_argument("--pred_dir", type=Path, default=Path("./pred/"), help="Root directory of the dataset to predict")
    parser.add_argument("--imgsz", type=int, default=224, help="Size of images")
    parser.add_argument("--model", type=Path, default=Path("./data/model.pth"), help="Trained model path")
    parser.add_argument("--classifier", type=str, default="mahalanobis", help="Classifier type")

    # Sampler parameters
    parser.add_argument("--channels", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of the embedding")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    transform = transforms.Compose([transforms.Resize((args["imgsz"], args["imgsz"])), transforms.ToTensor()])

    df = pd.read_csv((args["support_dir"] / "train.csv").__str__())

    X_train = []
    y_train = []

    model = PrototypicalNetwork(input_channels=args.channels, output_dim=args.embedding_dim)

    model.load_state_dict(torch.load(args["model"]))

    for index, row in df.iterrows():
        image_path = args["support_dir"] / row['image_column_name']
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        label = row['label']

        X_train.append(image)
        y_train.append(label)

    classifier = classifier_factory('cosine_similarity')

    classifier.train(X_train, y_train)

    for image_path in args["pred_dir"].rglob("*.jpg"):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        predictions = classifier.predict([image])
