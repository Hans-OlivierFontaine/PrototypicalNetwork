from tqdm import tqdm
import argparse
from pathlib import Path

import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.PrototypicalSampler import PrototypicalSampler
from utils.PrototypicalDataset import PrototypicalDataset
from utils.batch_conversion import split_batch_to_support_query
from utils.PrototypicalLoss import prototypical_loss
from utils.PrototypicalNetwork import PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Training")

    # Dataset parameters
    parser.add_argument("--root_dir", type=Path, default=Path("./data/"), help="Root directory of the dataset")
    parser.add_argument("--imgsz", type=int, default=224, help="Size of images")

    # Sampler parameters
    parser.add_argument("--n_ways", type=int, default=5, help="Number of classes in few-shot task")
    parser.add_argument("--k_shots", type=int, default=5, help="Number of support examples per class")
    parser.add_argument("--m_queries", type=int, default=5, help="Number of query examples per class")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--tasks_per_episodes", type=int, default=100, help="Number of tasks per episode")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"],
                        help="Distance metric")
    parser.add_argument("--channels", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of the embedding")

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_save", type=Path, default=Path("./data/model.pth"), help="Trained model path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Define transformations for your images
    transform = transforms.Compose([transforms.Resize((args["imgsz"], args["imgsz"])), transforms.ToTensor()])

    # Create the dataset
    dataset = PrototypicalDataset(root_dir=args.root_dir, csv_name="train.csv", transform=transform)

    model = PrototypicalNetwork(input_channels=args.channels, output_dim=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Create the prototypical sampler
    prototypical_sampler = PrototypicalSampler(dataset, args.n_ways, args.k_shots, args.m_queries,
                                               args.tasks_per_episodes)

    # Create a DataLoader with the prototypical sampler
    batch_size = args.n_ways * (args.k_shots + args.m_queries)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=prototypical_sampler)

    for no_episode in range(1, args.num_episodes + 1):
        for batch in tqdm(dataloader, desc=f"Training {no_episode}/{args.num_episodes}"):
            support_set, query_set = split_batch_to_support_query(batch, args.n_ways, args.k_shots, args.m_queries)
            optimizer.zero_grad()
            support_embeddings, query_embeddings = model(support_set), model(query_set)
            loss = prototypical_loss(support_embeddings, query_embeddings, args.n_ways, args.k_shots, args.metric)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args["model_save"])
