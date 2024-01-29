from tqdm import tqdm

import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.PrototypicalSampler import PrototypicalSampler
from utils.PrototypicalDataset import PrototypicalDataset
from utils.batch_conversion import split_batch_to_support_query
from utils.PrototypicalLoss import prototypical_loss
from utils.PrototypicalNetwork import PrototypicalNetwork

if __name__ == "__main__":
    # Specify your CSV file and root directory
    csv_file = "your_dataset.csv"
    root_dir = "/path/to/your/dataset"

    # Define transformations for your images
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Create the dataset
    dataset = PrototypicalDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Define sampler parameters
    n_ways = 5
    k_shots = 5
    m_queries = 5
    num_episodes = 10
    tasks_per_episodes = 100
    metric = "euclidian"
    channels = 3
    embedding_dim = 256
    lr = 0.001

    model = PrototypicalNetwork(input_channels=channels, output_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    # Create the prototypical sampler
    prototypical_sampler = PrototypicalSampler(dataset, n_ways, k_shots, m_queries, tasks_per_episodes)

    # Create a DataLoader with the prototypical sampler
    batch_size = n_ways * (k_shots + m_queries)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=prototypical_sampler)

    for no_episode in range(1, num_episodes + 1):
        for batch in dataloader:
            support_set, query_set = split_batch_to_support_query(batch, n_ways, k_shots, m_queries)
            optimizer.zero_grad()
            support_embeddings, query_embeddings = model(support_set), model(query_set)
            loss = prototypical_loss(support_embeddings, query_embeddings, n_ways, k_shots, metric)
            loss.backward()
            optimizer.step()
