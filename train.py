from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.PrototypicalSampler import PrototypicalSampler
from utils.PrototypicalDataset import PrototypicalDataset

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

    # Create the prototypical sampler
    prototypical_sampler = PrototypicalSampler(dataset, n_ways, k_shots, m_queries)

    # Create a DataLoader with the prototypical sampler
    batch_size = n_ways * (k_shots + m_queries)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=prototypical_sampler)

    # Now you can use 'dataloader' for your prototypical network training
