import random
import numpy as np
from torch.utils.data import Sampler


class PrototypicalSampler(Sampler):
    def __init__(self, data_source, n_ways, k_shots, m_queries, tasks_per_episode: int = None):
        self.data_source = data_source
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.m_queries = m_queries
        self.unique_labels = np.unique(data_source.data["label"])
        self.label_to_indices = {label: np.where(data_source.data["label"] == label)[0] for label in self.unique_labels}
        self.tasks_per_episode = tasks_per_episode if tasks_per_episode is not None else 100

    def __iter__(self):
        episode_indices = []

        for _ in range(self.n_ways):
            selected_label = random.choice(self.unique_labels)
            selected_indices = random.sample(self.label_to_indices[selected_label], self.k_shots + self.m_queries)
            episode_indices.extend(selected_indices)

        random.shuffle(episode_indices)
        return iter(episode_indices)

    def __len__(self):
        return self.tasks_per_episode
