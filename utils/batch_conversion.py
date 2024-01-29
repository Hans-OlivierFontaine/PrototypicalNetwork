import torch


def split_batch_to_support_query(batch, n_ways, k_shots, m_queries):
    """
    Split a batch into a support set and a query set for few-shot learning.

    Args:
    - batch (tuple): A batch of data from the DataLoader, where the first element is 'images' and the second element is 'labels'.
    - n_ways (int): Number of classes in the few-shot task.
    - k_shots (int): Number of support examples per class.
    - m_queries (int): Number of query examples per class.

    Returns:
    - support_set (Tensor): Support set data of shape (n_ways, k_shots, feature_dim).
    - query_set (Tensor): Query set data of shape (n_ways, m_queries, feature_dim).
    """

    # Unpack the batch
    images, labels = batch

    # Determine the number of unique classes in the batch
    unique_labels = torch.unique(labels)

    # Randomly select n_ways classes from the unique labels
    selected_classes = torch.randperm(unique_labels.size(0))[:n_ways]

    # Initialize support and query sets
    support_set = []
    query_set = []

    for class_idx in selected_classes:
        # Select images and labels for the current class
        class_mask = (labels == unique_labels[class_idx])
        class_images = images[class_mask]
        class_labels = labels[class_mask]

        # Randomly shuffle the indices to select k_shots and m_queries
        shuffled_indices = torch.randperm(class_images.size(0))

        # Split the indices into support and query indices
        support_indices = shuffled_indices[:k_shots]
        query_indices = shuffled_indices[k_shots:(k_shots + m_queries)]

        # Select k_shots and m_queries examples for support and query sets
        support_set.append(class_images[support_indices])
        query_set.append(class_images[query_indices])

    # Stack tensors to create support and query sets
    support_set = torch.stack(support_set, dim=0)
    query_set = torch.stack(query_set, dim=0)

    return support_set, query_set
