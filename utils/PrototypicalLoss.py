import torch


def prototypical_loss(support_set, query_set, n_ways, k_shots, distance_metric="euclidean"):
    """
    Calculate the prototypical loss for few-shot learning.

    Args:
    - support_set (Tensor): Tensor containing support set data of shape (n_ways, k_shots, feature_dim).
    - query_set (Tensor): Tensor containing query set data of shape (n_ways, m_queries, feature_dim).
    - n_ways (int): Number of classes in the few-shot task.
    - k_shots (int): Number of support examples per class.
    - distance_metric (str): Metric to use for computing distances ('euclidean' or 'cosine').

    Returns:
    - prototypical_loss (Tensor): Prototypical loss scalar.
    """

    # Calculate prototypes for each class
    prototypes = torch.mean(support_set, dim=1)  # Shape: (n_ways, feature_dim)

    # Compute distances between query examples and prototypes
    if distance_metric == "euclidean":
        distances = torch.cdist(query_set, prototypes, p=2)  # Euclidean distance
    elif distance_metric == "cosine":
        support_norm = torch.norm(support_set, dim=2, keepdim=True)
        query_norm = torch.norm(query_set, dim=2, keepdim=True)
        distances = 1 - torch.bmm(query_set, prototypes.unsqueeze(2)) / (query_norm * support_norm)

    # Calculate softmax over distances to get class probabilities
    softmax_logits = -distances  # Negative distances for softmax
    softmax_logits = softmax_logits - torch.max(softmax_logits, dim=1, keepdim=True)[0]  # Stability
    softmax_probs = torch.nn.functional.softmax(softmax_logits, dim=1)

    # Calculate log probabilities for the true classes
    true_class_idx = torch.arange(n_ways).unsqueeze(1).expand(-1, query_set.size(1)).to(query_set.device)
    log_probs = torch.log(softmax_probs.gather(1, true_class_idx))

    # Calculate the loss as the negative log likelihood
    prototypical_loss = -log_probs.mean()

    return prototypical_loss
