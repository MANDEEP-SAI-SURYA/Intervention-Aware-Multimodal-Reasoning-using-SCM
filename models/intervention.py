import torch

from configs.config import EDGE_THRESHOLD


def find_descendants(adjacency: torch.Tensor, node_index: int, threshold: float = EDGE_THRESHOLD) -> list[int]:
    graph = adjacency.detach().cpu()
    seen: set[int] = set()
    queue = [node_index]

    while queue:
        current = queue.pop(0)
        children = torch.nonzero(graph[current] > threshold, as_tuple=False).flatten().tolist()
        for child in children:
            if child != node_index and child not in seen:
                seen.add(child)
                queue.append(child)

    return sorted(seen)


class InterventionEngine:
    """Implements do(X_k = x'): cut incoming edges and replace the node embedding."""

    def intervene(
        self,
        objects: torch.Tensor,
        adjacency: torch.Tensor,
        node_indices: torch.Tensor,
        replacement_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if node_indices.dim() == 0:
            node_indices = node_indices.unsqueeze(0)
        if replacement_embeddings.dim() == 1:
            replacement_embeddings = replacement_embeddings.unsqueeze(0)

        intervened_objects = objects.clone()
        intervened_adjacency = adjacency.clone()
        descendant_sets: list[list[int]] = []

        for batch_index in range(objects.shape[0]):
            node = int(node_indices[batch_index].item())
            if node < 0 or node >= objects.shape[1]:
                raise IndexError(f"node index {node} is outside the object range")

            # do(X_k = x') removes every parent -> k influence.
            intervened_adjacency[batch_index, :, node] = 0.0
            intervened_objects[batch_index, node] = replacement_embeddings[batch_index]

            # Keep the causal propagation graph normalized after the cut.
            row_sum = intervened_adjacency[batch_index].sum(dim=-1, keepdim=True)
            intervened_adjacency[batch_index] = intervened_adjacency[batch_index] / (row_sum + 1e-8)
            descendant_sets.append(find_descendants(intervened_adjacency[batch_index], node))

        return intervened_objects, intervened_adjacency, descendant_sets
