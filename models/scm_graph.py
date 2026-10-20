import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class SCMGraph(nn.Module):

    def __init__(self, object_dim, question_dim, hidden_dim=512, top_k=2, threshold=0.5):
        super().__init__()

        self.project_question = nn.Linear(question_dim, object_dim)
        self.edge_mlp = EdgeMLP(object_dim * 5, hidden_dim)

        self.top_k = top_k
        self.threshold = threshold

    def forward(self, objects, question):

        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if question.dim() == 1:
            question = question.unsqueeze(0)

        B, N, D = objects.shape

        # -------------------------
        # question projection
        # -------------------------
        q = self.project_question(question)

        # -------------------------
        # pair construction
        # -------------------------
        source = objects.unsqueeze(2).expand(B, N, N, D)
        target = objects.unsqueeze(1).expand(B, N, N, D)
        q_pair = q[:, None, None, :].expand(B, N, N, D)

        edge_input = torch.cat(
            [source, target, target - source, target * source, q_pair],
            dim=-1
        )

        scores = self.edge_mlp(edge_input).squeeze(-1)

        # -------------------------
        # remove self loops
        # -------------------------
        eye = torch.eye(N, device=objects.device, dtype=torch.bool)
        scores = scores.masked_fill(eye.unsqueeze(0), -1e9)

        probs = torch.sigmoid(scores)

        # -------------------------
        # SAFE TOP-K (IMPORTANT FIX)
        # -------------------------
        adjacency = torch.zeros_like(probs)

        # FIX: prevent k out of range
        k = min(self.top_k, max(N - 1, 1))

        if N > 1:
            topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
            adjacency.scatter_(-1, topk_idx, topk_vals)

        # -------------------------
        # thresholding
        # -------------------------
        adjacency = adjacency * (adjacency > self.threshold).float()

        # remove self loops again
        adjacency = adjacency * (~eye.unsqueeze(0)).float()

        # -------------------------
        # normalization
        # -------------------------
        adjacency = adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            "adjacency": adjacency,
            "probs": probs,
            "scores": scores
        }