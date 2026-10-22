import torch
import torch.nn as nn

from configs.config import (
    HIDDEN_DIM,
    NUM_HEADS,
    OBJECT_DIM,
    QUESTION_DIM,
)

from models.causal_attention import CausalAttentionBlock
from models.intervention import InterventionEngine
from models.scm_graph import SCMGraph
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder


class QuestionAwareAnswerHead(nn.Module):
    """Y = Decoder(h, q).

    1. question-guided attention pooling over the causal node states
           alpha_i = softmax_i( <W_k h_i, W_q q> / sqrt(d) )
           v       = sum_i alpha_i W_v h_i
    2. fusion of the pooled scene evidence with the question
           z = [v, q', v * q']
    3. MLP classifier over the answer vocabulary.

    The original head (``legacy``) only saw mean(h), so the predicted answer
    did not depend on the question at all.
    """

    def __init__(self, hidden_dim: int, question_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(question_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
        self.scale = hidden_dim ** -0.5

    def forward(self, hidden: torch.Tensor, question: torch.Tensor,
                node_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if question.dim() == 1:
            question = question.unsqueeze(0)
        hidden = self.norm(hidden)
        q = self.q_proj(question)                                         # [B, D]
        scores = torch.einsum("bnd,bd->bn", self.key(hidden), self.query(q)) * self.scale
        if node_mask is not None:
            scores = scores.masked_fill(~node_mask, -1e9)
        attention = torch.softmax(scores, dim=-1)                          # [B, N]
        pooled = torch.einsum("bn,bnd->bd", attention, self.value(hidden))
        fused = torch.cat([pooled, q, pooled * q], dim=-1)
        return self.classifier(fused), attention


class SCMVLM(nn.Module):

    def __init__(self, num_classes, num_layers: int = 2, freeze_backbones: bool = True,
                 question_aware: bool = True):
        super().__init__()

        self.question_aware = question_aware

        # -------------------------
        # ENCODERS
        # -------------------------
        self.vision = VisionEncoder(freeze=freeze_backbones)
        self.text = TextEncoder(freeze=freeze_backbones)

        # -------------------------
        # PROJECTIONS
        # -------------------------
        self.project_objects = nn.Sequential(
            nn.Linear(OBJECT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

        self.project_replacement = nn.Sequential(
            nn.Linear(OBJECT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

        # -------------------------
        # GRAPH
        # -------------------------
        self.graph = SCMGraph(HIDDEN_DIM, QUESTION_DIM)

        # -------------------------
        # CAUSAL LAYERS
        # -------------------------
        self.layers = nn.ModuleList([
            CausalAttentionBlock(HIDDEN_DIM, NUM_HEADS)
            for _ in range(num_layers)
        ])

        # -------------------------
        # DECODER
        # -------------------------
        if question_aware:
            self.answer_head = QuestionAwareAnswerHead(HIDDEN_DIM, QUESTION_DIM, num_classes)
        else:
            # legacy head (checkpoints trained before the fix): mean(h) only
            self.decoder = nn.Sequential(
                nn.LayerNorm(HIDDEN_DIM),
                nn.Linear(HIDDEN_DIM, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )

        # -------------------------
        # INTERVENTION ENGINE
        # -------------------------
        self.intervention = InterventionEngine()

    # =====================================================
    # CAUSAL PROPAGATION
    # =====================================================
    def run_causal_layers(self, objects, adjacency):
        hidden = objects
        attention_maps = []

        for layer in self.layers:
            hidden, attention = layer(hidden, adjacency)
            attention_maps.append(attention)

        return hidden, attention_maps

    # =====================================================
    # DECODER  Y = Decoder(h, q)
    # =====================================================
    def decode(self, hidden, question_embedding=None, node_mask=None):
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if self.question_aware:
            if question_embedding is None:
                raise ValueError("The question-aware answer head needs the question embedding.")
            logits, _ = self.answer_head(hidden, question_embedding, node_mask)
            return logits
        pooled = hidden.mean(dim=1)
        return self.decoder(pooled)

    # =====================================================
    # FORWARD PASS
    # =====================================================
    def forward(self, image_path, question):

        vision_output = self.vision(image_path)
        question_embedding = self.text(question)

        output = self.forward_features(vision_output["embeddings"], question_embedding)
        output.update({
            "labels": vision_output.get("labels"),
            "boxes": vision_output.get("boxes"),
            "scores": vision_output.get("scores"),
        })
        return output

    def forward_features(self, raw_objects, question_embedding):
        """Same as forward() but from pre-computed DETR object embeddings
        [B, N, OBJECT_DIM] and BERT question embeddings [B, QUESTION_DIM]
        (used by the cached training loop)."""

        if raw_objects.dim() == 2:
            raw_objects = raw_objects.unsqueeze(0)
        if raw_objects.shape[1] == 0:
            raise ValueError("No objects detected in the image.")

        objects = self.project_objects(raw_objects)

        graph_output = self.graph(objects, question_embedding)

        adjacency = graph_output["adjacency"]

        hidden, attention_maps = self.run_causal_layers(objects, adjacency)

        logits = self.decode(hidden, question_embedding)

        return {
            "logits": logits,
            "adjacency": adjacency,
            "objects": objects,
            "question_embedding": question_embedding,
            "hidden": hidden,
            "attention_maps": attention_maps,
        }

    # =====================================================
    # COUNTERFACTUAL  Y_do(X_k = x')
    # =====================================================
    def counterfactual(
        self,
        objects,
        adjacency,
        node_indices,
        replacement,
        question_embedding
    ):
        """do(X_k = x'): cut the parents of k (row k), set h_k = x', keep every
        other structural equation fixed and propagate through the same graph.
        The graph is NOT rebuilt, otherwise the cut would be undone."""

        cf_objects, cf_adj, descendants = self.intervention.intervene(
            objects,
            adjacency,
            node_indices,
            replacement
        )

        cf_hidden, _ = self.run_causal_layers(cf_objects, cf_adj)

        logits = self.decode(cf_hidden, question_embedding)

        return {
            "logits": logits,
            "adjacency": cf_adj,
            "hidden": cf_hidden,
            "descendants": descendants
        }

    # =====================================================
    # PROPAGATE ONLY
    # =====================================================
    def propagate(self, objects, adjacency):
        hidden = objects
        for layer in self.layers:
            hidden, _ = layer(hidden, adjacency)
        return hidden

    # =====================================================
    # FROM OBJECTS ONLY
    # =====================================================
    def forward_from_objects(self, objects, question):

        question_embedding = self.text(question)
        output = self.forward_features(objects, question_embedding)
        return {
            "logits": output["logits"],
            "adjacency": output["adjacency"],
            "hidden": output["hidden"]
        }
