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


class SCMVLM(nn.Module):

    def __init__(self, num_classes, num_layers: int = 2, freeze_backbones: bool = True):
        super().__init__()

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
    # DECODER
    # =====================================================
    def decode(self, hidden):
        pooled = hidden.mean(dim=1)
        return self.decoder(pooled)

    # =====================================================
    # FORWARD PASS
    # =====================================================
    def forward(self, image_path, question):

        vision_output = self.vision(image_path)

        objects = vision_output["embeddings"]

        # ensure [B, N, D]
        if objects.dim() == 2:
            objects = objects.unsqueeze(0)

        objects = self.project_objects(objects)

        question_embedding = self.text(question)

        graph_output = self.graph(objects, question_embedding)

        adjacency = graph_output["adjacency"]

        hidden, attention_maps = self.run_causal_layers(objects, adjacency)

        logits = self.decode(hidden)

        return {
            "logits": logits,
            "adjacency": adjacency,
            "objects": objects,
            "labels": vision_output.get("labels"),
            "boxes": vision_output.get("boxes"),
            "scores": vision_output.get("scores"),
            "question_embedding": question_embedding,
            "hidden": hidden,
            "attention_maps": attention_maps,
        }

    # =====================================================
    # COUNTERFACTUAL
    # =====================================================
    def counterfactual(
        self,
        objects,
        adjacency,
        node_indices,
        replacement,
        question_embedding
    ):

        # intervention
        cf_objects, cf_adj, descendants = self.intervention.intervene(
            objects,
            adjacency,
            node_indices,
            replacement
        )

        # recompute graph
        cf_graph = self.graph(cf_objects, question_embedding)
        cf_adj = cf_graph["adjacency"]

        # propagation
        cf_hidden, _ = self.run_causal_layers(cf_objects, cf_adj)

        logits = self.decode(cf_hidden)

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
        objects = self.project_objects(objects)

        graph_output = self.graph(objects, question_embedding)
        adjacency = graph_output["adjacency"]

        hidden, _ = self.run_causal_layers(objects, adjacency)

        logits = self.decode(hidden)

        return {
            "logits": logits,
            "adjacency": adjacency,
            "hidden": hidden
        }