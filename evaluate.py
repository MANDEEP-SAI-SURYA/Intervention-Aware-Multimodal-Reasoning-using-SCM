import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from configs.config import DEVICE
from models.scm_vlm import SCMVLM
from utils.visualization import visualize_graph

DEMO_NUM_CLASSES = 3_129


def main() -> None:

    model = SCMVLM(num_classes=DEMO_NUM_CLASSES).to(DEVICE)

    model.eval()

    image_path = "images/Man_riding_bicycle.png"

    question = "Is the traffic light red?"

    with torch.no_grad():

        output = model(image_path, question)

    prediction = torch.argmax(
        output["logits"],
        dim=-1
    )

    print("\nPrediction:")
    print(prediction.item())

    # =========================
    # SAVE GRAPH IMAGE
    # =========================

    visualize_graph(
        output["adjacency"],
        save_path="images/causal_graph.png"
    )

    print(
        "\nSaved graph visualization "
        "to images/causal_graph.png"
    )

    # =========================
    # PRINT TOP CAUSAL RELATIONS
    # =========================

    adj = output["adjacency"][0]

    labels = output["labels"][0]

    print("\n=========================")
    print("TOP CAUSAL RELATIONS")
    print("=========================\n")

    for i in range(len(labels)):

        row = adj[i]

        top_indices = torch.argsort(
            row,
            descending=True
        )[:3]

        printed = 0

        for j in top_indices:

            j = j.item()

            if i == j:
                continue

            print(
                f"{labels[i]} ---> "
                f"{labels[j]} : "
                f"{adj[i][j].item():.3f}"
            )

            printed += 1

            if printed >= 2:
                break


if __name__ == "__main__":
    main()
