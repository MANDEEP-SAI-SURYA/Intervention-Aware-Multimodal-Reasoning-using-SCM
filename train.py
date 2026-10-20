import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.backends.cudnn.benchmark = True

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np

from configs.config import DEVICE
from models.scm_vlm import SCMVLM
from datasets.datasetBuilder import VQACOCO


# =====================================================
# SETTINGS
# =====================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "scm_vlm_latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 4
NUM_WORKERS = 0
TRAIN_SAMPLES = 2000
SAVE_LAST_N_RESULTS = 16
LEARNING_RATE = 5e-6
USE_COUNTERFACTUAL_PROB = 0.5

TOP_K = 3
RANDOM_NODE_PROB = 0.3

LAMBDA_CF = 0.5
LAMBDA_SPARSE = 0.1
LAMBDA_ACYCLIC = 0.2
LAMBDA_SHIFT = 0.3

print(f"\nUSING DEVICE: {DEVICE}")


# =====================================================
# COLLATE
# =====================================================

def collate_fn(batch):
    images, questions, labels = zip(*batch)
    return list(images), list(questions), torch.stack(labels)


# =====================================================
# SAFE HELPERS
# =====================================================

def safe_scalar(x):
    try:
        if torch.is_tensor(x):
            if x.numel() == 1:
                return float(x.item())
            return float(x.mean().item())
        return float(x)
    except:
        return 0.0


def safe_label(x):
    try:
        return str(x)
    except:
        return "unknown"


def save_checkpoint(model, dataset, epoch, accuracy):
    """Save learned layers and the answer vocabulary for standalone inference.

    DETR and BERT are frozen pretrained backbones, so they are loaded again
    from Hugging Face at inference time instead of being copied into every
    checkpoint file.
    """
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }

    torch.save({
        "format_version": 1,
        "epoch": epoch,
        "training_accuracy": accuracy,
        "num_classes": len(dataset.answer2idx),
        "idx2answer": dataset.idx2answer,
        "trainable_state_dict": trainable_state,
    }, CHECKPOINT_PATH)
    print(f"Saved trained checkpoint: {CHECKPOINT_PATH}")


# =====================================================
# STABLE DAG LOSS
# =====================================================

def stable_acyclicity_loss(A):
    try:
        A = A.float()

        while A.dim() > 2:
            A = A[0]

        if A.shape[0] != A.shape[1]:
            return torch.tensor(0.0, device=DEVICE)

        A = torch.clamp(A, 0.0, 1.0)

        A2 = A @ A

        return torch.norm(A2, p="fro")

    except:
        return torch.tensor(0.0, device=DEVICE)


# =====================================================
# DATASET
# =====================================================

full_dataset = VQACOCO(
    image_dir="coco/train2014/train2014",
    question_file="coco/vqa/v2_OpenEnded_mscoco_train2014_questions.json",
    annotation_file="coco/vqa/v2_mscoco_train2014_annotations.json"
)

dataset = Subset(full_dataset, list(range(min(TRAIN_SAMPLES, len(full_dataset)))))

NUM_CLASSES = len(full_dataset.answer2idx)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    # Windows uses spawn for workers.  Each spawned process would reload the
    # large VQA JSON files and can exhaust RAM, so keep loading in-process.
    pin_memory=(DEVICE == "cuda"),
    num_workers=NUM_WORKERS
)


# =====================================================
# MODEL
# =====================================================

model = SCMVLM(num_classes=NUM_CLASSES).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =====================================================
# TRAINING (FULLY FIXED + STABLE + IMPROVED)
# =====================================================

model.train()

print("\n========================")
print("TRAINING STARTED (FIXED + SCM UPGRADED)")
print("========================")

for epoch in range(EPOCHS):

    total_loss = 0
    correct = 0
    total = 0

    for images, questions, labels in dataloader:

        labels = labels.to(DEVICE)

        for i in range(len(images)):

            try:
                optimizer.zero_grad()

                # =====================
                # FACTUAL FORWARD
                # =====================
                out = model(images[i], questions[i])

                logits = torch.clamp(out["logits"], -30, 30)
                pred = torch.argmax(logits, dim=-1)

                correct += (pred == labels[i]).item()

                loss_task = F.cross_entropy(
                    logits,
                    labels[i].unsqueeze(0)
                )

                objects = out["objects"]
                adjacency = torch.clamp(out["adjacency"].float(), 0.0, 1.0)
                q_emb = out["question_embedding"]

                # =====================
                # GRAPH REGULARIZATION
                # =====================

                loss_sparse = torch.mean(adjacency)

                A = adjacency
                while A.dim() > 2:
                    A = A[0]

                loss_acyclic = stable_acyclicity_loss(A)

                edge_prob = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
                loss_entropy = -torch.mean(edge_prob * torch.log(edge_prob + 1e-6))

                # =====================
                # COUNTERFACTUAL LOSS
                # =====================

                loss_cf = torch.tensor(0.0, device=DEVICE)
                loss_shift = torch.tensor(0.0, device=DEVICE)
                loss_consistency = torch.tensor(0.0, device=DEVICE)

                if objects.shape[1] > 0:

                    node_scores = A.mean(dim=-1)

                    k = min(TOP_K, node_scores.shape[0])
                    _, topk_idx = torch.topk(node_scores, k=k)

                    if torch.rand(1).item() < RANDOM_NODE_PROB:
                        node_idx = torch.randint(
                            0,
                            objects.shape[1],
                            (1,),
                            device=DEVICE
                        )
                    else:
                        node_idx = topk_idx[torch.randint(0, k, (1,))].view(1)

                    intervention_strength = torch.rand(1).item() * 0.2

                    replacement = torch.randn_like(objects[:, 0]) * intervention_strength

                    cf_out = model.counterfactual(
                        objects,
                        adjacency,
                        node_idx,
                        replacement,
                        q_emb
                    )

                    cf_logits = torch.clamp(cf_out["logits"], -30, 30)

                    loss_cf = F.kl_div(
                        F.log_softmax(cf_logits, dim=-1),
                        F.softmax(logits.detach(), dim=-1),
                        reduction="batchmean"
                    ) * (1.0 + intervention_strength)

                    loss_shift = torch.mean(
                        torch.abs(
                            F.softmax(cf_logits, dim=-1)
                            - F.softmax(logits.detach(), dim=-1)
                        )
                    )

                    if "question_embedding" in cf_out:
                        loss_consistency = F.mse_loss(
                            cf_out["question_embedding"],
                            q_emb.detach()
                        )

                # =====================
                # FINAL LOSS
                # =====================

                loss = (
                    loss_task
                    + LAMBDA_SPARSE * loss_sparse
                    + LAMBDA_ACYCLIC * loss_acyclic
                    + 0.05 * loss_entropy
                    + LAMBDA_CF * loss_cf
                    + LAMBDA_SHIFT * loss_shift
                    + 0.1 * loss_consistency
                )

                if torch.isnan(loss):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                total += 1

                if total % 5 == 0:
                    print(
                        f"[TRAIN] Epoch={epoch+1} "
                        f"Sample={total} "
                        f"Loss={loss.item():.4f} "
                        f"Task={loss_task.item():.4f} "
                        f"CF={loss_cf.item():.4f} "
                        f"Shift={loss_shift.item():.4f}"
                    )

            except Exception as e:
                print(f"TRAIN ERROR: {e}")

    accuracy = correct / max(total, 1)

    print("\n========================")
    print(f"EPOCH {epoch+1}")
    print("========================")
    print(f"Loss     : {total_loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    save_checkpoint(model, full_dataset, epoch + 1, accuracy)


# =====================================================
# EVALUATION (UNCHANGED)
# =====================================================

model.eval()

factual_acc = 0
cf_acc = 0
eval_total = 0
shift_list = []

print("\n========================")
print("EVALUATION STARTED")
print("========================")

with torch.no_grad():

    for images, questions, labels in dataloader:

        labels = labels.to(DEVICE)

        for i in range(len(images)):

            try:
                out = model(images[i], questions[i])

                logits = torch.clamp(out["logits"], -30, 30)

                pred = torch.argmax(logits, dim=-1)

                factual_acc += (pred == labels[i]).item()

                objects = out["objects"]

                if objects.shape[1] == 0:
                    continue

                adjacency = torch.clamp(out["adjacency"], 0.0, 1.0)

                q_emb = out["question_embedding"]

                A = adjacency
                while A.dim() > 2:
                    A = A[0]

                node_scores = A.mean(dim=-1)
                node_idx = torch.argmax(node_scores).view(1)

                replacement = torch.randn_like(objects[:, 0]) * 0.05

                cf_out = model.counterfactual(
                    objects,
                    adjacency,
                    node_idx,
                    replacement,
                    q_emb
                )

                cf_logits = torch.clamp(cf_out["logits"], -30, 30)

                cf_pred = torch.argmax(cf_logits, dim=-1)

                cf_acc += (cf_pred == labels[i]).item()

                shift_list.append(torch.abs(logits - cf_logits).mean().item())

                eval_total += 1

                if eval_total % 5 == 0:
                    print(f"[EVAL] Processed {eval_total}")

            except Exception as e:
                print(f"EVAL ERROR: {e}")


# =====================================================
# FINAL RESULTS
# =====================================================

factual_accuracy = 100.0 * factual_acc / max(eval_total, 1)
counterfactual_accuracy = 100.0 * cf_acc / max(eval_total, 1)
avg_shift = sum(shift_list) / max(len(shift_list), 1)

results_df = pd.DataFrame([{
    "Factual Accuracy": factual_accuracy,
    "Counterfactual Accuracy": counterfactual_accuracy,
    "Avg Intervention Shift": avg_shift
}])

results_df.to_csv(os.path.join(RESULTS_DIR, "final_metrics.csv"), index=False)

print("\n========================")
print("FINAL RESULTS")
print("========================")

print(results_df.to_string(index=False))

print("\nSaved everything in:", RESULTS_DIR)
