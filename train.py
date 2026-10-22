"""Train SCM-VLM on VQA v2 / COCO.

    python train.py                          # defaults below
    python train.py --samples 20000 --epochs 10
    python train.py --samples 3000 --epochs 20 --lr 1e-4

What changed compared with the first version
---------------------------------------------
* The answer head now receives the question (Y = Decoder(h, q)).  The old head
  only saw mean(h), so every question got the same answer (mostly yes/no).
* Answer vocabulary = the 3129 most frequent answers (standard for VQA v2)
  instead of all 22k answers, most of which were never seen in training.
* DETR and BERT are frozen, so their outputs are cached once (cache/) instead
  of being recomputed for every sample in every epoch.
* An image-level train / validation split: accuracies in results/ are measured
  on held-out questions, not on the training samples.
* The acyclicity term is the NOTEARS constraint tr(e^{A o A}) - n from the paper.
* do() interventions cut the parents of the intervened node (row k), and the
  graph is no longer rebuilt after the cut (models/scm_vlm.py).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from configs.config import DEVICE
from datasets.datasetBuilder import VQACOCO
from losses.losses import acyclicity_loss
from models.scm_vlm import SCMVLM


# =====================================================
# SETTINGS (defaults; override on the command line)
# =====================================================

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "scm_vlm_latest.pt")
CACHE_DIR = Path("cache")

IMAGE_DIR = "coco/train2014/train2014"
QUESTION_FILE = "coco/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_FILE = "coco/vqa/v2_mscoco_train2014_annotations.json"

EPOCHS = 10
BATCH_SIZE = 16            # gradients are accumulated over the batch
TRAIN_SAMPLES = 20000      # questions used (train + validation)
VAL_FRACTION = 0.1         # held-out fraction, split by image
MAX_ANSWERS = 3129
LEARNING_RATE = 1e-4
SEED = 42

TOP_K = 3
RANDOM_NODE_PROB = 0.3

LAMBDA_CF = 0.5
LAMBDA_SPARSE = 0.1
LAMBDA_ACYCLIC = 0.2
LAMBDA_SHIFT = 0.3
LAMBDA_ENTROPY = 0.05


# =====================================================
# FEATURE CACHE (frozen DETR + BERT)
# =====================================================

def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, model: SCMVLM, root: Path) -> None:
        self.model = model
        self.image_dir = root / "detr"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.questions: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def image(self, image_path: str) -> torch.Tensor:
        path = self.image_dir / f"{_key(os.path.abspath(image_path))}.pt"
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)["embeddings"]
        output = self.model.vision(image_path)
        embeddings = output["embeddings"][0].detach().cpu()
        torch.save({"embeddings": embeddings, "labels": output["labels"][0]}, path)
        return embeddings

    @torch.no_grad()
    def question(self, question: str) -> torch.Tensor:
        if question not in self.questions:
            self.questions[question] = self.model.text(question)[0].detach().cpu()
        return self.questions[question]


# =====================================================
# HELPERS
# =====================================================

def split_by_image(dataset: VQACOCO, num_samples: int, val_fraction: float, seed: int):
    """Pick up to ``num_samples`` questions and split them so that no image
    appears in both train and validation."""
    by_image = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_image[str(record["image_path"])].append(index)
    images = sorted(by_image)
    random.Random(seed).shuffle(images)

    chosen = []
    for image in images:
        chosen.append(image)
        if sum(len(by_image[i]) for i in chosen) >= num_samples:
            break
    num_val = max(1, int(round(len(chosen) * val_fraction)))
    val_images, train_images = chosen[:num_val], chosen[num_val:]
    train = [i for image in train_images for i in by_image[image]]
    val = [i for image in val_images for i in by_image[image]]
    return train, val


def save_checkpoint(model, dataset, epoch, train_accuracy, val_accuracy):
    """Save learned layers + answer vocabulary. DETR and BERT are frozen
    pretrained backbones, so they are reloaded from Hugging Face instead."""
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }
    torch.save({
        "format_version": 2,
        "question_aware": True,
        "epoch": epoch,
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "num_classes": len(dataset.answer2idx),
        "idx2answer": dataset.idx2answer,
        "trainable_state_dict": trainable_state,
    }, CHECKPOINT_PATH)
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")


def sample_losses(model, raw_objects, q_emb, label):
    """Factual forward pass + SCM regularisers + counterfactual terms for one sample."""
    out = model.forward_features(raw_objects.unsqueeze(0).to(DEVICE), q_emb.unsqueeze(0).to(DEVICE))

    logits = torch.clamp(out["logits"], -30, 30)
    loss_task = F.cross_entropy(logits, label.view(1))

    objects = out["objects"]
    adjacency = torch.clamp(out["adjacency"].float(), 0.0, 1.0)
    A = adjacency[0]

    loss_sparse = A.mean()
    loss_acyclic = acyclicity_loss(A)
    edge_prob = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    loss_entropy = -torch.mean(edge_prob * torch.log(edge_prob + 1e-6))

    # counterfactual: do(X_k = x') on an important or random node
    node_scores = A.sum(dim=0)                      # out-degree (A[i, j]: j -> i)
    k = min(TOP_K, node_scores.shape[0])
    _, topk_idx = torch.topk(node_scores, k=k)
    if random.random() < RANDOM_NODE_PROB:
        node_idx = torch.randint(0, objects.shape[1], (1,), device=DEVICE)
    else:
        node_idx = topk_idx[torch.randint(0, k, (1,))].view(1)

    strength = random.random() * 0.2
    replacement = objects[:, node_idx.item()] + torch.randn_like(objects[:, 0]) * strength
    cf_out = model.counterfactual(objects, adjacency, node_idx, replacement, out["question_embedding"])
    cf_logits = torch.clamp(cf_out["logits"], -30, 30)

    loss_cf = F.kl_div(
        F.log_softmax(cf_logits, dim=-1),
        F.softmax(logits.detach(), dim=-1),
        reduction="batchmean",
    ) * (1.0 + strength)
    loss_shift = torch.mean(torch.abs(F.softmax(cf_logits, dim=-1) - F.softmax(logits.detach(), dim=-1)))

    loss = (
        loss_task
        + LAMBDA_SPARSE * loss_sparse
        + LAMBDA_ACYCLIC * loss_acyclic
        + LAMBDA_ENTROPY * loss_entropy
        + LAMBDA_CF * loss_cf
        + LAMBDA_SHIFT * loss_shift
    )
    parts = {"task": loss_task, "cf": loss_cf, "shift": loss_shift, "acyclic": loss_acyclic}
    return loss, logits, parts


@torch.no_grad()
def evaluate(model, dataset, indices, cache):
    """Factual accuracy, counterfactual accuracy (answer still correct after a
    small do() perturbation of the most influential node) and mean logit shift."""
    model.eval()
    factual = counterfactual = total = 0
    shifts = []
    for index in indices:
        image_path, question, label = dataset[index]
        try:
            raw = cache.image(image_path)
            if raw.shape[0] == 0:
                continue
            out = model.forward_features(raw.unsqueeze(0).to(DEVICE),
                                         cache.question(question).unsqueeze(0).to(DEVICE))
        except Exception as error:  # unreadable image etc.
            print(f"EVAL SKIP {image_path}: {error}")
            continue
        label = label.to(DEVICE)
        logits = out["logits"]
        factual += int(logits.argmax(-1).item() == label.item())

        A = torch.clamp(out["adjacency"], 0.0, 1.0)
        node_idx = torch.argmax(A[0].sum(dim=0)).view(1)
        objects = out["objects"]
        replacement = objects[:, node_idx.item()] + torch.randn_like(objects[:, 0]) * 0.05
        cf_out = model.counterfactual(objects, A, node_idx, replacement, out["question_embedding"])
        counterfactual += int(cf_out["logits"].argmax(-1).item() == label.item())
        shifts.append(torch.abs(logits - cf_out["logits"]).mean().item())
        total += 1
    model.train()
    total = max(total, 1)
    return 100.0 * factual / total, 100.0 * counterfactual / total, sum(shifts) / max(len(shifts), 1)


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    global LAMBDA_CF
    parser = argparse.ArgumentParser(description="Train SCM-VLM")
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-answers", type=int, default=MAX_ANSWERS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--no-cf", action="store_true", help="ablation: disable the counterfactual loss")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if args.no_cf:
        LAMBDA_CF = 0.0

    print(f"\nUSING DEVICE: {DEVICE}")

    dataset = VQACOCO(IMAGE_DIR, QUESTION_FILE, ANNOTATION_FILE, max_answers=args.max_answers)
    train_idx, val_idx = split_by_image(dataset, args.samples, args.val_fraction, SEED)
    print(f"Answer vocabulary: {len(dataset.answer2idx)} | train questions: {len(train_idx)} "
          f"| validation questions: {len(val_idx)}")

    model = SCMVLM(num_classes=len(dataset.answer2idx), question_aware=True).to(DEVICE)
    cache = FeatureCache(model, CACHE_DIR)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_val = -1.0
    model.train()

    print("\n========================")
    print("TRAINING STARTED")
    print("========================")

    for epoch in range(args.epochs):
        order = list(train_idx)
        random.shuffle(order)
        total_loss, correct, seen = 0.0, 0, 0
        optimizer.zero_grad()
        in_batch = 0

        for step, index in enumerate(order, start=1):
            image_path, question, label = dataset[index]
            try:
                raw = cache.image(image_path)
                if raw.shape[0] == 0:
                    continue
                loss, logits, parts = sample_losses(model, raw, cache.question(question), label.to(DEVICE))
            except Exception as error:
                print(f"TRAIN SKIP {image_path}: {error}")
                continue
            if torch.isnan(loss):
                continue

            (loss / args.batch_size).backward()
            in_batch += 1
            correct += int(logits.argmax(-1).item() == label.item())
            total_loss += loss.item()
            seen += 1

            if in_batch == args.batch_size or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                in_batch = 0

            if seen % 500 == 0:
                print(f"[TRAIN] epoch={epoch + 1} sample={seen} loss={loss.item():.4f} "
                      f"task={parts['task'].item():.4f} cf={parts['cf'].item():.4f} "
                      f"acyc={parts['acyclic'].item():.4f} running_acc={correct / seen:.4f}")

        scheduler.step()
        train_acc = correct / max(seen, 1)
        val_factual, val_cf, val_shift = evaluate(model, dataset, val_idx, cache)
        history.append({"epoch": epoch + 1, "loss": total_loss, "train_accuracy": train_acc,
                        "val_factual_accuracy": val_factual, "val_counterfactual_accuracy": val_cf,
                        "val_intervention_shift": val_shift})

        print("\n========================")
        print(f"EPOCH {epoch + 1}")
        print("========================")
        print(f"Loss                     : {total_loss:.4f}")
        print(f"Train accuracy           : {train_acc:.4f}")
        print(f"Val factual accuracy     : {val_factual:.2f}%")
        print(f"Val counterfactual acc.  : {val_cf:.2f}%")

        pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)
        if val_factual > best_val:
            best_val = val_factual
            save_checkpoint(model, dataset, epoch + 1, train_acc, val_factual)

    best = max(history, key=lambda row: row["val_factual_accuracy"]) if history else {}
    results = pd.DataFrame([{
        "Factual Accuracy": best.get("val_factual_accuracy", 0.0),
        "Counterfactual Accuracy": best.get("val_counterfactual_accuracy", 0.0),
        "Avg Intervention Shift": best.get("val_intervention_shift", 0.0),
        "Best Epoch": best.get("epoch", 0),
        "Validation Questions": len(val_idx),
    }])
    results.to_csv(os.path.join(RESULTS_DIR, "final_metrics.csv"), index=False)

    print("\n========================")
    print("FINAL RESULTS (held-out validation, best epoch)")
    print("========================")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
