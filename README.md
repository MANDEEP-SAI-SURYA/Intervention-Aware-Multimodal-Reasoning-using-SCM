# SCM-VLM Research Prototype

Implementation of the core architecture proposed in:

**Intervention Aware Multimodal Reasoning via Structural Causal Graph Integration in Vision Language Transformers**

This version follows the requested research-prototype layout and includes:

- DETR-based object extraction
- BERT-based question encoding
- learnable SCM graph construction
- DAG regularization
- multi-head causal attention
- intervention operators with descendant recomputation
- counterfactual-ready training pipeline
- evaluation utilities
- graph visualization

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place a test image at:

```text
images/test.jpg
```

## Run

```bash
python main.py
python train.py
python evaluate.py
```

The first run downloads DETR and BERT weights from Hugging Face.

## Google Collab NoteBook
https://colab.research.google.com/drive/1gc8_qWXL_sxYWe8thv1v4v5BfdpCYWNU?usp=sharing
