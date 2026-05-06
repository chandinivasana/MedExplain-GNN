import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from .inference import (
        TEMPERATURE,
        _class_names,
        _load_checkpoint_metadata,
        _load_model,
        load_graph_data,
    )
except ImportError:
    from inference import (
        TEMPERATURE,
        _class_names,
        _load_checkpoint_metadata,
        _load_model,
        load_graph_data,
    )


def _select_mask(data, requested_mask: str) -> tuple[str, torch.Tensor]:
    candidates = [requested_mask, "test_mask", "val_mask", "train_mask"]
    for name in candidates:
        if hasattr(data, name):
            mask = getattr(data, name)
            if mask is not None and int(mask.sum().item()) > 0:
                return name, mask.bool()
    raise ValueError("No non-empty evaluation mask found.")


def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for actual, predicted in zip(y_true.cpu(), y_pred.cpu()):
        matrix[int(actual.item()), int(predicted.item())] += 1
    return matrix


def _per_class_metrics(confusion: torch.Tensor):
    recalls = []
    precisions = []
    f1_scores = []
    for index in range(confusion.size(0)):
        tp = confusion[index, index].float()
        fp = confusion[:, index].sum().float() - tp
        fn = confusion[index, :].sum().float() - tp
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        precisions.append(float(precision.item()))
        recalls.append(float(recall.item()))
        f1_scores.append(float(f1.item()))
    return precisions, recalls, f1_scores


def _save_confusion_matrix(path: Path, confusion: torch.Tensor, class_names: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, confusion.tolist()):
            writer.writerow([name, *row])


def _rare_class_indices(data, num_classes: int, max_count: Optional[int] = None) -> list[int]:
    if hasattr(data, "train_mask") and int(data.train_mask.sum().item()) > 0:
        train_y = data.y[data.train_mask.bool()]
    else:
        train_y = data.y

    counts = torch.bincount(train_y.cpu(), minlength=num_classes)
    present = counts[counts > 0]
    if present.numel() == 0:
        return []

    threshold = max_count if max_count is not None else max(1, int(torch.quantile(present.float(), 0.25).item()))
    return [index for index, count in enumerate(counts.tolist()) if 0 < count <= threshold]


def _top_k_rows(
    probabilities: torch.Tensor,
    y_true: torch.Tensor,
    class_names: list[str],
    source_indices: torch.Tensor,
    top_k: int,
):
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
    rows = []
    for row_index in range(probabilities.size(0)):
        rows.append(
            {
                "node_index": int(source_indices[row_index].item()),
                "actual": class_names[int(y_true[row_index].item())],
                "top_predictions": [
                    {
                        "disease": class_names[int(indices[row_index, k].item())],
                        "confidence": round(float(values[row_index, k].item()), 4),
                    }
                    for k in range(values.size(1))
                ],
            }
        )
    return rows


@torch.no_grad()
def evaluate(
    graph_data_path: str,
    checkpoint_path: str,
    mask_name: str,
    temperature: float,
    output_dir: str,
    top_k: int,
    rare_max_count: Optional[int],
):
    if temperature < 0.5:
        raise ValueError("temperature below 0.5 makes predictions overconfident.")

    missing = [
        path
        for path in (checkpoint_path, graph_data_path)
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s): "
            f"{', '.join(missing)}. Place best_model.pth and graph_data.pt in "
            "/app inside the ai-service container, or pass explicit paths with "
            "--checkpoint-path and --graph-data-path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint_metadata(checkpoint_path, device)
    data = load_graph_data(graph_data_path, device)
    model = _load_model(checkpoint_path, device)
    selected_mask_name, mask = _select_mask(data, mask_name)

    logits = model(data.x, data.edge_index)
    probabilities = F.softmax(logits / temperature, dim=1)
    predictions = probabilities.argmax(dim=1)

    y_true = data.y[mask]
    y_pred = predictions[mask]
    eval_probs = probabilities[mask]
    source_indices = mask.nonzero(as_tuple=False).view(-1).cpu()

    num_classes = probabilities.size(1)
    class_names = _class_names(checkpoint, data, num_classes)
    confusion = _confusion_matrix(y_true, y_pred, num_classes)
    precisions, recalls, f1_scores = _per_class_metrics(confusion)

    accuracy = float((y_true == y_pred).float().mean().item())
    macro_f1 = sum(f1_scores) / len(f1_scores)
    mean_confidence = float(eval_probs.max(dim=1).values.mean().item())
    min_confidence = float(eval_probs.max(dim=1).values.min().item())
    max_confidence = float(eval_probs.max(dim=1).values.max().item())

    rare_indices = _rare_class_indices(data, num_classes, rare_max_count)
    rare_mask = torch.zeros_like(mask)
    for class_index in rare_indices:
        rare_mask |= mask & (data.y == class_index)
    rare_count = int(rare_mask.sum().item())
    rare_accuracy = None
    rare_rows = []
    if rare_count > 0:
        rare_accuracy = float((predictions[rare_mask] == data.y[rare_mask]).float().mean().item())
        rare_source_indices = rare_mask.nonzero(as_tuple=False).view(-1).cpu()
        rare_rows = _top_k_rows(
            probabilities[rare_mask].cpu(),
            data.y[rare_mask].cpu(),
            class_names,
            rare_source_indices,
            top_k,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    confusion_path = output_path / "confusion_matrix.csv"
    _save_confusion_matrix(confusion_path, confusion, class_names)

    per_class = [
        {
            "class_index": index,
            "disease": class_names[index],
            "precision": round(precisions[index], 4),
            "recall": round(recalls[index], 4),
            "f1": round(f1_scores[index], 4),
            "support": int(confusion[index, :].sum().item()),
        }
        for index in range(num_classes)
    ]

    report = {
        "checkpoint_path": checkpoint_path,
        "graph_data_path": graph_data_path,
        "mask": selected_mask_name,
        "saved_validation_loss": checkpoint.get("val_loss"),
        "focal_gamma": checkpoint.get("focal_gamma"),
        "temperature": temperature,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confidence": {
            "mean": round(mean_confidence, 4),
            "min": round(min_confidence, 4),
            "max": round(max_confidence, 4),
        },
        "rare_class_indices": rare_indices,
        "rare_accuracy": None if rare_accuracy is None else round(rare_accuracy, 4),
        "confusion_matrix_csv": str(confusion_path),
        "per_class_metrics": per_class,
        "top3_examples": _top_k_rows(
            eval_probs[: min(10, eval_probs.size(0))].cpu(),
            y_true[: min(10, y_true.size(0))].cpu(),
            class_names,
            source_indices[: min(10, source_indices.numel())],
            top_k,
        ),
        "rare_examples": rare_rows[:10],
    }

    report_path = output_path / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-data-path", default="graph_data.pt")
    parser.add_argument("--checkpoint-path", default="best_model.pth")
    parser.add_argument("--mask", default="test_mask")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--output-dir", default="evaluation_report")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rare-max-count", type=int, default=None)
    args = parser.parse_args()
    evaluate(
        graph_data_path=args.graph_data_path,
        checkpoint_path=args.checkpoint_path,
        mask_name=args.mask,
        temperature=args.temperature,
        output_dir=args.output_dir,
        top_k=args.top_k,
        rare_max_count=args.rare_max_count,
    )
