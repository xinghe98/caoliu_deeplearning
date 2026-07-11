"""Binary classification metrics, threshold selection and probability calibration."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, fbeta_score, precision_recall_fscore_support,
    roc_auc_score,
)


def apply_temperature(logits, temperature=1.0):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)))


def fit_temperature(logits, labels):
    """Fit one temperature on validation logits; return 1.0 for degenerate data."""
    logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    if len(logits) == 0 or labels.min() == labels.max():
        return 1.0
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def select_threshold(labels, probabilities, target_precision=0.90):
    """Maximise recall subject to precision target, otherwise maximise F0.5."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scored = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        f05 = fbeta_score(labels, predictions, beta=0.5, zero_division=0)
        scored.append((float(threshold), float(precision), float(recall), float(f05)))
    feasible = [row for row in scored if row[1] >= target_precision and row[2] > 0]
    if feasible:
        threshold, precision, recall, f05 = max(feasible, key=lambda row: (row[2], row[1], row[3]))
        return threshold, {'selection': 'precision_constraint', 'precision': precision, 'recall': recall, 'f0_5': f05}
    threshold, precision, recall, f05 = max(scored, key=lambda row: (row[3], row[1], row[2]))
    return threshold, {'selection': 'best_f0_5', 'precision': precision, 'recall': recall, 'f0_5': f05}


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'f0_5': float(fbeta_score(labels, predictions, beta=0.5, zero_division=0)),
        'brier_score': float(brier_score_loss(labels, probabilities)),
        'positive_rate': float(predictions.mean()),
        'probability_quantiles': {
            str(q): float(np.quantile(probabilities, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
    if len(np.unique(labels)) == 2:
        metrics['pr_auc'] = float(average_precision_score(labels, probabilities))
        metrics['roc_auc'] = float(roc_auc_score(labels, probabilities))
    else:
        metrics['pr_auc'] = math.nan
        metrics['roc_auc'] = math.nan
    return metrics
