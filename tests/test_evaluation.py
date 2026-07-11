import numpy as np
import pandas as pd

from dataset import _add_group_ids, split_train_validation
from utils.evaluation import binary_metrics, select_threshold


def test_group_split_keeps_duplicates_together():
    df = pd.DataFrame({
        'title': ['same', 'same', 'p1', 'n1', 'p2', 'n2', 'p3', 'n3', 'p4', 'n4'],
        'download_link': ['', '', 'p1', 'n1', 'p2', 'n2', 'p3', 'n3', 'p4', 'n4'],
        'label': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    train, validation = split_train_validation(_add_group_ids(df), 0.2, 42)
    assert not (set(train['_group_id']) & set(validation['_group_id']))


def test_threshold_honours_precision_constraint():
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    threshold, info = select_threshold(labels, probabilities, target_precision=1.0)
    metrics = binary_metrics(labels, probabilities, threshold)
    assert info['selection'] == 'precision_constraint'
    assert metrics['precision'] == 1.0
