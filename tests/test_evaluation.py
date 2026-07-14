import numpy as np
import pandas as pd
import pytest

from dataset import _add_group_ids, split_train_validation
import train
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


@pytest.mark.parametrize(
    ('selected_threshold', 'expected_threshold', 'floor_applied'),
    [
        (0.381, 0.45, True),
        (0.62, 0.62, False),
    ],
)
def test_evaluate_applies_minimum_decision_threshold(
    monkeypatch, selected_threshold, expected_threshold, floor_applied
):
    labels = np.array([0, 1])
    logits = np.array([-1.0, 1.0])
    monkeypatch.setattr(
        train,
        'validate',
        lambda *_args: (
            0.1,
            1.0,
            np.array([0, 1]),
            labels,
            np.array([0.2, 0.8]),
            ['negative', 'positive'],
            ['negative', 'positive'],
            ['test', 'test'],
            logits,
        ),
    )
    monkeypatch.setattr(train, 'fit_temperature', lambda *_args: 1.0)
    monkeypatch.setattr(train, 'apply_temperature', lambda *_args: np.array([0.4, 0.6]))
    monkeypatch.setattr(
        train,
        'select_threshold',
        lambda *_args: (selected_threshold, {'selection': 'test'}),
    )

    metrics, samples = train.evaluate(None, None, None, None, 0.9, 0.45)

    assert metrics['threshold'] == pytest.approx(expected_threshold)
    assert metrics['threshold_selection'] == {
        'selection': 'test',
        'selected_threshold': selected_threshold,
        'minimum_threshold': 0.45,
        'final_threshold': expected_threshold,
        'floor_applied': floor_applied,
    }
    assert samples['prediction'].tolist() == [0, int(0.6 >= expected_threshold)]


def test_prediction_artifact_uses_final_threshold(tmp_path):
    samples = pd.DataFrame({
        'true_label': [0, 1],
        'probability': [0.4, 0.6],
        'prediction': [0, 1],
    })

    train.write_prediction_artifacts(samples, str(tmp_path), 'validation', 0.45)

    written = pd.read_csv(tmp_path / 'validation_predictions.csv', encoding='utf-8-sig')
    assert written['decision_threshold'].tolist() == [0.45, 0.45]
