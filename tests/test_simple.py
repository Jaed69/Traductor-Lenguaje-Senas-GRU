"""
Test Suite for LSP System (Simple)
Básic tests to verify the main modules of the system.

Author: LSP Team
Version: 2.2 - July 2025
"""

import sys
import os
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_dependencies():
    """Tests that the main dependencies are installed."""
    dependencies = [
        'cv2', 'mediapipe', 'numpy', 'h5py',
        'tensorflow', 'scipy', 'sklearn'
    ]
    failed_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            failed_deps.append(dep)
    assert len(failed_deps) == 0, f"Missing dependencies: {', '.join(failed_deps)}"

def test_data_collection_module():
    """Tests the basic import and instantiation of the data collection module."""
    try:
        from src.data_collection.main_collector import LSPDataCollector
        _ = LSPDataCollector()
    except Exception as e:
        if "MediaPipe" in str(e):
            pytest.skip(f"Skipping data collection test: MediaPipe models not found. {e}")
        else:
            pytest.fail(f"Failed to initialize Data Collector: {e}")

def test_training_module():
    """Tests the basic import and instantiation of the training module."""
    try:
        from src.training.train_gru import GRUTrainer
        _ = GRUTrainer()
    except Exception as e:
        pytest.fail(f"Failed to initialize GRU Trainer: {e}")

def test_evaluation_module():
    """Tests the basic import and instantiation of the evaluation module."""
    try:
        from src.evaluation.evaluate_model import ModelEvaluator
        _ = ModelEvaluator()
    except Exception as e:
        pytest.fail(f"Failed to initialize Model Evaluator: {e}")

def test_inference_module():
    """Tests the basic import and instantiation of the inference module."""
    try:
        from src.inference.real_time_translator import RealTimeTranslator
        _ = RealTimeTranslator()
    except Exception as e:
        if "MediaPipe" in str(e):
            pytest.skip(f"Skipping inference test: MediaPipe models not found. {e}")
        else:
            pytest.fail(f"Failed to initialize Real Time Translator: {e}")
