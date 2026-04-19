# from Constants import *

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from pathlib import Path
import pandas as pd
# import matplotlib.pyplot as plt
from PIL import Image

def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed) # type: ignore
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def compute_mean_std(frame: pd.DataFrame, batch_size: int, num_workers: int, device: str | torch.device | None):
	"""
	Compute the per-channel mean and standard deviation of the training set for normalization.
	"""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(device=device)
	
	print(device, type(device))

	"================================================================="

	class ImageDataset(Dataset[torch.Tensor]):
		def __init__(self, frame: pd.DataFrame):
			self.frame = frame.reset_index(drop=True)
		
		def __len__(self) -> int:
			return len(self.frame)
		
		def __getitem__(self, idx: int) -> torch.Tensor:
			path = self.frame.iloc[idx]["path"]
			
			with Image.open(path) as image:
				image = image.convert("RGB")
				image = transforms.ToTensor()(pic=image)

			return torch.as_tensor(image, dtype=torch.float32)
	
	"================================================================="

	loader = DataLoader(
		dataset=ImageDataset(frame=frame),
		batch_size=batch_size,
		shuffle=True,			# or False, doesn't matter
		num_workers=num_workers,
		persistent_workers=(num_workers > 0),
		pin_memory=(device.type == "cuda"),
	)

	channel_sums = torch.zeros(3, device=device, dtype=torch.float64)
	channel_squared_sums = torch.zeros(3, device=device, dtype=torch.float64)
	total_pixels = 0

	for images in loader:
		images = images.to(device, non_blocking=True, dtype=torch.float64)  # [B, C, H, W]
		batchsize, channels, height, width = images.shape
		pixels_per_batch = batchsize * height * width

		channel_sums += images.sum(dim=[0, 2, 3])
		channel_squared_sums += (images ** 2).sum(dim=[0, 2, 3])
		total_pixels += pixels_per_batch

	mean = channel_sums / total_pixels
	std = torch.sqrt( (channel_squared_sums / total_pixels) - (mean ** 2) )

	return mean, std

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
	"""
	Measures how well model ranks positive samples over negative samples across all thresholds
	"""
	labels = np.asarray(labels, dtype=np.int64)
	scores = np.asarray(scores, dtype=np.float64)

	positive_mask = labels == 1
	negative_mask = labels == 0
	n_positive = int(positive_mask.sum())
	n_negative = int(negative_mask.sum())

	if n_positive == 0 or n_negative == 0:
		return float("nan")

	ranks = pd.Series(scores).rank(method="average").to_numpy()

	auc = (ranks[positive_mask].sum() - (n_positive * (n_positive + 1) / 2.0)) / (n_positive * n_negative)

	return float(auc)

def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
	"""
	Measures the average precision gained through ranking positive samples high
	"""
	labels = np.asarray(labels, dtype=np.int64)
	scores = np.asarray(scores, dtype=np.float64)

	total_positive = int(labels.sum())
	if total_positive == 0:
		return float("nan")

	order = np.argsort(-scores)         # Descending order of indices
	sorted_labels = labels[order]

	true_positives = np.cumsum(sorted_labels == 1)
	false_positives = np.cumsum(sorted_labels == 0)

	precision = true_positives / np.maximum(true_positives + false_positives, 1)
	recall = true_positives / total_positive

	precision = np.concatenate([[1.0], precision])
	recall = np.concatenate([[0.0], recall])

	ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

	return float(ap)

def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, int | float]:
	"""
	Compute a full set of binary classification metrics at a specific threshold.
	"""
	labels = np.asarray(labels, dtype=np.int64)
	probs = np.asarray(probs, dtype=np.float64)
	
	preds = (probs >= threshold).astype(np.int64)

	tp = int( ((preds == 1) & (labels == 1)).sum() )
	tn = int( ((preds == 0) & (labels == 0)).sum() )
	fp = int( ((preds == 1) & (labels == 0)).sum() )
	fn = int( ((preds == 0) & (labels == 1)).sum() )

	accuracy = (tp + tn) / len(labels)
	precision = tp / max(tp + fp, 1)
	recall = tp / max(tp + fn, 1)
	specificity = tn / max(tn + fp, 1)
	f1 = 2 * (precision * recall) / max(precision + recall, 1e-70)
	balanced_accuracy = (recall + specificity) / 2

	return {
		"threshold": float(threshold),
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"specificity": float(specificity),
		"f1": float(f1),
		"balanced_accuracy": float(balanced_accuracy),
		"auroc": compute_auc(labels, probs),
		"average_precision": compute_average_precision(labels, probs),
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
	}

def select_best_threshold(labels: np.ndarray, probs: np.ndarray, metric_name: str = "f1") -> tuple[float, dict[str, int | float]]:
	"""
	Select the best threshold for binary classification based on a given metric.
	"""
	thresholds = np.linspace(0.05, 0.95, 181)
	best_threshold = 0.5
	best_metrics: dict[str, int | float] | None = None
	best_value = float("-inf")

	for threshold in thresholds:
		metrics = compute_binary_metrics(labels, probs, threshold=threshold)
		metric_value = float(metrics[metric_name])
		
		if metric_value > best_value:
			best_value = metric_value
			best_threshold = float(threshold)
			best_metrics = metrics

	assert best_threshold is not None
	assert best_metrics is not None

	return best_threshold, best_metrics

def roc_curve_points(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	Generate points needed to plot the ROC curve.
	"""
	labels = np.asarray(labels, dtype=np.int64)
	probs = np.asarray(probs, dtype=np.float64)
	order = np.argsort(-probs)
	sorted_labels = labels[order]

	true_positives = np.cumsum(sorted_labels == 1)
	false_positives = np.cumsum(sorted_labels == 0)
	total_positive = max(int((labels == 1).sum()), 1)
	total_negative = max(int((labels == 0).sum()), 1)

	tpr = true_positives / total_positive
	fpr = false_positives / total_negative
	return np.concatenate([[0.0], fpr, [1.0]]), np.concatenate([[0.0], tpr, [1.0]])

def pr_curve_points(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	Generate points needed to plot the Precision-Recall curve.
	"""
	labels = np.asarray(labels, dtype=np.int64)
	probs = np.asarray(probs, dtype=np.float64)
	order = np.argsort(-probs)
	sorted_labels = labels[order]

	true_positives = np.cumsum(sorted_labels == 1)
	false_positives = np.cumsum(sorted_labels == 0)
	total_positive = max(int((labels == 1).sum()), 1)

	precision = true_positives / np.maximum(true_positives + false_positives, 1)
	recall = true_positives / total_positive
	return np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision])