"""Shallow feed-forward neural network (PyTorch, CPU) as a scikit-learn compatible classifier.

Why a self-contained estimator: it takes the raw feature DataFrame, so the evaluation harness,
permutation importance and SHAP can call fit / predict_proba on it exactly as on the sklearn
pipelines. The same preprocessing as the other models (median imputation + StandardScaler) is built
inside and fitted on the rows passed to fit only.

Training: Adam (weight decay = L2 penalty), binary cross-entropy on a single output. The sigmoid
output is applied inside BCEWithLogitsLoss for numerical stability; predict_proba applies it
explicitly, so the model is "sigmoid output + BCE" mathematically. Dropout and weight decay are the
regularisation knobs. If an eval_set is given, the epoch with the highest validation PR-AUC is kept
(early stopping with `patience`, at most `max_epochs`); the best weights are restored.

Determinism: seeds for random / numpy / torch, CPU only, a single thread, a seeded shuffling
generator and torch.use_deterministic_algorithms(True) during fit. Global torch settings are restored
afterwards.
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from torch import nn

from f1podium import config
from f1podium.models import build_preprocessor


def _make_net(n_in: int, hidden_sizes, dropout: float) -> nn.Sequential:
    layers, width = [], n_in
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class ShallowNNClassifier(ClassifierMixin, BaseEstimator):
    """1-2 hidden ReLU layers, dropout, sigmoid output, BCE loss, Adam. See the module docstring."""

    uses_validation_for_early_stopping = True      # evaluation.fit_on_train then passes the VALIDATION rows

    def __init__(self, hidden_sizes=(32,), dropout=0.0, weight_decay=0.0, lr=1e-3, batch_size=256,
                 max_epochs=100, patience=10, seed=config.RANDOM_SEED):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def _tensor(self, X: pd.DataFrame) -> torch.Tensor:
        return torch.from_numpy(self.prep_.transform(X).to_numpy(dtype=np.float32))

    def fit(self, X: pd.DataFrame, y, eval_set=None):
        """Fit on (X, y). `eval_set=(X_val, y_val)` enables early stopping on validation PR-AUC."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        n_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        try:
            self._fit(X, np.asarray(y, dtype=np.float32), eval_set)
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
            torch.set_num_threads(n_threads)
        return self

    def _fit(self, X, y, eval_set):
        self.prep_ = build_preprocessor(scale=True)
        Xt = torch.from_numpy(self.prep_.fit_transform(X).to_numpy(dtype=np.float32))
        yt = torch.from_numpy(y)
        if eval_set is not None:
            Xv, yv = self._tensor(eval_set[0]), np.asarray(eval_set[1], dtype=np.float32)
            yv_t = torch.from_numpy(yv)

        self.net_ = _make_net(Xt.shape[1], self.hidden_sizes, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        shuffle = torch.Generator().manual_seed(self.seed)

        history, best, best_state, wait = [], -np.inf, None, 0
        self.best_epoch_ = self.max_epochs
        for epoch in range(1, self.max_epochs + 1):
            self.net_.train()
            order = torch.randperm(len(Xt), generator=shuffle)
            total = 0.0
            for start in range(0, len(Xt), self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            row = {"epoch": epoch, "train_loss": total / len(Xt), "val_loss": np.nan, "val_pr_auc": np.nan}
            if eval_set is not None:
                self.net_.eval()
                with torch.no_grad():
                    logits = self.net_(Xv).squeeze(1)
                row["val_loss"] = loss_fn(logits, yv_t).item()
                row["val_pr_auc"] = float(average_precision_score(yv, torch.sigmoid(logits).numpy()))
                if row["val_pr_auc"] > best:
                    best, best_state, self.best_epoch_, wait = row["val_pr_auc"], copy.deepcopy(self.net_.state_dict()), epoch, 0
                else:
                    wait += 1
            history.append(row)
            if eval_set is not None and wait >= self.patience:
                break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.history_ = pd.DataFrame(history)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = Xt.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "net_"):
            raise AttributeError("ShallowNNClassifier is not fitted")
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        n_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        try:
            self.net_.eval()
            with torch.no_grad():
                p = torch.sigmoid(self.net_(self._tensor(X)).squeeze(1)).numpy().astype(np.float64)
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
            torch.set_num_threads(n_threads)
        return np.column_stack([1.0 - p, p])
