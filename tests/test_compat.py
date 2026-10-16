"""The known-warning filter must silence exactly its message and nothing else."""

import warnings

import pytest

from f1podium import compat


def test_only_the_known_scipy_message_is_silenced():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compat.silence_known_external_warnings()
        msg = ("scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be "
               "removed in SciPy 1.18.0.")
        warnings.warn_explicit(msg, DeprecationWarning, "_lbfgsb_py.py", 1, module="sklearn.linear_model._logistic")
        warnings.warn_explicit("something else is deprecated", DeprecationWarning, "_lbfgsb_py.py", 1,
                               module="sklearn.linear_model._logistic")
        warnings.warn_explicit(msg, DeprecationWarning, "elsewhere.py", 1, module="mypackage")   # other module
    assert [str(w.message) for w in caught] == ["something else is deprecated", msg]


def test_logistic_regression_fit_raises_no_warning_with_the_filter():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(0).normal(size=(50, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compat.silence_known_external_warnings()
        LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))


def test_unrelated_deprecations_still_fail_in_tests():
    with pytest.raises(DeprecationWarning):
        warnings.warn("an unrelated deprecation", DeprecationWarning)
