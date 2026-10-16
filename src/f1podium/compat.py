"""Known, harmless warnings from third-party version combinations, silenced narrowly and in one place.

Each entry is (message regex, category, module regex). Only these exact messages are ignored; every other warning
still surfaces (tests run with warnings as errors).
"""

import warnings

KNOWN_EXTERNAL_WARNINGS = (
    # scikit-learn 1.6.1 passes `iprint` to scipy's L-BFGS-B (LogisticRegression, solver="lbfgs"); scipy 1.16 deprecates
    # the option (removal in scipy 1.18). Both versions are pinned to match Kaggle; the option still works and does not
    # affect results. scipy raises it with stacklevel=3, so it is attributed to the CALLER, sklearn's logistic module;
    # the filter is scoped to that module. Remove this entry when scikit-learn is upgraded.
    (r"scipy\.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated", DeprecationWarning,
     r"sklearn\.linear_model\._logistic"),
)


def silence_known_external_warnings() -> None:
    """Ignore exactly the KNOWN_EXTERNAL_WARNINGS (used by the notebook's Setup cell and by tests/conftest.py)."""
    for message, category, module in KNOWN_EXTERNAL_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=category, module=module)

