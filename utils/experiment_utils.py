"""
                               ★■╬▂▂▂▂▂◓□                                                           
                              ☆◕◓◊◊▇▅◕⬤▽■●⬤                                                         
                                   ▽■◑▅▆◑★■╬◒.                                                      
                                       ⬤▂▄◔▽▽▅◒◕★                                                   
                                         ⬤▄▄○◈◔◓◊○⬤▼                                                
                                          .▲◈█▆▅▇██▇▂■                                              
                                             ☆☆■◑█▇███○                                             
                                         ★★★★.  □█▲○██▄                                             
                                        ◒▇╬◑●◊■□▂▲★███◐                                             
                                        ■○╬╬■▇▄□▂ ☆◊██△                                             
                                          ★▆█▅█○▂ ▼◈█▇◑                                             
                                           ▲█★██▂  ☆██▆                                             
                                           ▼█◐◐█▂  ▲██▆                                             
                                           ★█□☆▼▽  ▲██▄                                             
                                           ★█◓     ▲██▄                                             
                                           ▲█◒     ▽◈█▆◕★                                           
                                          ◓▄○.      ▽▅██▲                                           
                                         ☆◒▅   ▽□□□■.□▇█▇△.                                         
                                         ◊○⬤   ⬤▅▄▄◊  □███▂△                                        
                                         ◊◊▄           ◑███▄△                                       
                                        ◓▇◒△           △▇███◒                                       
                                        ◐█◒            ▽△◒██▂                                       
                                        ◕█◒    ☆○◈◈◈     ◒██▂                                       
                                        ◕█◒    ▲████★    ◒██▂                                       
                                       △●○▼    ▲████★  . ◒██▅△                                      
                                     ★□▂▅◕     ▲████★  . ◒███▆△▽                                    
                                    ◓◈◊◕◑     ☆▲████▅◈   ◓█████▂◑★                                  
                                   ◕◊▼  ◒      ▲█████▂   ◓▆██████◐☆                                 
                                  ●▆▽△▄▇●      ▲████▇╬   □◊███████▂                                 
                                  ▽█.◑██●     .▼███◈◑◒   △⬤███████⬤                                 
                                   ▆☆◑██●      .▽▽▽       ■███▄███▼                                 
                                   ◔●◐▲▄●          .      ■██▇▼╬▄◐★                                 
                                      ▆█●                 ▲╬██▂☆.                                   
                                      ▆█●              .  ▼▇██▄                                     
                                  .☆□●▇●◕                 ▽▆██▇◊■☆.                                 
                                 ▽◑○⬤▼╬○△                 .□████▆█◑▲.                               
                                ⬤◈★▽◔▅█╬▼                   ████▇███◐☆                              
                              .⬤●▽△○█▇██○▲                 ■████▂████◒▽                             
                              □▆.▼▂▅⬤╬██◕    ☆☆★★★★★☆★★.  .▂████◒◓████●                             
                             □◐▽☆◈○▲▂██◑●◓▼▲◓██████████╬◒◊███████◈□███▇◔                            
                             ⬤●□█◊▼☆◐▆█◐◓██████████████████████▆●△☆◔███◒                            
                             ⬤▆▄█⬤   ☆▂████◕◓◈◈◈◈◈◓■◈◈◈◈◒○███▆⬤▽    ▅██◒                            
                             ▽◒◊◒★    .◔▅█▇▲             ◓█▇●▲      ⬤╬◐▼                            
                                        ☆▲★               ▼▼                                        
                                                    /               
                                ___       ___  ___ (___       _ _   
                                |   )|   )|   )|   )|    |   )| | )  
                                |__/||__/ |__/||  / |__  |__/ |  /   
                                    |                                
                                                                    
                                    /           /    /             
                                ___ (  ___  ___ (    (___  ___  ___ 
                                |   )| |   )|   )|___)|    |   )|   )
                                |__/ | |__/||  / | \  |__  |__/ |  / 
                                |                                    
                                                                    
                                                /    /               
                                _ _  ___  ___ (___    ___  ___      
                                | | )|   )|    |   )| |   )|___)     
                                |  / |__/||__  |  / | |  / |__       
                                                                    
                                                                    
                                /                     /             
                                (  ___  ___  ___  ___    ___  ___    
                                | |___)|   )|   )|   )| |   )|   )   
                                | |__  |__/||    |  / | |  / |__/    
                                                            __/                                                                                                         
                                                                                                    
                                            by Julian Henry                                                        
"""

"""Shared utilities for scientifically rigorous experiment execution.

Provides baselines, metrics, statistical testing, seeding, and logging
used by both Phase 4 (binary) and Phase 5 (multi-class) experiments.
"""

import os
import random
import datetime
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set all random seeds for reproducibility.

    Imports TensorFlow lazily so this module can be used in contexts
    where TF is not installed (e.g. unit-test subsets).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def majority_baseline(y_test):
    """Accuracy achieved by always predicting the most common class."""
    counts = Counter(y_test)
    majority_count = max(counts.values())
    return majority_count / len(y_test)


def random_baseline(y_test, k=2):
    """Expected accuracy of uniform random predictions (= 1/k).

    Also returns the empirical accuracy of a random draw so the caller
    can verify the analytical value.
    """
    analytical = 1.0 / k
    rng = np.random.RandomState(0)
    random_preds = rng.randint(0, k, size=len(y_test))
    empirical = accuracy_score(y_test, random_preds)
    return {"analytical": analytical, "empirical": empirical}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, k=2):
    """Compute a comprehensive set of classification metrics.

    Returns a dict containing:
        accuracy, macro_f1, macro_precision, macro_recall,
        per_class_f1, per_class_precision, per_class_recall,
        confusion_matrix, classification_report (str)
    """
    labels = list(range(k))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)

    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_f1": per_f1.tolist(),
        "per_class_precision": per_prec.tolist(),
        "per_class_recall": per_rec.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Statistical Testing
# ---------------------------------------------------------------------------

def paired_significance_test(scores_a, scores_b):
    """Non-parametric paired test between two sets of fold-level scores.

    Uses the Wilcoxon signed-rank test when n >= 6 (the minimum for the
    test to be meaningful).  Falls back to a paired t-test for smaller n.

    Parameters
    ----------
    scores_a, scores_b : array-like of float
        Matched accuracy (or F1) scores, one per CV fold.

    Returns
    -------
    dict with keys: statistic, p_value, test_used
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diff = scores_a - scores_b

    # If all differences are zero, no test is meaningful
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "test_used": "none (identical)"}

    if len(scores_a) >= 6:
        try:
            stat, p = wilcoxon(scores_a, scores_b, alternative='two-sided')
            return {"statistic": float(stat), "p_value": float(p), "test_used": "wilcoxon"}
        except ValueError:
            # wilcoxon can fail if all differences are zero after rounding
            pass

    # Fallback: paired t-test
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p), "test_used": "paired_ttest"}


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : dict mapping label -> raw p-value

    Returns
    -------
    dict mapping label -> dict(raw_p, corrected_p, significant_05)
    """
    labels = list(p_values.keys())
    raw = np.array([p_values[l] for l in labels])
    n = len(raw)
    order = np.argsort(raw)

    corrected = np.ones(n)
    for rank, idx in enumerate(order):
        corrected[idx] = raw[idx] * (n - rank)

    # Enforce monotonicity and cap at 1.0
    corrected = np.minimum(corrected, 1.0)

    result = {}
    for i, label in enumerate(labels):
        result[label] = {
            "raw_p": float(raw[i]),
            "corrected_p": float(corrected[i]),
            "significant_05": bool(corrected[i] < 0.05),
        }
    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_experiment_metadata(model_name, model, n_train, n_test, class_names=None):
    """Print and return experiment metadata for reproducibility logging.

    Parameters
    ----------
    model_name : str
    model : keras Model (or any object with count_params())
    n_train, n_test : int
    class_names : list of str, optional

    Returns
    -------
    dict of metadata
    """
    try:
        param_count = model.count_params()
    except Exception:
        param_count = "unknown"

    meta = {
        "model_name": model_name,
        "param_count": param_count,
        "n_train": n_train,
        "n_test": n_test,
        "class_names": class_names,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    print(f"  [{model_name}] params={param_count}  train={n_train}  test={n_test}")
    return meta


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute a bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like of float
        Sample values (e.g. fold-level accuracies).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95 for a 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean, ci_lower, ci_upper, ci_level, n_bootstrap
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(values)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()

    alpha = 1.0 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def save_confusion_matrix(cm, filepath):
    """Save a confusion matrix as a CSV file."""
    import pandas as pd
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(cm).to_csv(filepath, index=False)
