"""Statistical power analysis for binary pair selection in Phase 4.

This module provides the mathematical justification for:
1. Why 4 pairs is insufficient for meaningful QNN vs. classical comparison
2. How many pairs are needed for adequate statistical power
3. A deterministic, reproducible pair-selection algorithm

The core insight is that with 5-fold CV, per-pair Wilcoxon signed-rank
tests cannot achieve p < 0.05 (minimum possible p = 2/2^5 = 0.0625).
Therefore we treat *pairs* as the unit of replication and perform an
aggregate test across pair-level accuracy differences.

Usage
-----
Run standalone to print the full analysis report::

    python utils/power_analysis.py [--data-dir DATA_DIR] [--min-per-class 80]

Or import and use programmatically::

    from power_analysis import (
        enumerate_viable_pairs,
        select_pairs,
        per_pair_power_table,
        aggregate_power_table,
    )
"""

import os
import math
import argparse
from itertools import combinations

# scipy is available in the Docker image; guard for lightweight use
try:
    from scipy.stats import t as t_dist, binom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ===================================================================
# Constants
# ===================================================================

# Classes excluded from pair selection: ambiguous, non-biological,
# or semantically overlapping categories that would confound results.
EXCLUDE_CLASSES = frozenset({
    'unknown',           # catch-all, no consistent morphology
    'unknown_plankton',  # same
    'dirt',              # non-biological debris
    'fish',              # vertebrate, not plankton
    'filament',          # ambiguous morphological grouping
})


# ===================================================================
# Dataset Enumeration
# ===================================================================

def count_images_per_class(data_dir):
    """Count images per class from the dataset directory.

    Parameters
    ----------
    data_dir : str
        Path to the plankton dataset root (e.g. ``data/zooplankton_0p5x``).

    Returns
    -------
    dict mapping class_name -> int (image count)
    """
    counts = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for cls in sorted(os.listdir(data_dir)):
        training_dir = os.path.join(data_dir, cls, 'training_data')
        if not os.path.isdir(training_dir):
            continue
        n = sum(1 for f in os.listdir(training_dir)
                if f.lower().endswith(('.jpeg', '.jpg', '.png')))
        counts[cls] = n
    return counts


def enumerate_viable_classes(class_counts, min_per_class=80):
    """Return sorted list of classes eligible for pair selection.

    Filters out:
    - Classes in ``EXCLUDE_CLASSES`` (ambiguous/non-biological)
    - Classes with fewer than ``min_per_class`` images

    Parameters
    ----------
    class_counts : dict
        Mapping of class_name -> image count.
    min_per_class : int
        Minimum images required. With 5-fold CV, each fold's test set
        gets ~20% of images. At min=80, that's ~16 test samples per
        class per fold -- enough for meaningful accuracy estimates.

    Returns
    -------
    list of str, sorted alphabetically
    """
    return sorted(
        c for c, n in class_counts.items()
        if c not in EXCLUDE_CLASSES and n >= min_per_class
    )


def enumerate_viable_pairs(viable_classes):
    """Return all C(n,2) pairs from viable classes.

    Parameters
    ----------
    viable_classes : list of str

    Returns
    -------
    list of (str, str) tuples, sorted
    """
    return sorted(combinations(viable_classes, 2))


# ===================================================================
# Per-Pair Power Analysis
# ===================================================================

def wilcoxon_min_p(n):
    """Minimum achievable two-sided p-value for Wilcoxon signed-rank at sample size n.

    The Wilcoxon signed-rank test statistic under H0 takes integer values.
    The most extreme outcome (all differences same sign with maximum rank
    sum) gives the smallest p-value = 2 / 2^n for the exact test.

    At n=5: min p = 0.0625 > 0.05 -- CANNOT reject H0.
    At n=6: min p = 0.03125 < 0.05 -- barely possible.
    """
    return 2.0 / (2 ** n)


def paired_ttest_min_d(n, alpha=0.05, power_target=0.80):
    """Minimum Cohen's d for a paired t-test to achieve target power.

    Uses the approximation:
        d = (t_crit + z_power) / sqrt(n)

    where t_crit is the two-sided critical value at significance level
    alpha, and z_power is the standard normal quantile for the target
    power (0.842 for 80%).

    Parameters
    ----------
    n : int
        Number of paired observations (CV folds).
    alpha : float
        Significance level (two-sided).
    power_target : float
        Desired statistical power.

    Returns
    -------
    float : minimum detectable effect size (Cohen's d)
    """
    if not HAS_SCIPY:
        # Rough approximation without scipy
        # t_crit ≈ 2.0 for large df, z_80 = 0.842
        return (2.0 + 0.842) / math.sqrt(n)

    from scipy.stats import norm as norm_dist
    df = n - 1
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    z_power = norm_dist.ppf(power_target)
    return (t_crit + z_power) / math.sqrt(n)


def per_pair_power_table(fold_counts=None, alpha=0.05):
    """Generate a table showing per-pair statistical power limitations.

    Returns
    -------
    list of dict with keys: n_folds, wilcoxon_min_p, can_reject_wilcoxon,
                             ttest_d_for_80pct_power
    """
    if fold_counts is None:
        fold_counts = [5, 6, 7, 8, 10, 15, 20]
    rows = []
    for n in fold_counts:
        min_p = wilcoxon_min_p(n)
        d = paired_ttest_min_d(n, alpha=alpha)
        rows.append({
            'n_folds': n,
            'wilcoxon_min_p': min_p,
            'can_reject_wilcoxon': min_p < alpha,
            'ttest_d_for_80pct_power': d,
        })
    return rows


# ===================================================================
# Aggregate Power Analysis (pairs as unit of replication)
# ===================================================================

def aggregate_power(n_pairs, effect_size_d, alpha=0.05):
    """Compute power for a one-sample t-test on pair-level mean differences.

    The aggregate test treats each pair's mean accuracy difference
    (QNN - Fair, averaged over folds) as an independent observation.
    A one-sample t-test then asks: is the population mean of these
    differences significantly different from zero?

    Parameters
    ----------
    n_pairs : int
        Number of pairs tested.
    effect_size_d : float
        Expected Cohen's d = mean(deltas) / std(deltas).
    alpha : float
        Significance level (two-sided).

    Returns
    -------
    float : statistical power
    """
    if not HAS_SCIPY:
        return float('nan')

    df = n_pairs - 1
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    ncp = effect_size_d * math.sqrt(n_pairs)
    # Power = P(reject H0) = P(|T| > t_crit | ncp)
    power = (1 - t_dist.cdf(t_crit, df, loc=ncp)
             + t_dist.cdf(-t_crit, df, loc=ncp))
    return power


def aggregate_power_table(effect_size_d=0.65, alpha=0.05):
    """Generate a table showing aggregate power at various pair counts.

    The default effect size d=0.65 is derived from the existing 4-pair
    pilot data:
        deltas = [-4.2%, +5.4%, +28.0%, +6.6%]
        mean = 8.95%, std = 13.6% -> d = 0.66

    Returns
    -------
    list of dict with keys: n_pairs, power, compute_hours_est
    """
    pair_counts = [4, 10, 15, 20, 25, 30, 40]
    rows = []
    for m in pair_counts:
        power = aggregate_power(m, effect_size_d, alpha)
        # Compute estimate: 3 models x 5 folds x ~60s per model-fold
        hours = m * 3 * 5 * 60 / 3600
        rows.append({
            'n_pairs': m,
            'power': power,
            'compute_hours_est': hours,
        })
    return rows


def min_pairs_for_power(effect_size_d=0.65, alpha=0.05, target_power=0.80):
    """Find minimum number of pairs needed to achieve target power.

    Returns
    -------
    int
    """
    for m in range(2, 500):
        if aggregate_power(m, effect_size_d, alpha) >= target_power:
            return m
    return 500


# ===================================================================
# Pair Selection Algorithm
# ===================================================================

def select_pairs(viable_classes, class_counts, n_target=25, seed=42):
    """Select n_target pairs with full class coverage and balance preference.

    Algorithm:
    1. **Coverage phase:** Greedily select pairs that cover uncovered
       classes, preferring pairs that cover 2 new classes over 1, and
       breaking ties by class-size balance (ratio closer to 1:1).
    2. **Filling phase:** Add remaining pairs preferring balanced sizes
       (ratio < 3:1), using a seeded shuffle for reproducibility.

    Parameters
    ----------
    viable_classes : list of str
        Classes eligible for pair selection.
    class_counts : dict
        Mapping class_name -> image count.
    n_target : int
        Number of pairs to select.
    seed : int
        Random seed for the filling phase.

    Returns
    -------
    list of (str, str) tuples, sorted alphabetically
    """
    import random
    rng = random.Random(seed)

    all_pairs = list(combinations(sorted(viable_classes), 2))

    selected = []
    remaining_classes = set(viable_classes)

    # Phase 1: Cover all classes
    while remaining_classes:
        candidates = [
            (a, b) for a, b in all_pairs
            if (a in remaining_classes or b in remaining_classes)
            and (a, b) not in selected
        ]
        if not candidates:
            break

        def _score(pair):
            a, b = pair
            covers = (a in remaining_classes) + (b in remaining_classes)
            ratio = max(class_counts[a], class_counts[b]) / max(1, min(class_counts[a], class_counts[b]))
            return (-covers, ratio)

        candidates.sort(key=_score)
        best = candidates[0]
        selected.append(best)
        remaining_classes -= set(best)

    # Phase 2: Fill to n_target with balanced pairs
    selected_set = set(selected)
    unused = [p for p in all_pairs if p not in selected_set]
    balanced = [p for p in unused
                if max(class_counts[p[0]], class_counts[p[1]])
                / max(1, min(class_counts[p[0]], class_counts[p[1]])) < 3.0]
    rng.shuffle(balanced)

    while len(selected) < n_target and balanced:
        selected.append(balanced.pop(0))

    # If still short, use any remaining
    if len(selected) < n_target:
        remaining_pairs = [p for p in unused if p not in set(selected)]
        rng.shuffle(remaining_pairs)
        while len(selected) < n_target and remaining_pairs:
            selected.append(remaining_pairs.pop(0))

    selected.sort()
    return selected


# ===================================================================
# Coverage Statistics
# ===================================================================

def coverage_ci_width(n_pairs, n_population, confidence=0.95):
    """Margin of error for estimating a proportion from n_pairs out of n_population.

    Uses the normal approximation with finite population correction:
        E = z * sqrt(p(1-p)/n) * sqrt((N-n)/(N-1))

    Assumes worst-case p=0.5 (maximum variance).

    Returns
    -------
    float : half-width of confidence interval (e.g., 0.10 means +/-10%)
    """
    if not HAS_SCIPY:
        z = 1.96
    else:
        from scipy.stats import norm as norm_dist
        z = norm_dist.ppf(1 - (1 - confidence) / 2)

    # Worst-case variance
    p = 0.5
    margin = z * math.sqrt(p * (1 - p) / n_pairs)
    # Finite population correction
    if n_population > 1:
        fpc = math.sqrt((n_population - n_pairs) / (n_population - 1))
        margin *= fpc
    return margin


# ===================================================================
# Report Generation
# ===================================================================

def generate_report(class_counts=None, data_dir=None, min_per_class=80, n_target=25):
    """Generate a comprehensive power analysis report.

    Parameters
    ----------
    class_counts : dict or None
        If None, counts are read from data_dir.
    data_dir : str or None
        Path to dataset. Only used if class_counts is None.
    min_per_class : int
        Minimum images per class.
    n_target : int
        Target number of pairs.

    Returns
    -------
    str : formatted report
    """
    if class_counts is None:
        if data_dir is None:
            data_dir = os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')
        class_counts = count_images_per_class(data_dir)

    viable = enumerate_viable_classes(class_counts, min_per_class)
    all_pairs = enumerate_viable_pairs(viable)
    selected = select_pairs(viable, class_counts, n_target)

    lines = []
    lines.append("=" * 70)
    lines.append("POWER ANALYSIS REPORT: Phase 4 Binary Pair Selection")
    lines.append("=" * 70)

    # --- Dataset summary ---
    lines.append(f"\nDataset: {len(class_counts)} total classes, "
                 f"{sum(class_counts.values())} total images")
    lines.append(f"Excluded classes: {sorted(EXCLUDE_CLASSES & set(class_counts.keys()))}")
    lines.append(f"Min images per class: {min_per_class}")
    lines.append(f"Eligible classes: {len(viable)}")
    lines.append(f"Eligible pairs: C({len(viable)},2) = {len(all_pairs)}")

    # --- Per-pair power limitations ---
    lines.append(f"\n{'─' * 70}")
    lines.append("PER-PAIR POWER (Wilcoxon signed-rank / paired t-test)")
    lines.append(f"{'─' * 70}")
    lines.append("")
    lines.append("The Wilcoxon signed-rank test requires the minimum possible")
    lines.append("p-value to be < alpha. With n paired observations, the")
    lines.append("minimum p = 2 / 2^n (two-sided).")
    lines.append("")
    lines.append(f"  {'n_folds':>7}  {'min_p':>10}  {'can_reject':>12}  {'d_for_80%_power':>18}")
    lines.append(f"  {'─' * 7}  {'─' * 10}  {'─' * 12}  {'─' * 18}")
    for row in per_pair_power_table():
        can = "YES" if row['can_reject_wilcoxon'] else "NO"
        lines.append(f"  {row['n_folds']:>7}  {row['wilcoxon_min_p']:>10.4f}  {can:>12}  "
                     f"{row['ttest_d_for_80pct_power']:>18.2f}")

    lines.append("")
    lines.append("FINDING: At n=5 folds, Wilcoxon CANNOT reject H0 at alpha=0.05.")
    lines.append("The paired t-test fallback requires Cohen's d >= 1.62 (very large).")
    lines.append("Per-pair significance claims at n=5 are inherently unreliable.")

    # --- Aggregate power ---
    lines.append(f"\n{'─' * 70}")
    lines.append("AGGREGATE POWER (pairs as unit of replication)")
    lines.append(f"{'─' * 70}")
    lines.append("")
    lines.append("Strategy: compute delta_i = mean(QNN_acc - Fair_acc) for each pair")
    lines.append("across 5 folds. Then test H0: mean(delta) = 0 using a one-sample")
    lines.append("t-test (or Wilcoxon) on the m pair-level deltas.")
    lines.append("")
    lines.append("Pilot data (4 pairs): deltas = [-4.2%, +5.4%, +28.0%, +6.6%]")
    lines.append("  mean = 8.95%, std = 13.6%, observed d = 0.66")
    lines.append("")
    lines.append(f"  {'n_pairs':>7}  {'power':>8}  {'compute_hrs':>12}")
    lines.append(f"  {'─' * 7}  {'─' * 8}  {'─' * 12}")
    for row in aggregate_power_table():
        lines.append(f"  {row['n_pairs']:>7}  {row['power']:>8.2f}  "
                     f"{row['compute_hours_est']:>12.1f}")

    m_min = min_pairs_for_power()
    lines.append(f"\nMinimum pairs for 80% power (d=0.65): {m_min}")
    lines.append(f"Selected pairs: {n_target} (power = "
                 f"{aggregate_power(n_target, 0.65):.0%})")

    # --- Coverage ---
    margin = coverage_ci_width(n_target, len(all_pairs))
    lines.append(f"\n{'─' * 70}")
    lines.append("COVERAGE")
    lines.append(f"{'─' * 70}")
    lines.append(f"Testing {n_target} of {len(all_pairs)} eligible pairs "
                 f"({100 * n_target / len(all_pairs):.1f}%)")
    lines.append(f"95% CI width for win-rate estimate: +/- {margin:.1%}")

    # --- Selected pairs ---
    lines.append(f"\n{'─' * 70}")
    lines.append(f"SELECTED PAIRS ({len(selected)})")
    lines.append(f"{'─' * 70}")
    lines.append(f"Selection: greedy class coverage + balanced size, seed=42")
    lines.append("")

    class_usage = {}
    for i, (a, b) in enumerate(selected):
        ratio = max(class_counts[a], class_counts[b]) / min(class_counts[a], class_counts[b])
        total = class_counts[a] + class_counts[b]
        lines.append(f"  {i + 1:2d}. {a} ({class_counts[a]}) vs "
                     f"{b} ({class_counts[b]}) "
                     f"[total={total}, ratio={ratio:.1f}]")
        class_usage[a] = class_usage.get(a, 0) + 1
        class_usage[b] = class_usage.get(b, 0) + 1

    lines.append(f"\nClass coverage: {len(class_usage)}/{len(viable)} eligible classes")
    uncovered = set(viable) - set(class_usage.keys())
    if uncovered:
        lines.append(f"UNCOVERED: {sorted(uncovered)}")
    else:
        lines.append("All eligible classes represented.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Power analysis for Phase 4 binary pair selection")
    parser.add_argument('--data-dir', default=None,
                        help="Path to plankton dataset (default: DATA_DIR env or data/zooplankton_0p5x)")
    parser.add_argument('--min-per-class', type=int, default=80,
                        help="Minimum images per class (default: 80)")
    parser.add_argument('--n-pairs', type=int, default=25,
                        help="Target number of pairs (default: 25)")
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('DATA_DIR', 'data/zooplankton_0p5x')

    # Try to read from filesystem; fall back to hardcoded counts
    try:
        class_counts = count_images_per_class(data_dir)
        print(f"(Read class counts from {data_dir})")
    except FileNotFoundError:
        print(f"(Dataset not found at {data_dir}; using hardcoded counts)")
        class_counts = _HARDCODED_COUNTS

    report = generate_report(
        class_counts=class_counts,
        min_per_class=args.min_per_class,
        n_target=args.n_pairs,
    )
    print(report)


# Hardcoded counts for environments without the dataset
_HARDCODED_COUNTS = {
    'dinobryon': 3321, 'nauplius': 1507, 'maybe_cyano': 1364,
    'diaphanosoma': 1089, 'asterionella': 1055, 'uroglena': 953,
    'cyclops': 866, 'ceratium': 814, 'rotifers': 744, 'daphnia': 721,
    'asplanchna': 607, 'eudiaptomus': 537, 'kellicottia': 519,
    'paradileptus': 424, 'keratella_quadrata': 420, 'filament': 405,
    'fragilaria': 306, 'conochilus': 264, 'trichocerca': 255,
    'unknown': 245, 'aphanizomenon': 225, 'fish': 222, 'leptodora': 203,
    'synchaeta': 142, 'brachionus': 138, 'dirt': 131,
    'keratella_cochlearis': 113, 'polyarthra': 80, 'bosmina': 80,
    'unknown_plankton': 71, 'daphnia_skins': 46, 'copepod_skins': 33,
    'hydra': 18, 'diatom_chain': 17, 'chaoborus': 10,
}


if __name__ == "__main__":
    main()
