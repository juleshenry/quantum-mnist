r"""
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
r"""

import numpy as np
import cirq
import sympy
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import os

def create_pqc_subsystem(n_qubits, n_layers=1):
    """
    Creates a small PQC for unit tests.  Uses a linear chain of
    ``cirq.GridQubit(i, 0)`` so that tests can run with any qubit
    count without paying the cost of a 17-qubit simulation.

    For the full production architecture that mirrors Phase 5, use
    ``create_production_pqc`` instead.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (including readout).
    n_layers : int
        Number of parametric layers.

    Returns
    -------
    circuit, qubits, symbols
    """
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    readout = qubits[-1]
    data_qubits = qubits[:-1]
    
    circuit = cirq.Circuit()
    symbols = []
    
    # Entanglement layer (linear chain for simplicity in subsystem)
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
    if len(data_qubits) > 0:
        circuit.append(cirq.CZ(data_qubits[-1], readout))
    
    # Parametric layers
    for l in range(n_layers):
        for i, q in enumerate(data_qubits):
            # XX gates
            symbol_xx = sympy.Symbol(f'xx-{l}-{i}')
            circuit.append(cirq.XX(q, readout)**symbol_xx)
            symbols.append(symbol_xx)
            # ZZ gates
            symbol_zz = sympy.Symbol(f'zz-{l}-{i}')
            circuit.append(cirq.ZZ(q, readout)**symbol_zz)
            symbols.append(symbol_zz)
            
    return circuit, qubits, symbols


def create_production_pqc(n_layers=1):
    """
    Creates the exact PQC architecture used in Phase 5 experiments.

    Architecture
    ------------
    * **17 qubits**: 16 data qubits arranged as a 4x4 ``GridQubit`` grid
      plus 1 readout qubit at ``GridQubit(-1, -1)``.
    * **Entanglement layer**: Linear CZ chain across the 16 data qubits
      (15 CZ gates) plus one CZ connecting the last data qubit to the
      readout (16 CZ gates total).
    * **Parametric layers** (repeated ``n_layers`` times): For each of the
      16 data qubits, two parameterised two-qubit gates are applied
      against the readout qubit:
        - ``cirq.XX(data_qubit, readout) ** symbol``
        - ``cirq.ZZ(data_qubit, readout) ** symbol``
      giving 32 trainable parameters per layer.

    This is a direct replica of
    ``phase5/quantum_k_classifier.py::create_k_category_quantum_model``,
    stripped of the observable definitions (not needed for expressibility /
    entanglement analysis).

    Parameters
    ----------
    n_layers : int
        Number of parametric layers.

    Returns
    -------
    circuit : cirq.Circuit
    qubits : list[cirq.GridQubit]
        All 17 qubits (data_qubits + [readout]).
    symbols : list[sympy.Symbol]
        Trainable symbols (32 per layer).
    """
    data_qubits = cirq.GridQubit.rect(4, 4)       # 16 qubits
    readout = cirq.GridQubit(-1, -1)               # 1 readout qubit

    circuit = cirq.Circuit()
    symbols = []

    # Entanglement layer — identical to Phase 5
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i + 1]))
    circuit.append(cirq.CZ(data_qubits[-1], readout))

    # Parametric layers — identical to Phase 5
    for l in range(n_layers):
        for i, q in enumerate(data_qubits):
            symbol_xx = sympy.Symbol(f'xx-{l}-{i}')
            circuit.append(cirq.XX(q, readout) ** symbol_xx)
            symbols.append(symbol_xx)

            symbol_zz = sympy.Symbol(f'zz-{l}-{i}')
            circuit.append(cirq.ZZ(q, readout) ** symbol_zz)
            symbols.append(symbol_zz)

    qubits = list(data_qubits) + [readout]
    return circuit, qubits, symbols

def get_haar_distribution(n_qubits, n_bins=75):
    """Returns the Haar fidelity distribution for n_qubits."""
    d = 2**n_qubits
    fidelities = np.linspace(0, 1, n_bins)
    p_haar = (d - 1) * (1 - fidelities)**(d - 2)
    # Normalize
    p_haar /= np.sum(p_haar)
    return fidelities, p_haar

def sample_pqc_fidelities(circuit, qubits, symbols, n_samples=1000, rng=None):
    """Samples fidelities between pairs of states generated by random parameters."""
    if rng is None:
        rng = np.random.RandomState(42)
    simulator = cirq.Simulator()
    fidelities = []
    
    for _ in range(n_samples):
        # Sample two sets of random parameters in [0, 2] (full rotation range for Cirq's exponent)
        params1 = {s: rng.uniform(0, 2) for s in symbols}
        params2 = {s: rng.uniform(0, 2) for s in symbols}
        
        # Get state vectors
        result1 = simulator.simulate(circuit, param_resolver=params1)
        state1 = result1.final_state_vector
        
        result2 = simulator.simulate(circuit, param_resolver=params2)
        state2 = result2.final_state_vector
        
        # Fidelity F = |<psi1|psi2>|^2
        fidelity = np.abs(np.vdot(state1, state2))**2
        fidelities.append(fidelity)
        
    return np.array(fidelities)

def calculate_expressibility(pqc_fidelities, n_qubits, n_bins=75):
    """Calculates the KL divergence between PQC and Haar distributions."""
    # Empirical distribution
    p_pqc, bin_edges = np.histogram(pqc_fidelities, bins=n_bins, range=(0, 1), density=True)
    p_pqc /= np.sum(p_pqc)
    
    # Haar distribution
    _, p_haar = get_haar_distribution(n_qubits, n_bins)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_pqc += epsilon
    p_haar += epsilon
    
    # KL Divergence: sum(P * log(P/Q))
    kl_div = np.sum(p_pqc * np.log(p_pqc / p_haar))
    return kl_div

def calculate_meyer_wallach(circuit, qubits, symbols, n_samples=100, rng=None):
    """Calculates the Meyer-Wallach entanglement measure averaged over random parameters.

    Returns
    -------
    dict with keys: mean, std, values (raw per-sample measures)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    simulator = cirq.Simulator()
    n = len(qubits)
    q_measures = []
    
    for _ in range(n_samples):
        params = {s: rng.uniform(0, 2) for s in symbols}
        result = simulator.simulate(circuit, param_resolver=params)
        state = result.final_state_vector
        
        sum_det = 0
        for i in range(n):
            # Reshape to (2, 2, 2, ...)
            reshaped_state = state.reshape([2]*n)
            # Move index i to the front
            permuted_state = np.moveaxis(reshaped_state, i, 0)
            # Flatten the rest
            reshaped_for_trace = permuted_state.reshape(2, -1)
            # rho_i = Tr_{not i} (|psi><psi|)
            rho_i = np.dot(reshaped_for_trace, reshaped_for_trace.T.conj())
            
            # det(rho_i)
            det_rho_i = np.linalg.det(rho_i)
            sum_det += det_rho_i
            
        q_psi = (4.0 / n) * np.real(sum_det)
        q_measures.append(q_psi)

    q_measures = np.array(q_measures)
    return {
        "mean": float(np.mean(q_measures)),
        "std": float(np.std(q_measures)),
        "values": q_measures,
    }

def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, rng=None):
    """Compute bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.RandomState(42)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()
    alpha = 1.0 - ci
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "ci_level": ci,
    }


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    # ── Configuration ───────────────────────────────────────────────
    # Use the exact Phase 5 production architecture (17 qubits: 16
    # data in a 4x4 grid + 1 readout).  Sample counts are reduced
    # relative to a 4-qubit toy circuit because 17-qubit state-vector
    # simulation is ~32 000x larger (2^17 vs 2^4).
    N_FIDELITY_SAMPLES = 500       # pairs for expressibility
    N_ENTANGLEMENT_SAMPLES = 50    # states for Meyer-Wallach
    N_BOOT_EXPR = 500              # bootstrap resamples (expressibility)
    N_BOOT_ENT = 5000              # bootstrap resamples (entanglement)
    LAYER_SWEEP = [1, 2, 3]        # layers to evaluate

    results = []
    os.makedirs('phase7', exist_ok=True)

    circuit_0, qubits_0, _ = create_production_pqc(n_layers=1)
    n_qubits = len(qubits_0)

    print(f"Running Expressibility & Entanglement Analysis")
    print(f"Architecture: Phase 5 production PQC ({n_qubits} qubits, "
          f"4x4 data grid + readout)")
    print(f"Global seed: {SEED}")
    print(f"Fidelity samples: {N_FIDELITY_SAMPLES}, "
          f"Entanglement samples: {N_ENTANGLEMENT_SAMPLES}")
    print(f"Layer sweep: {LAYER_SWEEP}")
    print()

    for layers in LAYER_SWEEP:
        # Per-layer deterministic RNG streams
        rng_expr = np.random.RandomState(SEED + layers)
        rng_ent = np.random.RandomState(SEED + 100 + layers)
        rng_boot_expr = np.random.RandomState(SEED + 200 + layers)
        rng_boot_ent = np.random.RandomState(SEED + 300 + layers)

        circuit, qubits, symbols = create_production_pqc(layers)

        fidelities = sample_pqc_fidelities(
            circuit, qubits, symbols,
            n_samples=N_FIDELITY_SAMPLES, rng=rng_expr,
        )
        expr = calculate_expressibility(fidelities, n_qubits)
        ent_result = calculate_meyer_wallach(
            circuit, qubits, symbols,
            n_samples=N_ENTANGLEMENT_SAMPLES, rng=rng_ent,
        )

        # Bootstrap CI on expressibility: resample fidelity sets
        expr_boot = []
        for _ in range(N_BOOT_EXPR):
            boot_idx = rng_boot_expr.choice(
                len(fidelities), size=len(fidelities), replace=True,
            )
            expr_boot.append(calculate_expressibility(fidelities[boot_idx], n_qubits))
        expr_boot = np.array(expr_boot)
        expr_ci = {
            "mean": float(expr),
            "std": float(np.std(expr_boot)),
            "ci_lower": float(np.percentile(expr_boot, 2.5)),
            "ci_upper": float(np.percentile(expr_boot, 97.5)),
            "ci_level": 0.95,
        }

        # Bootstrap CI on entanglement
        ent_ci = bootstrap_ci(
            ent_result["values"],
            n_bootstrap=N_BOOT_ENT, ci=0.95, rng=rng_boot_ent,
        )

        results.append({
            'layers': layers,
            'expressibility': expr_ci,
            'entanglement': ent_ci,
        })

        print(f"Layers: {layers} | "
              f"Expr: {expr_ci['mean']:.4f} "
              f"[{expr_ci['ci_lower']:.4f}, {expr_ci['ci_upper']:.4f}] | "
              f"Ent: {ent_ci['mean']:.4f} "
              f"[{ent_ci['ci_lower']:.4f}, {ent_ci['ci_upper']:.4f}]")

    # ── Summary report ──────────────────────────────────────────────
    print(f"\n--- Phase 7 Summary ({n_qubits}-qubit production PQC, "
          f"95% CIs) ---")
    for res in results:
        e = res['expressibility']
        t = res['entanglement']
        print(f"L={res['layers']}: "
              f"Expr={e['mean']:.4f} "
              f"95%CI[{e['ci_lower']:.4f},{e['ci_upper']:.4f}], "
              f"Ent={t['mean']:.4f} "
              f"95%CI[{t['ci_lower']:.4f},{t['ci_upper']:.4f}]")

    # ── Structured JSON output ──────────────────────────────────────
    json_results = {
        "phase": 7,
        "description": "Expressibility & Entanglement Analysis",
        "architecture": (
            f"Phase 5 production PQC ({n_qubits} qubits: "
            f"16 data in 4x4 GridQubit grid + 1 readout at (-1,-1))"
        ),
        "seed": SEED,
        "n_fidelity_samples": N_FIDELITY_SAMPLES,
        "n_entanglement_samples": N_ENTANGLEMENT_SAMPLES,
        "n_bootstrap_expr": N_BOOT_EXPR,
        "n_bootstrap_ent": N_BOOT_ENT,
        "layer_sweep": LAYER_SWEEP,
        "results": results,
        "limitations": [
            "Noiseless statevector simulation — real hardware noise "
            "(decoherence, gate errors) is not modelled.",
            "Sample counts reduced for computational feasibility under "
            "AMD64 emulation on ARM64.",
        ],
        "analysis": (
            "As the number of layers increases, the Expressibility value "
            "(KL divergence from Haar) should decrease, indicating the "
            "PQC explores the Hilbert space more uniformly.  The "
            "Meyer-Wallach entanglement measure should increase, showing "
            "the circuit generates more global entanglement.  These "
            "metrics are computed on the exact 17-qubit architecture "
            "deployed in Phase 5 experiments."
        ),
    }

    with open("phase7/results_rigor.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # ── Human-readable text report ──────────────────────────────────
    with open("phase7/results_rigor.txt", "w") as f:
        f.write("Phase 7: Expressibility & Entanglement Analysis\n")
        f.write(f"Architecture: {json_results['architecture']}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Fidelity samples: {N_FIDELITY_SAMPLES}, "
                f"Entanglement samples: {N_ENTANGLEMENT_SAMPLES}\n")
        f.write("--------------------------------------------------\n")
        for res in results:
            e = res['expressibility']
            t = res['entanglement']
            f.write(
                f"Layers: {res['layers']}, "
                f"Expr: {e['mean']:.4f} "
                f"95%CI[{e['ci_lower']:.4f},{e['ci_upper']:.4f}], "
                f"Ent: {t['mean']:.4f} +/- {t['std']:.4f} "
                f"95%CI[{t['ci_lower']:.4f},{t['ci_upper']:.4f}]\n"
            )

        f.write(f"\n{json_results['analysis']}\n")
        f.write("\nLimitations:\n")
        for lim in json_results["limitations"]:
            f.write(f"  - {lim}\n")
