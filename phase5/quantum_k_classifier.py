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
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

def add_rotation_layer(circuit, qubits, rot_fn, prefix, layer_idx):
    for i, qubit in enumerate(qubits):
        symbol = sympy.Symbol(f"{prefix}-{layer_idx}-{i}")
        circuit.append(rot_fn(symbol)(qubit))

def create_k_category_quantum_model(k, n_layers=1):
    # Total 17 qubits: 16 data + 1 readout
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    
    circuit = cirq.Circuit()
    
    # Entanglement layer
    for i in range(len(data_qubits) - 1):
        circuit.append(cirq.CZ(data_qubits[i], data_qubits[i+1]))
    circuit.append(cirq.CZ(data_qubits[-1], readout))
    
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    # Parametric layers
    for l in range(n_layers):
        for i, q in enumerate(data_qubits):
            # XX gates
            symbol_xx = sympy.Symbol(f'xx-{l}-{i}')
            circuit.append(cirq.XX(q, readout)**symbol_xx)
            # ZZ gates
            symbol_zz = sympy.Symbol(f'zz-{l}-{i}')
            circuit.append(cirq.ZZ(q, readout)**symbol_zz)

        add_rotation_layer(circuit, data_qubits, cirq.rx, "rx", l)
        add_rotation_layer(circuit, data_qubits, cirq.ry, "ry", l)

    circuit.append(cirq.H(readout))

    # Observables: We need k observables for k categories
    # We use the readout qubit (index 0) and data qubits (indices 1 to k-1)
    all_qubits = [readout] + data_qubits
    observables = [cirq.Z(all_qubits[i]) for i in range(k)]
        
    return circuit, observables

def convert_to_circuit(pca_features):
    """
    Expects 16 PCA features scaled to [0, 1].
    Converts them to a 4x4 grid of qubits.
    """
    values = np.ndarray.flatten(pca_features)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        # Map [0, 1] to [0, pi] for Ry rotations
        circuit.append(cirq.ry(np.pi * value)(qubits[i]))
    return circuit

class MultiClassPQC(tf.keras.layers.Layer):
    def __init__(self, circuit, observables, **kwargs):
        super().__init__(**kwargs)
        self.pqc = tfq.layers.PQC(circuit, observables)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, inputs):
        expectations = self.pqc(inputs)
        return self.softmax(expectations)

def create_qnn_multiclass_model(k, n_layers=1, learning_rate=0.01):
    circuit, observables = create_k_category_quantum_model(k, n_layers)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        MultiClassPQC(circuit, observables)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model
