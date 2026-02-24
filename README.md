# quantum-mnist
This project seeks to confirm results published in https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

We will explore quantum image processing via plankton data set found in this paper: https://arxiv.org/pdf/2108.05258.pdf

Other relevant research:
https://arxiv.org/pdf/2011.02831.pdf

### Project Phases
1. **confirm ipynb**: Confirm research conclusions in google colab using the MNIST dataset.
2. **basic binary quantum**: Apply a basic binary quantum classifier to the plankton dataset and compare with a "fair" classical neural net.
3. **optimise via param sweep**: Optimize the binary classifier through a hyperparameter sweep on neural architectures.
4. **compare to classical**: Perform the generalized quantum algorithm on the plankton dataset and compare results to established classical deep learning approaches.

---

# Phase 1: confirm ipynb
Done. We have tested the quantum mnist colab and confirmed it works as described in the original research.

# Phase 2: basic binary quantum
Done. We have implemented a binary quantum classifier template (`phasetwo/binary_quantum_classifier.py`) and verified its execution using a Docker environment with `tensorflow-quantum`. For the test pair 'aphanizomenon' vs 'bosmina', the quantum model achieved a baseline accuracy of approximately 48% on the test set, while the "fair" classical FFN often struggles with similar low-parameter configurations.

# Phase 3: optimise via param sweep
Done. We have implemented a comprehensive hyperparameter sweep script (`phasethree/optimize_binary_classifier.py`) that explores various neural architectures (hidden layers, neurons, activation functions, learning rates). The script has been verified in the Docker environment. Initial runs show that even with optimization, very shallow architectures remain limited in their classification power for complex plankton data.

# Phase 4: compare to classical
In this phase, we perform the generalized quantum algorithm on the plankton dataset and compare its performance to the classical deep learning approach found [here](https://arxiv.org/pdf/2108.05258.pdf).

---

## Plankton Gallery

Here's a random selection of 36 plankton from our dataset, showcased in a 6x6 grid:

<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; max-width: 100%;">
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1591776170518580-12521544422925-001609-017-916-676-160-308.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1590660477402556-11405867850399-004679-019-1060-1054-32-68.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1591718415538670-12463790306260-000059-044-1448-0-504-556.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1590660187371049-11405577826290-001779-038-2008-950-36-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1591718414551843-12463789306176-000049-033-2596-582-468-228.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1590652819713403-11398210261089-000099-075-1516-294-40-80.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1573585686719661-6767617524116-004769-013-1984-1996-216-500.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589976433667298-10721834338067-004239-008-1136-2184-32-76.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1573585686719661-6767617524116-004769-012-1926-2024-208-488.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589796523122270-10541926569271-005139-011-834-1302-24-76.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590473083169582-11218476391510-002739-076-1822-0-916-1548.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589551567916283-10296975108580-003579-016-2430-1436-56-60.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1591718414551843-12463789306176-000049-027-3040-920-264-124.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589551628905770-10297036113651-004189-026-2108-578-56-52.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1567483715216289-665738550349-005059-557-648-634-1316-1768.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589551536895555-10296944106003-003269-020-2538-816-48-56.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539136233589211-4818765048294-066239-027-628-1760-928-676.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589551495906546-10296903102595-002859-012-2782-1084-60-48.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1566187746252926-503915672838-005369-249-1028-618-96-228.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589537362893565-10282770296281-005529-020-1542-732-36-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539130710029141-4813241589152-011009-124-1610-68-804-1204.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589537268878602-10282676288466-004589-030-792-976-28-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539136234567396-4818766048377-066249-104-0-486-2212-740.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589537187855337-10282595281732-003779-013-906-1342-32-72.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539129903975531-4812435522148-002949-113-1646-0-512-684.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589537023909969-10282431268099-002139-017-2128-1294-40-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539130706069138-4813237588820-010969-044-0-1914-1296-188.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589530094453942-10275501932232-004849-004-1596-1764-52-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539129903975531-4812435522148-002949-091-1454-338-360-308.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589472448532789-10217856917166-004389-024-2382-824-48-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590660477402556-11405867850399-004679-019-1060-1054-32-68.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589472178489239-10217586894720-001689-036-1538-610-32-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590660187371049-11405577826290-001779-038-2008-950-36-64.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589472082484854-10217490886740-000729-013-2204-1520-52-36.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590660104417191-11405494819390-000949-020-612-1378-28-68.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589472074476026-10217482886075-000649-030-2206-984-56-48.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590652819713403-11398210261089-000099-075-1516-294-40-80.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589450883230134-10196291894563-004739-031-2082-588-56-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1590652815696860-11398206260757-000059-057-2600-52-60-48.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589464834111996-10210242629499-000249-022-1752-950-40-60.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1589976433667298-10721834338067-004239-008-1136-2184-32-76.jpeg" alt="Hydra" style="width: 100%; height: auto; border-radius: 4px;" />
  <img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1589450874216853-10196282893815-004649-039-2364-176-52-64.jpeg" alt="Ceratium" style="width: 100%; height: auto; border-radius: 4px;" />
</div>

---

## How to Run (Docker)

First, build the unified Docker image:

```bash
docker build -t quantum-mnist .
```

### Run Phase 2: Basic Binary Quantum (Data Ingress)
To verify the plankton data loading and class pairs:
```bash
docker run --rm quantum-mnist python phasetwo/plankton_ingress.py
```

### Run Phase 3: Optimise via Param Sweep
To see the hyperparameter sweep configuration:
```bash
docker run --rm quantum-mnist python phasethree/optimize_binary_classifier.py
```

### Run Phase 4: Compare to Classical (Full Experiments)
To run the full experiment suite and save results to your local machine:
```bash
docker run --rm -v $(pwd)/phasefour/results:/app/phasefour/results quantum-mnist python phasefour/run_experiments.py
```
