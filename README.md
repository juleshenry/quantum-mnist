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

A diverse 6x6 grid showing unique samples from 36 different plankton classes:

<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; max-width: 600px;">
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/aphanizomenon/training_data/SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg" alt="aphanizomenon" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">aphanizomenon</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/asplanchna/training_data/SPC-EAWAG-0P5X-1526947882588679-1089736153896-006729-006-2164-1964-132-160.jpeg" alt="asplanchna" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">asplanchna</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/asterionella/training_data/SPC-EAWAG-0P5X-1559498410191177-6403834470952-000009-061-1220-2378-52-40.jpeg" alt="asterionella" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">asterionella</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/bosmina/training_data/SPC-EAWAG-0P5X-1538179277542653-3861823145465-000679-094-3456-1308-108-60.jpeg" alt="bosmina" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">bosmina</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/brachionus/training_data/SPC-EAWAG-0P5X-1536022667859413-1705244249832-034579-003-2952-2218-48-28.jpeg" alt="brachionus" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">brachionus</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/ceratium/training_data/SPC-EAWAG-0P5X-1526947357532857-1089211110251-001479-018-2234-622-60-52.jpeg" alt="ceratium" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">ceratium</span></div>
  
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/chaoborus/training_data/SPC-EAWAG-0P5X-1561363726179016-8269122585331-005169-056-1368-1298-404-112.jpeg" alt="chaoborus" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">chaoborus</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/conochilus/training_data/SPC-EAWAG-0P5X-1542072407670616-7754895449918-051979-018-1142-2042-172-172.jpeg" alt="conochilus" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">conochilus</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/copepod_skins/training_data/SPC-EAWAG-0P5X-1556787796114071-3693261041663-001859-020-1102-800-100-100.jpeg" alt="copepod_skins" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">copepod_skins</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/cyclops/training_data/SPC-EAWAG-0P5X-1526948087602056-1089941170938-008779-020-3268-256-108-112.jpeg" alt="cyclops" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">cyclops</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/daphnia/training_data/SPC-EAWAG-0P5X-1526947464531326-1089318119146-002549-019-2166-472-240-216.jpeg" alt="daphnia" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">daphnia</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/daphnia_skins/training_data/SPC-EAWAG-0P5X-1563876012798986-10781374001834-000029-064-1496-1116-124-56.jpeg" alt="daphnia_skins" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">daphnia_skins</span></div>
  
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/diaphanosoma/training_data/SPC-EAWAG-0P5X-1529021007771870-735003526491-001979-103-2846-1084-304-344.jpeg" alt="diaphanosoma" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">diaphanosoma</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/diatom_chain/training_data/SPC-EAWAG-0P5X-1580983255524793-1728789601675-000459-052-1312-604-36-96.jpeg" alt="diatom_chain" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">diatom_chain</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/dinobryon/training_data/SPC-EAWAG-0P5X-1527038002688896-1179854948616-043930-019-1020-888-68-92.jpeg" alt="dinobryon" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">dinobryon</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/dirt/training_data/SPC-EAWAG-0P5X-1555333505560851-2238992078654-002959-007-1798-2014-40-60.jpeg" alt="dirt" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">dirt</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/eudiaptomus/training_data/SPC-EAWAG-0P5X-1526947642556033-1089496133944-004329-010-2198-1492-240-112.jpeg" alt="eudiaptomus" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">eudiaptomus</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/filament/training_data/SPC-EAWAG-0P5X-1526994875836773-1136728737520-044659-010-1038-1550-120-216.jpeg" alt="filament" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">filament</span></div>
  
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/fish/training_data/SPC-EAWAG-0P5X-1528334145951139-48151026950-045359-119-542-1040-456-448.jpeg" alt="fish" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">fish</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/fragilaria/training_data/SPC-EAWAG-0P5X-1529626015843011-1340003537417-004059-093-378-1238-52-44.jpeg" alt="fragilaria" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">fragilaria</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/hydra/training_data/SPC-EAWAG-0P5X-1539129903975531-4812435522148-002949-091-1454-338-360-308.jpeg" alt="hydra" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">hydra</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/kellicottia/training_data/SPC-EAWAG-0P5X-1526949277702744-1091131269866-020679-000-1994-2460-116-40.jpeg" alt="kellicottia" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">kellicottia</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/keratella_cochlearis/training_data/SPC-EAWAG-0P5X-1539561773899643-5244299076217-001649-171-2074-256-40-60.jpeg" alt="keratella_cochlearis" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">keratella_cochlearis</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/keratella_quadrata/training_data/SPC-EAWAG-0P5X-1526948227598243-1090081182577-010179-025-658-1240-40-52.jpeg" alt="keratella_quadrata" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">keratella_quadrata</span></div>
  
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/leptodora/training_data/SPC-EAWAG-0P5X-1530927400893106-2641370428069-057899-046-1744-1978-316-260.jpeg" alt="leptodora" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">leptodora</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/maybe_cyano/training_data/SPC-EAWAG-0P5X-1569405610760714-2587605614311-000009-082-1100-2068-56-36.jpeg" alt="maybe_cyano" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">maybe_cyano</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/nauplius/training_data/SPC-EAWAG-0P5X-1526948725652431-1090579223977-015159-008-2182-1826-64-116.jpeg" alt="nauplius" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">nauplius</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/paradileptus/training_data/SPC-EAWAG-0P5X-1560416504906225-7321915289151-000949-078-1148-1434-28-88.jpeg" alt="paradileptus" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">paradileptus</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/polyarthra/training_data/SPC-EAWAG-0P5X-1555333322564272-2238809063441-001129-039-1080-972-52-56.jpeg" alt="polyarthra" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">polyarthra</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/rotifers/training_data/SPC-EAWAG-0P5X-1528160429184132-496927842240-036189-031-1014-654-40-36.jpeg" alt="rotifers" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">rotifers</span></div>
  
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/synchaeta/training_data/SPC-EAWAG-0P5X-1559498420193480-6403844471783-000109-076-988-2298-100-68.jpeg" alt="synchaeta" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">synchaeta</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/trichocerca/training_data/SPC-EAWAG-0P5X-1530192651067651-1906630816850-054409-014-1856-1848-80-64.jpeg" alt="trichocerca" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">trichocerca</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/unknown/training_data/SPC-EAWAG-0P5X-1555333262507280-2238749058453-000529-016-114-1470-36-64.jpeg" alt="unknown" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">unknown</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/unknown_plankton/training_data/SPC-EAWAG-0P5X-1555333225521933-2238712055377-000159-056-1322-88-64-92.jpeg" alt="unknown_plankton" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">unknown_plankton</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/uroglena/training_data/SPC-EAWAG-0P5X-1559793761797298-6699181574335-001519-084-770-1430-124-164.jpeg" alt="uroglena" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">uroglena</span></div>
  <div style="text-align: center;"><img src="./data/zooplankton_0p5x/daphnia/training_data/SPC-EAWAG-0P5X-1526947603555156-1089457130702-003939-002-3054-1764-384-412.jpeg" alt="daphnia_2" style="width: 100%; border-radius: 2px;" /><span style="font-size: 10px;">daphnia</span></div>
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
