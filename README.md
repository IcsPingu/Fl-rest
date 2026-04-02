# ASTRA-FL: Containerized Federated Learning Simulation Framework

## Summary

This artifact implements **ASTRA-FL**, a comprehensive, containerized simulation framework designed to run end-to-end Federated Learning (FL) experiments. ASTRA-FL evaluates the performance of a proposed Hybrid graduated Knowledge Distillation strategy against state-of-the-art baselines (FedAvg, FedProx, MOON) under realistic, volatile conditions. The framework simulates hardware heterogeneity (CPU vs. GPU clients), non-IID data distributions (via Dirichlet), client dropouts, and network instability. 

By orchestrating PyTorch training loops inside isolated Docker containers, ASTRA-FL provides a highly reproducible environment to demonstrate how the proposed hybrid approach improves convergence speed, robustness, and communication efficiency.

---

# README Structure

The artifact is organized as follows:

```text
ASTRA-FL/
├── client/                 # Client training logic (PyTorch)
├── server/                 # Server aggregation & evaluation logic
├── shared/                 # Shared PyTorch models (SimpleCNN, etc.)
│
├── scripts/                
│   ├── plotting/           # Scripts to generate paper figures
│   └── setup/              # Setup and utility scripts (prepare_data.py)
│
├── results/                
│   ├── figures/            # Generated plots and diagrams
│   ├── logs/               # Raw simulation logs
│   └── metrics/            # CSV outputs of accuracy/loss/time
│
├── Dockerfile              # Unified FL Docker image
├── config.py               # Central configuration (auto-modified by run.sh)
├── run.sh                  # 🚀 Master control script
└── README.md               # This file
```

---

# Considered Seals

The authors consider the following labels to be part of the evaluation process:
* **Available Artifacts (Label D)**
* **Functional Artifacts (Label F)**
* **Reproducible Experiments (Label R)**: Based on the fully automated, containerized scripts and plotting modules provided in this repository.

---

# Basic Information

## Execution Environment

* **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows 10/11 (with WSL2)
* **Python Version**: Python 3.10+ (Inside Docker containers)

**Hardware Requirements**:
* **Minimum**: 8 GB RAM, 4 CPU cores (Sufficient for CPU-only baseline tests)
* **Recommended**: 16 GB RAM, 8 CPU cores, 10 GB disk space
* **GPU (optional)**: NVIDIA GPU with CUDA support for simulating High-Perf clients.

**Software Requirements**:
* Docker & Docker Compose
* NVIDIA Container Toolkit (Required ONLY if simulating GPU-enabled clients)
* Git (for cloning)

---

# Dependencies

## External Libraries (Containerized)

| Dependency | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | `>= 2.0.0` | Deep learning framework for client training and server evaluation |
| `torchvision` | `>= 0.15.0` | Computer vision utilities (CIFAR-10 dataset) |
| `pandas` | `>= 1.5.0` | Metrics tracking and CSV aggregation |
| `matplotlib` | `>= 3.4.0` | Result plotting and visualization (Host machine) |

## Dataset and Benchmarks
* **CIFAR-10 Dataset**: Automatically downloaded during the data preparation step.
* **Data Distribution**: Non-IID simulation using Dirichlet distribution ($\alpha$).

## Security Concerns & Safety Measures

**Identified Risks:**
1. **Unrestricted File I/O**: The framework creates output directories and writes logs/metrics to `./results/`. Ensure write permissions exist in the execution directory.
2. **Resource Consumption**: Running 20+ isolated Docker containers simultaneously consumes substantial RAM and CPU overhead.

**Safety Measures:**
* **Container Isolation**: All training code runs sandboxed inside Docker.
* **Graceful Teardown**: The master script (`run.sh`) explicitly handles orphan removal and network cleanup to prevent resource leaks.

---

# Installation

## Step 1: Clone the Repository

```bash
git clone [https://github.com/your-username/ASTRA-FL.git](https://github.com/your-username/ASTRA-FL.git)
cd ASTRA-FL
```

## Step 2: Ensure Docker is Running
Verify your Docker daemon is active and you have Docker Compose installed:

```bash
docker-compose --version
```

## Step 3: Verify Output Directories
The master script will automatically generate these, but ensure your user has standard permissions:

```bash
mkdir -p results/logs results/metrics results/figures
```

---

# Minimal Test

This test validates that the Docker network configures properly and the training loop executes without errors.

### Test Execution (Expected Time: < 2 minutes)

1. Open `run.sh` in a text editor.
2. Ensure **[CENÁRIO 1] FedAvg** is uncommented, and all other scenarios are commented out.
3. Set the system parameters to a minimal footprint:
   ```bash
   CLIENTS_HIGH_PERF=0 
   CLIENTS_LOW_PERF=2   
   TOTAL_ROUNDS=2
   ```
4. Run the script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

### Expected Output
* **Console Output**: Docker build logs followed by progress messages for rounds 1 and 2.
* **Result Files**: `results/logs/1_FedAvg.log` and `results/metrics/1_FedAvg_metrics.csv`.

### Expected Success Criteria
* No runtime errors or `ENOENT` exceptions.
* The system shuts down cleanly and returns to the host prompt.

---

# Experiments

This section describes how to reproduce the main claims presented in the article.

## Claim #1: ASTRA-FL Outperforms Baseline Methods

**Objective**: Demonstrate that the proposed Hybrid method (ASTRA-FL) achieves higher target accuracy and faster convergence compared to FedAvg, FedProx, MOON, and Constant KD.

### Configuration

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `TOTAL_ROUNDS` | 50 | Full training duration |
| `CLIENTS_HIGH_PERF` | 10 | Simulated GPU capability |
| `CLIENTS_LOW_PERF` | 10 | Simulated CPU capability |
| `Scenario` | All 5 | Sequential execution |

### Execution Commands

You must run the script 5 separate times, editing `run.sh` to uncomment exactly **one** Scenario block (1 through 5) before each execution:

```bash
# 1. Edit run.sh -> Uncomment [CENÁRIO 1]
./run.sh

# 2. Edit run.sh -> Uncomment [CENÁRIO 2]
./run.sh

# ... Repeat for Scenarios 3, 4 (Hybrid), and 5.
# Total expected execution time: ~1-2 hours (depending on GPU availability)
```

### Result Analysis

```bash
# Generate the accuracy convergence plot
python scripts/plotting/plot_results.py

# Generate the loss trajectory plot
python scripts/plotting/overleaf_loss.py
```

### Expected Results
**Accuracy Trajectory**:
* **FedAvg**: Lowest baseline convergence.
* **MOON / FedProx**: Moderate improvement handling hardware variance.
* **ASTRA-FL (Hybrid)**: Reaches highest target accuracy significantly faster than baselines.
*(Check `results/figures/Figure_Accuracy.png` to verify).*

---

## Claim #2: Robustness to Data Heterogeneity

**Objective**: Show that ASTRA-FL maintains its predictive power even under extreme Non-IID data distributions across clients.

### Configuration

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `Scenario` | 4 (Hybrid) | Test proposed method |
| `DIRICHLET_ALPHA` | 0.3 vs 0.5 | Severe vs. Moderate Skew |

### Execution Command

Modify the hardcoded `DIRICHLET_ALPHA` inside the client application code, or define it dynamically. Run the simulation twice: once with $\alpha=0.3$ and once with $\alpha=0.5$.

```bash
# Generate the Alpha sensitivity comparison plot
python scripts/plotting/plot_alpha.py
```

### Expected Results
**Robustness Metric**: 
The performance gap between $\alpha=0.3$ and $\alpha=0.5$ is minimized using the ASTRA-FL hybrid strategy, demonstrating resilience to severe label skew compared to standard FedAvg.

---

## Claim #3: Time and Communication Efficiency

**Objective**: Validate that by handling stragglers and using graduated knowledge distillation, ASTRA-FL reduces total wall-clock training time despite hardware differences.

### Execution Commands

Assuming you have already run the scenarios from **Claim #1**:

```bash
# Analyze time efficiency and communication overhead
python scripts/plotting/overleaf_time.py
```

### Expected Results
* **Time per Round**: The `training_times.csv` logs will show that ASTRA-FL reduces average round latency.
* **Efficiency**: CPU-limited (Low-Perf) clients exhibit less straggler-induced delay on the global aggregation clock.

---

# LICENSE

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

```text
MIT License

Copyright (c) 2025 [Authors]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## References

1. Sousa, J., et al. "Enhancing robustness in federated learning using minimal repair and dynamic adaptation" - *Annals of Telecommunications* (2025)
2. [Authors]. "ASTRA-FL: Proactive Client Selection and Distillation for Vehicular Federated Learning" - *[Conference/Journal]*
```