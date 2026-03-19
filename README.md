# IEEE-CIS Fraud Detection: High-Throughput Engineering Pipeline

A production-oriented data mining project focused on detecting fraudulent e-commerce transactions using the Vesta real-world dataset.

### Engineering Constraints & Strategy
* **Hardware:** Optimized for **8GB RAM** (MacBook Air).
* **Core Strategy:** Implemented a **Memory-First Ingestion Pipeline** using automated type-downcasting and garbage collection to handle 500k+ records on consumer hardware.
* **Tech Stack:** Python (Pandas, NumPy, Scikit-Learn), LaTeX (for formal reporting).

---

## Daily Progress Log

| Date | Phase | Milestone | Key Engineering Takeaway |
| :--- | :--- | :--- | :--- |
| **Mar 18** | Inception | Initialized Repo & Env | Designed a modular folder structure for scalability. |
| **Mar 19** | Ingestion | Memory Optimization | Reduced memory footprint by 65% using `float16` downcasting. |
| **Mar 20** | EDA | Feature Distribution | Identified class imbalance (3.5% Fraud) and log-transformed `TransactionAmt`. |

---
