import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class QuantumUnityCatalog:
    def __init__(self, n_records=1000, sensitivity=1000):
        self.data = np.random.normal(50000, 15000, n_records) # Sensitive Salary Data
        self.sensitivity = sensitivity

    def query(self, epsilon):
        """
        Epsilon (e) is the Privacy Budget.
        Small e = High Privacy (High Noise).
        Large e = Low Privacy (Low Noise).
        """
        true_mean = np.mean(self.data)
        # Laplace Noise: scale = sensitivity / epsilon
        noise = np.random.laplace(0, self.sensitivity / epsilon)
        return true_mean + noise

def run_simulation():
    print("\n--- 2. Running Heisenberg Governance Simulation ---")
    catalog = QuantumUnityCatalog(n_records=10000)
    
    epsilons = np.linspace(0.1, 5.0, 50)
    errors = []
    
    for eps in epsilons:
        observed = catalog.query(eps)
        true_val = np.mean(catalog.data)
        error = abs(observed - true_val)
        errors.append(error)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, errors, color='crimson', linewidth=2)
    plt.title("The Heisenberg Curve: Privacy vs. Utility")
    plt.xlabel("Privacy Budget Spent (Epsilon)")
    plt.ylabel("Query Error (Noise)")
    plt.axvline(x=1.0, color='grey', linestyle='--', label='Standard Limit')
    plt.text(0.5, max(errors)/2, 'High Privacy\n(High Uncertainty)', color='red')
    plt.text(3.5, min(errors), 'High Precision\n(Low Privacy)', color='green')
    plt.grid(True, alpha=0.3)
    plt.savefig('heisenberg_result.png')
    print("Graph saved to heisenberg_result.png")

if __name__ == "__main__":
    run_simulation()
