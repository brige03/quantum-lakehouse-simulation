import numpy as np
import matplotlib.pyplot as plt

class PrivacyBudgetManager:
    """
    Simulates a Unity Catalog Governance Layer that uses Differential Privacy.
    """
    def __init__(self, n_records=1000, sensitivity=5000):
        # Sensitive Data (e.g., Salaries)
        self.data = np.random.normal(70000, 20000, n_records)
        self.sensitivity = sensitivity # Max impact of one individual

    def query_with_budget(self, epsilon):
        """
        Executes a query.
        epsilon (e): Privacy Budget spent.
        Lower e = Higher Privacy (More Noise).
        Higher e = Lower Privacy (Less Noise).
        """
        true_mean = np.mean(self.data)
        
        # The Laplace Mechanism
        # Noise scale is inversely proportional to epsilon
        scale = self.sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return true_mean + noise

print("--- Simulation: Differential Privacy Budgeting ---")
manager = PrivacyBudgetManager(n_records=10000)

epsilons = np.linspace(0.1, 5.0, 50)
errors = []

for eps in epsilons:
    observed_val = manager.query_with_budget(eps)
    true_val = np.mean(manager.data)
    # Calculate % Error
    error_pct = abs(observed_val - true_val) / true_val * 100
    errors.append(error_pct)

plt.figure(figsize=(10, 6))
plt.plot(epsilons, errors, color='#1b3d6d', linewidth=2.5) # Databricks Blueish
plt.title("The Privacy-Utility Tradeoff (Unity Catalog Concept)")
plt.xlabel("Privacy Budget Spent ($\epsilon$)")
plt.ylabel("Query Error (%)")
plt.axvline(x=1.0, color='red', linestyle='--', label='Standard Privacy Threshold')
plt.text(0.5, max(errors)/2, 'High Privacy\n(High Noise)', color='gray')
plt.text(3.5, 0.5, 'High Utility\n(Low Noise)', color='gray')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('privacy_budget_result.png')
print("Graph saved to 'privacy_budget_result.png'")
