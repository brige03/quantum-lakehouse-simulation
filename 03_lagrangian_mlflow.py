import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def physics_law(t, h0, v0):
    """The Lagrangian Constraint: y = h0 + v0*t - 0.5*g*t^2"""
    g = 9.8 
    return h0 + v0*t - 0.5 * g * t**2

def run_simulation():
    print("\n--- 3. Running Lagrangian MLflow Simulation ---")
    
    # 1. Ground Truth & Noisy Data
    t = np.linspace(0, 4, 20)
    y_true = physics_law(t, h0=100, v0=0)
    y_noisy = y_true + np.random.normal(0, 8, len(t))
    
    # 2. Standard AI (The Hallucinator)
    # Using high-degree polynomial to simulate an overfitting Neural Net
    model = make_pipeline(PolynomialFeatures(15), LinearRegression())
    model.fit(t.reshape(-1, 1), y_noisy)
    
    # 3. Lagrangian AI (The Physicist)
    # Constrained optimization
    params, _ = curve_fit(physics_law, t, y_noisy)
    
    # 4. Future Prediction (The Test)
    t_future = np.linspace(0, 5, 100)
    pred_standard = model.predict(t_future.reshape(-1, 1))
    pred_physics = physics_law(t_future, *params)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(t, y_noisy, color='black', label='Noisy Data (Delta Lake)')
    plt.plot(t_future, pred_standard, 'r-', linewidth=2, label='Standard AI (Hallucination)')
    plt.plot(t_future, pred_physics, 'g--', linewidth=3, label='Lagrangian AI (Physics-Informed)')
    plt.ylim(0, 150)
    plt.title("AI Safety: Standard vs. Lagrangian Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lagrangian_result.png')
    print("Graph saved to lagrangian_result.png")
    print(f"Lagrangian discovered params: Height={params[0]:.2f}, Velocity={params[1]:.2f}")

if __name__ == "__main__":
    run_simulation()
