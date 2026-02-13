#### **File: `src/01_physics_informed_mlflow.py`**
*Renamed from "Lagrangian" to be more accessible. This is the star of the show.*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

print("--- Simulation: Physics-Informed AI (PINN) vs. Standard AI ---")

# 1. The Ground Truth (The Law of Physics: Gravity)
# y = h0 + v0*t - 0.5*g*t^2
def true_physics_law(t, h0=100, v0=0):
    g = 9.8  # Gravity is constant
    return h0 + v0*t - 0.5 * g * t**2

# 2. The Data (Noisy IoT Sensors)
# Simulating a drone sensor with noise
t_data = np.linspace(0, 4, 20)
y_true = true_physics_law(t_data)
noise = np.random.normal(0, 8, size=len(t_data))
y_noisy = y_true + noise

# 3. Standard AI (The "Hallucinator")
# A standard high-degree polynomial regression (simulating a deep neural net)
# It tries to fit every data point, including the noise.
standard_model = make_pipeline(PolynomialFeatures(degree=15), LinearRegression())
standard_model.fit(t_data.reshape(-1, 1), y_noisy)

# 4. Physics-Informed AI (The "Industrial Safe" Model)
# We constrain the optimization to the known equation form.
# The model only learns the parameters (h0, v0), while enforcing 'g' (gravity).
def physics_constraint_func(t, h0, v0):
    g = 9.8 # Hard constraint
    return h0 + v0*t - 0.5 * g * t**2

# Optimization using Scipy (mimicking a custom Loss Function in PyTorch/TensorFlow)
params_pinn, _ = curve_fit(physics_constraint_func, t_data, y_noisy)

# 5. Future Prediction (Extrapolation)
# Standard AI fails catastrophically when extrapolating because it fit the noise.
t_future = np.linspace(0, 5, 100)
pred_standard = standard_model.predict(t_future.reshape(-1, 1))
pred_pinn = physics_constraint_func(t_future, *params_pinn)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(t_data, y_noisy, color='black', label='Noisy Sensor Data (Delta Table)', alpha=0.6)
plt.plot(t_future, pred_standard, 'r-', linewidth=2, label='Standard AI (Hallucination Risk)')
plt.plot(t_future, pred_pinn, 'g--', linewidth=3, label='Physics-Informed Model (Safe)')

plt.title("Industrial AI Safety: Preventing Hallucination in Physical Systems")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.ylim(0, 150)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('physics_informed_result.png')

print(f"Physics-Informed Parameters Discovered: Height={params_pinn[0]:.2f}m, Velocity={params_pinn[1]:.2f}m/s")
print("Standard AI Prediction at t=5s (Crash Risk):", pred_standard[-1])
print("Physics AI Prediction at t=5s (Safe):", pred_pinn[-1])
print("Graph saved to 'physics_informed_result.png'")
