import numpy as np
import time
import pandas as pd

def fibonacci_hash(value, table_size):
    """
    Exploration: Using the Golden Ratio (phi) to disperse keys.
    Multiplicative Hashing: (value * phi) % 1
    """
    phi = (1 + 5 ** 0.5) / 2
    # Calculate hash using fractional part of the product
    hash_val = (value * phi) % 1
    return int(hash_val * table_size)

print("--- Simulation: Bio-Mimetic Indexing Concept ---")
print("Exploring Golden Ratio Hashing for uniform distribution...")

table_size = 1000
keys = np.random.randint(0, 10000, 500)

# Simulate Bucket Distribution
buckets = np.zeros(table_size)
for k in keys:
    idx = fibonacci_hash(k, table_size)
    buckets[idx] += 1

# Analyze Collisions (Variance)
# A lower variance means better distribution (fewer collisions)
variance = np.var(buckets)

print(f"Table Size: {table_size}")
print(f"Keys Inserted: {len(keys)}")
print(f"Bucket Variance (Distribution Quality): {variance:.4f}")
print("\nInsight: The Golden Ratio provides theoretical optimal spacing for linear probing.")
print("Research Question: Can this geometry apply to Photon Vector Search?")
