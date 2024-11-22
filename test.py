from joblib import Parallel, delayed
import time

# Function to simulate a CPU-heavy task
def compute_square(x):
    time.sleep(0.1)  # Simulate some processing time
    return x * x

if __name__ == "__main__":
    print("Starting Joblib Test...")

    # List of inputs
    inputs = list(range(10))

    # Parallel computation
    results = Parallel(n_jobs=-1)(delayed(compute_square)(i) for i in inputs)

    print("Inputs:", inputs)
    print("Results:", results)
