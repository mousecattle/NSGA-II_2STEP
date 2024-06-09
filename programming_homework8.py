import numpy as np

# Define the functions
def f(x):
    return 0.2 * (x - 3) ** 2 + 3

def g(x):
    return 0.01 * x ** 4 - 0.5 * x ** 2 - 0.1 * x - 2

def h(x):
    return 5 * np.sin(x) + 0.2 * x ** 2 + 2

# Define the gradients of the functions
def f_prime(x):
    return 0.4 * (x - 3)

def g_prime(x):
    return 0.04 * x ** 3 - x - 0.1

def h_prime(x):
    return 5 * np.cos(x) + 0.4 * x

# Gradient Descent Algorithm
def gradient_descent(func, grad_func, alpha, K, d, x0):
    x = x0
    for i in range(int(K)):  # Kを整数にキャスト
        grad = grad_func(x)
        if abs(grad) < d:
            break
        x = x - alpha * grad
    else:
        i = K  # ループが一度も実行されなかった場合にiをKに設定
    return x, func(x), i + 1

# Parameter ranges
alphas = np.arange(0.001, 1, 0.01)
Ks = np.arange(10, 1000, 1)
ds = np.arange(0.0001, 0.01, 0.0001)

# Initial point range
x0_range = np.arange(-10.0, 10.1, 1.0)

# Store the best results
best_result = {"f": None, "g": None, "h": None}
best_params = {"f": None, "g": None, "h": None}

for alpha in alphas:
    for K in Ks:
        for d in ds:
            results = {"f": [], "g": [], "h": []}
            for x0 in x0_range:
                xf, f_val, f_iter = gradient_descent(f, f_prime, alpha, K, d, x0)
                xg, g_val, g_iter = gradient_descent(g, g_prime, alpha, K, d, x0)
                xh, h_val, h_iter = gradient_descent(h, h_prime, alpha, K, d, x0)
                results["f"].append((x0, xf, f_val, f_iter))
                results["g"].append((x0, xg, g_val, g_iter))
                results["h"].append((x0, xh, h_val, h_iter))

            avg_iter = {
                "f": np.mean([r[3] for r in results["f"]]),
                "g": np.mean([r[3] for r in results["g"]]),
                "h": np.mean([r[3] for r in results["h"]])
            }

            if best_result["f"] is None or avg_iter["f"] < best_result["f"]:
                best_result["f"] = avg_iter["f"]
                best_params["f"] = (alpha, K, d)

            if best_result["g"] is None or avg_iter["g"] < best_result["g"]:
                best_result["g"] = avg_iter["g"]
                best_params["g"] = (alpha, K, d)

            if best_result["h"] is None or avg_iter["h"] < best_result["h"]:
                best_result["h"] = avg_iter["h"]
                best_params["h"] = (alpha, K, d)

# Output the best results
print("Best Parameters and Results:")
print(
    f"Function f: alpha={best_params['f'][0]}, K={best_params['f'][1]}, d={best_params['f'][2]}, iterations={best_result['f']}")
print(
    f"Function g: alpha={best_params['g'][0]}, K={best_params['g'][1]}, d={best_result['g']}")
print(
    f"Function h: alpha={best_params['h'][0]}, K={best_params['h'][1]}, d={best_params['h'][2]}, iterations={best_result['h']}")