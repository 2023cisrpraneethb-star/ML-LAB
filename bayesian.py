import numpy as np
import matplotlib.pyplot as plt
P_w1 = 0.6   
P_w2 = 0.4   

def likelihood(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


mean1, var1 = 2, 1
mean2, var2 = 5, 1.5

x_values = np.linspace(0, 8, 20)  

print("\n--- Bayesian Decision Results ---")
print("x-value\t\tP(ω1|x)\t\tP(ω2|x)\t\tDecision")
print("------------------------------------------------------------")

posterior1 = []
posterior2 = []
decisions = []

for x in x_values:
    px_w1 = likelihood(x, mean1, var1)
    px_w2 = likelihood(x, mean2, var2)

    px = px_w1 * P_w1 + px_w2 * P_w2

    P_w1_x = (px_w1 * P_w1) / px
    P_w2_x = (px_w2 * P_w2) / px

    decision = "ω1" if P_w1_x > P_w2_x else "ω2"

    print(f"{x:.2f}\t\t{P_w1_x:.4f}\t\t{P_w2_x:.4f}\t\t{decision}")

    posterior1.append(P_w1_x)
    posterior2.append(P_w2_x)
    decisions.append(decision)

plt.figure(figsize=(8,5))
plt.plot(x_values, posterior1, label='P(ω1|x)', linewidth=2)
plt.plot(x_values, posterior2, label='P(ω2|x)', linewidth=2)
plt.axvline(x=x_values[np.argmin(np.abs(np.array(posterior1) - np.array(posterior2)))],
            color='gray', linestyle='--', label='Decision Boundary')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.title('Bayesian Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()