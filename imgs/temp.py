import matplotlib.pyplot as plt
import numpy as np

# Sample data
ptq4vit = [71.41, 74.94, 75.95, 77.69, 78.24]  # , 78.84]
ptmq_paper = [71.67, 75.14, 76.09, 77.14, 78.16]  # , 78.84]
ptmq_ourcode = [74.33, 73.27, 69.38, 69.74, 66.51]  # , 78.84]

x = np.arange(5)  # Precision values (W4A6, W6A^, etc.)
width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(x - width, ptq4vit, width, color="#4c8bf5", label="PTQ4ViT (Paper)")
ax.bar(x, ptmq_paper, width, color="#8e44ad", label="PTMQ (Paper)")
ax.bar(x + width, ptmq_ourcode, width, color="#e67e22", label="PTMQ (Our Code)")
# Add line plot with correct x-positions
ax.plot(x + width, ptmq_ourcode, "-o", color="#e67e22", linewidth=2, markersize=6)

ax.set_xlabel("Precision")
ax.set_ylabel("Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(["W4A6", "W5A6", "W6A6", "W7A7", "W8A8"])  # , 'FP32'])
ax.legend()

plt.title("ImageNet-1K Accuracy (iters=1000, lr=4e-2)")

plt.tight_layout()
plt.show()
plt.savefig("ptmq_comparison.png")
