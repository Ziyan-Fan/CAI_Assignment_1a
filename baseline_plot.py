import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Use actual training fractions for proportional x-axis spacing
x = [0.05, 0.10, 0.25, 1.00]
labels = ["5%", "10%", "25%", "100%"]

crf_f1 = [0.8358, 0.8804, 0.9139, 0.9522]
bilstm_f1 = [0.8219, 0.8686, 0.9271, 0.9686]

plt.figure(figsize=(7, 5))

plt.plot(x, crf_f1, marker="o", label="CRF")
plt.plot(x, bilstm_f1, marker="o", label="BiLSTM")

plt.xticks(x, labels)

plt.xlabel("Training Data Fraction")
plt.ylabel("Dev Span F1")
plt.title("Data Efficiency: CRF vs. BiLSTM")

plt.legend()
plt.grid(True, alpha=0.4)

plt.xlim(0, 1.02)

plt.tight_layout()

plt.savefig(
    "part_c_data_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved plot to part_c_data_efficiency.png")