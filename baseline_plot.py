import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

labels = ["5%", "10%", "25%", "100%"]
x = [0, 1, 2, 3]

crf_f1 = [0.8281, 0.8725, 0.9116, 0.9451]
bilstm_f1 = [0.5170, 0.7402, 0.8845, 0.9637]

plt.figure(figsize=(7, 5))

plt.plot(x, crf_f1, marker="o", label="CRF")
plt.plot(x, bilstm_f1, marker="o", label="BiLSTM")

plt.xticks(x, labels)

plt.xlabel("Training Data Fraction")
plt.ylabel("Dev Slot F1")
plt.title("Data Efficiency: CRF vs. BiLSTM")

plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

plt.savefig(
    "part_c_data_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved plot to part_c_data_efficiency.png")