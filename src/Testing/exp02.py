import seaborn as sns
import matplotlib.pyplot as plt

# Load a built‑in dataset from seaborn
df = sns.load_dataset("tips")

# Create a simple scatter plot
sns.scatterplot(data=df, x="total_bill", y="tip")

plt.title("Seaborn Test Plot")
plt.show()
