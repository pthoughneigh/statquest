from pathlib import Path
from matplotlib import pyplot as plt
from statquest.phase0_foundation.load_soyabeans import load_soyabeans_csv

plots_path = (Path(__file__).parents[3] / 'figures')
plots_path.mkdir(parents=True, exist_ok=True)

data = load_soyabeans_csv()
class_column_value_counts = data['class'].value_counts()
print(f"\nColumn 'class' value counts: \n{class_column_value_counts}\n")

# 13 percent of accuracy exists for 'free'
baseline_accuracy = class_column_value_counts.max() / len(data)
print(f"Baseline accuracy: {baseline_accuracy}")

baseline_macro_recall = 1 / data['class'].nunique()
print(f"Macro recall: {baseline_macro_recall}")

fig, ax = plt.subplots(figsize=(25,6))
b = ax.barh(class_column_value_counts.index, class_column_value_counts)
ax.set_title("Disease distribution")
ax.set_xlabel("Number of cases")
ax.set_ylabel("Disease")
ax.bar_label(b)
fig.savefig(plots_path / 'baseline.png', bbox_inches='tight')