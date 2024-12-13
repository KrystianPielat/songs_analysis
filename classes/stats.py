import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BootstrapEvaluator:
    def __init__(self, data, group_col, value_col, control_group):
        """
        Initialize the BootstrapEvaluator.

        Parameters:
        - data: pd.DataFrame, the dataset containing the data.
        - group_col: str, column name indicating group membership (categorical).
        - value_col: str, column name indicating the values to evaluate.
        - control_group: str, value in group_col to be treated as the control group.
        """
        self.data = data
        self.group_col = group_col
        self.value_col = value_col
        self.control_group = control_group
        self.results = None

    def bootstrap_means(self, data, n_resamples=10000):
        """Generate bootstrap samples and calculate means."""
        return np.array([
            np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)
        ])

    def evaluate(self, n_resamples=10000):
        """
        Perform bootstrap evaluation.

        Parameters:
        - n_resamples: int, number of bootstrap resamples.

        Returns:
        - results_table: pd.DataFrame, containing evaluation results.
        """
        groups = self.data[self.group_col].unique()
        bootstrap_results = {}
        observed_means = {}
        ci_results = {}

        for group in groups:
            group_data = self.data[self.data[self.group_col] == group][self.value_col]
            bootstrap_means = self.bootstrap_means(group_data, n_resamples)
            bootstrap_results[group] = bootstrap_means
            observed_means[group] = group_data.mean()
            ci_results[group] = np.percentile(bootstrap_means, [2.5, 97.5])

        diff_results = []
        for group in groups:
            if group == self.control_group:
                continue

            diff_means = bootstrap_results[group] - bootstrap_results[self.control_group]
            diff_ci = np.percentile(diff_means, [2.5, 97.5])
            mean_diff_percentage = (observed_means[group] - observed_means[self.control_group]) / observed_means[self.control_group] * 100
            percentage_ci = np.percentile(
                (bootstrap_results[group] - bootstrap_results[self.control_group]) / observed_means[self.control_group] * 100, [2.5, 97.5]
            )

            diff_results.append({
                "control_group": self.control_group,
                "treatment_group": group,
                "pct_diff": mean_diff_percentage,
                "pct_ci_lower": percentage_ci[0],
                "pct_ci_upper": percentage_ci[1],
                "abs_diff": observed_means[group] - observed_means[self.control_group],
                "abs_diff_ci_lower": diff_ci[0],
                "abs_diff_ci_upper": diff_ci[1]
            })

        self.results = pd.DataFrame(diff_results)

    def plot_means_distribution(self, n_resamples=10000):
        """Plot bootstrap distributions of group means."""
        plt.figure(figsize=(12, 6))
        for group in self.data[self.group_col].unique():
            group_data = self.data[self.data[self.group_col] == group][self.value_col]
            bootstrap_means = self.bootstrap_means(group_data, n_resamples)
            sns.histplot(bootstrap_means, kde=True, label=f"{group}", stat="density", alpha=0.7)
            plt.axvline(x=group_data.mean(), linestyle="--", label=f"{group} Mean")

        plt.title("Bootstrap Distributions of Group Means", fontsize=16)
        plt.xlabel("Mean", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title="Group")
        plt.tight_layout()
        plt.show()

    def plot_interval(self):
        """Plot confidence intervals for all treatments vs. control."""
        if self.results is None:
            raise ValueError("Please run evaluate() before plotting.")

        plt.figure(figsize=(12, 6))
        x_labels = []
        for idx, row in self.results.iterrows():
            x_labels.append(row["treatment_group"])
            plt.errorbar(
                x=len(x_labels) - 1, 
                y=row["pct_diff"], 
                yerr=[[row["pct_diff"] - row["pct_ci_lower"]], [row["pct_ci_upper"] - row["pct_diff"]]], 
                fmt='o'
            )

        plt.axhline(y=0, color="black", linestyle="--", label="No Difference")
        plt.title("Confidence Intervals of Percentage Differences", fontsize=16)
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45)
        plt.xlabel("Treatment Groups", fontsize=12)
        plt.ylabel("Percentage Difference", fontsize=12)
        plt.tight_layout()
        plt.show()
            
    def get_results_table(self, prettify: bool = True):
        """Format results for display with highlighting."""
        if self.results is None:
            raise ValueError("Please run evaluate() before formatting results.")

        if not prettify:
            return self.results
            
        styled_df = self.results.style.format(
            formatter={
                "pct_diff": "{:,.2f}%",
                "pct_ci_lower": "{:,.2f}%",
                "pct_ci_upper": "{:,.2f}%",
                "abs_diff": "{:,.2f}",
                "abs_ci_lower": "{:,.2f}",
                "abs_ci_upper": "{:,.2f}",
            }
        )

        styled_df = styled_df.bar(subset=['pct_diff'], color=('red', 'green'), align='mid')
        return styled_df
