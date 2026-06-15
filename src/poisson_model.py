import math
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


class PoissonPredictor:
    def __init__(self):
        self.lambda_ = None

    def fit(self, historical_data):
        self.lambda_ = max(float(np.mean(historical_data)), 1e-9)
        return self

    def predict_distribution(self, k_range):
        if self.lambda_ is None:
            raise ValueError("Model not fitted yet.")
        return poisson.pmf(k_range, self.lambda_)

    def plot_distribution(self, k_range=None):
        if k_range is None:
            # Extend range to ±4σ around λ to capture the full distribution shape
            sigma = math.sqrt(self.lambda_)
            k_max = max(int(self.lambda_ + 4 * sigma) + 1, 10)
            k_range = np.arange(0, k_max)

        probs = self.predict_distribution(k_range)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1A1F2E')
        ax.bar(k_range, probs, alpha=0.85, color='#2E86AB', edgecolor='none')
        ax.axvline(self.lambda_, color='#FF9800', linestyle='--', linewidth=1.5,
                   label=f'λ = {self.lambda_:.1f}')
        ax.set_title(f'Poisson Arrival Distribution  (λ = {self.lambda_:.2f})',
                     color='#E8EAF0', fontsize=12)
        ax.set_xlabel('Daily Admissions', color='#8892a4')
        ax.set_ylabel('Probability', color='#8892a4')
        ax.tick_params(colors='#8892a4')
        ax.spines[:].set_visible(False)
        ax.legend(facecolor='#1A1F2E', labelcolor='#E8EAF0')
        ax.grid(True, alpha=0.15, color='#E8EAF0')
        plt.tight_layout()
        return fig

    def get_most_likely_range(self, confidence=0.80):
        if self.lambda_ is None:
            raise ValueError("Model not fitted yet.")

        # Extend k_range to ±5σ so the PMF sums close to 1 even for large λ
        sigma = math.sqrt(self.lambda_)
        k_max = int(self.lambda_ + 5 * sigma) + 1
        k_range = np.arange(0, k_max)
        probs = self.predict_distribution(k_range)

        total_prob = float(probs.sum())
        if total_prob < 1e-10:
            # Numerical underflow for very large λ: fall back to normal approximation
            z = 1.28  # ~80% two-tailed
            return (max(0, int(self.lambda_ - z * sigma)),
                    int(self.lambda_ + z * sigma))

        sorted_indices = np.argsort(probs)[::-1]
        cumsum_probs = np.cumsum(probs[sorted_indices])
        mask = cumsum_probs <= confidence
        likely_indices = sorted_indices[mask]

        # If the very first PMF value already exceeds the confidence threshold,
        # mask is all-False → return just the mode
        if len(likely_indices) == 0:
            likely_indices = sorted_indices[:1]

        return int(min(k_range[likely_indices])), int(max(k_range[likely_indices]))
