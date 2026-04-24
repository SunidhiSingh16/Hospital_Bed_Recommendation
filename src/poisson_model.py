import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class PoissonPredictor:
    def __init__(self):
        self.lambda_ = None
    
    def fit(self, historical_data):
        """
        Fit Poisson distribution to historical data
        Args:
            historical_data (array-like): Historical daily admissions data
        """
        self.lambda_ = np.mean(historical_data)
        return self
    
    def predict_distribution(self, k_range):
        """
        Calculate probability distribution for different numbers of admissions
        Args:
            k_range (array-like): Range of possible admission numbers
        Returns:
            array: Probability for each number in k_range
        """
        if self.lambda_ is None:
            raise ValueError("Model not fitted yet!")
        
        probabilities = poisson.pmf(k_range, self.lambda_)
        return probabilities
    
    def plot_distribution(self, k_range=None):
        """
        Plot Poisson distribution
        Args:
            k_range (array-like, optional): Range of values to plot
        """
        if k_range is None:
            k_range = np.arange(0, int(self.lambda_ * 2))
        
        probs = self.predict_distribution(k_range)
        
        plt.figure(figsize=(10, 6))
        plt.bar(k_range, probs, alpha=0.8, color='skyblue')
        plt.title(f'Poisson Distribution (λ = {self.lambda_:.2f})')
        plt.xlabel('Number of Admissions')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        return plt
    
    def get_most_likely_range(self, confidence=0.80):
        """
        Get the most likely range of admissions for a given confidence level
        Args:
            confidence (float): Confidence level (0-1)
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        k_range = np.arange(0, int(self.lambda_ * 3))
        probs = self.predict_distribution(k_range)
        
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        cumsum_probs = np.cumsum(probs[sorted_indices])
        
        # Find indices where cumsum exceeds confidence
        mask = cumsum_probs <= confidence
        likely_indices = sorted_indices[mask]
        
        return min(k_range[likely_indices]), max(k_range[likely_indices]) 