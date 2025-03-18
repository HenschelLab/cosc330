# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

class PerpendicularBisectorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.b = None
    
    def fit(self, X, y):

        
        M1, M2 = X
        midpoint = (np.mean([M1[0], M2[0]]), np.mean([M1[1], M2[1]]))
        
        # Compute the slope of the line connecting M1 and M2
        if M2[0] - M1[0] == 0:
            self.w1, self.w2, self.b = 1, 0, -midpoint[0]  # Vertical bisector
        else:
            slope = (M2[1] - M1[1]) / (M2[0] - M1[0])
            perp_slope = -1 / slope
            self.w1, self.w2 = -perp_slope, 1
            self.b = -(self.w1 * midpoint[0] + self.w2 * midpoint[1])
        return self
    
    def predict(self, X):
        return np.sign(self.w1 * X[:, 0] + self.w2 * X[:, 1] + self.b)
    
    def plot_decision_boundary(self, X):
        x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_vals = (-self.w1 * x_vals - self.b) / self.w2
        
        plt.plot(x_vals, y_vals, 'g--', label=f"{self.w1:.2f}x1 + {self.w2:.2f}x2 + {self.b:.2f} = 0")
        plt.scatter(X[:, 0], X[:, 1], c=['red', 'blue'], marker='o', edgecolors='k', label="Points")
        plt.legend()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Perpendicular Bisector Classifier")
        plt.show()

# %%
print(__name__)
if __name__ == "__main__":
    import pandas as pd

    # Create the DataFrame
    data = {
        "Transaction amount": [60, 900, 10, 250, 500, 30],
        "Transaction Frequency": [7, 29, 8, 15, 13, 7],
        "Case": ["Legitimate", "Fraudulent", "Legitimate", "Fraudulent", "Fraudulent", "Legitimate"]
    }

    df = pd.DataFrame(data)

    # Example usage
    X = np.array([[1, 2], [4, 3]])
    y = np.array([1, -1])
    
    classifier = PerpendicularBisectorClassifier()
    classifier.fit(X, y)
    print(f"Learned parameters: w1={classifier.w1}, w2={classifier.w2}, b={classifier.b}")
    
    classifier.plot_decision_boundary(X)

# %%
