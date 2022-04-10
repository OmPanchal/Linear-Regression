import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
  def __init__(self, X, y) -> None:
    assert len(X) == len(y), "Both arrays should be the same size"
    assert len(X.shape) > 1, "X should be a 2 dimentiona matrix"

    self.X = np.array(X)
    self.y = np.array(y)
    self.n = self.X.shape[0]

  def fit(self, L=0.001, epochs=1000):
    self.w = np.ones(shape=(self.X.shape[1] + 1))
    X = np.ones(shape=(self.X.shape[0], 1))
    self.X = np.hstack((X, self.X))
    self.loss = []

    for i in range(epochs):
      loss = (1/(2*self.n)) * np.sum((np.dot(self.X, self.w) - self.y) ** 2)
      self.loss.append(loss)

      dw = (1/self.n) * np.dot(self.X.T, (np.dot(self.X, self.w)) - self.y) 
      self.w = self.w - (L * dw)

    print("\n WEIGHTS LEARNT: ", self.w)

  def graph(self):
    plt.figure(figsize=[12, 6])

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(self.loss)
    plt.show()


np.random.seed(seed=102)

X = np.array([[1, 3], [1, 4], [1, 3], [1, 5]])
y = np.array([1, 4, 7, 8])

model = LinearRegression(X, y)

model.fit()
model.graph()
