import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, learning_rate: float, tolerance: float, dims: int | None = None):
        self.dims = dims
        self.intercept = 0
        self.slope = np.zeros(dims) if dims is not None else 0
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    @staticmethod
    def cost(predicted: np.ndarray | float, actual: np.ndarray | float) -> float:
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)

        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        )

    @staticmethod
    def cost_derivative_slope(
        slope: np.ndarray | float,
        inputs: np.ndarray | float,
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ) -> np.ndarray:
        # needs inputs too cause sigmoid is applied
        return np.dot(inputs.T, (predicted - actual)).reshape(slope.shape) / (
            inputs.shape[0] if inputs.ndim != 0 else 1
        )

    @staticmethod
    def cost_derivative_intercept(
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ) -> np.ndarray:

        return np.mean(predicted - actual)

    @staticmethod
    def sigmoid(inputs: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(-inputs))

    # @staticmethod
    # def sigmoid_derivative(inputs: np.ndarray | float) -> np.ndarray | float:
    #    return LogisticRegression.sigmoid(inputs) * (1 - LogisticRegression.sigmoid(inputs))

    def gradient_descent(
        self,
        inputs: np.ndarray | float,
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ):
        self.intercept -= self.learning_rate * self.cost_derivative_intercept(
            predicted, actual
        )
        self.slope -= self.learning_rate * self.cost_derivative_slope(
            self.slope, inputs, predicted, actual
        )

    def predict(self, inputs: np.ndarray | float):
        linear = np.dot(inputs, self.slope) + self.intercept

        return LogisticRegression.sigmoid(linear)

    def train(self, inputs, actual, epochs: int):
        inputs = np.array(inputs)
        actual = np.array(actual)

        prev_loss = float("inf")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            predicted = self.predict(inputs).reshape(actual.shape)
            self.gradient_descent(inputs, predicted, actual)

            loss = self.cost(predicted, actual)
            if abs(prev_loss - loss) < self.tolerance:
                return
            if loss > prev_loss:
                self.learning_rate *= 0.5
            prev_loss = loss
            print(f"Loss: {np.mean(loss)}")
    
    def score(self, inputs, actual):
        inputs = np.array(inputs)
        actual = np.array(actual)

        predicted = self.predict(inputs).reshape(actual.shape)

        return np.mean(predicted == actual)


if __name__ == "__main__":

    df = pd.read_csv("data.csv")
    inputs = df.drop(["actual"], axis=1)
    actual = df[["actual"]]

    num_inputs = inputs.shape[1]

    model = LogisticRegression(0.001, 0.0001, num_inputs)

    model.train(inputs, actual, 100000)
    print (f"Score: {model.score(inputs, actual)}")
    print(f"Prediction {np.round(model.predict([4]))}")
