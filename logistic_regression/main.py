import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, learning_rate: float, dims: int | None = None):
        self.dims = dims
        self.intercept = np.random.rand()
        self.slope = np.random.rand(dims) if dims is not None else np.random.rand()
        self.learning_rate = learning_rate

    @staticmethod
    def cost(predicted: np.ndarray | float, actual: np.ndarray | float) -> float:
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        
        epsilon = 1e-9
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        )

    @staticmethod
    def cost_derivative_slope(
        inputs: np.ndarray | float,
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ) -> np.ndarray:
        inputs = np.asarray(inputs)
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        # needs inputs too cause sigmoid is applied

        return np.dot(inputs.T, (predicted - actual)) / (len(inputs) if inputs.ndim != 0 else 1)

    @staticmethod
    def cost_derivative_intercept(
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ) -> np.ndarray:
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)

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
            inputs, predicted, actual
        )

    def predict(self, inputs: np.ndarray | float):
        linear = (
            np.dot(self.slope, inputs) + self.intercept
            if isinstance(inputs, np.ndarray)
            else self.slope * inputs + self.intercept
        )
        return LogisticRegression.sigmoid(linear)

    def train(self, data: str, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            df = pd.read_csv(data)
            losses = []
            for _, row in df.iterrows():
                inputs = (
                    row["inputs"]
                    if self.dims is None
                    else np.array([row[f"inputs_{i}"] for i in range(self.dims)])
                )

                actual = row["actual"]

                predicted = self.predict(inputs)
                self.gradient_descent(inputs, predicted, actual)
                loss = self.cost(predicted, actual)
                losses.append(loss)
            print(f"Average loss: {np.mean(losses)}")


if __name__ == "__main__":
    model = LogisticRegression(1e-5)
    model.train("data2.csv", 10)
    print(f"Prediction {model.predict(44)}")
    print(f"Prediction {model.predict(3)}")
