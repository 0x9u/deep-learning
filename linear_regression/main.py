import numpy as np
import csv


class LinearRegression:
    def __init__(self, learning_rate: float, dims: int | None = None):
        self.dims = dims
        self.intercept = np.random.rand()
        self.slope = np.random.rand(dims) if dims is not None else np.random.rand()
        self.learning_rate = learning_rate

    # not really used? only its derivative is used
    @staticmethod
    def cost(predicted: np.ndarray | float, actual: np.ndarray | float) -> float:
        return np.mean((predicted - actual) ** 2)

    @staticmethod
    def cost_derivative_slope(
        predicted: np.ndarray | float, actual: np.ndarray | float
    ) -> float:
        return 2 * np.mean(predicted - actual)

    @staticmethod
    def cost_derivative_intercept(
        inputs: np.ndarray | float,
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ) -> float:
        return 2 * np.mean(
            (predicted - actual) * inputs,
            axis=0 if isinstance(inputs, np.ndarray) else None,
        )

    def gradient_descent(
        self,
        inputs: np.ndarray | float,
        predicted: np.ndarray | float,
        actual: np.ndarray | float,
    ):
        self.intercept -= self.learning_rate * self.cost_derivative_intercept(
            inputs, predicted, actual
        )
        self.slope -= self.learning_rate * self.cost_derivative_slope(predicted, actual)

    def predict(self, inputs: np.ndarray | float):
        return (
            np.dot(self.slope, inputs) + self.intercept
            if isinstance(inputs, np.ndarray)
            else self.slope * inputs + self.intercept
        )

    def train(self, data: str, epochs: int):
        previous_loss = float("inf")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            losses = []
            with open(data, mode="r", encoding="utf-8-sig") as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    inputs = (
                        float(row["inputs"])
                        if self.dims is None
                        else np.array([row[f"inputs_{i}"] for i in range(self.dims)])
                    )
                    actual = float(row["actual"])

                    predicted = self.predict(inputs)
                    self.gradient_descent(inputs, predicted, actual)
                    loss = self.cost(predicted, actual)
                    if loss < previous_loss:
                        self.learning_rate *= 0.5
                        previous_loss = loss
                    losses.append(loss)
            print(f"Average loss: {np.mean(losses)}")


if __name__ == "__main__":
    model = LinearRegression(0.0001)
    model.train("data.csv", 10)
    print(f"Prediction {model.predict(2025)}")
