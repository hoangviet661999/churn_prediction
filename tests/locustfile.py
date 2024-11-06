from locust import HttpUser, between, task


class SimulatedUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 5)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def get_result(self) -> None:
        """A task that simulates a user get prediction result."""
        form_data = {
            "CreditScore": 0,
            "Geography": "France",
            "Gender": "Male",
            "Age": 10,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 2,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 5.0,
        }
        self.client.post(
            "/refer",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "accept": "application/json",
            },
            data=form_data,
        )
