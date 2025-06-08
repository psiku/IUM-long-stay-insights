import pandas as pd
import requests


def load_test_data(file_path: str) -> list:
    df = pd.read_csv(file_path)
    df = df.dropna()
    df["host_acceptance_rate"] = df["host_acceptance_rate"].astype(str).str.replace("%", "").astype(float)
    df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
    return df.to_dict(orient='records')


def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def run():
    test_data = load_test_data("microservice/data/test_data.csv")

    for batch_data in batch(test_data, 1):
        response = requests.post(
            "http://localhost:8000/AB-test",
            json=batch_data
        )

        if response.status_code == 200:
            for row, result in zip(batch_data, response.json()):
                print(f"Response for {row['id']}: {result}")
        else:
            print(f"Batch error: {response.status_code} - {response.text}")
        
if __name__ == "__main__":
    run()