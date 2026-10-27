import pandas as pd

data = {
    "CustomerID": list(range(1, 21)),
    "Annual_Income": [
        15, 16, 17, 18, 19, 20, 22, 24, 25, 28,
        55, 58, 60, 62, 65, 68, 70, 85, 90, 95
    ],
    "Spending_Score": [
        39, 81, 6, 77, 40, 76, 6, 94, 3, 14,
        50, 45, 49, 55, 35, 60, 29, 75, 20, 88
    ]
}

df = pd.DataFrame(data)

df.to_csv("customer_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)