import pandas as pd

# Load public UCI Online Retail dataset
df = pd.read_excel("data/raw/Online Retail.xlsx")

# Remove exact duplicates
df = df.drop_duplicates()

# Remove cancellation transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep valid sales records
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Required fields for customer-level analysis
df = df.dropna(subset=["CustomerID", "Description"])

# Convert date and create transaction value
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# IQR outlier detection
Q1 = df["TotalAmount"].quantile(0.25)
Q3 = df["TotalAmount"].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5*IQR
outliers = df[(df["TotalAmount"] < Q1 - 1.5*IQR) |
              (df["TotalAmount"] > upper_fence)]

df.to_csv("data/processed/online_retail_cleaned.csv", index=False)
print("Cleaned shape:", df.shape)
print("IQR outliers:", len(outliers))
