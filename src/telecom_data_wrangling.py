from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/raw/telecom_customer_usage_raw.csv")
df.columns=(df.columns.str.strip().str.lower().str.replace("%","pct",regex=False)
 .str.replace(r"[^a-z0-9]+","_",regex=True).str.strip("_"))
df=df.drop_duplicates("customer_id").copy()

for c in ["gender","region","plan","churn"]:
    df[c]=df[c].astype("string").str.strip().str.title()

numeric=["age","tenure_months","monthly_data_gb","call_minutes","sms_count",
"download_speed_mbps","latency_ms","dropped_calls_pct","monthly_bill_inr",
"support_calls","satisfaction_score"]
for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")

df.loc[df["latency_ms"]<0,"latency_ms"]=np.nan
df.loc[df["download_speed_mbps"]<0,"download_speed_mbps"]=np.nan
df.loc[df["dropped_calls_pct"]<0,"dropped_calls_pct"]=np.nan

for c in numeric: df[c]=df[c].fillna(df[c].median())
for c in ["gender","region","plan","churn"]: df[c]=df[c].fillna(df[c].mode()[0])

for c in ["monthly_data_gb","call_minutes","download_speed_mbps","latency_ms","monthly_bill_inr"]:
    df[c+"_norm"]=(df[c]-df[c].min())/(df[c].max()-df[c].min())

out=ROOT/"data/processed/telecom_customer_usage_clean.csv"
df.to_csv(out,index=False)

sns.set_theme(style="whitegrid")
# Generate five analysis figures here; see repository figures/ for outputs.
plt.figure(figsize=(8,5)); sns.histplot(df,x="monthly_data_gb",bins=30,kde=True); plt.title("Monthly Mobile Data Usage Distribution"); plt.tight_layout(); plt.savefig(ROOT/"figures/01_data_usage_distribution.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.barplot(data=df,x="plan",y="monthly_data_gb",estimator="mean",errorbar=None); plt.title("Average Monthly Data Usage by Plan"); plt.tight_layout(); plt.savefig(ROOT/"figures/02_usage_by_plan.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.scatterplot(data=df,x="download_speed_mbps",y="latency_ms",hue="churn",alpha=.65); plt.title("Network Performance: Download Speed vs Latency"); plt.tight_layout(); plt.savefig(ROOT/"figures/03_speed_vs_latency.png",dpi=180); plt.close()
plt.figure(figsize=(8,5)); sns.barplot(data=df,x="region",y="satisfaction_score",estimator="mean",errorbar=None); plt.title("Average Customer Satisfaction by Region"); plt.tight_layout(); plt.savefig(ROOT/"figures/04_satisfaction_by_region.png",dpi=180); plt.close()
ch=df.groupby("plan")["churn"].apply(lambda s:(s=="Yes").mean()*100).reset_index(name="churn_rate_pct")
plt.figure(figsize=(8,5)); sns.barplot(data=ch,x="plan",y="churn_rate_pct",errorbar=None); plt.title("Churn Rate by Telecom Plan"); plt.tight_layout(); plt.savefig(ROOT/"figures/05_churn_by_plan.png",dpi=180); plt.close()
