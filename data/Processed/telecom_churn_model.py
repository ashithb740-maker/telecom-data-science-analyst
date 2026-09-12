from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
df.loc[df["latency_ms"]<0,"latency_ms"]=pd.NA
df.loc[df["download_speed_mbps"]<0,"download_speed_mbps"]=pd.NA
for c in numeric: df[c]=df[c].fillna(df[c].median())

df["total_contact_activity"]=df["support_calls"]+df["call_minutes"]/100
df["network_stress_index"]=df["latency_ms"]/df["download_speed_mbps"]
df["bill_per_tenure_month"]=df["monthly_bill_inr"]/(df["tenure_months"]+1)
df["high_support_flag"]=(df["support_calls"]>=df["support_calls"].quantile(.75)).astype(int)

features=['age', 'gender', 'region', 'plan', 'tenure_months', 'monthly_data_gb', 'call_minutes', 'sms_count', 'download_speed_mbps', 'latency_ms', 'dropped_calls_pct', 'monthly_bill_inr', 'support_calls', 'satisfaction_score', 'total_contact_activity', 'network_stress_index', 'bill_per_tenure_month', 'high_support_flag']
cat=['gender', 'region', 'plan']
num_model=['age', 'tenure_months', 'monthly_data_gb', 'call_minutes', 'sms_count', 'download_speed_mbps', 'latency_ms', 'dropped_calls_pct', 'monthly_bill_inr', 'support_calls', 'satisfaction_score', 'total_contact_activity', 'network_stress_index', 'bill_per_tenure_month', 'high_support_flag']
X=df[features]
y=(df["churn"]=="Yes").astype(int)

pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_model),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
models={
 "Logistic Regression":Pipeline([("prep",pre),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))]),
 "Decision Tree":Pipeline([("prep",pre),("model",DecisionTreeClassifier(max_depth=5,min_samples_leaf=8,class_weight="balanced",random_state=42))])
}
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    proba=model.predict_proba(X_test)[:,1]
    print(name)
    print("Accuracy:",accuracy_score(y_test,pred))
    print("Precision:",precision_score(y_test,pred,zero_division=0))
    print("Recall:",recall_score(y_test,pred,zero_division=0))
    print("F1:",f1_score(y_test,pred,zero_division=0))
    print("ROC-AUC:",roc_auc_score(y_test,proba))

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# cross_validate(models["Logistic Regression"], X_train, y_train, cv=cv, scoring=["accuracy","precision","recall","f1","roc_auc"])
