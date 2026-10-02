import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:\\LPU\\7th Semester\\CSD 403\\data\\raw\\traffic_accident_risk.csv")

df.head()
print(df.info())
print(df.describe())
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Accident_Risk"].value_counts(normalize=True))
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nDuplicate rows:")
print(df.duplicated().sum())
plt.figure(figsize=(7, 5))
sns.countplot(x="Accident_Risk", data=df)
plt.title("Accident Risk Distribution")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="Weather_Condition", hue="Accident_Risk", data=df)
plt.title("Accident Risk by Weather Condition")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("artifacts/accident_risk_by_weather.png")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="Road_Condition", hue="Accident_Risk", data=df)
plt.title("Accident Risk by Road Condition")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("artifacts/accident_risk_by_road_condition.png")
plt.show()
plt.figure(figsize=(10, 5))
sns.countplot(x="Hour", hue="Accident_Risk", data=df)
plt.title("Accident Risk by Hour")
plt.tight_layout()
plt.savefig("artifacts/accident_risk_by_hour.png")
plt.show()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("artifacts/correlation_heatmap.png")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df_baseline = df.copy()

# Remove duplicate rows
df_baseline = df_baseline.drop_duplicates()

# Fill missing numerical values
numeric_cols = df_baseline.select_dtypes(include=["int64", "float64"]).columns
df_baseline[numeric_cols] = df_baseline[numeric_cols].fillna(
    df_baseline[numeric_cols].median()
)

# Convert categorical columns
categorical_cols = df_baseline.select_dtypes(include=["object", "str"]).columns

for col in categorical_cols:
    if col != "Accident_Risk":
        df_baseline[col] = LabelEncoder().fit_transform(
            df_baseline[col].astype(str)
        )

# Encode target
target_encoder = LabelEncoder()
df_baseline["Accident_Risk"] = target_encoder.fit_transform(
    df_baseline["Accident_Risk"]
)

# Drop timestamp
df_baseline = df_baseline.drop(columns=["Timestamp"])

X = df_baseline.drop(columns=["Accident_Risk"])
y = df_baseline["Accident_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nBaseline Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
