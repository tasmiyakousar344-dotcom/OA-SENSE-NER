from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

DATA_DIR = Path("data/oa_ml")

DEMO_FILE = DATA_DIR / "DEMO_J.XPT"
MCQ_FILE = DATA_DIR / "MCQ_J.XPT"
BMX_FILE = DATA_DIR / "BMX_J.XPT"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "oa_model.pkl"

for file in [DEMO_FILE, MCQ_FILE, BMX_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file.resolve()}")

print("Loading NHANES data...")

demo = pd.read_sas(DEMO_FILE)
mcq = pd.read_sas(MCQ_FILE)
bmx = pd.read_sas(BMX_FILE)

print(f"Demographics rows: {len(demo)}")
print(f"Medical condition rows: {len(mcq)}")
print(f"Body measures rows: {len(bmx)}")

df = pd.merge(demo, mcq, on="SEQN", how="inner")
df = pd.merge(df, bmx[["SEQN", "BMXBMI"]], on="SEQN", how="inner")

print(f"Merged rows: {len(df)}")

required_columns = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "MCQ160A",
    "MCQ195"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[
    (df["RIDAGEYR"] >= 20) &
    (df["RIDAGEYR"] <= 80)
].copy()

df["OA"] = 0

df.loc[
    (df["MCQ160A"] == 1) &
    (df["MCQ195"] == 1),
    "OA"
] = 1

df = df[
    (df["MCQ160A"].isin([1, 2])) &
    (df["MCQ195"].isin([1, 2, 3, 4]))
]

features = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI"
]

X = df[features]
y = df["OA"]

print("\nClass distribution:")
print(y.value_counts())

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No OA", "OA"],
        zero_division=0
    )
)

MODEL_DIR.mkdir(exist_ok=True)

model_package = {
    "model": model,
    "imputer": imputer,
    "features": features,
    "classes": {
        0: "No OA",
        1: "OA"
    }
}

joblib.dump(model_package, MODEL_PATH)

print("\nTraining completed.")
print(f"Model saved at:\n{MODEL_PATH.resolve()}")