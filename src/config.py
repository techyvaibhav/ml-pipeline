"""Central config: all paths, params and constants live here."""
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent  # project root dir
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"  # raw csv
SRC_PATH = ROOT / "data" / "raw" / "source.txt"  # "real" or "synthetic"
PROC_DIR = ROOT / "data" / "processed"  # train/test csv dir
MODEL_DIR = ROOT / "models"  # model artifacts dir
MODEL_PATH = MODEL_DIR / "model.joblib"  # saved pipeline bundle
METRICS_PATH = MODEL_DIR / "metrics.json"  # saved metrics
CARD_PATH = MODEL_DIR / "model_card.md"  # saved model card
REPORT_DIR = ROOT / "reports"  # plots dir

# --- data ---
DATA_URL = (  # public telco churn csv
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
ID_COL = "customerID"  # row id (not a feature)
TARGET = "Churn"  # target column (1 = churned)

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]  # numeric features
YES_NO = ["Yes", "No"]  # plain binary values
ADDON = ["Yes", "No", "No internet service"]  # internet add-on values
CATS = {  # categorical feature -> allowed values
    "gender": ["Male", "Female"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON,
    "OnlineBackup": ADDON,
    "DeviceProtection": ADDON,
    "TechSupport": ADDON,
    "StreamingTV": ADDON,
    "StreamingMovies": ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CAT_COLS = list(CATS)  # categorical feature names
FEATURES = NUM_COLS + CAT_COLS  # all model input columns

# --- training ---
SEED = 42  # random seed
TEST_SIZE = 0.2  # hold-out fraction
N_FOLDS = 5  # CV folds for model comparison
N_ITER = 12  # RandomizedSearch iterations
THRESH = 0.5  # decision threshold for churn label
MAJOR = "1.0"  # major model version
