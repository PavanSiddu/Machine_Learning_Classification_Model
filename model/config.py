"""Shared constants used by the training script and the Streamlit app."""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

RAW_DATA_PATH = MODEL_DIR / "spambase.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "metrics_summary.csv"

UCI_SPAMBASE_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"

RANDOM_STATE = 2026
TEST_SIZE = 0.20

TARGET_COLUMN = "is_spam"

_WORDS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
]
_CHARS = ["semicolon", "parenthesis", "bracket", "exclamation", "dollar", "hash"]

FEATURE_COLUMNS = (
    [f"word_freq_{w}" for w in _WORDS]
    + [f"char_freq_{c}" for c in _CHARS]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)

# Display name -> pickle file inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
