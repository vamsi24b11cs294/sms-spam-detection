"""Train and compare MultinomialNB, LogisticRegression, and LinearSVC on a labeled SMS CSV."""
import os, sys, csv, json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from database import init_db, get_db

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nRead data/README.md and place a labeled CSV at data/spam.csv.")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    # Common datasets: v1/v2 (UCI SMS Spam Collection) or label/message.
    columns = {str(c).strip().lower(): c for c in df.columns}
    label_col = next((columns[x] for x in ("label","v1","category","class") if x in columns), None)
    text_col = next((columns[x] for x in ("message","v2","sms","text") if x in columns), None)
    if label_col is None or text_col is None:
        raise ValueError("CSV must contain label/message or v1/v2 columns. See data/README.md.")
    df = df[[label_col, text_col]].dropna()
    df.columns = ["label", "message"]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"ham":"legitimate","legit":"legitimate"})
    df = df[df["label"].isin(["spam","legitimate"])]
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].drop_duplicates()
    if df["label"].nunique() < 2 or len(df) < 20:
        raise ValueError("Need at least 20 labeled rows and both spam and legitimate examples.")
    return df

def main():
    init_db()
    df = load_dataset(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced")
    }
    results = []
    best = None
    for name, estimator in models.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2),
                       strip_accents="unicode", sublinear_tf=True, max_features=50000)),
                         ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "model_name": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, pos_label="spam", zero_division=0)),
            "recall": float(recall_score(y_test, pred, pos_label="spam", zero_division=0)),
            "f1": float(f1_score(y_test, pred, pos_label="spam", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=["legitimate","spam"]).tolist(),
            "report": classification_report(y_test, pred, zero_division=0)
        }
        results.append(metrics)
        print(f"\n{name}: accuracy={metrics['accuracy']:.3f}, precision={metrics['precision']:.3f}, "
              f"recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        print("Confusion matrix (rows=true [legitimate, spam], columns=predicted [legitimate, spam]):")
        print(metrics["confusion_matrix"])
        if best is None or metrics["f1"] > best[0]:
            best = (metrics["f1"], name, pipe)
    chosen = best[2]
    joblib.dump({"model": chosen.named_steps["model"], "vectorizer": chosen.named_steps["tfidf"],
                 "model_name": best[1]}, os.path.join(MODEL_DIR, "spam_model.joblib"))
    with open(os.path.join(MODEL_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with get_db() as conn:
        conn.execute("DELETE FROM model_metrics")
        for m in results:
            conn.execute("INSERT INTO model_metrics(model_name,accuracy,precision,recall,f1,confusion_matrix) VALUES(?,?,?,?,?,?)",
                (m["model_name"], m["accuracy"], m["precision"], m["recall"], m["f1"], json.dumps(m["confusion_matrix"])))
    print(f"\nSaved best model: {best[1]} (F1={best[0]:.3f})")
    print("Restart Flask to load the newly trained model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
