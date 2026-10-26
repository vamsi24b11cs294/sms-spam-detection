import os, re, joblib
from database import get_settings

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.joblib")
CATEGORIES = ["Banking SMS","UPI and Payment SMS","OTP and Verification","Telecom and Network SMS",
 "Promotional SMS","Loan and Financial Offers","Delivery and E-commerce","Government and Service Alerts",
 "Personal SMS","Spam and Phishing","Unknown Sender"]

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_name = "Rule-based fallback"
        self.ready = False
        try:
            bundle = joblib.load(MODEL_PATH)
            self.model, self.vectorizer = bundle["model"], bundle["vectorizer"]
            self.model_name = bundle.get("model_name", "TF-IDF model")
            self.ready = True
        except (FileNotFoundError, ValueError, KeyError, OSError):
            pass

    def predict(self, message):
        text = message.strip()
        lower = text.lower()
        scam = any(x in lower for x in ["claim your prize", "won a prize", "verify your account",
            "click here to claim", "send your password", "account suspended", "urgent action required",
            "gift card", "lottery winner", "investment guaranteed"])
        if self.ready:
            vector = self.vectorizer.transform([text])
            pred = str(self.model.predict(vector)[0])
            # Calibrated probabilities are not assumed for SVM; score is a risk estimate based on
            # model output + explicit indicators, not a guaranteed probability.
            score = 0.15
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(vector)[0]
                classes = list(self.model.classes_)
                if "spam" in classes:
                    score = float(probs[classes.index("spam")])
                elif "Spam" in classes:
                    score = float(probs[classes.index("Spam")])
            elif hasattr(self.model, "decision_function"):
                margin = float(self.model.decision_function(vector)[0])
                score = 1 / (1 + __import__("math").exp(-max(-20, min(20, margin))))
            label = "Spam" if pred.lower() == "spam" else "Legitimate"
            if scam:
                label = "Fraudulent"
                score = max(score, 0.9)
            elif label == "Spam":
                score = max(score, 0.65)
            else:
                score = min(score, 0.45)
            return {"label": label, "score": round(score, 3), "model": self.model_name,
                    "note": "Risk estimate; not a guaranteed probability."}
        # Safe fallback enables the app to run before training, but is not a replacement for ML.
        suspicious = scam or any(x in lower for x in ["bit.ly", "tinyurl", "free entry", "cash prize",
                      "claim now", "won £", "won $", "click this link", "bank details"])
        return {"label": "Fraudulent" if scam else ("Spam" if suspicious else "Legitimate"),
                "score": 0.92 if scam else (0.72 if suspicious else 0.12),
                "model": self.model_name, "note": "Rule-based demo fallback; train the ML model for evaluation."}

def classify_sms_type(message, sender_id="UNKNOWN"):
    # The website intentionally does not ask for a sender ID or phone number.
    # SMS type is inferred from message content only.
    text = message.lower()
    rules = [
      ("Spam and Phishing", ["phishing","verify your account","account suspended","claim your prize","lottery winner","send password","click here to claim","urgent action required"]),
      ("OTP and Verification", ["otp","one time password","verification code","login code","do not share this code"]),
      ("UPI and Payment SMS", ["upi","paid successfully","payment received","transaction id","wallet","paytm","phonepe","gpay"]),
      ("Banking SMS", ["debited","credited","available balance","account ending","a/c","bank transaction","withdrawn"]),
      ("Telecom and Network SMS", ["data balance","recharge","validity","plan expires","sim card","network","gb data"]),
      ("Promotional SMS", ["discount","sale","offer","promo code","buy now","limited time"]),
      ("Loan and Financial Offers", ["loan offer","instant loan","credit card offer","guaranteed returns","investment scheme"]),
      ("Delivery and E-commerce", ["delivery","shipped","tracking number","order confirmed","out for delivery","courier"]),
      ("Government and Service Alerts", ["government","public service","municipal","official notice","service update"]),
      ("Personal SMS", ["hey","hello","how are you","see you","call me","thank you"])
    ]
    for category, words in rules:
        if any(word in text for word in words):
            return category
    return "General SMS"

def assess_risk(message, prediction, category):
    score = float(prediction["score"])
    lower = message.lower()
    link = bool(re.search(r"(https?://|www\.|bit\.ly|tinyurl)", lower))
    credential = any(x in lower for x in ["password", "pin", "cvv", "share otp", "send otp"])
    severity = "Low"
    if score >= 0.95 and (credential or "suspended" in lower or "verify your account" in lower):
        severity = "Critical"
    elif score >= 0.75 or prediction["label"] == "Fraudulent":
        severity = "High"
    elif score >= 0.45 or link:
        severity = "Medium"
    # User-configurable threshold affects high alerts without making sender ID alone suspicious.
    threshold = float(get_settings().get("alert_threshold", "0.75"))
    if score >= threshold and severity == "Medium":
        severity = "High"
    return {"score": round(score, 3), "severity": severity, "has_link_indicator": link}

def explain_prediction(message, prediction, category, risk):
    reasons = []
    low = message.lower()
    if any(k in low for k in ["verify your account","account suspended","claim your prize","lottery winner","send your password"]):
        reasons.append("message contains a common impersonation or prize-scam phrase")
    if any(k in low for k in ["password","pin","cvv","share otp","send otp"]):
        reasons.append("message requests sensitive credentials or authentication information")
    if risk["has_link_indicator"]:
        reasons.append("message contains a URL/link indicator; links require verification")
    if prediction["label"] == "Legitimate" and not reasons:
        reasons.append("no strong suspicious phrase was identified")
    if not reasons:
        reasons.append("classification based on message text and available model signals")
    return "; ".join(reasons) + f". Category: {category}."
