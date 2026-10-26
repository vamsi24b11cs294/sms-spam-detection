import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from database import init_db, get_db, create_message
from spam_detector import SpamDetector, classify_sms_type, assess_risk, explain_prediction

load_dotenv()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "local-sms-demo-change-this")
DETECTOR = SpamDetector()
init_db()

@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        message = (request.form.get("message") or "").strip()
        if not message:
            flash("Type a message first.", "warning")
        elif len(message) > 5000:
            flash("Message is too long (maximum 5,000 characters).", "danger")
        else:
            prediction = DETECTOR.predict(message)
            category = classify_sms_type(message, "UNKNOWN")
            risk = assess_risk(message, prediction, category)
            reason = explain_prediction(message, prediction, category, risk)
            message_id = create_message("Messages", message, prediction["label"], category,
                                        prediction["score"], prediction["model"], risk["severity"], reason)
            result = {"id": message_id, "message": message, "prediction": prediction,
                      "category": category, "risk": risk, "reason": reason}
            flash("Message checked. See the safety label below.", "success")
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM sms_messages ORDER BY created_at DESC LIMIT 30").fetchall()
    messages = [dict(row) for row in rows]
    # The latest submitted message is displayed as a chat bubble after the form is submitted.
    return render_template("index.html", messages=messages, result=result)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    from flask import jsonify
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message or len(message) > 5000:
        return jsonify({"error": "message is required and must be at most 5000 characters"}), 400
    prediction = DETECTOR.predict(message)
    category = classify_sms_type(message, "UNKNOWN")
    risk = assess_risk(message, prediction, category)
    reason = explain_prediction(message, prediction, category, risk)
    mid = create_message("Messages", message, prediction["label"], category,
                         prediction["score"], prediction["model"], risk["severity"], reason)
    return jsonify({"id": mid, "prediction": prediction, "category": category,
                    "risk": risk, "reason": reason})

@app.route("/health")
def health():
    from flask import jsonify
    return jsonify({"status": "ok", "model_ready": DETECTOR.ready})

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", host="127.0.0.1", port=5000)
