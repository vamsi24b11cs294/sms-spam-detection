# Messages — SMS Spam Detection Website

A simple Flask website styled like a modern Android messaging app. There is no admin login or admin dashboard, and the user does not need to enter a phone number or sender ID. Type or paste an SMS into the chat composer and tap send to classify it.

## Features
- Android Messages-inspired inbox and chat interface, responsive for desktop and mobile.
- No admin phase, login screen, sender ID field, or phone number field.
- SMS type classification from message text: Banking, UPI/Payment, OTP/Verification, Telecom/Network, Promotional, Loan/Financial, Delivery/E-commerce, Government/Service, Personal, Spam/Phishing, and General SMS.
- Spam/fraud risk label, risk score, and a short explanation.
- Recent checked messages shown in the inbox and saved locally in SQLite (`history.db`).
- Optional TF-IDF machine-learning model training with Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
- JSON endpoint: `POST /api/analyze` with `{"message":"Your SMS text here"}`.

## Important limitations
- This is a local academic demo. It does not read SMS messages from a real Android phone or send messages through a mobile network; messages are entered manually in the website.
- Before a model is trained, the app uses a rule-based fallback. This is not a trained ML model.
- Risk scores are estimates, not guarantees. Do not use the app as the sole basis for financial or security decisions.
- Avoid entering real OTPs, passwords, account details, or private messages. Submitted text is saved locally in SQLite.

## Run in VS Code on Windows
1. Extract the ZIP and open the extracted `sms_spam_detection_project` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Start the website:

```powershell
.\.venv\Scripts\python.exe app.py
```

5. Open **http://127.0.0.1:5000** in your browser.

If `py -3.12` is unavailable but `py --version` works, use `py -m venv .venv` instead. Using the virtual environment's `python.exe` directly avoids PATH and PowerShell activation issues.

## Train a real ML model (optional)
1. Obtain a labeled SMS dataset, such as the UCI SMS Spam Collection, and save it as `data/spam.csv`.
2. The CSV must contain `label,message` columns or the original `v1,v2` columns. Labels should identify legitimate/ham and spam messages.
3. Run:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script compares the supported models on a hold-out test split and saves the selected model to `model/spam_model.joblib`. Restart the website after training. Results depend on the dataset supplied.

## Troubleshooting
- `ModuleNotFoundError`: run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` from the project folder.
- `Dataset not found`: add the dataset at exactly `data/spam.csv`; training is optional.
- Port 5000 is busy: close the other local Flask app or change the port at the bottom of `app.py`.
