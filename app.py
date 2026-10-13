
from pathlib import Path
import re

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "spam_classifier.joblib"

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Spam Message Detector")
st.write(
    "Classify SMS messages using machine learning "
    "and simple phishing warning rules."
)

if not MODEL_FILE.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)

message = st.text_area(
    "Enter your message",
    placeholder="Enter a message to check...",
    height=150,
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.error("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = float(
            probabilities[classes.index(prediction)]
        )

        text = message.lower()

        patterns = {
            "Urgent account or security language":
                r"\b(urgent|immediately|suspended|suspicious login|"
                r"verify your identity|secure your account)\b",

            "Request to verify account details":
                r"\b(verify|confirm|update)\b.{0,45}"
                r"\b(account|identity|password|details)\b",

            "Link included in the message":
                r"https?://\S+|www\.\S+",

            "Prize or reward language":
                r"\b(won|winner|prize|claim your reward|"
                r"free cash|cash reward)\b",
        }

        matched_warnings = [
            label
            for label, pattern in patterns.items()
            if re.search(pattern, text)
        ]

        # A link alone is not enough to trigger a warning.
        warnings = [
            warning
            for warning in matched_warnings
            if warning != "Link included in the message"
        ]

        # Flag links combined with suspicious account language.
        account_warnings = {
            "Urgent account or security language",
            "Request to verify account details",
        }

        if (
            "Link included in the message" in matched_warnings
            and any(
                warning in matched_warnings
                for warning in account_warnings
            )
        ):
            warnings.append(
                "Link combined with suspicious account language"
            )

        st.subheader("Detection Results")

        if prediction == "spam" or warnings:
            st.error("Overall Result: SUSPICIOUS")
        else:
            st.success(
                "Overall Result: No configured warning detected"
            )

        st.caption(
            f"ML model prediction: {prediction.upper()}"
        )
        st.metric("ML model confidence", f"{confidence:.1%}")

        if warnings:
            st.warning(
                "Potential phishing indicators were detected."
            )

            for warning in warnings:
                st.write(f"- {warning}")

            st.info(
                "Avoid clicking unexpected links or sharing "
                "sensitive information. Verify the sender "
                "through an official channel."
            )
        else:
            st.info(
                "No configured warning patterns were detected. "
                "This does not guarantee the message is safe."
            )
