import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OA-SENSE NER",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("models/xray_model.keras")

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# THINGSPEAK CONFIGURATION
# ============================================================
# Create a ThingSpeak channel with:
#
# Field 1 = Thigh Angle
# Field 2 = Lower Leg Angle
# Field 3 = Knee Angle
#
# Put your channel ID below.
# If your channel is PUBLIC, leave READ_API_KEY empty.
# ============================================================

THINGSPEAK_CHANNEL_ID = " 3502394"

THINGSPEAK_READ_API_KEY = ""

# ============================================================
# GET HARDWARE DATA FROM THINGSPEAK
# ============================================================

def get_hardware_data():

    if (
        not THINGSPEAK_CHANNEL_ID
        or THINGSPEAK_CHANNEL_ID == "3502394"
    ):
        return {
            "error": "ThingSpeak Channel ID has not been configured."
        }

    url = (
        "https://api.thingspeak.com/channels/"
        + str(3502394)
        + "/feeds/last.json"
    )

    params = {}

    if THINGSPEAK_READ_API_KEY:
        params["api_key"] = THINGSPEAK_READ_API_KEY

    try:

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        thigh = data.get("field1")
        lower_leg = data.get("field2")
        knee = data.get("field3")

        if (
            thigh is None
            or lower_leg is None
            or knee is None
        ):
            return {
                "error": "ThingSpeak does not contain sensor data yet."
            }

        return {
            "thigh": float(thigh),
            "lower_leg": float(lower_leg),
            "knee": float(knee),
            "updated": data.get(
                "created_at",
                "Unknown"
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }
    # ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🦴 OA-SENSE NER")

st.sidebar.caption(
    "AI-Assisted Osteoarthritis Risk Marker System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📝 Patient Assessment",
        "🧠 NER Analysis",
        "🤖 ML Prediction",
        "📷 X-ray Analysis",
        "🦵 Hardware Monitoring",
        "📊 Analytics",
        "📄 Reports",
        "🗃️ Patient History",
        "⚙️ Settings"
    ]
)

# ============================================================
# MAIN TITLE
# ============================================================

st.title("🦴 OA-SENSE NER")

st.caption(
    "AI-Assisted Early Detection System for "
    "Osteoarthritis Risk Markers"
)

st.divider()

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Assessments",
            len(st.session_state.history)
        )

    with c2:
        st.metric(
            "NER Module",
            "Ready"
        )

    with c3:
        st.metric(
            "ML Model",
            "Not Connected"
        )

    with c4:

        if MODEL_PATH.exists():

            st.metric(
                "X-ray Model",
                "Connected"
            )

        else:

            st.metric(
                "X-ray Model",
                "Not Connected"
            )

    st.divider()

    st.subheader("System Overview")

    st.write(
        """
        OA-SENSE NER is a prototype platform designed
        to support osteoarthritis risk-marker assessment
        using structured patient information, clinical
        text analysis, X-ray image analysis and
        hardware-based knee movement monitoring.
        """
    )

    st.info(
        "The current application is a prototype. "
        "Risk results are not a medical diagnosis."
    )

    st.subheader("Available Modules")

    a, b, c = st.columns(3)

    with a:

        st.markdown("### 📝 Assessment")

        st.write(
            "Enter patient information and OA-related "
            "risk factors."
        )

    with b:

        st.markdown("### 🧠 NER")

        st.write(
            "Analyze clinical text for important "
            "OA-related terms."
        )

    with c:

        st.markdown("### 🦵 Hardware")

        st.write(
            "Monitor knee movement using ESP32 "
            "and MPU6050 sensors."
        )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "📝 Patient Assessment":

    st.header("Patient Assessment")

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        patient_id = st.text_input(
            "Patient ID"
        )

        age = st.number_input(
            "Age",
            1,
            120,
            40
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Other"
            ]
        )

    with col2:

        height = st.number_input(
            "Height (cm)",
            50.0,
            250.0,
            165.0
        )

        weight = st.number_input(
            "Weight (kg)",
            10.0,
            250.0,
            65.0
        )

    st.divider()

    st.subheader("OA Risk Factors")

    col1, col2, col3 = st.columns(3)

    with col1:

        joint_pain = st.checkbox(
            "Joint pain"
        )

        morning_stiffness = st.checkbox(
            "Morning stiffness"
        )

        swelling = st.checkbox(
            "Joint swelling"
        )

    with col2:

        previous_injury = st.checkbox(
            "Previous joint injury"
        )

        family_history = st.checkbox(
            "Family history"
        )

        movement_difficulty = st.checkbox(
            "Difficulty in movement"
        )

    with col3:

        knee_pain = st.checkbox(
            "Knee symptoms"
        )

        hip_pain = st.checkbox(
            "Hip symptoms"
        )

        hand_pain = st.checkbox(
            "Hand symptoms"
        )

    clinical_text = st.text_area(
        "Clinical Notes",
        placeholder=(
            "Enter relevant clinical observations "
            "or symptoms..."
        )
    )

    st.divider()

    if st.button(
        "🔍 Analyze Assessment",
        use_container_width=True
    ):

        bmi = weight / (
            (height / 100) ** 2
        )

        score = 0

        if age >= 50:
            score += 2

        elif age >= 40:
            score += 1

        if bmi >= 30:
            score += 2

        elif bmi >= 25:
            score += 1

        if joint_pain:
            score += 2

        if morning_stiffness:
            score += 1

        if swelling:
            score += 1

        if previous_injury:
            score += 1

        if family_history:
            score += 1

        if movement_difficulty:
            score += 2

        if (
            knee_pain
            or hip_pain
            or hand_pain
        ):
            score += 1

        if score <= 2:
            risk = "Low Risk"

        elif score <= 5:
            risk = "Moderate Risk"

        else:
            risk = "High Risk"

        factors = []

        if age >= 40:
            factors.append(
                "Age-related factor"
            )

        if bmi >= 25:
            factors.append(
                "Higher BMI"
            )

        if joint_pain:
            factors.append(
                "Joint pain"
            )

        if morning_stiffness:
            factors.append(
                "Morning stiffness"
            )

        if swelling:
            factors.append(
                "Joint swelling"
            )

        if previous_injury:
            factors.append(
                "Previous joint injury"
            )

        if family_history:
            factors.append(
                "Family history"
            )

        if movement_difficulty:
            factors.append(
                "Movement difficulty"
            )

        if knee_pain:
            factors.append(
                "Knee symptoms"
            )

        if hip_pain:
            factors.append(
                "Hip symptoms"
            )

        if hand_pain:
            factors.append(
                "Hand symptoms"
            )

        result = {
            "Patient ID":
                patient_id
                if patient_id
                else "Not provided",

            "Age":
                age,

            "Gender":
                gender,

            "BMI":
                round(bmi, 2),

            "Risk Score":
                score,

            "Risk Level":
                risk,

            "Date":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
        }

        st.session_state.history.append(
            result
        )

        st.success(
            "Assessment completed successfully."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )

        with c2:

            st.metric(
                "Risk Score",
                score
            )

        with c3:

            st.metric(
                "Risk Level",
                risk
            )

        st.subheader(
            "Identified Risk Factors"
        )

        if factors:

            for factor in factors:

                st.write(
                    "• " + factor
                )

        else:

            st.write(
                "No selected risk factors."
            )

        if clinical_text:

            st.subheader(
                "Clinical Text"
            )

            st.write(
                clinical_text
            )

        st.warning(
            "This is an educational prototype "
            "and does not provide a medical diagnosis."
        )


# ============================================================
# NER ANALYSIS
# ============================================================

elif page == "🧠 NER Analysis":

    st.header(
        "Clinical NER Analysis"
    )

    st.write(
        "Enter clinical text to identify common "
        "osteoarthritis-related entities."
    )

    text = st.text_area(
        "Clinical Text",
        height=200,
        placeholder=(
            "Example: Patient reports knee pain, "
            "morning stiffness and difficulty walking."
        )
    )

    if st.button(
        "🧠 Extract Entities",
        use_container_width=True
    ):

        if not text.strip():

            st.warning(
                "Please enter clinical text."
            )

        else:

            text_lower = text.lower()

            symptoms = []
            body_parts = []
            risk_factors = []

            symptom_words = [
                "pain",
                "stiffness",
                "swelling",
                "difficulty walking",
                "difficulty in movement"
            ]

            body_words = [
                "knee",
                "hip",
                "hand",
                "joint"
            ]

            risk_words = [
                "obesity",
                "overweight",
                "injury",
                "family history",
                "age",
                "previous injury"
            ]

            for word in symptom_words:

                if word in text_lower:
                    symptoms.append(word)

            for word in body_words:

                if word in text_lower:
                    body_parts.append(word)

            for word in risk_words:

                if word in text_lower:
                    risk_factors.append(word)

            data = {
                "Entity Type": [],
                "Detected Entity": []
            }

            for item in symptoms:

                data["Entity Type"].append(
                    "Symptom"
                )

                data["Detected Entity"].append(
                    item
                )

            for item in body_parts:

                data["Entity Type"].append(
                    "Body Location"
                )

                data["Detected Entity"].append(
                    item
                )

            for item in risk_factors:

                data["Entity Type"].append(
                    "Risk Factor"
                )

                data["Detected Entity"].append(
                    item
                )

            if data["Detected Entity"]:

                st.dataframe(
                    pd.DataFrame(data),
                    use_container_width=True
                )

            else:

                st.info(
                    "No predefined OA-related "
                    "entities were detected."
                )

            st.caption(
                "Current NER is a prototype "
                "keyword-based module."
            )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header(
        "Machine Learning Prediction"
    )

    st.info(
        "ML model integration area. A trained and "
        "validated OA prediction model should be "
        "placed in the models folder."
    )

    st.write(
        "Expected model location:"
    )

    st.code(
        "models/oa_model.pkl"
    )

    st.subheader(
        "Model Status"
    )

    model_path = Path(
        "models/oa_model.pkl"
    )

    if model_path.exists():

        st.success(
            "OA machine-learning model detected."
        )

    else:

        st.warning(
            "No trained OA model is currently connected."
        )

    st.write(
        "Once a validated model is available, "
        "this module can generate model-based "
        "predictions and probabilities."
    )


# ============================================================
# X-RAY ANALYSIS
# ============================================================

elif page == "📷 X-ray Analysis":

    st.header(
        "X-ray Analysis"
    )

    st.write(
        "Upload a knee X-ray image for prototype "
        "AI-based OA grade prediction."
    )

    class_names = [
        "Normal",
        "Doubtful",
        "Mild",
        "Moderate",
        "Severe"
    ]

    uploaded_file = st.file_uploader(
        "Upload Knee X-ray Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Knee X-ray",
            use_container_width=True
        )

        if MODEL_PATH.exists():

            st.success(
                "X-ray AI model connected successfully."
            )

            if st.button(
                "🔬 Analyze X-ray",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Analyzing X-ray..."
                    ):

                        model = (
                            tf.keras.models.load_model(
                                MODEL_PATH
                            )
                        )

                        image_resized = image.resize(
                            (224, 224)
                        )

                        image_array = np.array(
                            image_resized
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        image_array = (
                            image_array.astype(
                                "float32"
                            ) / 255.0
                        )

                        prediction = (
                            model.predict(
                                image_array,
                                verbose=0
                            )
                        )

                        predicted_index = int(
                            np.argmax(
                                prediction[0]
                            )
                        )

                        predicted_class = (
                            class_names[
                                predicted_index
                            ]
                        )

                        confidence = (
                            float(
                                prediction[0][
                                    predicted_index
                                ]
                            ) * 100
                        )

                    st.subheader(
                        "X-ray Prediction"
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.metric(
                            "Predicted OA Grade",
                            predicted_class
                        )

                    with c2:

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                    st.subheader(
                        "Prediction Probabilities"
                    )

                    probability_data = {
                        "OA Grade":
                            class_names,

                        "Probability (%)": [
                            float(value) * 100
                            for value
                            in prediction[0]
                        ]
                    }

                    probability_df = (
                        pd.DataFrame(
                            probability_data
                        )
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "OA Grade"
                        )
                    )

                    st.warning(
                        "This AI prediction is for "
                        "educational prototype testing "
                        "and is not a medical diagnosis."
                    )

                except Exception as e:

                    st.error(
                      f"X-ray analysis error: {e}"
                      )

        else:

            st.error(
                "X-ray model not found."
            )

            st.code(
                "models/xray_model.keras"
            )


# ============================================================
# HARDWARE MONITORING
# ============================================================

elif page == "🦵 Hardware Monitoring":

    st.header(
        "🦵 Hardware Monitoring"
    )

    st.write(
        "Real-time knee movement data from the "
        "ESP32 and dual MPU6050 sensors."
    )

    st.divider()

    # --------------------------------------------------------
    # GET SENSOR DATA
    # --------------------------------------------------------

    hardware_data = (
        get_hardware_data()
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if hardware_data is None:

        st.error(
            "No hardware data received."
        )

    elif "error" in hardware_data:

        st.error(
            "Hardware connection error."
        )

        st.warning(
            hardware_data["error"]
        )

        st.info(
            """
            Make sure:

            1. ThingSpeak Channel ID is correct.
            2. ESP32 is connected to Wi-Fi.
            3. ESP32 is sending data to ThingSpeak.
            4. ThingSpeak contains Field 1, Field 2 and Field 3.
            """
        )

    else:

        thigh = hardware_data["thigh"]

        lower_leg = (
            hardware_data["lower_leg"]
        )

        knee = hardware_data["knee"]

        # ----------------------------------------------------
        # CONNECTION STATUS
        # ----------------------------------------------------

        st.success(
            "🟢 ESP32 / MPU6050 data received"
        )
        # ----------------------------------------------------
        # SENSOR VALUES
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Thigh Angle",
                f"{thigh:.1f}°"
            )

        with c2:

            st.metric(
                "Lower Leg Angle",
                f"{lower_leg:.1f}°"
            )

        with c3:

            st.metric(
                "Knee Angle",
                f"{knee:.1f}°"
            )

        st.divider()

        # ----------------------------------------------------
        # KNEE MOVEMENT
        # ----------------------------------------------------

        st.subheader(
            "Knee Movement Status"
        )

        if knee > 60:

            st.warning(
                "⚠️ Knee angle is above the "
                "configured prototype monitoring limit."
            )

        else:

            st.success(
                "✅ Knee angle is within the "
                "configured prototype monitoring limit."
            )

        # ----------------------------------------------------
        # HARDWARE DATA TABLE
        # ----------------------------------------------------

        st.subheader(
            "Current Sensor Data"
        )

        hardware_table = pd.DataFrame(
            {
                "Parameter": [
                    "Thigh Angle",
                    "Lower Leg Angle",
                    "Knee Angle"
                ],

                "Value": [
                    f"{thigh:.1f}°",
                    f"{lower_leg:.1f}°",
                    f"{knee:.1f}°"
                ]
            }
        )

        st.dataframe(
            hardware_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # UPDATE TIME
        # ----------------------------------------------------

        st.caption(
            "Last sensor update: "
            + str(
                hardware_data["updated"]
            )
        )

        # ----------------------------------------------------
        # REFRESH
        # ----------------------------------------------------

        if st.button(
            "🔄 Refresh Sensor Data",
            use_container_width=True
        ):

            st.rerun()

        st.info(
            "Hardware measurements are prototype "
            "sensor data and should not be interpreted "
            "as a medical diagnosis."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header(
        "Analytics"
    )

    if not st.session_state.history:

        st.info(
            "No assessment data available yet."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.subheader(
            "Assessment Data"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader(
            "Risk Distribution"
        )

        risk_counts = (
            df["Risk Level"]
            .value_counts()
        )

        fig, ax = plt.subplots()

        ax.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax.set_xlabel(
            "Risk Level"
        )

        ax.set_ylabel(
            "Number of Assessments"
        )

        ax.set_title(
            "OA Risk Distribution"
        )

        st.pyplot(fig)

        st.subheader(
            "Risk Score"
        )

        fig2, ax2 = plt.subplots()

        ax2.plot(
            range(
                1,
                len(df) + 1
            ),
            df["Risk Score"],
            marker="o"
        )

        ax2.set_xlabel(
            "Assessment"
        )

        ax2.set_ylabel(
            "Risk Score"
        )

        ax2.set_title(
            "Risk Score Trend"
        )

        st.pyplot(fig2)
        # ============================================================
# REPORTS
# ============================================================

elif page == "📄 Reports":

    st.header(
        "Assessment Reports"
    )

    if not st.session_state.history:

        st.info(
            "Complete at least one patient "
            "assessment first."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        report_text = (
            "OA-SENSE NER ASSESSMENT REPORT\n"
        )

        report_text += (
            "=" * 40
            + "\n\n"
        )

        for index, row in df.iterrows():

            report_text += (
                f"Assessment {index + 1}\n"
            )

            report_text += (
                f"Patient ID: "
                f"{row['Patient ID']}\n"
            )

            report_text += (
                f"Age: "
                f"{row['Age']}\n"
            )

            report_text += (
                f"Gender: "
                f"{row['Gender']}\n"
            )

            report_text += (
                f"BMI: "
                f"{row['BMI']}\n"
            )

            report_text += (
                f"Risk Score: "
                f"{row['Risk Score']}\n"
            )

            report_text += (
                f"Risk Level: "
                f"{row['Risk Level']}\n"
            )

            report_text += (
                f"Date: "
                f"{row['Date']}\n"
            )

            report_text += (
                "-" * 40
                + "\n"
            )

        report_file = (
            REPORT_DIR
            / "oa_sense_assessment_report.txt"
        )

        report_file.write_text(
            report_text,
            encoding="utf-8"
        )

        st.download_button(
            "⬇️ Download Assessment Report",
            data=report_text,
            file_name=(
                "oa_sense_assessment_report.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "🗃️ Patient History":

    st.header(
        "Patient History"
    )

    if not st.session_state.history:

        st.info(
            "No patient assessment history available."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.header(
        "Settings"
    )

    st.subheader(
        "System Status"
    )

    st.write(
        "Streamlit application: ✅ Running"
    )

    st.write(
        "Assessment module: ✅ Available"
    )

    st.write(
        "NER prototype: ✅ Available"
    )
    model_path = Path(
        "models/oa_model.pkl"
    )

    if model_path.exists():

        st.write(
            "ML model: ✅ Connected"
        )

    else:

        st.write(
            "ML model: ⚠️ Not connected"
        )

    if MODEL_PATH.exists():

        st.write(
            "X-ray model: ✅ Connected"
        )

    else:

        st.write(
            "X-ray model: ⚠️ Not connected"
        )

    st.divider()

    st.subheader(
        "Hardware Status"
    )

    if (
        THINGSPEAK_CHANNEL_ID
        and THINGSPEAK_CHANNEL_ID
        != "YOUR_CHANNEL_ID"
    ):

        st.write(
            "ThingSpeak: ✅ Configured"
        )

    else:

        st.write(
            "ThingSpeak: ⚠️ Not configured"
        )

    st.divider()

    st.subheader(
        "About"
    )

    st.write(
        """
        OA-SENSE NER is an educational prototype
        for exploring osteoarthritis risk-marker
        assessment, clinical text analysis,
        machine-learning integration, medical-image
        analysis and hardware-based knee movement
        monitoring.
        """
    )

    st.warning(
        "This software is not a substitute for "
        "professional medical diagnosis."
    )
