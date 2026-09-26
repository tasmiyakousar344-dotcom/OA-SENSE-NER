import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OA-SENSE NER",
    page_icon="🦵",
    layout="wide"
)


# ============================================================
# THINGSPEAK CONFIGURATION
# ============================================================

# Replace these with your actual ThingSpeak details

THINGSPEAK_CHANNEL_ID = "3502394"

THINGSPEAK_READ_API_KEY = ""


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = Path(
    "models/xray_model.keras"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# GET HARDWARE DATA FROM THINGSPEAK
# ============================================================

def get_hardware_data():

    try:

        if (
            not THINGSPEAK_CHANNEL_ID
            or THINGSPEAK_CHANNEL_ID == "3502394"
        ):

            return (
                None,
                "ThingSpeak Channel ID is not configured."
            )


        url = (
            f"https://api.thingspeak.com/channels/"
            f"{3502394}/feeds/last.json"
        )


        params = {}


        if THINGSPEAK_READ_API_KEY:

            params["api_key"] = (
                THINGSPEAK_READ_API_KEY
            )


        response = requests.get(
            url,
            params=params,
            timeout=10
        )


        response.raise_for_status()


        data = response.json()


        # Field 1 = Knee Angle

        field1 = data.get("field1")


        if field1 is None:

            return (
                None,
                "No knee-angle data found in ThingSpeak Field 1."
            )


        knee_angle = float(field1)


        timestamp = data.get(
            "created_at",
            ""
        )


        return {

            "knee_angle": knee_angle,

            "timestamp": timestamp

        }, None


    except requests.exceptions.RequestException as e:

        return (
            None,
            f"ThingSpeak connection error: {e}"
        )


    except ValueError:

        return (
            None,
            "Invalid knee-angle value received from ThingSpeak."
        )


    except Exception as e:

        return (
            None,
            f"Hardware data error: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🦵 OA-SENSE NER")

st.sidebar.write(
    "AI-Assisted Early Detection System "
    "for Osteoarthritis Risk Markers"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Patient Assessment",
        "🌎 NER Analysis",
        "🤖 ML Prediction",
        "🩻 X-ray Analysis",
        "🔧 Hardware Monitoring",
        "📊 Analytics",
        "📄 Reports",
        "📋 Patient History",
        "⚙️ Settings"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🦵 OA-SENSE NER")

    st.subheader(
        "AI-Assisted Early Detection System "
        "for Osteoarthritis Risk Markers"
    )


    st.write(
        "This prototype combines patient assessment, "
        "motion monitoring and data analysis."
    )


    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "System",
            "OA-SENSE NER"
        )


    with col2:

        st.metric(
            "Hardware",
            "ESP32 + MPU6050"
        )


    with col3:

        st.metric(
            "Cloud",
            "ThingSpeak"
        )


    st.divider()


    st.info(
        "Use the Hardware Monitoring page to view "
        "the latest knee-angle measurement received "
        "from the ESP32."
    )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "👤 Patient Assessment":

    st.header("Patient Assessment")


    name = st.text_input(
        "Patient Name"
    )


    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


    pain = st.slider(
        "Knee Pain Level",
        0,
        10,
        0
    )


    stiffness = st.selectbox(
        "Morning Stiffness",
        [
            "No",
            "Mild",
            "Moderate",
            "Severe"
        ]
    )


    difficulty = st.selectbox(
        "Difficulty in Movement",
        [
            "No difficulty",
            "Mild",
            "Moderate",
            "Severe"
        ]
    )


    if st.button(
        "Save Assessment"
    ):

        assessment = {

            "Date":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "Patient":
                name,

            "Age":
                age,

            "Pain":
                pain,

            "Stiffness":
                stiffness,

            "Movement Difficulty":
                difficulty
        }


        st.session_state.history.append(
            assessment
        )


        st.success(
            "Patient assessment saved successfully."
        )


# ============================================================
# NER ANALYSIS
# ============================================================

elif page == "🌎 NER Analysis":

    st.header(
        "North Eastern Region Analysis"
    )


    st.write(
        "This section can be used to analyze "
        "regional osteoarthritis risk-marker data."
    )


    col1, col2 = st.columns(2)


    with col1:

        region = st.text_input(
            "Region",
            "North Eastern Region"
        )


    with col2:

        population = st.number_input(
            "Population / Sample Count",
            min_value=0,
            value=0
        )


    st.divider()


    st.info(
        "Regional analysis can be expanded with "
        "validated datasets and statistical models."
    )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header(
        "ML Risk-Marker Prediction"
    )


    st.write(
        "Prototype rule-based screening interface."
    )


    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


    pain = st.slider(
        "Pain Level",
        0,
        10,
        0
    )


    stiffness = st.selectbox(
        "Stiffness",
        [
            "No",
            "Mild",
            "Moderate",
            "Severe"
        ]
    )


    if st.button(
        "Analyze Risk Markers"
    ):

        score = 0


        if age >= 50:
            score += 1


        if pain >= 5:
            score += 1


        if stiffness in [
            "Moderate",
            "Severe"
        ]:
            score += 1


        if score >= 2:

            st.warning(
                "Multiple risk markers detected."
            )

        else:

            st.success(
                "Fewer risk markers detected."
            )


        st.info(
            "This prototype output is not a medical diagnosis."
        )


# ============================================================
# X-RAY ANALYSIS
#elif page == "🩻 X-ray Analysis":

    st.header(
        "X-ray Analysis"
    )


    uploaded_file = st.file_uploader(
        "Upload X-ray Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded X-ray",
            use_container_width=True
        )


        st.info(
            "X-ray prediction functionality can be "
            "connected to the trained model."
        )


        if MODEL_PATH.exists():

            st.success(
                "Model file found."
            )

        else:

            st.warning(
                "X-ray model file was not found at: "
                f"{MODEL_PATH}"
            )


# ============================================================
# HARDWARE MONITORING
# ============================================================

elif page == "🔧 Hardware Monitoring":

    st.header(
        "Hardware Monitoring"
    )


    st.write(
        "Real-time knee-angle monitoring from "
        "ESP32 + MPU6050 through ThingSpeak."
    )


    # Refresh button

    if st.button(
        "🔄 Refresh Hardware Data"
    ):

        st.rerun()


    # Get latest hardware data

    hardware_data, error = (
        get_hardware_data()
    )


    if error:

        st.error(
            error
        )


        st.info(
            "Check your ThingSpeak Channel ID, "
            "Read API Key, and make sure the ESP32 "
            "is sending knee-angle data to Field 1."
        )


    elif hardware_data is not None:

        knee_angle = hardware_data[
            "knee_angle"
        ]


        timestamp = hardware_data[
            "timestamp"
        ]


        # Connection status

        st.success(
            "🟢 ESP32 / MPU6050 data received"
        )


        st.divider()


        # Main measurements

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🦵 Knee Angle",
                f"{knee_angle:.1f}°"
            )


        with col2:

            if knee_angle > 60:

                st.error(
                    "⚠️ WARNING"
                )

                st.write(
                    "Angle is above the prototype "
                    "warning threshold."
                )

            else:

                st.success(
                    "✅ Normal Range"
                )


        st.divider()


        # Hardware information

        st.subheader(
            "Hardware Information"
        )


        hardware_info = pd.DataFrame(
            {

                "Component": [

                    "Controller",

                    "Motion Sensor",

                    "Display",

                    "Cloud Platform",

                    "ThingSpeak Field",

                    "Measurement"

                ],


                "Details": [

                    "ESP32-WROOM-32",

                    "MPU6050",

                    "OLED 128×64",

                    "ThingSpeak",

                    "Field 1",

                    "Knee Angle"

                ]

            }
        )


        st.dataframe(
            hardware_info,
            use_container_width=True,
            hide_index=True
        )


        # Last update time

        if timestamp:

            st.caption(
                "Last ThingSpeak update: "
                f"{timestamp}"
            )


        st.divider()


        # Data flow

        st.subheader(
            "Data Flow"
        )


        st.write(
            "MPU6050 → ESP32 → Wi-Fi → "
            "ThingSpeak → Streamlit"
        )


        st.info(
            "The displayed angle is a prototype "
            "motion measurement and should not be "
            "interpreted as a clinical diagnosis."
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
            "Complete at least one patient "
            "assessment to view analytics."
        )


    else:

        df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(
            df,
            use_container_width=True
        )


        if "Pain" in df.columns:

            st.subheader(
                "Pain Level"
            )


            st.bar_chart(
                df["Pain"]
            )


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
            "OA-SENSE NER Assessment Report\n\n"
        )


        report_text += (
            df.to_string(index=False)
        )


        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name="oa_sense_ner_report.txt",
            mime="text/plain"
        )


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.header(
        "Patient History"
    )


    if not st.session_state.history:

        st.info(
            "No patient history available."
        )


    else:

        df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(
            df,
            use_container_width=True
        )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.header(
        "Settings"
    )


    st.subheader(
        "System Information"
    )


    st.write(
        "Application: OA-SENSE NER"
    )


    st.write(
        "Controller: ESP32-WROOM-32"
    )


    st.write(
        "Sensor: MPU6050"
    )


    st.write(
        "Display: OLED 128×64"
    )


    st.write(
        "Cloud Platform: ThingSpeak"
    )


    st.write(
        "ThingSpeak Measurement: Field 1 – Knee Angle"
    )


    st.divider()


    st.subheader(
        "Model Information"
    )


    model_path = Path(
        "models/oa_model.pkl"
    )


    if model_path.exists():

        st.success(
            "Model file found."
        )

    else:

        st.info(
            "No additional model file configured."
        )


    st.divider()


    st.info(
        "OA-SENSE NER is a prototype system for "
        "early risk-marker screening and motion "
        "monitoring. It is not a replacement for "
        "professional medical evaluation."
    )
