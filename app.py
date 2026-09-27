import streamlit as st
import requests
import pandas as pd
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

THINGSPEAK_CHANNEL_ID = "3502394"

# Leave empty if your ThingSpeak channel is PUBLIC.
# If the channel is PRIVATE, put your READ API KEY here.
THINGSPEAK_READ_API_KEY = ""

HIGH_RISK_ANGLE = 25.0


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# GET HARDWARE DATA
# ============================================================

def get_hardware_data():

    try:

        url = (
            "https://api.thingspeak.com/channels/"
            "3502394/feeds/last.json"
        )

        params = {}

        if THINGSPEAK_READ_API_KEY:
            params["api_key"] = THINGSPEAK_READ_API_KEY

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        field1 = data.get("field1")

        if field1 is None:
            return None, "No knee-angle data found in ThingSpeak Field 1."

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
            "Invalid knee-angle value received."
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

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Patient Assessment",
        "🧠 NER Analysis",
        "🤖 ML Prediction",
        "🩻 X-ray Analysis",
        "📡 Hardware Monitoring",
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
        "North Eastern Region (NER)"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🦵 Knee Movement Monitoring")

    with col2:
        st.info("📡 ESP32 + MPU6050")

    with col3:
        st.info("☁️ ThingSpeak Cloud")


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "👤 Patient Assessment":

    st.title("👤 Patient Assessment")

    name = st.text_input("Patient Name")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    symptoms = st.text_area(
        "Symptoms / Observations"
    )

    if st.button("Save Assessment"):

        st.session_state.history.append(
            {
                "Patient Name": name,
                "Age": age,
                "Symptoms": symptoms,
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
            }
        )

        st.success(
            "Patient assessment saved successfully."
        )


# ============================================================
# NER ANALYSIS
# ============================================================

elif page == "🧠 NER Analysis":

    st.title("🧠 NER Analysis")

    st.info(
        "Regional analysis and patient information "
        "can be displayed here."
    )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.title("🤖 ML Prediction")

    st.info(
        "Machine-learning prediction module."
    )

    st.warning(
        "This prototype output should not be "
        "treated as a medical diagnosis."
    )


# ============================================================
# X-RAY ANALYSIS
# ============================================================

elif page == "🩻 X-ray Analysis":

    st.title("🩻 X-ray Analysis")

    uploaded_file = st.file_uploader(
        "Upload X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded X-ray",
            use_container_width=True
        )

        st.info(
            "X-ray analysis module ready."
        )


# ============================================================
# HARDWARE MONITORING
# =================================================
elif page == "📡 Hardware Monitoring":

    st.title("📡 Hardware Monitoring")

    st.write(
        "Live knee-angle data from ESP32 → ThingSpeak"
    )

    st.divider()

    # --------------------------------------------------------
    # REFRESH BUTTON
    # --------------------------------------------------------

    if st.button("🔄 Refresh Data"):

        st.rerun()

    # --------------------------------------------------------
    # GET DATA
    # --------------------------------------------------------

    hardware_data, hardware_error = (
        get_hardware_data()
    )

    # --------------------------------------------------------
    # DISPLAY DATA
    # --------------------------------------------------------

    if hardware_data is not None:

        knee_angle = hardware_data["knee_angle"]

        timestamp = hardware_data["timestamp"]

        st.success(
            "🟢 ESP32 / ThingSpeak data received"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🦵 Knee Angle",
                f"{knee_angle:.1f}°"
            )

        with col2:

            st.metric(
                "⚠️ Prototype Threshold",
                f"{HIGH_RISK_ANGLE:.0f}°"
            )

        st.divider()

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if knee_angle > HIGH_RISK_ANGLE:

            st.error(
                "🚨 HIGH-RISK WARNING"
            )

            st.error(
                f"Knee angle is "
                f"{knee_angle:.1f}°."
            )

            st.warning(
                "The measured angle is above "
                "the prototype warning threshold "
                f"of {HIGH_RISK_ANGLE:.0f}°."
            )

            st.info(
                "🔔 The ESP32 buzzer should activate "
                "when the measured angle exceeds "
                f"{HIGH_RISK_ANGLE:.0f}°."
            )

        else:

            st.success(
                f"✅ NORMAL RANGE — "
                f"Knee angle: {knee_angle:.1f}°"
            )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        if timestamp:

            st.caption(
                f"Last ThingSpeak update: {timestamp}"
            )

    else:

        st.error(
            "❌ Hardware data not received"
        )

        st.warning(
            hardware_error
        )

        st.info(
            "Check that ESP32 is sending "
            "knee-angle data to ThingSpeak Field 1."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No assessment data available."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "📄 Reports":

    st.title("📄 Assessment Reports")

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


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.title("📋 Patient History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No patient history available."
        )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.write(
        "### Hardware Configuration"
    )

    st.write(
        "ESP32-WROOM-32"
    )

    st.write(
        "MPU6050 — I2C"
    )

    st.write(
        "OLED — 128 × 64"
    )

    st.write(
        "ThingSpeak Channel ID: 3502394"
    )

    st.write(
        f"Prototype warning threshold: "
        f"{HIGH_RISK_ANGLE:.0f}°"
    )

    st.warning(
        "The threshold used in this prototype "
        "is not a clinically validated diagnostic cutoff."
    )
