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

if "historical_df" not in st.session_state:
    st.session_state.historical_df = None

if "historical_error" not in st.session_state:
    st.session_state.historical_error = None


# ============================================================
# GET LATEST HARDWARE DATA
# ============================================================

def get_hardware_data():

    try:

        url = (
            f"https://api.thingspeak.com/channels/"
            f"{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
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

        # Optional fields
        fsr1 = data.get("field2")
        fsr2 = data.get("field3")
        total_load = data.get("field4")
        status_code = data.get("field5")
        buzzer = data.get("field6")

        try:
            fsr1 = float(fsr1) if fsr1 not in (None, "") else None
        except (ValueError, TypeError):
            fsr1 = None

        try:
            fsr2 = float(fsr2) if fsr2 not in (None, "") else None
        except (ValueError, TypeError):
            fsr2 = None

        try:
            total_load = (
                float(total_load)
                if total_load not in (None, "")
                else None
            )
        except (ValueError, TypeError):
            total_load = None

        try:
            status_code = (
                int(float(status_code))
                if status_code not in (None, "")
                else None
            )
        except (ValueError, TypeError):
            status_code = None

        try:
            buzzer = (
                int(float(buzzer))
                if buzzer not in (None, "")
                else None
            )
        except (ValueError, TypeError):
            buzzer = None

        timestamp = data.get(
            "created_at",
            ""
        )

        return {
            "knee_angle": knee_angle,
            "fsr1": fsr1,
            "fsr2": fsr2,
            "total_load": total_load,
            "status_code": status_code,
            "buzzer": buzzer,
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
            "Invalid sensor value received."
        )

    except Exception as e:

        return (
            None,
            f"Hardware data error: {e}"
        )


# ============================================================
# GET HISTORICAL THINGSPEAK DATA
# ============================================================

def get_historical_data(results=100):

    try:

        url = (
            f"https://api.thingspeak.com/channels/"
            f"{THINGSPEAK_CHANNEL_ID}/feeds.json"
        )

        params = {
            "results": results
        }

        if THINGSPEAK_READ_API_KEY:
            params["api_key"] = THINGSPEAK_READ_API_KEY

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        feeds = data.get("feeds", [])

        if not feeds:

            return (
                None,
                "No historical data available in ThingSpeak."
            )

        rows = []

        for feed in feeds:

            rows.append(
                {
                    "Time": feed.get("created_at"),

                    "Knee Angle (°)": (
                        pd.to_numeric(
                            feed.get("field1"),
                            errors="coerce"
                        )
                    ),

                    "FSR1": (
                        pd.to_numeric(
                            feed.get("field2"),
                            errors="coerce"
                        )
                    ),

                    "FSR2": (
                        pd.to_numeric(
                            feed.get("field3"),
                            errors="coerce"
                        )
                    ),

                    "Total Load": (
                        pd.to_numeric(
                            feed.get("field4"),
                            errors="coerce"
                        )
                    ),

                    "Status Code": (
                        pd.to_numeric(
                            feed.get("field5"),
                            errors="coerce"
                        )
                    ),

                    "Buzzer": (
                        pd.to_numeric(
                            feed.get("field6"),
                            errors="coerce"
                        )
                    )
                }
            )

        df = pd.DataFrame(rows)

        df["Time"] = pd.to_datetime(
            df["Time"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Time"]
        )

        df = df.sort_values(
            "Time"
        )

        return df, None

    except requests.exceptions.RequestException as e:

        return (
            None,
            f"ThingSpeak connection error: {e}"
        )

    except Exception as e:

        return (
            None,
            f"Historical data error: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🦵 OA-SENSE NER")

page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "🏠 Dashboard",
        "👤 Patient Assessment",
        "🧠 NER Analysis",
        "🤖 ML Prediction",
        "🩻 X-ray Analysis",
        "📡 Hardware Monitoring",
        "📊 Analytics",
        "🎤 Voice Assistant",
        "📄 Reports",
        "📋 Patient History",
        "⚙️ Settings"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    f"☁️ ThingSpeak Channel: {THINGSPEAK_CHANNEL_ID}"
)

st.sidebar.caption(
    "📡 Live source: ESP32 → ThingSpeak"
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
        st.info("📡 ESP32 + MPU6500")

    with col3:
        st.info("☁️ ThingSpeak Cloud")

    st.divider()

    st.subheader("📡 Latest Hardware Status")

    hardware_data, hardware_error = get_hardware_data()

    if hardware_data is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🦵 Knee Angle",
                f"{hardware_data['knee_angle']:.1f}°"
            )

        with col2:

            if hardware_data["fsr1"] is not None:
                st.metric(
                    "FSR1",
                    f"{hardware_data['fsr1']:.2f}"
                )
            else:
                st.metric(
                    "FSR1",
                    "N/A"
                )

        with col3:

            if hardware_data["fsr2"] is not None:
                st.metric(
                    "FSR2",
                    f"{hardware_data['fsr2']:.2f}"
                )
            else:
                st.metric(
                    "FSR2",
                    "N/A"
                )

        with col4:

            if hardware_data["total_load"] is not None:
                st.metric(
                    "⚖️ Total Load",
                    f"{hardware_data['total_load']:.2f}"
                )
            else:
                st.metric(
                    "⚖️ Total Load",
                    "N/A"
                )

        if hardware_data["knee_angle"] > HIGH_RISK_ANGLE:

            st.warning(
                "⚠️ Prototype warning threshold exceeded."
            )

        else:

            st.success(
                "✅ Current reading is below the prototype warning threshold."
            )

        if hardware_data["timestamp"]:

            st.caption(
                f"Last ThingSpeak update: "
                f"{hardware_data['timestamp']}"
            )

    else:

        st.info(
            "📡 Hardware data is not currently available."
        )

        if hardware_error:
            st.caption(hardware_error)


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

    if st.button("💾 Save Assessment"):

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
            "✅ Patient assessment saved successfully."
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
        "⚠️ This prototype output should not be "
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
            "🩻 X-ray analysis module ready."
        )


# ============================================================
# HARDWARE MONITORING
# ============================================================

elif page == "📡 Hardware Monitoring":

    st.title("📡 Hardware Monitoring")

    st.write(
        "Live sensor data from ESP32 → ThingSpeak"
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

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🦵 Knee Angle",
                f"{knee_angle:.1f}°"
            )

        with col2:

            if hardware_data["fsr1"] is not None:

                st.metric(
                    "FSR1",
                    f"{hardware_data['fsr1']:.2f}"
                )

            else:

                st.metric(
                    "FSR1",
                    "N/A"
                )

        with col3:

            if hardware_data["fsr2"] is not None:

                st.metric(
                    "FSR2",
                    f"{hardware_data['fsr2']:.2f}"
                )

            else:

                st.metric(
                    "FSR2",
                    "N/A"
                )

        with col4:

            if hardware_data["total_load"] is not None:

                st.metric(
                    "⚖️ Total Load",
                    f"{hardware_data['total_load']:.2f}"
                )

            else:

                st.metric(
                    "⚖️ Total Load",
                    "N/A"
                )

        st.divider()

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if knee_angle > HIGH_RISK_ANGLE:

            st.error(
                "🚨 PROTOTYPE WARNING"
            )

            st.warning(
                f"Knee angle is {knee_angle:.1f}°."
            )

            st.info(
                "The measured angle is above the "
                "prototype warning threshold of "
                f"{HIGH_RISK_ANGLE:.0f}°."
            )

        else:

            st.success(
                f"✅ Below prototype warning threshold — "
                f"Knee angle: {knee_angle:.1f}°"
            )

        # ----------------------------------------------------
        # BUZZER
        # ----------------------------------------------------

        if hardware_data["buzzer"] is not None:

            if hardware_data["buzzer"] == 1:

                st.warning(
                    "🔔 Buzzer status: ON"
                )

            else:

                st.success(
                    "🔕 Buzzer status: OFF"
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
            "Check that ESP32 is sending data "
            "to ThingSpeak."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    st.write(
        "Historical sensor data from "
        "ESP32 → ThingSpeak."
    )

    st.divider()

    # --------------------------------------------------------
    # NUMBER OF READINGS
    # --------------------------------------------------------

    results = st.selectbox(
        "📌 Number of ThingSpeak readings",
        [50, 100, 200, 500],
        index=1
    )

    # --------------------------------------------------------
    # REFRESH GRAPH DATA
    # --------------------------------------------------------

    if st.button(
        "🔄 Load / Refresh Graphs",
        type="primary"
    ):

        historical_df, historical_error = (
            get_historical_data(
                results=results
            )
        )

        st.session_state.historical_df = historical_df

        st.session_state.historical_error = (
            historical_error
        )

    # --------------------------------------------------------
    # AUTOMATIC FIRST LOAD
    # --------------------------------------------------------

    if st.session_state.historical_df is None:

        historical_df, historical_error = (
            get_historical_data(
                results=results
            )
        )

        st.session_state.historical_df = (
            historical_df
        )

        st.session_state.historical_error = (
            historical_error
        )

    historical_df = st.session_state.historical_df

    historical_error = (
        st.session_state.historical_error
    )

    # --------------------------------------------------------
    # DISPLAY HISTORICAL DATA
    # --------------------------------------------------------

    if (
        historical_df is not None
        and not historical_df.empty
    ):

        st.success(
            f"✅ {len(historical_df)} ThingSpeak readings loaded."
        )

        # ====================================================
        # GRAPH 1 — KNEE ANGLE
        # ====================================================

        st.subheader(
            "📈 Knee Angle vs Time"
        )

        angle_df = historical_df[
            [
                "Time",
                "Knee Angle (°)"
            ]
        ].dropna(
            subset=["Knee Angle (°)"]
        )

        if not angle_df.empty:

            angle_df = angle_df.set_index(
                "Time"
            )

            st.line_chart(
                angle_df,
                use_container_width=True
            )

        else:

            st.info(
                "No knee-angle history available."
            )

        # ====================================================
        # GRAPH 2 — FSR1 AND FSR2
        # ====================================================

        st.subheader(
            "📊 FSR1 & FSR2 vs Time"
        )

        fsr_df = historical_df[
            [
                "Time",
                "FSR1",
                "FSR2"
            ]
        ].dropna(
            how="all",
            subset=[
                "FSR1",
                "FSR2"
            ]
        )

        if not fsr_df.empty:

            fsr_df = fsr_df.set_index(
                "Time"
            )

            st.line_chart(
                fsr_df,
                use_container_width=True
            )

        else:

            st.info(
                "No FSR historical data available."
            )

        # ====================================================
        # GRAPH 3 — TOTAL LOAD
        # ====================================================

        st.subheader(
            "⚖️ Total Load vs Time"
        )

        load_df = historical_df[
            [
                "Time",
                "Total Load"
            ]
        ].dropna(
            subset=["Total Load"]
        )

        if not load_df.empty:
           load_df = load_df.set_index(
                "Time"
            )

           st.line_chart(
                load_df,
                use_container_width=True
            )

        else:

            st.info(
                "No total-load historical data available."
            )

        # ====================================================
        # DATA SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📌 Sensor Data Summary"
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        with summary_col1:

            valid_angles = historical_df[
                "Knee Angle (°)"
            ].dropna()

            if not valid_angles.empty:

                st.metric(
                    "Average Angle",
                    f"{valid_angles.mean():.2f}°"
                )

            else:

                st.metric(
                    "Average Angle",
                    "N/A"
                )

        with summary_col2:

            if not valid_angles.empty:

                st.metric(
                    "Maximum Angle",
                    f"{valid_angles.max():.2f}°"
                )

            else:

                st.metric(
                    "Maximum Angle",
                    "N/A"
                )

        with summary_col3:

            valid_fsr1 = historical_df[
                "FSR1"
            ].dropna()

            if not valid_fsr1.empty:

                st.metric(
                    "Average FSR1",
                    f"{valid_fsr1.mean():.2f}"
                )

            else:

                st.metric(
                    "Average FSR1",
                    "N/A"
                )

        with summary_col4:

            valid_load = historical_df[
                "Total Load"
            ].dropna()

            if not valid_load.empty:

                st.metric(
                    "Average Total Load",
                    f"{valid_load.mean():.2f}"
                )

            else:

                st.metric(
                    "Average Total Load",
                    "N/A"
                )

        # ====================================================
        # HISTORICAL SENSOR DATA TABLE
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Historical Sensor Data"
        )

        display_df = historical_df.copy()

        display_df["Time"] = display_df[
            "Time"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD CSV
        # ====================================================

        st.divider()

        csv_data = historical_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Sensor History CSV",
            data=csv_data,
            file_name="oa_sense_ner_sensor_history.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "⚠️ No historical ThingSpeak data available."
        )

        if historical_error:

            st.info(
                historical_error
            )

        st.write(
            "Make sure the ESP32 is uploading data "
            "to ThingSpeak."
        )

        st.write(
            "Required ThingSpeak fields:"
        )

        st.write(
            "• Field 1 → Knee Angle"
        )

        st.write(
            "• Field 2 → FSR1"
        )

        st.write(
            "• Field 3 → FSR2"
        )

        st.write(
            "• Field 4 → Total Load"
        )

        st.write(
            "• Field 5 → Status Code"
        )

        st.write(
            "• Field 6 → Buzzer"
        )
        
elif page == "🎤 Voice Assistant":

    st.header("🎤 OA-SENSE NER Voice Assistant")
    st.caption("Ask OA-SENSE NER questions using your voice.")

    import streamlit.components.v1 as components
    import json

    # ---------------------------------------------------------
    # CURRENT OA-SENSE NER DATA
    # ---------------------------------------------------------
    try:
        voice_data = get_hardware_data()

        if voice_data:
            current_angle = float(voice_data.get("angle", 0))
            current_fsr1 = float(voice_data.get("fsr1", 0))
            current_fsr2 = float(voice_data.get("fsr2", 0))
            current_total = float(voice_data.get("total_load", 0))
            current_status = str(voice_data.get("status", "NORMAL"))
        else:
            current_angle = 0
            current_fsr1 = 0
            current_fsr2 = 0
            current_total = 0
            current_status = "NO DATA"

    except Exception:
        current_angle = 0
        current_fsr1 = 0
        current_fsr2 = 0
        current_total = 0
        current_status = "NO DATA"


    # ---------------------------------------------------------
    # LANGUAGE SETTINGS
    # ---------------------------------------------------------
    languages = {
        "English": "en-IN",
        "Kannada": "kn-IN",
        "Hindi": "hi-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "Malayalam": "ml-IN",
        "Marathi": "mr-IN",
        "Bengali": "bn-IN",
        "Gujarati": "gu-IN",
        "Punjabi": "pa-IN",
        "Urdu": "ur-IN",
        "Spanish": "es-ES",
        "French": "fr-FR",
        "German": "de-DE",
        "Arabic": "ar-SA",
        "Chinese": "zh-CN",
        "Japanese": "ja-JP"
    }

    selected_language = st.selectbox(
        "🌐 Select Voice Language",
        list(languages.keys()),
        index=0
    )

    selected_language_code = languages[selected_language]


    # ---------------------------------------------------------
    # OA-SENSE NER INFORMATION FOR VOICE RESPONSE
    # ---------------------------------------------------------
    voice_info = {
        "angle": current_angle,
        "fsr1": current_fsr1,
        "fsr2": current_fsr2,
        "total": current_total,
        "status": current_status
    }

    voice_info_json = json.dumps(voice_info)


    # ---------------------------------------------------------
    # MOBILE-FRIENDLY VOICE ASSISTANT
    # ---------------------------------------------------------
    voice_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport"
          content="width=device-width,
                   initial-scale=1.0,
                   maximum-scale=1.0">

    <style>

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        padding: 8px;
        font-family: Arial, sans-serif;
        background: transparent;
    }}

    .voice-container {{
        width: 100%;
        max-width: 700px;
        margin: auto;
    }}

    .voice-card {{
        width: 100%;
        padding: 18px;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #dddddd;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }}

    .title {{
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 6px;
    }}

    .subtitle {{
        font-size: 14px;
        color: #666;
        margin-bottom: 18px;
    }}

    .language-label {{
        font-weight: bold;
        margin-bottom: 6px;
        display: block;
    }}

    select {{
        width: 100%;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #cccccc;
        font-size: 16px;
        background: white;
        margin-bottom: 15px;
    }}

    .button-row {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 10px;
    }}

    button {{
        border: none;
        border-radius: 12px;
        padding: 13px 8px;
        font-size: 15px;
        font-weight: bold;
        cursor: pointer;
        min-height: 48px;
    }}

    #startBtn {{
        background: #4caf50;
        color: white;
    }}

    #stopBtn {{
        background: #f44336;
        color: white;
    }}

    #speakBtn {{
        background: #2196f3;
        color: white;
    }}

    #clearBtn {{
        background: #777;
        color: white;
    }}

    #copyBtn {{
        background: #673ab7;
        color: white;
    }}

    textarea {{
        width: 100%;
        min-height: 130px;
        resize: vertical;
        border: 1px solid #cccccc;
        border-radius: 12px;
        padding: 13px;
        font-size: 16px;
        line-height: 1.5;
        outline: none;
        margin-bottom: 10px;
    }}

    textarea:focus {{
        border: 2px solid #2196f3;
    }}

    .status {{
        padding: 12px;
        border-radius: 10px;
        background: #f5f5f5;
        font-size: 14px;
        margin-top: 8px;
    }}

    .response {{
        margin-top: 12px;
        padding: 14px;
        border-radius: 12px;
        background: #eef7ff;
        border-left: 4px solid #2196f3;
        font-size: 15px;
        line-height: 1.5;
    }}

    @media (max-width: 480px) {{

        body {{
            padding: 3px;
        }}

        .voice-card {{
            padding: 13px;
            border-radius: 14px;
        }}

        .title {{
            font-size: 19px;
        }}

        .button-row {{
            grid-template-columns: 1fr 1fr;
            gap: 7px;
        }}

        button {{
            font-size: 14px;
            padding: 11px 5px;
        }}

        textarea {{
            min-height: 120px;
            font-size: 15px;
        }}
    }}

    </style>
    </head>

    <body>

    <div class="voice-container">

        <div class="voice-card">

            <div class="title">
                🎤 OA-SENSE NER Voice Assistant
            </div>

            <div class="subtitle">
                Ask questions about your OA-SENSE NER screening.
            </div>

            <label class="language-label">
                🌐 Voice Language
            </label>

            <select id="languageSelect">

                <option value="en-IN">English</option>
                <option value="kn-IN">Kannada</option>
                <option value="hi-IN">Hindi</option>
                <option value="ta-IN">Tamil</option>
                <option value="te-IN">Telugu</option>
                <option value="ml-IN">Malayalam</option>
                <option value="mr-IN">Marathi</option>
                <option value="bn-IN">Bengali</option>
                <option value="gu-IN">Gujarati</option>
                <option value="pa-IN">Punjabi</option>
                <option value="ur-IN">Urdu</option>
                <option value="es-ES">Spanish</option>
                <option value="fr-FR">French</option>
                <option value="de-DE">German</option>
                <option value="ar-SA">Arabic</option>
                <option value="zh-CN">Chinese</option>
                <option value="ja-JP">Japanese</option>

            </select>


            <div class="button-row">

                <button id="startBtn">
                    🎤 Start Voice
                </button>

                <button id="stopBtn">
                    ⏹ Stop Voice
                </button>

            </div>


            <textarea
                id="questionBox"
                placeholder="Speak or type your question here...">
            </textarea>


            <div class="button-row">

                <button id="answerBtn"
                        style="background:#009688;color:white;">
                    🤖 Get OA-SENSE Response
                </button>

                <button id="speakBtn">
                    🔊 Speak Response
                </button>

            </div>


            <div class="button-row">

                <button id="copyBtn">
                    📋 Copy Text
                </button>

                <button id="clearBtn">
                    🧹 Clear
                </button>

            </div>


            <div class="status" id="status">
                🟢 Ready to listen.
            </div>


            <div class="response" id="responseBox">
                OA-SENSE NER response will appear here.
            </div>

        </div>

    </div>


    <script>

    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    const questionBox =
        document.getElementById("questionBox");

    const responseBox =
        document.getElementById("responseBox");

    const statusBox =
        document.getElementById("status");

    const startBtn =
        document.getElementById("startBtn");

    const stopBtn =
        document.getElementById("stopBtn");

    const answerBtn =
        document.getElementById("answerBtn");

    const speakBtn =
        document.getElementById("speakBtn");

    const copyBtn =
        document.getElementById("copyBtn");

    const clearBtn =
        document.getElementById("clearBtn");

    const languageSelect =
        document.getElementById("languageSelect");


    const sensorData =
        {voice_info_json};


    let recognition = null;


    // -------------------------------------------------------
    // VOICE RECOGNITION
    // -------------------------------------------------------

    if (SpeechRecognition) {{

        recognition = new SpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = true;

        recognition.onstart = function() {{
            statusBox.innerHTML =
                "🟢 Listening... Speak now.";
        }};


        recognition.onresult = function(event) {{

            let text = "";

            for (
                let i = event.resultIndex;
                i < event.results.length;
                i++
            ) {{

                text += event.results[i][0].transcript;
            }}

            questionBox.value = text;

            statusBox.innerHTML =
                "🗣️ Voice captured.";
        }};


        recognition.onerror = function(event) {{

            statusBox.innerHTML =
                "⚠️ Voice error: " + event.error;
        }};


        recognition.onend = function() {{

            statusBox.innerHTML =
                "⏹ Voice recognition stopped.";
        }};

    }} else {{

        statusBox.innerHTML =
            "⚠️ Voice recognition is not supported by this browser.";
    }}


    // -------------------------------------------------------
    // START VOICE
    // -------------------------------------------------------

    startBtn.onclick = function() {{

        if (!recognition) {{
            statusBox.innerHTML =
                "⚠️ Voice recognition is unavailable.";
            return;
        }}

        recognition.lang =
            languageSelect.value;

        try {{
            recognition.start();

            statusBox.innerHTML =
                "🟢 Listening...";
        }}
        catch(error) {{
            statusBox.innerHTML =
                "🎤 Already listening.";
        }}
    }};


    // -------------------------------------------------------
    // STOP VOICE
    // -------------------------------------------------------

    stopBtn.onclick = function() {{

        if (recognition) {{
            recognition.stop();
        }}

        window.speechSynthesis.cancel();

        statusBox.innerHTML =
            "⏹ Voice stopped.";
    }};


    // -------------------------------------------------------
    // OA-SENSE RESPONSE ENGINE
    // -------------------------------------------------------

    answerBtn.onclick = function() {{

        const question =
            questionBox.value.toLowerCase().trim();

        if (!question) {{

            responseBox.innerHTML =
                "🎤 Please speak or type a question first.";

            return;
        }}


        let answer = "";


        // ANGLE
        if (
            question.includes("angle") ||
            question.includes("knee angle") ||
            question.includes("bend")
        ) {{

            answer =
                "The current knee angle measured by OA-SENSE NER is "
                + sensorData.angle.toFixed(1)
                + " degrees.";
        }}


        // FSR
        else if (
            question.includes("fsr") ||
            question.includes("pressure") ||
            question.includes("force")
        ) {{

            answer =
                "The current sensor readings are FSR one "
                + sensorData.fsr1.toFixed(1)
                + ", FSR two "
                + sensorData.fsr2.toFixed(1)
                + ", with total load "
                + sensorData.total.toFixed(1)
                + ".";
        }}


        // STATUS
        else if (
            question.includes("status") ||
            question.includes("condition") ||
            question.includes("result") ||
            question.includes("risk")
        ) {{

            answer =
                "The current OA-SENSE NER screening status is "
                + sensorData.status
                + ". This is a prototype screening result and is not a medical diagnosis.";
        }}


        // HARDWARE
        else if (
            question.includes("sensor") ||
            question.includes("hardware") ||
            question.includes("esp32")
        ) {{

            answer =
                "OA-SENSE NER uses an ESP32 with knee movement and pressure sensing. "
                + "The sensor data is transmitted to ThingSpeak and displayed on the dashboard.";
        }}


        // THINGSPEAK
        else if (
            question.includes("thingspeak") ||
            question.includes("cloud") ||
            question.includes("data")
        ) {{

            answer =
                "OA-SENSE NER sends live sensor data from the ESP32 to ThingSpeak. "
                + "The Streamlit dashboard reads the data from the ThingSpeak channel.";
        }}


        // OA-SENSE
        else if (
            question.includes("oa-sense") ||
            question.includes("osteoarthritis") ||
            question.includes("project")
        ) {{

            answer =
                "OA-SENSE NER is an AI-assisted prototype designed to screen "
                + "movement and sensor-based risk markers related to knee osteoarthritis. "
                + "It combines sensor data, cloud monitoring and software analysis.";
        }}


        // HELP
        else if (
            question.includes("help") ||
            question.includes("what can you do") ||
            question.includes("commands")
        ) {{

            answer =
                "You can ask me about the current knee angle, sensor readings, "
                + "screening status, hardware, ThingSpeak data, or the OA-SENSE NER project.";
        }}


        // DEFAULT
        else {{

            answer =
                "I can help with OA-SENSE NER information such as "
                + "knee angle, FSR sensor readings, screening status, "
                + "hardware, ThingSpeak data and project information.";
        }}


        responseBox.innerHTML = answer;

statusBox.innerHTML =
    "🤖 Response generated. 🔊 Speaking...";

window.speechSynthesis.cancel();

const speech = new SpeechSynthesisUtterance(answer);

speech.lang = languageSelect.value;
speech.rate = 0.9;
speech.pitch = 1.0;

speech.onend = function() {{
    statusBox.innerHTML =
        "✅ Response generated and spoken.";
}};

window.speechSynthesis.speak(speech);
    }};


    // -------------------------------------------------------
    // SPEAK RESPONSE
    // -------------------------------------------------------

    speakBtn.onclick = function() {{

        const text =
            responseBox.innerText;

        if (!text ||
            text === "OA-SENSE NER response will appear here.") {{

            statusBox.innerHTML =
                "🔊 Generate a response first.";

            return;
        }}


        window.speechSynthesis.cancel();

        const speech =
            new SpeechSynthesisUtterance(text);

        speech.lang =
            languageSelect.value;

        speech.rate = 0.9;
        speech.pitch = 1.0;

        speech.onstart = function() {{
            statusBox.innerHTML =
                "🔊 Speaking response...";
        }};

        speech.onend = function() {{
            statusBox.innerHTML =
                "✅ Response finished.";
        }};

        window.speechSynthesis.speak(speech);

    }};


    // -------------------------------------------------------
    // COPY
    // -------------------------------------------------------

    copyBtn.onclick = async function() {{

        const text =
            questionBox.value;

        if (!text) {{

            statusBox.innerHTML =
                "📋 Nothing to copy.";

            return;
        }}

        try {{

            await navigator.clipboard.writeText(text);

            statusBox.innerHTML =
                "✅ Question copied.";

        }} catch(error) {{

            questionBox.select();

            document.execCommand("copy");

            statusBox.innerHTML =
                "✅ Question copied.";
        }}
    }};


    // -------------------------------------------------------
    // CLEAR
    // -------------------------------------------------------

    clearBtn.onclick = function() {{

        questionBox.value = "";

        responseBox.innerHTML =
            "OA-SENSE NER response will appear here.";

        statusBox.innerHTML =
            "🧹 Cleared. Ready for a new question.";

        window.speechSynthesis.cancel();

    }};


    // -------------------------------------------------------
    // CHANGE LANGUAGE
    // -------------------------------------------------------

    languageSelect.onchange = function() {{

        if (recognition) {{
            recognition.lang =
                languageSelect.value;
        }}

        window.speechSynthesis.cancel();

        statusBox.innerHTML =
            "🌐 Language changed.";

    }};

    </script>

    </body>
    </html>
    """


    components.html(
        voice_html,
        height=800,
        scrolling=True
    )


    st.info(
        "💡 Allow microphone permission when your browser asks. "
        "Voice recognition and speech output depend on browser/device support."
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

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Assessment Report",
            data=csv_data,
            file_name="oa_sense_ner_assessment_report.csv",
            mime="text/csv"
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
        "### 🔧 Hardware Configuration"
    )

    st.write(
        "📡 ESP32-WROOM-32"
    )

    st.write(
        "🧭 MPU6500 — I2C Address 0x68"
    )

    st.write(
        "🖥️ OLED — 128 × 64 — I2C Address 0x3C"
    )

    st.write(
        "📍 FSR1 — GPIO34"
    )

    st.write(
        "📍 FSR2 — GPIO35"
    )

    st.write(
        "🔔 Buzzer — GPIO25"
    )

    st.write(
        f"☁️ ThingSpeak Channel ID: "
        f"{THINGSPEAK_CHANNEL_ID}"
    )

    st.write(
        "📊 ThingSpeak Field 1 — Knee Angle"
    )

    st.write(
        "📊 ThingSpeak Field 2 — FSR1"
    )

    st.write(
        "📊 ThingSpeak Field 3 — FSR2"
    )

    st.write(
        "📊 ThingSpeak Field 4 — Total Load"
    )

    st.write(
        "📊 ThingSpeak Field 5 — Status Code"
    )

    st.write(
        "📊 ThingSpeak Field 6 — Buzzer"
    )

    st.write(
        f"⚠️ Prototype warning threshold: "
        f"{HIGH_RISK_ANGLE:.0f}°"
    )

    st.warning(
        "The threshold used in this prototype "
        "is not a clinically validated diagnostic cutoff."
    )
