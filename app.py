import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
from datetime import datetime
import json

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

# Leave empty if ThingSpeak channel is PUBLIC
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
        "🎤 Voice Assistant",
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
        st.info("📡 ESP32 + MPU6500")

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
# ============================================================

elif page == "📡 Hardware Monitoring":

    st.title("📡 Hardware Monitoring")

    st.write(
        "Live knee-angle data from ESP32 → ThingSpeak"
    )

    st.divider()

    if st.button("🔄 Refresh Data"):

        st.rerun()

    hardware_data, hardware_error = (
        get_hardware_data()
    )

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
# VOICE ASSISTANT
# ============================================================

elif page == "🎤 Voice Assistant":

    st.title("🎤 OA-SENSE Voice Assistant")

    st.write(
        "Ask OA-SENSE NER about the current prototype "
        "hardware reading."
    )

    # --------------------------------------------------------
    # GET CURRENT HARDWARE DATA
    # --------------------------------------------------------

    hardware_data, hardware_error = (
        get_hardware_data()
    )

    if hardware_data is not None:

        current_angle = hardware_data["knee_angle"]

        if current_angle > HIGH_RISK_ANGLE:

            default_response = (
                f"The current knee angle measured by "
                f"OA-SENSE NER is {current_angle:.1f} degrees. "
                f"It is above the prototype warning threshold "
                f"of {HIGH_RISK_ANGLE:.0f} degrees."
            )

        else:

            default_response = (
                f"The current knee angle measured by "
                f"OA-SENSE NER is {current_angle:.1f} degrees. "
                f"It is below the prototype warning threshold "
                f"of {HIGH_RISK_ANGLE:.0f} degrees."
            )

    else:

        current_angle = None

        default_response = (
            "I could not retrieve the current knee-angle "
            "data from ThingSpeak."
        )

    # --------------------------------------------------------
    # LANGUAGE SELECTION
    # --------------------------------------------------------

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
        "🌐 Select Language",
        list(languages.keys())
    )

    language_code = languages[selected_language]

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_area(
        "🎤 Ask OA-SENSE NER",
        placeholder=(
            "Example: What is the current knee angle?"
        ),
        height=100
    )

    # --------------------------------------------------------
    # RESPONSE GENERATION
    # --------------------------------------------------------

    if st.button(
        "🤖 Get OA-SENSE Response",
        use_container_width=True
    ):

        question_lower = question.lower().strip()

        if not question_lower:

            response_text = default_response

        elif (
            "angle" in question_lower
            or "knee" in question_lower
            or "degree" in question_lower
        ):

            if current_angle is not None:

                response_text = (
                    f"The current knee angle measured by "
                    f"OA-SENSE NER is "
                    f"{current_angle:.1f} degrees."
                )

            else:

                response_text = (
                    "The current knee-angle data "
                    "is not available."
                )

        elif (
            "threshold" in question_lower
            or "limit" in question_lower
        ):

            response_text = (
                f"The prototype warning threshold "
                f"is {HIGH_RISK_ANGLE:.0f} degrees. "
                "This is a prototype value and is not "
                "a clinically validated diagnostic cutoff."
            )

        elif (
            "status" in question_lower
            or "condition" in question_lower
        ):

            if current_angle is not None:

                if current_angle > HIGH_RISK_ANGLE:

                    response_text = (
                        f"The current measured knee angle "
                        f"is {current_angle:.1f} degrees, "
                        "which is above the prototype "
                        "warning threshold."
                    )

                else:

                    response_text = (
                        f"The current measured knee angle "
                        f"is {current_angle:.1f} degrees, "
                        "which is below the prototype "
                        "warning threshold."
                    )

            else:

                response_text = (
                    "The current hardware status "
                    "is unavailable."
                )

        elif (
            "hardware" in question_lower
            or "sensor" in question_lower
        ):

            response_text = (
                "OA-SENSE NER uses an ESP32 with an "
                "MPU6500 motion sensor, FSR sensors, "
                "an OLED display and ThingSpeak cloud "
                "monitoring."
            )

        elif (
            "thingspeak" in question_lower
            or "cloud" in question_lower
        ):

            response_text = (
                "The ESP32 sends prototype sensor data "
                "to ThingSpeak, and OA-SENSE NER reads "
                "the latest data from the cloud."
            )

        else:

            response_text = (
                "OA-SENSE NER is an AI-assisted prototype "
                "for monitoring knee movement and risk markers. "
                "Ask me about the current knee angle, "
                "threshold, hardware, or ThingSpeak."
            )

        # Store response in session state
        st.session_state.voice_response = response_text

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    if "voice_response" not in st.session_state:

        st.session_state.voice_response = default_response

    response_text = st.session_state.voice_response

    st.divider()

    st.subheader("🤖 OA-SENSE Response")

    st.info(response_text)

    # --------------------------------------------------------
    # ANDROID NATIVE TTS + BROWSER FALLBACK
    # --------------------------------------------------------

    response_json = json.dumps(response_text)
    language_json = json.dumps(language_code)

    voice_html = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
        }}

        button {{
            width: 100%;
            padding: 14px;
            margin: 6px 0;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
        }}

        #speakButton {{
            background: #1f77b4;
            color: white;
        }}

        #stopButton {{
            background: #777;
            color: white;
        }}

        #status {{
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background: #f2f2f2;
        }}

    </style>

    </head>

    <body>

        <button id="speakButton">
            🔊 Speak Response
        </button>

        <button id="stopButton">
            ⏹ Stop Speaking
        </button>

        <div id="status">
            Ready
        </div>

    <script>

        const responseText = {response_json};
        const languageCode = {language_json};

        const speakButton =
            document.getElementById("speakButton");

        const stopButton =
            document.getElementById("stopButton");

        const statusBox =
            document.getElementById("status");


        // ====================================================
        // SPEAK
        // ====================================================

        speakButton.onclick = function() {{

            statusBox.innerHTML =
                "🔊 Speaking...";

            // -----------------------------------------------
            // ANDROID NATIVE TTS
            // -----------------------------------------------

            try {{

                if (
                    window.AndroidTTS &&
                    AndroidTTS.isReady &&
                    AndroidTTS.isReady()
                ) {{

                    AndroidTTS.speak(
                        responseText,
                        languageCode
                    );

                    statusBox.innerHTML =
                        "🔊 Android TTS is speaking.";

                    return;
                }}

            }} catch (error) {{

                console.log(
                    "Android TTS error:",
                    error
                );

            }}


            // -----------------------------------------------
            // BROWSER FALLBACK
            // -----------------------------------------------

            if (
                "speechSynthesis" in window
            ) {{

                window.speechSynthesis.cancel();

                const speech =
                    new SpeechSynthesisUtterance(
                        responseText
                    );

                speech.lang =
                    languageCode;

                speech.rate = 1.0;

                speech.pitch = 1.0;

                speech.onend = function() {{

                    statusBox.innerHTML =
                        "✅ Speech completed.";

                }};

                speech.onerror = function() {{

                    statusBox.innerHTML =
                        "❌ Browser speech failed.";

                }};

                window.speechSynthesis.speak(
                    speech
                );

            }} else {{

                statusBox.innerHTML =
                    "❌ Text-to-Speech is not available.";

            }}

        }};


        // ====================================================
        // STOP
        // ====================================================

        stopButton.onclick = function() {{

            try {{

                if (
                    window.AndroidTTS &&
                    AndroidTTS.stop
                ) {{

                    AndroidTTS.stop();

                }}

            }} catch (error) {{

                console.log(
                    "Android TTS stop error:",
                    error
                );

            }}


            try {{

                if (
                    "speechSynthesis" in window
                ) {{

                    window.speechSynthesis.cancel();

                }}

            }} catch (error) {{

                console.log(
                    "Browser speech stop error:",
                    error
                );

            }}

            statusBox.innerHTML =
                "⏹ Speech stopped.";

        }};

    </script>

    </body>
    </html>
    """

    components.html(
        voice_html,
        height=230,
        scrolling=False
    )

    st.caption(
        "The Android app uses native Android TTS. "
        "The normal web version uses browser speech synthesis."
    )

    st.warning(
        "OA-SENSE NER is a prototype screening and "
        "monitoring system. Its output is not a medical diagnosis."
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
        "MPU6500 — I2C"
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
