import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OA-SENSE NER",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

THINGSPEAK_CHANNEL_ID = "3502394"

# Keep empty if your ThingSpeak channel is PUBLIC.
# For a PRIVATE channel, enter the READ API KEY here.
THINGSPEAK_READ_API_KEY = ""

HIGH_RISK_ANGLE = 25.0

THINGSPEAK_LAST_URL = (
    "https://api.thingspeak.com/channels/"
    "3502394/feeds/last.json"
)

THINGSPEAK_FEEDS_URL = (
    "https://api.thingspeak.com/channels/"
    "3502394/feeds.json"
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "live_readings" not in st.session_state:
    st.session_state.live_readings = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        background-color: #ffffff;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        text-align: center;
        background-color: #ffffff;
    }

    .big-number {
        font-size: 36px;
        font-weight: 800;
    }

    .normal-box {
        padding: 18px;
        border-radius: 14px;
        background-color: #eaf7ee;
        border: 1px solid #9bd3aa;
    }

    .warning-box {
        padding: 18px;
        border-radius: 14px;
        background-color: #fff3e0;
        border: 1px solid #f0b45b;
    }

    .danger-box {
        padding: 18px;
        border-radius: 14px;
        background-color: #fdecec;
        border: 1px solid #e09b9b;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# THINGSPEAK - LATEST DATA
# ============================================================

def get_latest_hardware_data():

    try:

        params = {}

        if THINGSPEAK_READ_API_KEY:
            params["api_key"] = THINGSPEAK_READ_API_KEY

        response = requests.get(
            THINGSPEAK_LAST_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        field1 = data.get("field1")

        if field1 is None:
            return None, "No knee-angle data found in ThingSpeak Field 1."

        angle = float(field1)

        return {
            "knee_angle": angle,
            "timestamp": data.get("created_at", ""),
            "entry_id": data.get("entry_id", "")
        }, None

    except requests.exceptions.RequestException as e:

        return None, f"ThingSpeak connection error: {e}"

    except ValueError:

        return None, "Invalid knee-angle value received from ThingSpeak."

    except Exception as e:

        return None, f"Hardware data error: {e}"


# ============================================================
# THINGSPEAK - RECENT DATA
# ============================================================

def get_recent_hardware_data(results=30):

    try:

        params = {
            "results": results
        }

        if THINGSPEAK_READ_API_KEY:
            params["api_key"] = THINGSPEAK_READ_API_KEY

        response = requests.get(
            THINGSPEAK_FEEDS_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        feeds = data.get("feeds", [])

        rows = []

        for feed in feeds:

            if feed.get("field1") is not None:

                try:

                    rows.append(
                        {
                            "Time": feed.get("created_at", ""),
                            "Knee Angle": float(
                                feed.get("field1")
                            ),
                            "Entry": feed.get(
                                "entry_id", ""
                            )
                        }
                    )

                except ValueError:
                    pass

        if not rows:

            return None, "No recent Field 1 data available."

        df = pd.DataFrame(rows)

        return df, None

    except Exception as e:

        return None, f"Unable to read recent data: {e}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🦵 OA-SENSE NER"
)

st.sidebar.caption(
    "AI-Assisted Osteoarthritis Risk Marker Monitoring"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📡 Live Monitoring",
        "👤 Patient Assessment",
        "📋 Patient History",
        "📊 Analytics",
        "🤖 AI Analysis",
        "🩻 X-ray Analysis",
        "📄 Reports",
        "🌎 NER Insights",
        "⚙️ Settings"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Prototype system for research and demonstration. "
    "Outputs are not a clinical diagnosis."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">OA-SENSE NER</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Assisted Early Detection & Monitoring Platform'
        '</div>',
        unsafe_allow_html=True
    )

    # Get latest data
    hardware_data, hardware_error = (
        get_latest_hardware_data()
    )

    if hardware_data:

        angle = hardware_data["knee_angle"]
        timestamp = hardware_data["timestamp"]

        connection_status = "🟢 ONLINE"

    else:

        angle = None
        timestamp = ""
        connection_status = "🔴 OFFLINE"

    # Status cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "ESP32",
            connection_status
        )

    with c2:
        st.metric(
            "ThingSpeak",
            "🟢 CONNECTED"
            if hardware_data
            else "🔴 ERROR"
        )

    with c3:
        if angle is not None:
            st.metric(
                "Knee Angle",
                f"{angle:.1f}°"
            )
        else:
            st.metric(
                "Knee Angle",
                "--"
            )

    with c4:
        st.metric(
            "Prototype Threshold",
            f"{HIGH_RISK_ANGLE:.0f}°"
        )

    st.divider()

    # Main status
    if angle is not None:

        if angle > HIGH_RISK_ANGLE:

            st.error(
                f"🚨 ATTENTION — Knee angle: {angle:.1f}°"
            )

            st.warning(
                f"The reading is above the prototype "
                f"warning threshold of"
            f"{HIGH_RISK_ANGLE:.0f}°."
            )

        else:

            st.success(
                f"🟢 NORMAL PROTOTYPE RANGE — "
                f"Knee angle: {angle:.1f}°"
            )

    else:

        st.error(
            "Hardware data is currently unavailable."
        )

        if hardware_error:
            st.caption(hardware_error)

    st.divider()

    st.subheader("📈 Recent Movement")

    df, error = get_recent_hardware_data(30)

    if df is not None:

        chart_df = df.copy()

        chart_df["Time"] = pd.to_datetime(
            chart_df["Time"],
            errors="coerce"
        )

        chart_df = chart_df.dropna(
            subset=["Time"]
        )

        chart_df = chart_df.set_index("Time")

        st.line_chart(
            chart_df["Knee Angle"],
            height=350
        )

    else:

        st.info(
            "Recent movement data is not available."
        )

    if timestamp:

        st.caption(
            f"Last ThingSpeak update: {timestamp}"
        )


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "📡 Live Monitoring":

    st.title("📡 Live Hardware Monitoring")

    st.caption(
        "ESP32 → MPU6050 → ThingSpeak → OA-SENSE NER"
    )

    if st.button("🔄 Refresh Live Data"):

        st.rerun()

    st.divider()

    hardware_data, hardware_error = (
        get_latest_hardware_data()
    )

    if hardware_data:

        angle = hardware_data["knee_angle"]

        timestamp = hardware_data["timestamp"]

        entry_id = hardware_data["entry_id"]

        # Current values
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "🦵 Current Angle",
                f"{angle:.1f}°"
            )

        with c2:
            st.metric(
                "⚠️ Threshold",
                f"{HIGH_RISK_ANGLE:.0f}°"
            )

        with c3:
            st.metric(
                "☁️ Channel",
                THINGSPEAK_CHANNEL_ID
            )

        with c4:
            st.metric(
                "Entry ID",
                str(entry_id)
            )

        st.divider()

        if angle > HIGH_RISK_ANGLE:

            st.error(
                "🚨 HIGH-ATTENTION STATE"
            )

            st.warning(
                f"Measured angle: {angle:.1f}°"
            )

            st.info(
                "The ESP32 prototype buzzer is configured "
                "to activate above the selected threshold."
            )

        else:

            st.success(
                "🟢 NORMAL PROTOTYPE RANGE"
            )

        st.divider()

        st.subheader("📈 Live Knee-Angle Trend")

        df, error = get_recent_hardware_data(50)

        if df is not None:

            plot_df = df.copy()

            plot_df["Time"] = pd.to_datetime(
                plot_df["Time"],
                errors="coerce"
            )

            plot_df = plot_df.dropna(
                subset=["Time"]
            )

            plot_df = plot_df.set_index(
                "Time"
            )

            st.line_chart(
                plot_df["Knee Angle"],
                height=400
            )

            st.subheader(
                "Recent Sensor Readings"
            )

            display_df = df.copy()

            display_df["Status"] = display_df[
                "Knee Angle"
            ].apply(
                lambda x:
                "Attention"
                if x > HIGH_RISK_ANGLE
                else "Normal"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            angles = df["Knee Angle"]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Average",
                    f"{angles.mean():.1f}°"
                )

            with c2:
                st.metric(
                    "Maximum",
                    f"{angles.max():.1f}°"
                )

            with c3:
                st.metric(
                    "Minimum",
                    f"{angles.min():.1f}°"
                )

        else:

            st.warning(error)

        if timestamp:

            st.caption(
                f"Last update: {timestamp}"
            )

    else:

        st.error(
            "❌ Hardware data not received."
        )

        st.info(
            hardware_error
        )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "👤 Patient Assessment":

    st.title("👤 Patient Assessment")

    st.caption(
        "Create a structured prototype assessment record."
    )

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        with col1:

            patient_id = st.text_input(
                "Patient ID",
                placeholder="OA-001"
            )

            patient_name = st.text_input(
                "Patient Name"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        with col2:

            region = st.selectbox(
                "Region",
                [
                    "Assam",
                    "Arunachal Pradesh",
                    "Manipur",
                    "Meghalaya",
                    "Mizoram",
                    "Nagaland",
                    "Sikkim",
                    "Tripura",
                    "Other"
                ]
            )

            pain_score = st.slider(
                "Reported Pain Score",
                0,
                10,
                0
            )

            mobility = st.selectbox(
                "Mobility Observation",
                [
                    "Normal",
                    "Mild difficulty",
                    "Moderate difficulty",
                    "Severe difficulty"
                ]
            )

        symptoms = st.text_area(
            "Symptoms / Observations"
        )

        submitted = st.form_submit_button(
            "💾 Save Assessment"
        )

    if submitted:

        if not patient_id:

            st.error(
                "Please enter a Patient ID."
            )

        else:

            record = {
                "Patient ID": patient_id,
                "Patient Name": patient_name,
                "Age": age,
                "Region": region,
                "Pain Score": pain_score,
                "Mobility": mobility,
                "Symptoms": symptoms,
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
            }

            st.session_state.history.append(
                record
            )

            st.success(
                "✅ Patient assessment saved."
            )


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.title("📋 Patient History")

    if not st.session_state.history:

        st.info(
            "No patient assessments have been saved yet."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        search = st.text_input(
            "🔎 Search Patient ID"
        )

        if search:

            df = df[
                df["Patient ID"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Movement Analytics")

    df, error = get_recent_hardware_data(100)

    if df is None:

        st.warning(error)

    else:

        angles = df["Knee Angle"]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Average Angle",
                f"{angles.mean():.1f}°"
            )

        with c2:
            st.metric(
                "Maximum Angle",
                f"{angles.max():.1f}°"
            )

        with c3:
            st.metric(
                "Minimum Angle",
                f"{angles.min():.1f}°"
            )

        with c4:

            alerts = (
                angles > HIGH_RISK_ANGLE
            ).sum()

            st.metric(
                "Threshold Crossings",
                int(alerts)
            )

        st.divider()

        st.subheader(
            "📈 Movement Trend"
        )

        plot_df = df.copy()

        plot_df["Time"] = pd.to_datetime(
            plot_df["Time"],
            errors="coerce"
        )

        plot_df = plot_df.dropna(
            subset=["Time"]
        )

        plot_df = plot_df.set_index(
            "Time"
        )

        st.line_chart(
            plot_df["Knee Angle"],
            height=400
        )

        st.divider()

        st.subheader(
            "📊 Statistical Summary"
        )

        st.dataframe(
            angles.describe().to_frame(
                "Knee Angle (°)"
            ),
            use_container_width=True
        )
        # ============================================================
# AI ANALYSIS
# ============================================================

elif page == "🤖 AI Analysis":

    st.title("🤖 AI-Assisted Analysis")

    st.info(
        "This section is designed for the project's "
        "future trained ML model."
    )

    hardware_data, error = (
        get_latest_hardware_data()
    )

    if hardware_data:

        angle = hardware_data["knee_angle"]

        st.subheader(
            "Current Movement Indicator"
        )

        if angle > HIGH_RISK_ANGLE:

            st.warning(
                f"Current sensor reading: {angle:.1f}°"
            )

        else:

            st.success(
                f"Current sensor reading: {angle:.1f}°"
            )

        st.divider()

        st.subheader(
            "Explainable Prototype Indicators"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "### Movement"
            )

            st.write(
                "Current sensor measurement"
            )

            st.write(
                f"**{angle:.1f}°**"
            )

        with col2:

            st.write(
                "### Threshold Status"
            )

            if angle > HIGH_RISK_ANGLE:

                st.write(
                    "⚠️ Above prototype threshold"
                )

            else:

                st.write(
                    "🟢 Within prototype range"
                )

        st.divider()

        st.warning(
            "A trained and validated ML model must be "
            "connected before presenting an AI prediction "
            "as a model-generated result."
        )

    else:

        st.error(error)


# ============================================================
# X-RAY ANALYSIS
# ============================================================

elif page == "🩻 X-ray Analysis":

    st.title("🩻 X-ray Analysis")

    st.write(
        "AI-assisted imaging module"
    )

    uploaded_file = st.file_uploader(
        "Upload X-ray image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded X-ray",
            use_container_width=True
        )

        st.success(
            "Image uploaded successfully."
        )

        st.subheader(
            "Image Quality Check"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write(
                "📷 Image received"
            )

        with c2:
            st.write(
                "✓ Supported format"
            )

        with c3:
            st.write(
                "✓ Ready for preprocessing"
            )

        st.info(
            "Connect your trained X-ray model here "
            "for actual prediction and Grad-CAM analysis."
        )

        st.warning(
            "Do not interpret this upload screen as "
            "a medical diagnosis."
        )

    else:

        st.info(
            "Upload an X-ray image to begin."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "📄 Reports":

    st.title("📄 Assessment Reports")

    hardware_data, error = (
        get_latest_hardware_data()
    )

    if hardware_data:

        angle = hardware_data["knee_angle"]

        report_data = {
            "Parameter": [
                "Patient",
                "Knee Angle",
                "Prototype Threshold",
                "Status",
                "ThingSpeak Channel",
                "Timestamp"
            ],
            "Value": [
                "Current Session",
                f"{angle:.1f}°",
                f"{HIGH_RISK_ANGLE:.0f}°",
                (
                    "Attention"
                    if angle > HIGH_RISK_ANGLE
                    else "Normal"
                ),
                THINGSPEAK_CHANNEL_ID,
                hardware_data["timestamp"]
            ]
        }

        st.subheader(
            "Current Assessment Report"
        )

        st.dataframe(
            pd.DataFrame(report_data),
            use_container_width=True,
            hide_index=True
        )

        if angle > HIGH_RISK_ANGLE:

            st.error(
                "🚨 Prototype attention state"
            )

        else:

            st.success(
                "🟢 Prototype normal state"
            )

    else:

        st.warning(
            "Hardware data is unavailable."
        )

        st.caption(error)

    st.divider()

    st.subheader(
        "Saved Patient Assessments"
    )

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.history
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No patient assessments saved."
        )


# ============================================================
# NER INSIGHTS
# ============================================================

elif page == "🌎 NER Insights":

    st.title(
        "🌎 North Eastern Region Insights"
    )

    st.caption(
        "Regional dashboard for prototype/demo data."
    )

    st.warning(
        "Use only authorized, de-identified or synthetic "
        "data unless appropriate permissions are available."
    )

    regions = [
        "Assam",
        "Arunachal Pradesh",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Sikkim",
        "Tripura"
    ]

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        regional_counts = (
            history_df["Region"]
            .value_counts()
            .reindex(
                regions,
                fill_value=0
            )
        )

        st.bar_chart(
            regional_counts
        )

    else:

        st.info(
            "Add assessment records to display "
            "regional analytics."
        )

    st.divider()

    st.subheader(
        "Target Region"
    )

    cols = st.columns(4)

    for i, region_name in enumerate(regions):

        with cols[i % 4]:

            st.info(
                region_name
            )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ System Settings")

    st.subheader(
        "Hardware Configuration"
    )

    settings_data = {
        "Component": [
            "Controller",
            "Motion Sensor",
            "Display",
            "Buzzer",
            "Cloud Platform"
        ],
        "Configuration": [
            "ESP32-WROOM-32",
            "MPU6050",
            "OLED 128×64",
            "GPIO 25",
            "ThingSpeak"
        ]
    }

    st.dataframe(
        pd.DataFrame(settings_data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "ThingSpeak Configuration"
    )

    st.write(
        f"Channel ID: **{THINGSPEAK_CHANNEL_ID}**"
    )

    st.write(
        "Data source: **Field 1 — Knee Angle**"
    )

    st.divider()

    st.subheader(
        "Prototype Threshold"
    )

    st.write(
        f"Current threshold: **{HIGH_RISK_ANGLE:.0f}°**"
    )

    st.divider()

    st.subheader(
        "System Disclaimer"
    )

    st.info(
        "OA-SENSE NER is a research and prototype "
        "system. Sensor measurements, thresholds and "
        "future AI outputs should not be presented as "
        "a standalone clinical diagnosis."
    )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "OA-SENSE NER 2.0 | Prototype"
)
