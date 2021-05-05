import streamlit as st
from PIL import Image
from streamlit_webrtc import ClientSettings
from video_mask import run_mask_page
from detect_obj import run_detect_page

WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
)

# get Streamlit logo for page icon
s_logo = Image.open("./images/streamlit.png")

# set up webapp page configuration
st.set_page_config(page_title="Live Streamlit Detection", page_icon=s_logo)

st.title("Live Detection of the Streamlit Logo")
#
# # make the sidebar
with st.sidebar:
    col1,col2,col3 = st.beta_columns(3)
    col1.image(s_logo, width=50)
    col2.markdown("# & ")
    col3.image(Image.open("./images/roboflow.png"), width=50)

    st.write("""
Welcome to the live Streamlit logo detection app with Roboflow!
     """)

    page = st.selectbox("Choose Page", options=["Video masks", "Object detection"])

if page == "Video masks":
    run_mask_page(page, WEBRTC_CLIENT_SETTINGS)
if page == "Object detection":
    run_detect_page(page, WEBRTC_CLIENT_SETTINGS)
