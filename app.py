import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from sequel import *
from io import BytesIO
from PIL import Image
import numpy as np
import yolov5 as yl


# st.set_page_config(page_title='Home') #ignore this
# model = torch.hub.load('ultralytics/yolov5','custom','./models/best.pt')
model = yl.load("./models/best.pt")

def convert_cv2(df):  # function to convert dataframe into a csv
    return df.to_csv().encode("utf-8")


def convert_xcel(df):  # fuction to convert a dataframe into an excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer)
    return output

def read_df(data):
    d = {}
    for key in data:
        d[key] = d.get(key,0)+1
    df = pd.DataFrame(d.items(),columns=["Name","Count"])
    return df

    

# rtc configuration !
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("Cold Drinks Inventory Management System")

option = st.selectbox("Mode", ["None", "Staff", "Admin"])

if option == "Staff":
    st.title("Staff")
    with st.sidebar:
        date = st.date_input("Date")  # defaults to today's date

        Threshold = st.slider(
            "Confidence Threshold",
            min_value=0.00,
            max_value=1.00,
            step=0.05,
            value=0.65,
        )

        media = st.radio("View Mode", ("🎥Video", "📊Data", "🖼️Image"))

    if media == "🎥Video":
        st.title("🎥Object Detection Video")
        ctx = webrtc_streamer(
            key="Staff-Cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        if st.checkbox("Store"):
            pass

        if st.checkbox("Show the detected labels"):
            pass

    if media == "📊Data":
        st.title("📊Data")

        d = read()
        st.table(d)

        d2 = read_()
        d2.set_index("", inplace=True)
        st.table(d2)

        download_mode = st.sidebar.radio("Download Mode", ("None", "Excel", "CSV"))

        if download_mode == "Excel":
            excel = convert_xcel(d)
            st.sidebar.download_button(
                label="Download as Excel",
                data=excel,
                file_name="data.xlsx",
                mime="application/vnd.ms-excel",
            )

        if download_mode == "CSV":
            csv = convert_cv2(d)
            st.sidebar.download_button(
                label="Download as CSV", data=csv, file_name="data.csv", mime="text/csv"
            )
        # mime - multipurpose internet mail extensions = formats of files such as audio,images,video and application programs

    if media == "🖼️Image":
        image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        orig_image,out_image = st.columns(2)

        if image is not None:
            image = Image.open(image)
            image = np.array(image)

            with orig_image:
                st.image(image)

            model.conf = Threshold  # probability of an event
            model.iou = 0.45  # Intersection over union = area of overlap/ area of union > 0.5 considered "good"
            # model.agnostic = False
            model.multi_label = False  # stops multiple class detections per box
            model.max_det = 1000  # maximum number of detections per image

            if st.button("Detect"):
                results = model(image)

                results.save(save_dir="./Output/")

                counted = results.pandas().xyxy[0]
                dd = counted["name"].tolist()

                dd = read_df(dd)

                dd.set_index("Name",inplace=True)

                with out_image:
                    st.image("./Output/image0.jpg")
                    
                st.sidebar.subheader("Detected!")
                st.sidebar.table(dd)
