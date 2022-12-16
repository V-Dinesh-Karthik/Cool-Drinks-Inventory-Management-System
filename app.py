import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from controller.sequel import *
from controller.userController import *
from PIL import Image
import numpy as np
import yolov5 as yl
import torch
import av
import time


# model = torch.hub.load('ultralytics/yolov5','custom','./models/best.pt')
@st.cache
def model():
    return yl.load("./models/model.pt")


model = model()

# rtc configuration !
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def __init__(self) -> None:
        self.res = None
        self.confidence = 0.5

    def getRes(self):
        return self.res

    def recv(self, frame):
        model.conf = self.confidence
        img = frame.to_ndarray(format="bgr24")
        flipped = img[:, ::-1, :]

        im_pil = Image.fromarray(flipped)
        results = model(im_pil, size=112)
        self.res = results
        b_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(b_img, format="bgr24")


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
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

        flag = 0

        if st.checkbox("Store", value=False):
            flag = 1

        if st.checkbox("Show the detected labels", value=True):
            empty = st.empty()
            if ctx.state.playing:
                while True:
                    if ctx.video_processor:
                        results = ctx.video_processor.getRes()
                        if results != None:
                            count = results.pandas().xyxy[0]
                            dj = count["name"].tolist()
                            dj = read_df(dj)

                            empty.table(dj)
                            for idx,row in dj.iterrows():
                                if flag:
                                    Insert(date, row["Name"], int(row["Count"]))
                                    time.sleep(3)
                        else:
                            empty.write("No labels detected")
                    else:
                        break

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

        orig_image, out_image = st.columns(2)

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

            results = model(image)
            # box_img = np.array(results.render()[0])

            results.save(save_dir="./Output")

            counted = results.pandas().xyxy[0]["name"].to_list()
            dd = read_df(counted)

            st.sidebar.subheader("Detected!")
            st.sidebar.table(dd)

            with out_image:
                st.image("./Output/image0.jpg")

            if st.button("Store"):
                for index, row in dd.iterrows():
                    Insert(date=date, text=row["Name"], count=int(row["Count"]))
