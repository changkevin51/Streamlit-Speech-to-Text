import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client

from streamlit_webrtc import WebRtcMode, webrtc_streamer
from transformers import pipeline

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

@st.cache_data  # type: ignore
def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

def main():
    st.title("Enhanced Real-Time Speech-to-Text and Sentiment Analysis")

    st.sidebar.header("Settings")
    st.sidebar.markdown(
        """
        Use the tools below to customize your experience:
        - **Select mode**: Choose between "Sound only" or "With video"
        - **Live Sentiment Analysis**: Toggle to enable or disable sentiment analysis
        """
    )

    app_mode = st.sidebar.selectbox("Choose the app mode", [
        "Sound only (sendonly)", "With video (sendrecv)"
    ])

    enable_sentiment = st.sidebar.checkbox(
        "Enable Sentiment Analysis", value=True,
        help="Enable real-time sentiment analysis for recognized speech."
    )

    if enable_sentiment:
        sentiment_analyzer = pipeline("sentiment-analysis")

    if app_mode == "Sound only (sendonly)":
        app_sst(enable_sentiment, sentiment_analyzer if enable_sentiment else None)
    elif app_mode == "With video (sendrecv)":
        app_sst_with_video(enable_sentiment, sentiment_analyzer if enable_sentiment else None)

def app_sst(enable_sentiment: bool, sentiment_analyzer):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()
    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading model...")
    text_output = st.empty()
    sentiment_output = st.empty()

    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                from deepspeech import Model

                model_path = "models/deepspeech-0.9.3-models.pbmm"
                lm_path = "models/deepspeech-0.9.3-models.scorer"

                model = Model(model_path)
                model.enableExternalScorer(lm_path)

                stream = model.createStream()

                status_indicator.write("Model loaded. Speak now!")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()

                text_output.markdown(f"**Recognized Text:** {text}")

                if enable_sentiment and text.strip():
                    sentiment = sentiment_analyzer(text)
                    sentiment_output.markdown(
                        f"**Sentiment:** {sentiment[0]['label']} (Score: {sentiment[0]['score']:.2f})"
                    )
        else:
            status_indicator.write("Audio Receiver is not set. Aborting.")
            break

def app_sst_with_video(enable_sentiment: bool, sentiment_analyzer):
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(frames: List[av.AudioFrame]) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading model...")
    text_output = st.empty()
    sentiment_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.state.playing:
            if stream is None:
                from deepspeech import Model

                model_path = "models/deepspeech-0.9.3-models.pbmm"
                lm_path = "models/deepspeech-0.9.3-models.scorer"

                model = Model(model_path)
                model.enableExternalScorer(lm_path)

                stream = model.createStream()

                status_indicator.write("Model loaded. Speak now!")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()

                text_output.markdown(f"**Recognized Text:** {text}")

                if enable_sentiment and text.strip():
                    sentiment = sentiment_analyzer(text)
                    sentiment_output.markdown(
                        f"**Sentiment:** {sentiment[0]['label']} (Score: {sentiment[0]['score']:.2f})"
                    )
        else:
            status_indicator.write("Stopped.")
            break

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()