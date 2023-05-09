import json
import logging
import os
import subprocess
import sys
import time

import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from plotly.graph_objs import Figure
from pydantic import BaseModel
from streamlit_chat import message

from chat2plot import ResponseType, chat2plot
from chat2plot.chat2plot import Chat2Vega

sys.path.append("../../")


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2Plot Demo", page_icon=":robot:", layout="wide")
st.header("Chat2Plot Demo")
st.subheader("Settings")

logger = logging.getLogger("root")
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

api_key = st.text_input("Step1: Input your OpenAI API-KEY", value="")
csv_file = st.file_uploader("Step2: Upload csv file", type={"csv"})

if api_key and csv_file:
    os.environ["OPENAI_API_KEY"] = api_key

    df = pd.read_csv(csv_file)

    st.write(df.head())

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.subheader("Chat")
    model_name = st.selectbox(
        "Model type",
        (
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0314",
        ),
        index=0,
    )

    def initialize_c2p():
        st.session_state["chat"] = chat2plot(
            df, st.session_state["chart_format"], verbose=True
        )

    chart_format = st.selectbox(
        "Chart format",
        ("default", "vega"),
        key="chart_format",
        index=0,
        on_change=initialize_c2p,
    )

    if "chat" not in st.session_state:
        initialize_c2p()

    c2p = st.session_state["chat"]

    c2p.session.set_chatmodel(ChatOpenAI(temperature=0, model_name=model_name))

    def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text

    user_input = get_text()

    def reset_history():
        initialize_c2p()
        st.session_state["generated"] = []
        st.session_state["past"] = []

    if user_input:
        with st.spinner(text="Wait for LLM response..."):
            if isinstance(c2p, Chat2Vega):
                res = c2p(user_input, config_only=True)
            else:
                res = c2p(user_input, config_only=False, show_plot=False)
        response_type = res.response_type

        st.session_state.past.append(user_input)
        st.session_state.generated.append(res)

    st.button("Reset history", on_click=reset_history)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            res = st.session_state["generated"][i]

            if res.response_type == ResponseType.SUCCESS:
                col1, col2 = st.columns([2, 1])

                with col2:
                    config = res.config
                    if isinstance(config, BaseModel):
                        st.code(config.json(indent=2), language="json")
                    else:
                        st.code(json.dumps(config, indent=2), language="json")
                with col1:
                    if isinstance(res.figure, Figure):
                        st.plotly_chart(res.figure, use_container_width=True)
                    else:
                        st.vega_lite_chart(df, res.config, use_container_width=True)
            else:
                if res.response_type == ResponseType.NOT_RELATED:
                    message(
                        "This chat accepts queries to visualize the given data. Please provide a question about the data.",
                        key=str(i),
                    )
                else:
                    message(res.raw_response, key=str(i))

            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
