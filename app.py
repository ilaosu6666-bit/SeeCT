import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import streamlit as st

st.set_page_config(
    page_title="智影溯源 - AI肺结节教学平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("智影溯源")
page = st.sidebar.radio(
    "选择模块",
    ["AI训练师 - 开发你的AI模型", "CT分析演示"],
)

if page == "AI训练师 - 开发你的AI模型":
    from game import main as game_main
    game_main()
else:
    from part1 import main as ct_main
    ct_main(standalone=False)
