import streamlit as st

st.set_page_config(
    page_title="seeCT - AI肺结节教学平台",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("智影溯源")
page = st.sidebar.radio(
    "选择模块",
    ["AI训练师 - 开发你的AI模型", "CT分析演示"],
)

if page == "AI训练师 - 开发你的AI模型":
    try:
        from game import main as game_main
        game_main()
    except ImportError as e:
        st.error(f"游戏模块加载失败 (依赖未安装): {e}")
        st.info("请确保 requirements.txt 中已包含所有依赖。")
    except Exception as e:
        st.error(f"游戏运行出错: {e}")
else:
    try:
        from part1 import main as ct_main
        ct_main(standalone=False)
    except ImportError as e:
        st.error(f"CT分析模块加载失败 (依赖未安装): {e}")
    except Exception as e:
        st.error(f"CT分析运行出错: {e}")
