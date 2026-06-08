import streamlit as st

st.set_page_config(
    page_title="seeCT - AI肺结节教学平台",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 响应式 CSS ----------
st.markdown("""
<style>
/* 移动端 (<768px) */
@media (max-width: 768px) {
    .stImage img { max-height: 280px !important; object-fit: contain; }
    .stMetric { font-size: 0.8rem !important; }
    .stButton button { padding: 0.4rem 0.8rem !important; font-size: 0.85rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.75rem !important; padding: 0.3rem 0.5rem !important; }
    [data-testid="stSidebar"] { min-width: 180px !important; max-width: 220px !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }
}
/* 桌面端 (>768px) */
@media (min-width: 769px) {
    .stImage img { max-height: 500px; max-width: 480px; object-fit: contain; }
}
</style>
""", unsafe_allow_html=True)

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
