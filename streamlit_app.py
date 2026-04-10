# -*- coding: utf-8 -*-
"""
NexusData | Streamlit 프론트엔드 (UI)
백엔드 로직은 backend/main.py에서 import
"""
import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
try:
    import koreanize_matplotlib
except ImportError:
    pass

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    _PLOTLY_AVAILABLE = False

from NexusData.backend.main import (
    init_session, init_data_manager, load_datasets, init_llm_client,
    run_analysis, auto_publish, save_history, save_current_chat_before_new,
    build_quality_chart, publish_chart, publish_recipe,
)
from NexusData.backend.service import CodeExecutor
from NexusData.backend.service.code_engine import build_error_guide
# ================================================================
# 페이지 설정 + CSS
# ================================================================
st.set_page_config(page_title="NexusData | LLM Dashboard", page_icon="N", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header { background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%); padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; color: white; }
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
.stChatMessage { border-radius: 10px; }
.stCode, .stCode > div, .stCode pre { max-height: none !important; overflow: visible !important; }
.stCode pre, .stCode code { white-space: pre-wrap !important; word-break: break-all !important; overflow-x: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 세션 초기화
# ================================================================
init_session()
if not st.session_state.get("_history_restored"):
    st.session_state["_history_restored"] = True

# ================================================================
# 사이드바
# ================================================================
with st.sidebar:
    st.markdown("# 📊 NexusData")
    solution_list = ["Dataiku", "솔루션1", "솔루션2"]
    st.selectbox("솔루션", solution_list, index=0, key="sol_select")
    st.markdown("---")
    st.markdown("### 대화 히스토리")

    if st.button("✏️ 새 채팅", use_container_width=True):
        save_current_chat_before_new()
        st.session_state.messages = []
        st.session_state.df = None
        st.session_state.dataset_info = None
        st.session_state.selected_dataset = None
        st.session_state.chat_id = None
        st.session_state.prompt_key_counter = st.session_state.get('prompt_key_counter', 0) + 1
        st.rerun()

    history_manager = st.session_state.history_manager
    user_id = st.session_state.user_id

    st.markdown("""
    <style>
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.2rem; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child button {
        padding: 0.25rem 0.4rem; font-size: 0.75rem; min-height: 0; border: none; background: transparent; color: #999; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child button:hover {
        color: #e74c3c; background: rgba(231,76,60,0.08); }
    </style>
    """, unsafe_allow_html=True)

    _del_target = st.session_state.get("_hist_delete_target")
    if _del_target:
        history_manager.delete_history(user_id, _del_target)
        if st.session_state.get("chat_id") == _del_target:
            st.session_state.messages = []
            st.session_state.chat_id = None
        st.session_state._hist_delete_target = None
        st.rerun()

    histories = history_manager.list_user_histories(user_id)
    if histories:
        for idx, hist in enumerate(histories):
            title = hist.get('title', hist.get('dataset', ''))
            chat_id = hist.get('chat_id', '')
            h_col1, h_col2 = st.columns([6, 1])
            with h_col1:
                if st.button(f"{title}", key=f"hist_{idx}_{chat_id}", use_container_width=True):
                    saved = history_manager.load_history(user_id, chat_id)
                    if saved:
                        st.session_state.messages = saved.get("messages", [])
                        st.session_state.selected_dataset = saved.get("dataset", "")
                        st.session_state.chat_id = chat_id
                        st.rerun()
            with h_col2:
                if st.button("✕", key=f"del_{idx}_{chat_id}", help="대화 삭제"):
                    st.session_state._hist_delete_target = chat_id
                    st.rerun()
    else:
        st.caption("저장된 대화가 없습니다.")

    st.markdown("---")

    if st.session_state.messages:
        code_msgs = [(i, m) for i, m in enumerate(st.session_state.messages)
                     if m.get("role") == "assistant" and m.get("code")]
        if code_msgs:
            st.markdown("### 분석 코드 로그")
            for i, (msg_idx, m) in enumerate(code_msgs):
                user_prompt = ""
                if msg_idx > 0:
                    prev = st.session_state.messages[msg_idx - 1]
                    user_prompt = prev.get("content", "")[:30] + "..." if len(prev.get("content", "")) > 30 else prev.get("content", "")
                with st.expander(f"#{i+1} {user_prompt}", expanded=False):
                    st.code(m["code"], language="python")
                    if st.button("재실행", key=f"rerun_{msg_idx}_{i}"):
                        st.session_state.pending_prompt = st.session_state.messages[msg_idx - 1].get("content", "")
                        st.rerun()

# ================================================================
# 메인 영역
# ================================================================
data_manager = init_data_manager()

# ── 데이터 미리보기 + 품질 리포트 + EDA ──
if st.session_state.df is not None:
    loaded_datasets = st.session_state.get("datasets", {})

    def _render_quality_report(ds_name, ds_info):
        """품질 리포트 UI 렌더링"""
        with st.expander("데이터 품질 리포트", expanded=False):
            q_col1, q_col2, q_col3 = st.columns(3)
            missing_detail = ds_info.get("missing_detail", {})
            total_missing = sum(v["count"] for v in missing_detail.values()) if missing_detail else 0
            with q_col1:
                st.metric("결측치", f"{total_missing:,}건", delta="양호" if total_missing == 0 else "주의",
                          delta_color="normal" if total_missing == 0 else "inverse")
                for col, v in missing_detail.items():
                    st.caption(f"`{col}`: {v['count']:,}건 ({v['pct']}%)")
            outlier_detail = ds_info.get("outlier_detail", {})
            total_outliers = sum(outlier_detail.values()) if outlier_detail else 0
            with q_col2:
                st.metric("이상치 (IQR)", f"{total_outliers:,}건", delta="양호" if total_outliers == 0 else "확인 필요",
                          delta_color="normal" if total_outliers == 0 else "inverse")
                for col, cnt in outlier_detail.items():
                    st.caption(f"`{col}`: {cnt:,}건")
            dup_count = ds_info.get("dup_count", 0)
            dup_pct = ds_info.get("dup_pct", 0)
            with q_col3:
                st.metric("중복 행", f"{dup_count:,}건", delta="양호" if dup_count == 0 else f"{dup_pct}%",
                          delta_color="normal" if dup_count == 0 else "inverse")
            if _PLOTLY_AVAILABLE:
                if st.button("📤 품질 리포트 게시", key=f"pub_quality_{ds_name}"):
                    fig = build_quality_chart(ds_name, total_missing, total_outliers, dup_count)
                    if fig:
                        ok, msg = publish_chart(fig, f"{ds_name} 품질 리포트", ds_name)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

    if len(loaded_datasets) > 1:
        st.markdown("### 데이터 미리보기")
        tabs = st.tabs(list(loaded_datasets.keys()))
        for tab, (ds_name, ds_df) in zip(tabs, loaded_datasets.items()):
            with tab:
                st.dataframe(ds_df.head(10), use_container_width=True, height=200)
                ds_info = st.session_state.datasets_info.get(ds_name, {})
                if ds_info:
                    st.caption(f"**컬럼**: {', '.join(ds_info['columns'])}  |  **행**: {ds_info['shape'][0]:,}  |  **열**: {ds_info['shape'][1]}")
                    _render_quality_report(ds_name, ds_info)
    else:
        st.markdown("### 데이터 미리보기")
        st.dataframe(st.session_state.df.head(10), use_container_width=True, height=250)
        info = st.session_state.dataset_info
        if info:
            st.caption(f"**컬럼**: {', '.join(info['columns'])}  |  **행**: {info['shape'][0]:,}  |  **열**: {info['shape'][1]}")
            _render_quality_report(st.session_state.selected_dataset or "dataset", info)

    # ── EDA ──
    with st.expander("EDA", expanded=False):
        _loaded_ds = st.session_state.get("datasets", {})
        _ds_names = list(_loaded_ds.keys()) if _loaded_ds else [st.session_state.selected_dataset or "default"]
        _eda_ds_name = st.selectbox("데이터셋", _ds_names, key="eda_ds_select") if len(_ds_names) > 1 else _ds_names[0]
        _eda_df = _loaded_ds.get(_eda_ds_name, st.session_state.df)
        _eda_info = st.session_state.datasets_info.get(_eda_ds_name, st.session_state.dataset_info)
        _eda_numeric = _eda_info.get("numeric_columns", [])

        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["기술통계", "분포", "상관관계", "이상치"])

        with eda_tab1:
            _desc_df = _eda_df[_eda_numeric].describe().T if _eda_numeric else pd.DataFrame()
            st.dataframe(_desc_df, use_container_width=True)
            cat_cols = _eda_df.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                st.caption(f"범주형 컬럼: {', '.join(cat_cols)}")
                for cc in cat_cols[:5]:
                    vc = _eda_df[cc].value_counts().head(10)
                    st.caption(f"`{cc}` 상위 {len(vc)}개: {', '.join(f'{k}({v})' for k, v in vc.items())}")
            if _eda_numeric and _PLOTLY_AVAILABLE:
                if st.button("📤 기술통계 게시", key=f"pub_desc_{_eda_ds_name}"):
                    _desc_reset = _desc_df.reset_index().rename(columns={"index": "변수"})
                    _fig = go.Figure(data=[go.Table(
                        header=dict(values=list(_desc_reset.columns), fill_color='#4472C4', font=dict(color='white', size=12), align='center', height=30),
                        cells=dict(values=[_desc_reset[c].round(2) if _desc_reset[c].dtype != 'object' else _desc_reset[c] for c in _desc_reset.columns],
                                   fill_color='#D9E2F3', align='center', font=dict(size=11), height=28))])
                    _fig.update_layout(title=dict(text=f"{_eda_ds_name} 기술통계", x=0.5), height=max(350, len(_desc_df) * 40 + 100))
                    ok, msg = publish_chart(_fig, f"{_eda_ds_name} 기술통계", _eda_ds_name)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        with eda_tab2:
            if _eda_numeric and _PLOTLY_AVAILABLE:
                _dist_col = st.selectbox("컬럼 선택", _eda_numeric, key=f"eda_dist_{_eda_ds_name}")
                if _dist_col:
                    _dist_data = _eda_df[[_dist_col]].dropna()
                    if len(_dist_data) > 10000:
                        _dist_data = _dist_data.sample(10000, random_state=42)
                    _fig_dist = px.histogram(_dist_data, x=_dist_col, marginal="box", title=f"{_dist_col} 분포")
                    _fig_dist.update_layout(height=400)
                    st.plotly_chart(_fig_dist, use_container_width=True)
                    if st.button("📤 분포 차트 게시", key=f"pub_dist_{_eda_ds_name}"):
                        ok, msg = publish_chart(_fig_dist, f"[{_eda_ds_name}] {_dist_col} 분포", _eda_ds_name)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
            else:
                st.info("수치형 컬럼이 없거나 Plotly가 설치되지 않았습니다.")

        with eda_tab3:
            if len(_eda_numeric) >= 2 and _PLOTLY_AVAILABLE:
                _corr = _eda_df[_eda_numeric].corr(numeric_only=True)
                _fig_corr = px.imshow(_corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="상관관계 히트맵", aspect="auto")
                _fig_corr.update_layout(height=max(400, len(_eda_numeric) * 40))
                st.plotly_chart(_fig_corr, use_container_width=True)
                _stat_tests = _eda_info.get("stat_tests", {})
                if _stat_tests:
                    st.caption("Pearson 상관관계 (|r| > 0.3)")
                    for pair, vals in _stat_tests.items():
                        st.caption(f"`{pair}`: r={vals['r']}, p={vals['p']} {'✅' if vals['significant'] else '❌'}")
                if st.button("📤 상관관계 히트맵 게시", key=f"pub_corr_{_eda_ds_name}"):
                    ok, msg = publish_chart(_fig_corr, f"[{_eda_ds_name}] 상관관계 히트맵", _eda_ds_name)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("수치형 컬럼이 2개 이상 필요합니다.")

        with eda_tab4:
            _outlier_detail = _eda_info.get("outlier_detail", {})
            if _outlier_detail and _PLOTLY_AVAILABLE:
                _box_cols = list(_outlier_detail.keys())[:8]
                _fig_box = px.box(_eda_df, y=_box_cols, title="이상치 분포 (IQR 기준)")
                _fig_box.update_layout(height=400)
                st.plotly_chart(_fig_box, use_container_width=True)
                for col, cnt in _outlier_detail.items():
                    st.caption(f"`{col}`: {cnt:,}건 ({round(cnt / len(_eda_df) * 100, 2)}%)")
                if st.button("📤 이상치 차트 게시", key=f"pub_outlier_{_eda_ds_name}"):
                    ok, msg = publish_chart(_fig_box, f"[{_eda_ds_name}] 이상치 분포", _eda_ds_name)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("IQR 기준 이상치가 감지되지 않았습니다.")

    st.markdown("---")

# ── 대화 내용 표시 ──
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("fig") is not None:
                fig = msg["fig"]
                if CodeExecutor.is_plotly_figure(fig):
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg_idx}")
                elif CodeExecutor.is_matplotlib_figure(fig):
                    st.pyplot(fig, use_container_width=True, key=f"pyplot_{msg_idx}")
            if msg.get("result_df") is not None:
                _rdf = msg["result_df"]
                st.dataframe(_rdf, use_container_width=True, height=min(400, 35 * len(_rdf) + 38))
                st.caption(f"{_rdf.shape[0]:,}행 × {_rdf.shape[1]}열")
            if msg.get("insight"):
                st.info(f"**인사이트**: {msg['insight']}")

            _has_code = bool(msg.get("code"))
            _has_fig = msg.get("fig") is not None
            _has_result_df = msg.get("result_df") is not None

            if _has_code:
                with st.expander("🔍 코드 보기", expanded=False):
                    st.code(msg["code"], language="python")

            if (_has_fig or _has_result_df) and _has_code:
                if st.button("🔧 Flow 레시피 게시", key=f"pub_recipe_{msg_idx}", use_container_width=True):
                    _user_q = st.session_state.messages[msg_idx - 1].get('content', '') if msg_idx > 0 else ''
                    ok, _pub_msg = publish_recipe(msg.get("code", ""), _user_q)
                    if ok:
                        st.success(_pub_msg)
                    else:
                        st.error(_pub_msg)

            if msg.get("error"):
                st.error(msg["error"])
                retry_guide = build_error_guide(msg["error"], st.session_state.dataset_info)
                st.warning(f"다시 시도하려면:\n{retry_guide}")

st.markdown("---")

# ── 프롬프트 입력 ──
if "prompt_key_counter" not in st.session_state:
    st.session_state.prompt_key_counter = 0

prompt_col1, prompt_col2 = st.columns([5, 1])
with prompt_col1:
    user_input_text = st.text_input("분석 요청", key=f"user_prompt_{st.session_state.prompt_key_counter}",
                                     label_visibility="collapsed", placeholder="요청을 입력하세요.")
with prompt_col2:
    send_button = st.button("전송", use_container_width=True)

user_input = user_input_text if send_button and user_input_text else None
if st.session_state.get("pending_prompt"):
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── LLM 선택 + 데이터셋 선택 ──
col1, col2 = st.columns(2)
with col1:
    st.markdown("**LLM 선택**")
    llm_col1, llm_col2 = st.columns(2)
    with llm_col1:
        llm_provider = st.selectbox("Provider", options=["openai", "groq", "claude"],
            format_func=lambda x: "Groq" if x == "groq" else ("OpenAI" if x == "openai" else "Claude"),
            label_visibility="collapsed")
    with llm_col2:
        model_options = {"openai": ["gpt-4o", "gpt-4o-mini"], "claude": ["claude-3-5-sonnet-20241022"],
                         "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]}
        selected_model = st.selectbox("Model", options=model_options[llm_provider], label_visibility="collapsed")
    try:
        init_llm_client(llm_provider, selected_model)
    except Exception as e:
        st.error(f"LLM 연결 실패: {str(e)}")

with col2:
    st.markdown("**데이터셋 선택**")
    try:
        dataset_list = data_manager.list_datasets()
    except Exception:
        dataset_list = []
    if dataset_list:
        ds_col1, ds_col2 = st.columns([2, 1])
        with ds_col1:
            selected_datasets = st.multiselect("Datasets", options=dataset_list, default=[], key="dataset_selector",
                                                label_visibility="collapsed", placeholder="데이터셋을 선택하세요")
        with ds_col2:
            if st.button("로드", use_container_width=True) and selected_datasets:
                try:
                    load_datasets(data_manager, selected_datasets)
                    st.rerun()
                except Exception as e:
                    st.error(f"로드 실패: {str(e)}")
        if st.session_state.datasets:
            loaded_names = list(st.session_state.datasets.keys())
            st.caption(f"로드됨: {', '.join(loaded_names)} ({len(loaded_names)}개)")
    else:
        st.warning("데이터셋 없음")

# ================================================================
# 사용자 입력 처리
# ================================================================
prompt = user_input
if prompt:
    if st.session_state.df is None:
        st.warning("데이터셋을 선택한 후 로드해주세요.")
        st.stop()
    if st.session_state.llm_client is None:
        st.warning("LLM을 선택해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("코드 생성 및 실행 중..."):
        assistant_msg = run_analysis(prompt)
        st.session_state.messages.append(assistant_msg)
        auto_publish(prompt, assistant_msg)
        save_history()

    st.session_state.prompt_key_counter = st.session_state.get('prompt_key_counter', 0) + 1
    st.rerun()
