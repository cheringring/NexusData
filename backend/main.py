# -*- coding: utf-8 -*-
"""
NexusData 백엔드 로직
streamlit_app.py (프론트엔드)에서 이 함수들을 호출
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    _PLOTLY_AVAILABLE = False

from .service import HistoryManager, PromptEngine, CodeValidator, CodeExecutor
from .service.prompt_engine import SYSTEM_PROMPT
from .service.code_engine import build_error_guide
from .solution_connector import (
    DataikuManager, DataikuFlowExporter,
    create_llm_client, load_api_key,
)
from .utils.constants import MAX_AUTO_RETRY


def init_session():
    defaults = {
        "messages": [], "df": None, "datasets": {}, "dataset_info": None,
        "datasets_info": {}, "llm_client": None, "selected_dataset": None,
        "chat_id": None, "error_count": 0,
        "history_manager": HistoryManager(), "user_id": HistoryManager.get_user_id(),
        "flow_exporter": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def init_data_manager() -> DataikuManager:
    dm = DataikuManager()
    if st.session_state.flow_exporter is None:
        st.session_state.flow_exporter = DataikuFlowExporter(dm)
    return dm


def load_datasets(data_manager: DataikuManager, selected_datasets: List[str]):
    loaded_datasets = {}
    loaded_infos = {}
    for ds_name in selected_datasets:
        df_loaded = data_manager.load_dataset(ds_name)
        loaded_datasets[ds_name] = df_loaded
        loaded_infos[ds_name] = data_manager.build_dataset_info(df_loaded)
    first_ds = selected_datasets[0]
    st.session_state.df = loaded_datasets[first_ds]
    st.session_state.dataset_info = loaded_infos[first_ds]
    st.session_state.datasets = loaded_datasets
    st.session_state.datasets_info = loaded_infos
    st.session_state.selected_dataset = first_ds
    st.session_state.messages = []


def init_llm_client(provider: str, model: str):
    _api_key = load_api_key(provider)
    if not _api_key:
        return
    _client_key = f"{provider}:{model}"
    if st.session_state.llm_client is None or st.session_state.get("_client_key") != _client_key:
        st.session_state.llm_client = create_llm_client(provider=provider, api_key=_api_key, model=model)
        st.session_state["_client_key"] = _client_key


def run_analysis(prompt: str) -> dict:
    llm_client = st.session_state.llm_client
    df = st.session_state.df
    dataset_info = st.session_state.dataset_info
    datasets = st.session_state.get("datasets", {})
    datasets_info = st.session_state.get("datasets_info", {})

    code = None; fig = None; result_df = None; insight = None; error_msg = None

    try:
        main_prompt = PromptEngine.build_main_prompt(prompt, dataset_info, datasets_info=datasets_info)
        raw_output = llm_client.generate(system_prompt=SYSTEM_PROMPT, user_prompt=main_prompt)

        if "ERROR_COLUMN_NOT_FOUND" in raw_output:
            col_name = raw_output.split("ERROR_COLUMN_NOT_FOUND:")[-1].strip()
            available = ", ".join(dataset_info.get('columns', []))
            error_msg = f"'{col_name}' 컬럼이 현재 데이터셋에 존재하지 않습니다.\n\n사용 가능한 컬럼: {available}"
        else:
            all_columns = list(dataset_info.get('columns', []))
            if datasets_info and len(datasets_info) > 1:
                for ds_info in datasets_info.values():
                    all_columns.extend(ds_info.get('columns', []))
                all_columns = list(set(all_columns))
            code, is_safe, validation_msg = CodeValidator.full_check(raw_output, all_columns, user_request=prompt)
            if not is_safe:
                error_msg = f"검증 실패: {validation_msg}"
            else:
                fig, result_df, exec_error = CodeExecutor.execute(code, df, datasets=datasets)
                if exec_error:
                    for retry in range(1, MAX_AUTO_RETRY + 1):
                        recovery_prompt = PromptEngine.build_error_recovery_prompt(
                            user_request=prompt, failed_code=code, error_message=exec_error, dataset_info=dataset_info)
                        raw_output2 = llm_client.generate(system_prompt=SYSTEM_PROMPT, user_prompt=recovery_prompt)
                        code, is_safe2, _ = CodeValidator.full_check(raw_output2, dataset_info.get('columns', []), user_request=prompt)
                        if not is_safe2:
                            break
                        fig, result_df, exec_error = CodeExecutor.execute(code, df, datasets=datasets)
                        if fig is not None or result_df is not None:
                            break
                    if exec_error and fig is None and result_df is None:
                        error_msg = f"코드 실행 실패 (자동 수정 {MAX_AUTO_RETRY}회 후):\n\n```\n{exec_error}\n```"
    except Exception as e:
        error_msg = f"LLM 호출 오류: {str(e)}"

    assistant_msg: dict = {"role": "assistant"}
    if error_msg:
        assistant_msg["error"] = error_msg
        if code:
            assistant_msg["code"] = code
    else:
        try:
            insight_prompt = PromptEngine.build_insight_prompt(prompt, dataset_info, datasets_info=datasets_info)
            insight = llm_client.generate(
                system_prompt="You are a data analyst. Summarize the analysis result in 1-2 SHORT sentences in Korean. "
                              "Do NOT mention outliers, missing values, correlations, or data quality. Just briefly describe what the result shows.",
                user_prompt=insight_prompt)
        except Exception:
            insight = None
        if fig is not None:
            assistant_msg["text"] = "차트가 생성되었습니다."
        elif result_df is not None:
            assistant_msg["text"] = f"데이터 처리 완료 ({result_df.shape[0]:,}행 × {result_df.shape[1]}열)"
        else:
            assistant_msg["text"] = "처리가 완료되었습니다."
        assistant_msg["fig"] = fig
        assistant_msg["result_df"] = result_df
        assistant_msg["code"] = code
        assistant_msg["insight"] = insight
    return assistant_msg


def auto_publish(prompt: str, assistant_msg: dict):
    fig = assistant_msg.get("fig")
    result_df = assistant_msg.get("result_df")
    if assistant_msg.get("error") or (fig is None and result_df is None):
        return
    try:
        exporter = st.session_state.flow_exporter
        uid = st.session_state.user_id
        ds = st.session_state.selected_dataset or ''
        code = assistant_msg.get("code", "")
        insight = assistant_msg.get("insight", "")
        if fig is not None:
            exporter.publish_chart(user_id=uid, fig=fig, code=code, question=prompt, insight=insight, dataset_name=ds)
        elif result_df is not None and _PLOTLY_AVAILABLE:
            tbl = go.Figure(data=[go.Table(
                header=dict(values=list(result_df.columns), fill_color='#4472C4', font=dict(color='white', size=12), align='center'),
                cells=dict(values=[result_df[c].head(100) for c in result_df.columns], fill_color='#D9E2F3', align='center', font=dict(size=11)))])
            tbl.update_layout(title=f"데이터 처리 결과 ({result_df.shape[0]:,}행)", height=max(350, min(len(result_df), 100) * 30 + 100))
            exporter.publish_chart(user_id=uid, fig=tbl, code=code, question=prompt, dataset_name=ds)
    except Exception as e:
        print(f"[AutoPublish] 에러: {e}")


def save_history():
    if not st.session_state.messages:
        return
    hm = st.session_state.history_manager
    uid = st.session_state.user_id
    if not st.session_state.get("chat_id"):
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages_to_save = [{k: v for k, v in msg.items() if k not in ("fig", "result_df")} for msg in st.session_state.messages]
    ds_names = list(st.session_state.get('datasets', {}).keys())
    ds_label = ', '.join(ds_names) if ds_names else (st.session_state.selected_dataset or "대화")
    first_user_msg = next((m.get('content', '') for m in messages_to_save if m.get('role') == 'user'), '')
    short_msg = first_user_msg[:20] + ('...' if len(first_user_msg) > 20 else '')
    hist_title = f"{ds_label} | {short_msg}" if short_msg else ds_label
    hm.save_history(uid, st.session_state.chat_id, messages_to_save, title=hist_title, dataset_name=st.session_state.selected_dataset or "")


def save_current_chat_before_new():
    if st.session_state.messages:
        if not st.session_state.get("chat_id"):
            st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_history()


def build_quality_chart(ds_name, total_missing, total_outliers, dup_count):
    if not _PLOTLY_AVAILABLE:
        return None
    values = [total_missing, total_outliers, dup_count]
    fig = go.Figure(data=[go.Bar(x=["결측치", "이상치 (IQR)", "중복 행"], y=values,
        marker_color=['#FF6B6B' if v > 0 else '#51CF66' for v in values], text=values, textposition='auto')])
    fig.update_layout(title=f"{ds_name} 데이터 품질 리포트", yaxis_title="건수", height=350)
    return fig


def publish_chart(fig, question, dataset_name):
    return st.session_state.flow_exporter.publish_chart(
        user_id=st.session_state.user_id, fig=fig, code="", question=question, dataset_name=dataset_name)


def publish_recipe(code, label):
    ds_names = list(st.session_state.get("datasets", {}).keys())
    if not ds_names:
        ds_names = [st.session_state.selected_dataset or "dataset"]
    return st.session_state.flow_exporter.publish_recipe(code=code, input_datasets=ds_names, label=label)
