"""
Dataiku LLM-Powered Visualization Dashboard
메인 Streamlit 애플리케이션

실행 방법:
  - Dataiku 웹앱: 이 파일을 webapp의 app.py로 등록
  - 로컬 테스트: streamlit run app.py
"""

import re
import ast
import io
import time
import traceback
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")  # pyplot import 전에 반드시 설정
import matplotlib.pyplot as plt

# 한글 폰트 자동 설정 (koreanize-matplotlib 사용)
plt.rcParams['axes.unicode_minus'] = False

try:
    import koreanize_matplotlib
    _KOREAN_FONT_OK = True
except ImportError:
    _KOREAN_FONT_OK = False
    print("Warning: koreanize-matplotlib not installed. Korean text may not display correctly.")

# plotly 선택적 import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    _PLOTLY_AVAILABLE = False

# seaborn 선택적 import
try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    sns = None
    _SNS_AVAILABLE = False


# ================================================================
# API Keys (Dataiku 웹앱 내부 전용)
# ================================================================

_OPENAI_API_KEY    = ""
_ANTHROPIC_API_KEY = ""
_GROQ_API_KEY      = ""
# ================================================================
# HistoryManager - 사용자별 채팅 히스토리 관리
# ================================================================

class HistoryManager:
    """사용자별 채팅 히스토리를 Managed Folder(Dataiku) 또는 로컬 JSON으로 저장/복원"""

    FOLDER_NAME = "nexusdata_charts"
    HISTORY_PREFIX = "_history"  # Managed Folder 내 경로: _history/{user_id}_{dataset}.json

    def __init__(self, storage_dir: str = ".chat_history"):
        self.storage_dir = storage_dir
        self._in_dataiku = False
        self._folder = None
        try:
            import dataiku
            self._folder = dataiku.Folder(self.FOLDER_NAME)
            self._in_dataiku = True
        except Exception:
            os.makedirs(storage_dir, exist_ok=True)

    @staticmethod
    def get_user_id() -> str:
        """Dataiku 사용자 ID 가져오기 (없으면 'default')"""
        try:
            import dataiku
            client = dataiku.api_client()
            auth_info = client.get_auth_info()
            return auth_info.get("authIdentifier", "default")
        except Exception:
            return "default"

    def _folder_path(self, user_id: str, dataset_name: str) -> str:
        safe_ds_name = re.sub(r'[^\w\-]', '_', dataset_name)
        return f"{self.HISTORY_PREFIX}/{user_id}_{safe_ds_name}.json"

    def get_history_file(self, user_id: str, dataset_name: str) -> str:
        safe_ds_name = re.sub(r'[^\w\-]', '_', dataset_name)
        return os.path.join(self.storage_dir, f"{user_id}_{safe_ds_name}.json")

    def save_history(self, user_id: str, dataset_name: str, messages: List[dict], title: str = None) -> bool:
        """히스토리 저장"""
        try:
            if title is None:
                first_user_msg = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
                short_msg = first_user_msg[:20] + ('...' if len(first_user_msg) > 20 else '')
                title = f"{short_msg}" if short_msg else dataset_name
            data = {
                "user_id": user_id,
                "dataset": dataset_name,
                "title": title,
                "last_updated": datetime.now().isoformat(),
                "messages": messages
            }
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

            if self._in_dataiku and self._folder:
                self._folder.upload_stream(self._folder_path(user_id, dataset_name), io.BytesIO(payload))
            else:
                filepath = self.get_history_file(user_id, dataset_name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(payload.decode("utf-8"))
            return True
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
            return False

    def load_history(self, user_id: str, dataset_name: str) -> List[dict]:
        """히스토리 복원"""
        try:
            if self._in_dataiku and self._folder:
                path = self._folder_path(user_id, dataset_name)
                try:
                    with self._folder.get_download_stream(path) as f:
                        data = json.loads(f.read().decode("utf-8"))
                    return data.get("messages", [])
                except Exception:
                    return []
            else:
                filepath = self.get_history_file(user_id, dataset_name)
                if not os.path.exists(filepath):
                    return []
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("messages", [])
        except Exception as e:
            print(f"히스토리 로드 실패: {e}")
            return []

    def list_user_histories(self, user_id: str) -> List[dict]:
        """사용자의 모든 히스토리 목록"""
        try:
            histories = []
            if self._in_dataiku and self._folder:
                prefix = f"{self.HISTORY_PREFIX}/{user_id}_"
                for path in self._folder.list_paths_in_partition():
                    if path.startswith(prefix) and path.endswith(".json"):
                        try:
                            with self._folder.get_download_stream(path) as f:
                                data = json.loads(f.read().decode("utf-8"))
                            histories.append({
                                "dataset": data.get("dataset"),
                                "title": data.get("title", data.get("dataset", "")),
                                "last_updated": data.get("last_updated"),
                                "message_count": len(data.get("messages", []))
                            })
                        except Exception:
                            continue
            else:
                for filename in os.listdir(self.storage_dir):
                    if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                        filepath = os.path.join(self.storage_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        histories.append({
                            "dataset": data.get("dataset"),
                            "title": data.get("title", data.get("dataset", "")),
                            "last_updated": data.get("last_updated"),
                            "message_count": len(data.get("messages", []))
                        })
            return sorted(histories, key=lambda x: x.get("last_updated", ""), reverse=True)
        except Exception:
            return []

    def delete_history(self, user_id: str, dataset_name: str) -> bool:
        """히스토리 삭제"""
        try:
            if self._in_dataiku and self._folder:
                path = self._folder_path(user_id, dataset_name)
                self._folder.delete_path(path)
            else:
                filepath = self.get_history_file(user_id, dataset_name)
                if os.path.exists(filepath):
                    os.remove(filepath)
            return True
        except Exception:
            return False


# ================================================================
# DataikuManager
# ================================================================

class DataikuManager:
    def __init__(self):
        self._client = None
        self._project = None
        self._in_dataiku = False
        self._try_connect()

    def _try_connect(self):
        try:
            import dataiku
            self._client = dataiku.api_client()
            self._project = self._client.get_default_project()
            self._in_dataiku = True
        except Exception:
            self._in_dataiku = False

    @property
    def is_connected(self) -> bool:
        return self._in_dataiku

    def list_datasets(self) -> List[str]:
        if not self._in_dataiku:
            return self._demo_dataset_names()
        try:
            datasets = self._project.list_datasets()
            return [ds["name"] for ds in datasets]
        except Exception as e:
            raise RuntimeError(f"데이터셋 목록 조회 실패: {str(e)}")

    def load_dataset(self, name: str, limit: int = 0) -> pd.DataFrame:
        if not self._in_dataiku:
            return self._demo_dataframe(name)
        try:
            import dataiku
            dataset = dataiku.Dataset(name)
            if limit > 0:
                return dataset.get_dataframe(limit=limit)
            return dataset.get_dataframe()
        except Exception as e:
            raise RuntimeError(f"데이터셋 '{name}' 로드 실패: {str(e)}")

    def get_schema(self, name: str) -> List[Dict]:
        if not self._in_dataiku:
            return []
        try:
            import dataiku
            dataset = dataiku.Dataset(name)
            return dataset.read_schema()
        except Exception:
            return []

    def build_dataset_info(self, df: pd.DataFrame) -> Dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        describe_str = df[numeric_cols].describe().to_string() if numeric_cols else "수치형 컬럼 없음"
        date_cols = self._detect_date_columns(df)
        cat_info = {}
        for col in df.select_dtypes(include="object").columns:
            cat_info[col] = df[col].dropna().unique()[:10].tolist()
        # 결측치 정보 (비율 포함)
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            missing_pct = (missing / len(df) * 100).round(1)
            missing_str = "\n".join(f"  {col}: {cnt}건 ({missing_pct[col]}%)" for col, cnt in missing.items())
        else:
            missing_str = "결측치 없음"
        # 중복도 정보
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / len(df) * 100, 1) if len(df) > 0 else 0
        # 상관관계 (숫자형 2개 이상)
        corr_str = "상관계수 정보 없음"
        stat_tests_str = "통계 검정 없음"
        stat_tests = {}
        if len(numeric_cols) >= 2:
            corr_str = df[numeric_cols].corr(numeric_only=True).round(3).to_string()
            # Pearson 상관관계 + p-value 자동 계산
            try:
                from scipy import stats as _scipy_stats
                sig_pairs = []
                for i, c1 in enumerate(numeric_cols[:8]):  # 최대 8개 컬럼만 체크 (성능)
                    for c2 in numeric_cols[i+1:8]:
                        valid = df[[c1, c2]].dropna()
                        if len(valid) > 10:
                            r, p = _scipy_stats.pearsonr(valid[c1], valid[c2])
                            if abs(r) > 0.3:  # 상관계수 0.3 이상만 보고
                                sig = "유의미 ✅" if p < 0.05 else "유의미하지 않음"
                                sig_pairs.append(f"  {c1}↔{c2}: r={r:.3f}, p={p:.4f} ({sig})")
                                stat_tests[f"{c1}↔{c2}"] = {"r": round(r, 3), "p": round(p, 4), "significant": p < 0.05}
                stat_tests_str = "\n".join(sig_pairs) if sig_pairs else "강한 상관관계 없음 (|r|>0.3 기준)"
            except Exception:
                stat_tests_str = "통계 검정 계산 불가"
        # 이상치 (IQR 방식)
        outlier_info = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            cnt = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if cnt > 0:
                outlier_info[col] = cnt
        outlier_str = ", ".join(f"{c}: {n}건" for c, n in outlier_info.items()) if outlier_info else "이상치 없음"
        return {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample_str": df.head(5).to_string(index=False),
            "describe_str": describe_str,
            "date_columns": date_cols,
            "categorical_samples": cat_info,
            "numeric_columns": numeric_cols,
            "missing_str": missing_str,
            "missing_detail": {col: {"count": int(cnt), "pct": round(cnt / len(df) * 100, 1)} for col, cnt in missing.items()} if not missing.empty else {},
            "corr_str": corr_str,
            "outlier_str": outlier_str,
            "outlier_detail": outlier_info,
            "dup_count": dup_count,
            "dup_pct": dup_pct,
            "stat_tests_str": stat_tests_str,
            "stat_tests": stat_tests,
        }

    @staticmethod
    def _detect_date_columns(df: pd.DataFrame) -> List[str]:
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif any(kw in col.lower() for kw in ["date", "time", "year", "month", "day"]):
                date_cols.append(col)
        return list(set(date_cols))

    @staticmethod
    def _demo_dataset_names() -> List[str]:
        return ["demo_sales", "demo_customers", "demo_products"]

    @staticmethod
    def _demo_dataframe(name: str) -> pd.DataFrame:
        np.random.seed(42)
        n = 200
        if name == "demo_customers":
            return pd.DataFrame({
                "customer_id": range(1, n + 1),
                "age": np.random.randint(18, 70, n),
                "gender": np.random.choice(["Male", "Female"], n),
                "region": np.random.choice(["Seoul", "Busan", "Daegu", "Incheon"], n),
                "total_purchase": np.random.uniform(10000, 500000, n).round(0),
            })
        if name == "demo_products":
            return pd.DataFrame({
                "product": [f"Product_{i}" for i in range(1, 21)],
                "category": np.random.choice(["Electronics", "Clothing", "Food", "Sports"], 20),
                "price": np.random.uniform(5000, 200000, 20).round(0),
                "stock": np.random.randint(0, 500, 20),
                "rating": np.random.uniform(1, 5, 20).round(1),
            })
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "date": dates,
            "sales": np.random.randint(100000, 1000000, n),
            "profit": np.random.randint(10000, 300000, n),
            "region": np.random.choice(["Seoul", "Busan", "Daegu", "Incheon"], n),
            "category": np.random.choice(["Electronics", "Clothing", "Food", "Sports"], n),
            "units": np.random.randint(1, 200, n),
        })


# ================================================================
# DataikuFlowExporter - 분석 결과를 Dataiku Flow 자산으로 내보내기
# ================================================================

class DataikuFlowExporter:
    """
    Streamlit 웹앱의 비정형 분석 결과를 Dataiku Flow 자산으로 게시.

    Flow 구조:
    └── 📁 Managed Folder: "nexusdata_charts"
          └── {user_id}/{insight_id}.json  (질의/코드/인사이트 메타)
          + Dataiku Static Insight → 대시보드 타일 자동 추가
    """

    MANAGED_FOLDER_NAME = "nexusdata_charts"

    def __init__(self, data_manager: DataikuManager):
        self._dm = data_manager
        self._in_dataiku = data_manager.is_connected
        if self._in_dataiku:
            self._ensure_assets_exist()

    def _ensure_assets_exist(self):
        """Managed Folder + Dataset 자동 생성"""
        try:
            import dataiku
            client = dataiku.api_client()
            project = client.get_default_project()

            # ── Managed Folder 자동 생성 ──
            existing_folders = [f["name"] for f in project.list_managed_folders()]
            if self.MANAGED_FOLDER_NAME not in existing_folders:
                project.create_managed_folder(self.MANAGED_FOLDER_NAME)
                print(f"[FlowExporter] Managed Folder '{self.MANAGED_FOLDER_NAME}' 생성 완료")

        except Exception as e:
            print(f"[FlowExporter] 자산 자동 생성 중 오류 (무시됨): {e}")

    def publish_chart(self, user_id: str, fig, code: str, question: str,
                      insight: str = "", dataset_name: str = "",
                      chart_type: str = "plotly") -> Tuple[bool, str]:
        """LLM 생성 차트를 Dataiku Static Insight로 게시 → 대시보드에서 바로 확인 가능"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        insight_id = f"nexusdata_{user_id}_{ts}"

        # 차트 제목 추출 (fig.layout.title)
        chart_title = ""
        try:
            if hasattr(fig, 'layout') and hasattr(fig.layout, 'title'):
                t = fig.layout.title
                if isinstance(t, str):
                    chart_title = t
                elif hasattr(t, 'text') and t.text:
                    chart_title = t.text
        except Exception:
            pass
        # 라벨: 데이터셋명 + 차트 제목 (또는 질문 요약), 중복 방지
        _ds_tag = f"[{dataset_name}]" if dataset_name else ""
        _title_or_q = chart_title if chart_title else (question[:60] if question else "분석 결과")
        # 차트 제목에 이미 데이터셋명이 포함되어 있으면 태그 생략
        if dataset_name and dataset_name in _title_or_q:
            label = _title_or_q
        elif _ds_tag:
            label = f"{_ds_tag} {_title_or_q}"
        else:
            label = _title_or_q

        if self._in_dataiku:
            try:
                import dataiku
                import dataiku.insights
                client = dataiku.api_client()
                project = client.get_default_project()

                # ── 차트 레이아웃 최적화 (대시보드 타일용) ──
                try:
                    if _PLOTLY_AVAILABLE and hasattr(fig, 'update_layout'):
                        # 차트 종류별 높이 동적 조절
                        chart_h = 400
                        if hasattr(fig, 'data') and fig.data:
                            trace_type = type(fig.data[0]).__name__.lower()
                            if 'heatmap' in trace_type:
                                n_vars = len(fig.data[0].z) if hasattr(fig.data[0], 'z') and fig.data[0].z is not None else 8
                                chart_h = max(400, n_vars * 60 + 100)
                        fig.update_layout(
                            height=chart_h,
                            margin=dict(t=60, b=80, l=80, r=30),
                            title=dict(font=dict(size=14)),
                        )
                        # 서브플롯 제목 겹침 방지
                        if hasattr(fig.layout, 'annotations'):
                            for ann in fig.layout.annotations:
                                if hasattr(ann, 'text') and ann.text and len(ann.text) > 20:
                                    ann.text = ann.text[:20] + '...'
                                if hasattr(ann, 'font'):
                                    ann.font.size = 10
                except Exception:
                    pass

                # ── 1) Static Insight 게시 ──
                if _PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                    # Plotly figure → save_plotly (HTML 자동 변환)
                    dataiku.insights.save_plotly(insight_id, fig, label=label)
                elif hasattr(fig, 'savefig'):
                    # Matplotlib figure → save_figure
                    dataiku.insights.save_figure(insight_id, fig, label=label)
                else:
                    return False, "지원하지 않는 차트 타입"

                # ── 2) Managed Folder에 메타 JSON 저장 ──
                try:
                    folder = dataiku.Folder(self.MANAGED_FOLDER_NAME)

                    meta = {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset_name,
                        "question": question,
                        "code": code,
                        "insight": insight,
                        "chart_type": chart_type,
                        "insight_id": insight_id,
                        "label": label,
                    }
                    folder.upload_stream(
                        f"{user_id}/{insight_id}.json",
                        io.BytesIO(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
                    )
                except Exception as e:
                    print(f"[FlowExporter] Managed Folder 저장 실패 (무시): {e}")

                # ── 3) 대시보드에 Insight 타일 자동 추가 ──
                try:
                    self._add_insight_to_dashboard(project, insight_id, label)
                except Exception as e:
                    print(f"[FlowExporter] 대시보드 타일 추가 실패: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

                return True, f"✅ 게시 완료: {label}"
            except Exception as e:
                return False, f"대시보드 게시 실패: {str(e)}"
        else:
            # 로컬 폴백
            local_dir = os.path.join(".nexusdata_charts", user_id)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{insight_id}.html")
            if _PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(fig.to_html(include_plotlyjs="cdn", full_html=True))
            return True, f"로컬 저장 완료: {local_path}"

    DASHBOARD_NAME = "NexusData Dashboard"

    def _add_insight_to_dashboard(self, project, insight_id: str, label: str):
        """NexusData 대시보드에 Insight 타일 자동 추가 (없으면 대시보드 생성)"""
        import dataiku
        client = dataiku.api_client()
        project_key = project.project_key

        # 대시보드 찾기
        dashboard = None
        for d in project.list_dashboards():
            if "nexusdata" in d.name.lower():
                dashboard = d.to_dashboard()
                break

        # 대시보드 없으면 생성
        if dashboard is None:
            print(f"[FlowExporter] 대시보드 '{self.DASHBOARD_NAME}' 생성 시도...")
            dashboard = project.create_dashboard(self.DASHBOARD_NAME)
            print(f"[FlowExporter] 대시보드 생성 완료: {dashboard.get_settings().id}")

        settings = dashboard.get_settings()
        raw = settings.get_raw()
        pages = raw.get("pages", [])

        # 첫 번째 페이지가 없으면 추가
        if not pages:
            pages.append({"title": "분석 차트", "grid": {"tiles": []}})
            raw["pages"] = pages

        page = pages[0]
        if "grid" not in page:
            page["grid"] = {"tiles": []}
        grid = page["grid"]
        if "tiles" not in grid:
            grid["tiles"] = []
        tiles = grid["tiles"]

        # 이미 같은 insight가 있으면 스킵
        for t in tiles:
            if t.get("insightId") == insight_id:
                print(f"[FlowExporter] 이미 대시보드에 존재: {insight_id}")
                return

        # 타일 위치 계산 (12열 그리드 기준, 아래로 쌓기)
        max_row = 0
        for t in tiles:
            box = t.get("box", {})
            bottom = box.get("top", 0) + box.get("height", 4)
            if bottom > max_row:
                max_row = bottom

        new_tile = {
            "insightId": insight_id,
            "insightType": "static_file",
            "tileType": "INSIGHT",
            "title": label,
            "autoTitle": False,
            "box": {
                "left": 0,
                "top": max_row,
                "width": 12,
                "height": 15,
            },
            "tileParams": {},
            "clickAction": "DO_NOTHING",
            "resizeImageMode": "FIT_SIZE",
            "displayMode": "INSIGHT",
        }
        tiles.append(new_tile)
        settings.save()
        print(f"[FlowExporter] 대시보드 타일 추가 완료: {label}")

    RECIPE_NAME = "compute_nexusdata_flow"

    def publish_recipe(self, code: str, input_datasets: List[str],
                       label: str = "",
                       connection: str = "filesystem_managed") -> Tuple[bool, str]:
        """LLM 생성 코드를 Dataiku Flow의 단일 Python 레시피에 누적 게시.

        구조:
        EL_Sensor ──┐
                    ├── [nexusdata_flow 레시피] ──┬── nexus_left_join
        EL_Vibration┘                            ├── nexus_merge
                                                 └── nexus_filter_top5

        - 레시피가 없으면 새로 생성
        - 레시피가 있으면 기존 코드 뒤에 새 코드 블록을 append + 출력 데이터셋 추가
        """
        if not self._in_dataiku:
            return False, "Dataiku 연결이 필요합니다 (로컬 모드에서는 사용 불가)"

        try:
            import dataiku
            client = dataiku.api_client()
            project = client.get_default_project()

            # 출력 데이터셋 이름 생성
            slug = self._make_slug(label) if label else datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"nexus_{slug}"
            display_label = label[:50] if label else out_name

            # 이름 중복 방지
            existing_ds = [d["name"] for d in project.list_datasets()]
            if out_name in existing_ds:
                out_name = f"{out_name}_{datetime.now().strftime('%H%M%S')}"

            # 이 블록의 코드 변환
            block_code = self._make_code_block(code, input_datasets, out_name, display_label)

            existing_recipes = [r["name"] for r in project.list_recipes()]

            if self.RECIPE_NAME in existing_recipes:
                # ── 기존 레시피에 코드 append + 출력 추가 ──
                recipe = project.get_recipe(self.RECIPE_NAME)
                settings = recipe.get_settings()

                # 기존 코드 뒤에 새 블록 추가
                old_code = settings.get_code()
                new_code = old_code + "\n\n" + block_code
                settings.set_code(new_code)

                # 출력 데이터셋 추가 (기존 출력은 유지)
                raw = settings.get_raw()
                outputs = raw.get("outputs", {})
                main_outputs = outputs.get("main", {}).get("items", [])
                main_outputs.append({"ref": out_name, "appendMode": False})
                outputs["main"] = {"items": main_outputs}
                raw["outputs"] = outputs

                # 출력 데이터셋 생성 (managed dataset)
                try:
                    project.create_dataset(out_name, type="Filesystem",
                                           params={"connection": connection,
                                                   "path": f"/{project.project_key}/{out_name}"})
                except Exception:
                    pass  # 이미 존재하면 무시

                settings.save()
                return True, f"✅ 레시피에 추가: {display_label} → {out_name}"
            else:
                # ── 새 레시피 생성 ──
                header = self._make_header(input_datasets)
                full_code = header + "\n\n" + block_code

                builder = project.new_recipe("python")
                for ds_name in input_datasets:
                    builder.with_input(ds_name)
                builder.with_new_output_dataset(out_name, connection)
                builder.with_script(full_code)
                recipe = builder.create()
                return True, f"✅ Flow 레시피 생성: {display_label} → {out_name}"
        except Exception as e:
            return False, f"레시피 게시 실패: {str(e)}"

    @staticmethod
    def _make_slug(text: str) -> str:
        """프롬프트에서 Flow 이름용 slug 생성"""
        keyword_map = {
            "left join": "left_join", "inner join": "inner_join",
            "right join": "right_join", "outer join": "outer_join",
            "병합": "merge", "조인": "join", "필터": "filter",
            "필터링": "filter", "그룹바이": "groupby", "그룹": "group",
            "피벗": "pivot", "정렬": "sort", "결측": "missing",
            "이상치": "outlier", "상관": "corr", "평균": "mean",
            "중앙값": "median", "분포": "dist", "추출": "extract",
            "상위": "top", "하위": "bottom", "구간": "range",
        }
        slug = text.lower()
        for kr, en in keyword_map.items():
            if kr in slug:
                slug = en
                break
        else:
            slug = re.sub(r'[^a-zA-Z0-9_\s]', '', slug)
            slug = '_'.join(slug.split()[:4])
        slug = re.sub(r'[^a-zA-Z0-9_]', '', slug)
        return slug[:30] if slug else datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _make_header(input_datasets: List[str]) -> str:
        """레시피 최초 생성 시 공통 헤더 (import + 데이터셋 로드)"""
        lines = [
            "import dataiku",
            "import pandas as pd",
            "import numpy as np",
            "",
        ]
        for ds_name in input_datasets:
            safe = re.sub(r'[^\w]', '_', ds_name)
            lines.append(f"{safe} = dataiku.Dataset('{ds_name}').get_dataframe()")
        if input_datasets:
            first_safe = re.sub(r'[^\w]', '_', input_datasets[0])
            lines.append(f"df = {first_safe}")
        return "\n".join(lines)

    @staticmethod
    def _make_code_block(code: str, input_datasets: List[str],
                         output_name: str, label: str) -> str:
        """하나의 게시 요청을 독립 코드 블록으로 변환"""
        lines = [
            f"# {'=' * 50}",
            f"# {label}",
            f"# {'=' * 50}",
        ]

        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import dataiku") or stripped.startswith("import pandas") or stripped.startswith("import numpy"):
                continue
            if any(kw in stripped for kw in ["fig =", "fig.", "px.", "go.", "plt.", "st.", "make_subplots"]):
                continue
            if "plotly" in stripped or "matplotlib" in stripped or "seaborn" in stripped:
                continue
            lines.append(line)

        lines.append("")
        lines.append(f"# 결과 저장 → {output_name}")
        lines.append(f"_out_{output_name} = dataiku.Dataset('{output_name}')")
        lines.append(f"if 'result_df' in dir() and result_df is not None:")
        lines.append(f"    _out_{output_name}.write_with_schema(result_df)")
        lines.append(f"elif 'merged_df' in dir() and merged_df is not None:")
        lines.append(f"    _out_{output_name}.write_with_schema(merged_df)")
        lines.append(f"else:")
        lines.append(f"    _out_{output_name}.write_with_schema(df)")

        return "\n".join(lines)



# ================================================================
# PromptEngine
# ================================================================

SYSTEM_PROMPT = """You are an expert Python data visualization engineer specializing in sensor and industrial data.
Your ONLY job is to generate Python code that creates charts using the provided DataFrame.

STRICT RULES:
1. The DataFrame is already loaded as variable 'df' — do NOT load any files.
2. You MUST use the EXACT column names provided in "Dataset Information" section below.
3. DO NOT make up column names. Use ONLY the actual columns from the dataset.
4. **CRITICAL**: If the user requests a column that does NOT exist in the dataset, do NOT guess or substitute.
   Instead, return ONLY this message (no code): ERROR_COLUMN_NOT_FOUND: [requested_column_name]
5. **MULTI-DATASET**: Multiple datasets may be available as separate variables.
   - 'df' is the primary (first loaded) dataset.
   - Other datasets are available by their name (e.g., EL_Sensor, EL_Vibration).
   - Also accessible via datasets dict: datasets['EL_Sensor'], datasets['EL_Vibration']
   - To merge: merged_df = pd.merge(EL_Sensor, EL_Vibration, on='ID', how='inner')
   - If only one dataset is loaded, 'datasets' dict is empty — use 'df' only.
   - Check "Available Datasets" section in the prompt for loaded dataset names and columns.
6. Allowed libraries: pandas, numpy, matplotlib, matplotlib.pyplot, plotly.express, plotly.graph_objects, plotly.subplots, seaborn, scipy.stats, sklearn (sklearn.linear_model, sklearn.preprocessing, sklearn.metrics, sklearn.model_selection 등)
7. Store the final figure in a variable named 'fig'.
   - plotly (DEFAULT): fig = px.scatter(...) or fig = go.Figure(...)
   - matplotlib (only if user explicitly requests): fig, ax = plt.subplots(...)
   - **DATA PROCESSING REQUESTS** (병합, 조인, 필터링, 피벗 등 차트 없이 데이터 결과만 필요한 경우):
     결과 DataFrame을 'result_df' 변수에 저장하세요. fig는 생성하지 않아도 됩니다.
     예: result_df = pd.merge(EL_Sensor, EL_Vibration, on='ID', how='left')
     예: result_df = df[df['vibration'] > 20].reset_index(drop=True)
     사용자가 "병합해줘", "조인해줘", "필터링해줘", "정렬해줘", "피벗해줘", "그룹바이해줘" 등
     데이터 처리만 요청하고 시각화를 명시하지 않으면 result_df만 저장하세요.
     단, 사용자가 "보여줘", "그려줘", "차트", "그래프", "플롯" 등 시각화를 요청하면 fig를 생성하세요.
8. NEVER use: os, sys, subprocess, open(), eval(), exec(), __import__, requests, urllib
9. NEVER call fig.show() or plt.show() — the framework handles rendering automatically.
9. NEVER call fig.show() or plt.show() — the framework handles rendering automatically.
10. Handle NaN/missing values with dropna() or fillna() before plotting.
11. Add meaningful titles, axis labels, and legends in Korean.
12. Return ONLY the Python code block — no explanation, no markdown text outside the code block.
13. Wrap your code inside ```python ... ``` tags.

**DEFAULT LIBRARY: Plotly** — ALL charts must use Plotly (px or go) by default.
This ensures every chart supports hover, zoom, and pan interactions.
Use matplotlib ONLY if the user explicitly says "정적", "static", or "matplotlib".

**PERFORMANCE OPTIMIZATION (CRITICAL for large data >5000 rows)**:
- ALWAYS use go.Scattergl instead of go.Scatter for line/scatter traces (WebGL 가속, 11만건도 즉시 렌더링)
- For px.scatter: ALWAYS add render_mode='webgl'
- For subplots with plotly.subplots.make_subplots: use go.Scattergl, go.Box (NOT go.Scatter)
- For multi-chart requests: use make_subplots to combine into ONE figure (여러 fig 생성 금지)
- NEVER create multiple separate figures — always ONE fig with subplots
- Keep trace count under 10 per figure to avoid rendering lag
- **COMPLEX MULTI-CHART REQUESTS** (히트맵+히스토그램, 상관관계+분포 등):
  - Do NOT combine imshow (heatmap) with histograms in make_subplots — Plotly does not support mixed subplot types well
  - Instead: pick the MOST informative single chart type, or use tabs/annotations
  - For "히트맵 + 히스토그램": just create the heatmap with text_auto=True (it already shows values)
  - For "전체 변수 분석": use correlation heatmap only — it covers relationships comprehensively
  - ❌ FORBIDDEN: make_subplots with specs=[[{"type": "heatmap"}], [{"type": "xy"}]] — this WILL crash
  - ❌ FORBIDDEN: mixing px.imshow() result with make_subplots — incompatible
  - ✅ CORRECT: use ONLY px.imshow(corr_matrix, text_auto=True) for correlation + values in one chart
- **CRITICAL: NEVER use add_vrect/add_vline/add_shape inside a loop** — browser will freeze!
  - ❌ FORBIDDEN: `for id in outlier_ids: fig.add_vrect(...)` — this creates 1000+ shapes
  - ❌ FORBIDDEN: `for start, end in ranges[:30]: fig.add_vrect(...)` — data loss (누락)
  - ✅ CORRECT: Use `fill='tozeroy'` with Scattergl trace to create background highlight
  - ✅ CORRECT: `outlier_fill = np.where(mask, y_max, np.nan)` then `fig.add_trace(go.Scattergl(..., fill='tozeroy'))`
  - See Example 10 for the ONLY correct way to highlight many outlier regions

PLOTLY DEFAULT SETTINGS (apply to ALL charts):
- Always add: fig.update_layout(dragmode='zoom') — enables drag-to-zoom for detailed inspection
- Always add: fig.update_layout(hovermode='x unified') for line charts, 'closest' for scatter/box
- NEVER use deprecated properties: titlefont, tickfont as direct dict. Use title_font, tickfont via update_yaxes/update_xaxes instead.
  - ❌ WRONG: yaxis2=dict(titlefont=dict(color='red'))
  - ✅ CORRECT: fig.update_yaxes(title_font=dict(color='red'), row=1, col=1) or yaxis2=dict(title=dict(font=dict(color='red')))

CHART SELECTION GUIDE:
- Trend / time-series / sequential → px.line() with hover
- Variable relationship → px.scatter() (small data) or px.density_heatmap() (large data >5000)
- Scatter + regression → px.scatter(trendline="ols") for automatic regression line and R²
- Distribution → px.histogram() with marginal="box" or marginal="violin"
- Distribution comparison → px.histogram() with color parameter or px.box() with multiple columns
- Outlier detection → px.box() or px.violin()
- Correlation heatmap → px.imshow() with text_auto=True
- Category comparison → px.bar() (horizontal preferred)
- Do NOT always default to bar charts. Pick the most informative chart type.
- **Statistical report / metric requests** (상관계수, 변화율, 통계 검정 등):
  NEVER use only print(). You MUST create a Plotly figure that visualizes the result.
  - Correlation / change rate → scatter plot with trendline + annotate r value
  - Detection / threshold → line chart highlighting detected regions
  - Comparison of groups → box/violin plot or grouped bar chart
  - Summary metrics → go.Table() or go.Indicator() inside a fig
  - Always combine the statistical calculation WITH a visualization in one fig.

TIME-SERIES / LINE CHART RULES:
- Always add a rolling mean (이동평균) trace using go.Scattergl for trend visibility (WebGL 가속)
- Typical window: 100 for >10k data, 50 for >1k data, 10 for small data
- **For large data (>10k points)**: Skip raw data line, show only rolling mean + confidence band
- **Confidence band (신뢰구간)**: Add ±2σ band using rolling std, fill area with light gray
- **Outlier highlighting**: Mark points outside ±2σ band with red markers (if <1000 outliers)
- Original line (if shown): very thin (width=0.5) + semi-transparent (opacity=0.3, color='lightgray')
- Rolling mean: bold (width=2.5) + solid color (steelblue)
- Use plotly.graph_objects (go.Figure) for multi-trace overlay

SCATTER PLOT RULES:
- For "관계", "산점도", "scatter", "correlation" → ALWAYS use px.scatter() with render_mode='webgl'
- Use px.density_heatmap() ONLY when user explicitly requests "밀도", "density", "히트맵", "heatmap"
- Scatter settings: opacity=0.4, marker size=2, render_mode='webgl' (WebGL 가속으로 대용량 데이터도 빠름)
- NEVER use plt.plot() or ax.plot() for scatter — it connects points with lines
- Always add: dragmode='zoom', hovermode='closest' for interactive exploration
- Always show total count in title: title=f'... (n={len(plot_df):,})'

REGRESSION ANALYSIS:
- px.scatter(trendline="ols") → automatic OLS regression line with R² in hover
- For residual analysis: calculate residuals and plot with px.scatter()
- For multiple regression comparison: use plotly.graph_objects with multiple traces

DISTRIBUTION ANALYSIS:
- Single variable: px.histogram(marginal="box") for distribution + outliers at once
- Two variables: px.histogram(x=col, color=group) for overlay comparison
- px.violin() or px.box() for group-level distribution comparison

PRE-FAILURE / WINDOW ANALYSIS (장애 구간 사전 분석):
- NEVER use Python for loops to extract windows — use vectorized pandas instead
- ✅ CORRECT (fast): use shift/diff/rolling with boolean masks
  ```python
  # 장애 구간 (vibration > 20)
  fault_mask = merged_df['vibration'] > 20.0
  # 장애 직전 50개 구간: shift(1) ~ shift(50)으로 마스크 확장
  pre_fault_mask = pd.concat([fault_mask.shift(-i) for i in range(1, 51)], axis=1).any(axis=1)
  pre_fault_df = merged_df[pre_fault_mask & ~fault_mask]
  normal_df = merged_df[~fault_mask & ~pre_fault_mask]
  ```
- ✅ For comparison: use go.Scattergl for both groups, add mean line with go.Scattergl
"""

FEW_SHOT_EXAMPLES = """
### EXAMPLES (ALL use Plotly for hover/zoom support) ###

Example 1 — Scatter plot (variable relationship):
User: "x1과 x2의 관계를 보여줘"
Dataset columns: x1 (float64), x2 (float64)

```python
import plotly.express as px

plot_df = df[['x1', 'x2']].dropna()

# 점으로 표현 (WebGL 가속으로 대용량 데이터도 빠르게 렌더링)
fig = px.scatter(plot_df, x='x1', y='x2', opacity=0.4,
                 title=f'x1과 x2의 관계 (n={len(plot_df):,})',
                 render_mode='webgl')
fig.update_traces(marker=dict(size=2))
fig.update_layout(xaxis_title='x1', yaxis_title='x2', dragmode='zoom', 
                  hovermode='closest')
```

Example 1-1 — Density heatmap (when user explicitly requests density):
User: "x1과 x2의 밀도를 보여줘"
Dataset columns: x1 (float64), x2 (float64)

```python
import plotly.express as px

plot_df = df[['x1', 'x2']].dropna()

fig = px.density_heatmap(plot_df, x='x1', y='x2', nbinsx=50, nbinsy=50,
                         title=f'x1과 x2의 관계 (밀도, n={len(plot_df):,})',
                         color_continuous_scale='Blues',
                         marginal_x='histogram', marginal_y='histogram')
fig.update_layout(xaxis_title='x1', yaxis_title='x2', dragmode='zoom')
```

Example 2 — Regression analysis (with trendline and R²):
User: "x1과 x2의 회귀분석 결과를 보여줘"
Dataset columns: x1 (float64), x2 (float64)

```python
import plotly.express as px

plot_df = df[['x1', 'x2']].dropna()

# trendline='ols'는 내부적으로 최적화되므로 전체 데이터 사용
fig = px.scatter(plot_df, x='x1', y='x2', trendline='ols', opacity=0.4,
                 title=f'x1과 x2 회귀분석 (n={len(plot_df):,})',
                 render_mode='webgl')
fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
fig.update_layout(xaxis_title='x1', yaxis_title='x2', dragmode='zoom')
```

Example 3 — Line chart with rolling mean + confidence band (time-series / trend):
User: "ID에 따른 x1 추이를 보여줘"
Dataset columns: ID (int64), x1 (float64)

```python
import plotly.graph_objects as go
import numpy as np

plot_df = df[['ID', 'x1']].dropna().sort_values('ID')
window = 100 if len(plot_df) > 10000 else 50 if len(plot_df) > 1000 else 10

# 이동평균 및 표준편차 계산
plot_df['rolling_mean'] = plot_df['x1'].rolling(window=window, min_periods=1).mean()
plot_df['rolling_std'] = plot_df['x1'].rolling(window=window, min_periods=1).std()
plot_df['upper_band'] = plot_df['rolling_mean'] + 2 * plot_df['rolling_std']
plot_df['lower_band'] = plot_df['rolling_mean'] - 2 * plot_df['rolling_std']

# 이상치 탐지 (±2σ 벗어난 점)
plot_df['is_outlier'] = (plot_df['x1'] > plot_df['upper_band']) | (plot_df['x1'] < plot_df['lower_band'])
outliers = plot_df[plot_df['is_outlier']]

fig = go.Figure()

# 신뢰구간 밴드 (±2σ)
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['upper_band'],
                         mode='lines', name='상한 (+2σ)',
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['lower_band'],
                         mode='lines', name='하한 (-2σ)',
                         line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.1)',
                         fill='tonexty', showlegend=False))

# 원본 데이터 (매우 얇게)
if len(plot_df) <= 10000:
    fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['x1'],
                             mode='lines', name='x1 (원본)',
                             line=dict(width=0.5, color='lightgray'), opacity=0.3))

# 이동평균선
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['rolling_mean'],
                         mode='lines', name=f'이동평균 ({window}구간)',
                         line=dict(width=2.5, color='steelblue')))

# 이상치 강조
if len(outliers) > 0 and len(outliers) < 1000:
    fig.add_trace(go.Scattergl(x=outliers['ID'], y=outliers['x1'],
                             mode='markers', name='이상치',
                             marker=dict(size=4, color='red')))

fig.update_layout(title=f'ID에 따른 x1 추이 (n={len(plot_df):,})', 
                  xaxis_title='ID', yaxis_title='x1',
                  hovermode='x unified', dragmode='zoom')
```

Example 4 — Histogram with box (distribution + outlier):
User: "x1 분포를 보여줘"
Dataset columns: x1 (float64)

```python
import plotly.express as px

data = df[['x1']].dropna()

fig = px.histogram(data, x='x1', nbins=50, marginal='box',
                   title='x1 분포', color_discrete_sequence=['steelblue'])
fig.update_layout(xaxis_title='x1', yaxis_title='빈도', dragmode='zoom')
```

Example 5 — Box plot (outlier detection, multiple variables):
User: "센서 변수들의 이상치를 비교해줘"
Dataset columns: x1 (float64), x2 (float64), x3 (float64)

```python
import plotly.express as px

melt_df = df[['x1', 'x2', 'x3']].melt(var_name='변수', value_name='값')
melt_df = melt_df.dropna()

fig = px.box(melt_df, x='변수', y='값', color='변수',
             title='센서 변수별 이상치 분포', points='outliers')
fig.update_layout(yaxis_title='값', dragmode='zoom')
```

Example 6 — Correlation heatmap:
User: "변수 간 상관관계를 히트맵으로 보여줘"
Dataset columns: x1 (float64), x2 (float64), x3 (float64)

```python
import plotly.express as px

corr = df[['x1', 'x2', 'x3']].corr()

fig = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, title='변수 간 상관관계 히트맵',
                labels=dict(color='상관계수'))
```

Example 7 — Distribution comparison (multiple variables overlay):
User: "x1과 x2의 분포를 비교해줘"
Dataset columns: x1 (float64), x2 (float64)

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(x=df['x1'].dropna(), name='x1', opacity=0.6, nbinsx=50))
fig.add_trace(go.Histogram(x=df['x2'].dropna(), name='x2', opacity=0.6, nbinsx=50))
fig.update_layout(barmode='overlay', title='x1과 x2 분포 비교',
                  xaxis_title='값', yaxis_title='빈도', dragmode='zoom')
```

Example 8 — Multi-dataset merge and comparison:
User: "두 데이터셋을 병합해서 vibration이 높은 구간의 x1 분포를 비교해줘"
Available datasets: EL_Sensor (ID, x1, x2, ...), EL_Vibration (ID, vibration, revolutions, ...)

```python
import plotly.express as px

# 두 데이터셋을 ID 기준으로 병합
merged_df = pd.merge(EL_Sensor, EL_Vibration, on='ID', how='inner')

# vibration 상위 10% vs 하위 90% 구분
threshold = merged_df['vibration'].quantile(0.9)
merged_df['vibration_group'] = merged_df['vibration'].apply(
    lambda v: '상위 10% (고진동)' if v >= threshold else '하위 90% (정상)')

fig = px.histogram(merged_df, x='x1', color='vibration_group', 
                   nbins=50, barmode='overlay', opacity=0.6,
                   title='진동 수준별 x1 센서 값 분포 비교',
                   color_discrete_map={'상위 10% (고진동)': 'red', '하위 90% (정상)': 'steelblue'})
fig.update_layout(xaxis_title='x1', yaxis_title='빈도', dragmode='zoom')
```

Example 9 — Multi-chart complex analysis (subplots + Scattergl):
User: "병합 후 vibration ≥ 20인 구간의 x1, revolutions 박스플롯 + vibration-x1 상관관계 산점도"
Available datasets: EL_Sensor (ID, x1, ...), EL_Vibration (ID, vibration, revolutions, ...)

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

merged_df = pd.merge(EL_Sensor, EL_Vibration, on='ID', how='inner')
high_vib = merged_df[merged_df['vibration'] >= 20.0].dropna(subset=['x1', 'revolutions', 'vibration'])

fig = make_subplots(rows=1, cols=3,
                    subplot_titles=['x1 분포 (vibration≥20)', 'revolutions 분포', 'vibration vs x1 상관관계'])

fig.add_trace(go.Box(y=high_vib['x1'], name='x1', marker_color='steelblue'), row=1, col=1)
fig.add_trace(go.Box(y=high_vib['revolutions'], name='revolutions', marker_color='coral'), row=1, col=2)
fig.add_trace(go.Scattergl(x=high_vib['vibration'], y=high_vib['x1'],
                            mode='markers', name='vibration vs x1',
                            marker=dict(size=3, opacity=0.4, color='steelblue')), row=1, col=3)

fig.update_layout(title=f'고진동 구간 분석 (vibration≥20, n={len(high_vib):,})',
                  showlegend=False, dragmode='zoom', height=450)
```

Example 10 — Highlighting outlier regions efficiently (NO vrect loop!):
User: "x4 이상치가 발생한 구간을 x1과 x2 그래프에 빨간색 배경으로 표시해줘"
Dataset columns: ID (int64), x1 (float64), x2 (float64), x4 (float64)

```python
import plotly.graph_objects as go
import numpy as np

plot_df = df[['ID', 'x1', 'x2', 'x4']].dropna().sort_values('ID')

# IQR 기반 이상치 탐지
Q1, Q3 = plot_df['x4'].quantile(0.25), plot_df['x4'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (plot_df['x4'] < Q1 - 1.5 * IQR) | (plot_df['x4'] > Q3 + 1.5 * IQR)
outlier_count = outlier_mask.sum()

# ❌ WRONG (브라우저 멈춤): for id in outlier_ids: fig.add_vrect(...)
# ❌ WRONG (데이터 누락): for start, end in ranges[:30]: fig.add_vrect(...)
# ✅ CORRECT: fill='tozeroy'로 배경 생성 (1063개 이상치 전체 표시, 빠름)

# 이상치 구간의 y축 최대값 (배경 fill용)
y_max = max(plot_df['x1'].max(), plot_df['x2'].max()) * 1.1

# 이상치 위치에만 y_max 값, 정상 위치는 NaN (끊김 처리)
outlier_fill = np.where(outlier_mask, y_max, np.nan)

fig = go.Figure()

# 이상치 배경 (빨간색 fill - 가장 먼저 추가해서 뒤에 배치)
# 이 방식으로 1063개 이상치 전체를 빠르게 표시 (vrect 1063개 대신 trace 1개)
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=outlier_fill,
    mode='lines', fill='tozeroy', name=f'x4 이상치 구간 ({outlier_count}건)',
    fillcolor='rgba(255, 0, 0, 0.15)', line=dict(width=0), showlegend=True))

# x1 변화 그래프
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['x1'],
    mode='lines', name='x1', line=dict(width=1.5, color='steelblue')))

# x2 변화 그래프
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['x2'],
    mode='lines', name='x2', line=dict(width=1.5, color='green')))

fig.update_layout(title=f'x1과 x2 변화 + x4 이상치 구간 ({outlier_count}건, 전체 표시)',
                  xaxis_title='ID', yaxis_title='값',
                  hovermode='x unified', dragmode='zoom')
```

Example 11 — Statistical report with visualization (변화율/상관계수 등 통계 리포트):
User: "humidity 급변 구간에서 vibration 변화율 상관계수를 계산해줘"
Dataset columns: ID (int64), humidity (float64), vibration (float64)

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

plot_df = df[['ID', 'humidity', 'vibration']].dropna().sort_values('ID')
plot_df['humidity_change'] = plot_df['humidity'].pct_change().fillna(0)
plot_df['vibration_change'] = plot_df['vibration'].pct_change().fillna(0)

threshold = 0.20
rapid_mask = plot_df['humidity_change'].abs() > threshold
rapid_df = plot_df[rapid_mask]

fig = make_subplots(rows=2, cols=1, subplot_titles=['습도/진동 추이 (급변 구간 강조)', '급변 구간 변화율 산점도'],
                    vertical_spacing=0.15)

# 상단: 추이 + 급변 구간 강조
fig.add_trace(go.Scattergl(x=plot_df['ID'], y=plot_df['humidity'], mode='lines',
    name='humidity', line=dict(width=1, color='steelblue')), row=1, col=1)
fig.add_trace(go.Scattergl(x=rapid_df['ID'], y=rapid_df['humidity'], mode='markers',
    name='급변 구간', marker=dict(color='red', size=5)), row=1, col=1)

# 하단: 변화율 산점도 + 상관계수 표시
if len(rapid_df) >= 2:
    r, p = pearsonr(rapid_df['humidity_change'], rapid_df['vibration_change'])
    fig.add_trace(go.Scattergl(x=rapid_df['humidity_change'], y=rapid_df['vibration_change'],
        mode='markers', name=f'r={r:.3f}, p={p:.4f}',
        marker=dict(color='orange', size=5, opacity=0.6)), row=2, col=1)
    fig.add_annotation(text=f'변화율 상관계수 r={r:.3f} (p={p:.4f})', xref='paper', yref='paper',
        x=0.5, y=-0.05, showarrow=False, font=dict(size=14))

fig.update_layout(title=f'습도 급변 구간 분석 (급변 {len(rapid_df)}건 / 전체 {len(plot_df):,}건)',
                  hovermode='closest', dragmode='zoom', height=700)
fig.update_xaxes(title_text='ID', row=1, col=1)
fig.update_xaxes(title_text='humidity 변화율', row=2, col=1)
fig.update_yaxes(title_text='vibration 변화율', row=2, col=1)
```

### END OF EXAMPLES ###
"""

TYPE_HINTS = {
    "timeseries": """
Detected TIME-SERIES or SEQUENTIAL data. Use Plotly:
- px.line() for trends over time/ID (hover shows exact values)
- Add rolling average: .rolling(window=50).mean()
- Convert to datetime if needed: pd.to_datetime(df['{col}'])
""",
    "categorical": """
Detected CATEGORICAL data. Use Plotly:
- px.bar() for comparison (horizontal preferred)
- px.pie() for proportions
- Filter top N for readability: .nlargest(10)
""",
    "numerical": """
Detected NUMERICAL/SENSOR data. Use Plotly:
- Distribution → px.histogram(marginal='box') for distribution + outlier at once
- Outlier detection → px.box() or px.violin()
- Variable relationships → px.scatter(trendline='ols') or px.density_heatmap() for large data
- Correlation overview → px.imshow(corr, text_auto='.3f')
- Distribution comparison → go.Histogram overlay
- Do NOT default to bar chart for numerical data.
""",
}


class PromptEngine:

    @staticmethod
    def build_main_prompt(user_request: str, dataset_info: Dict, datasets_info: Optional[Dict] = None) -> str:
        type_hint = PromptEngine._get_type_hint(dataset_info)
        
        # 멀티 데이터셋 정보 생성
        multi_ds_section = ""
        if datasets_info and len(datasets_info) > 1:
            multi_ds_section = "\n📦 AVAILABLE DATASETS (use variable name directly):\n"
            for ds_name, ds_info in datasets_info.items():
                cols = ", ".join(ds_info['columns'])
                multi_ds_section += f"  - {ds_name}: {ds_info['shape'][0]} rows × {ds_info['shape'][1]} cols → columns: [{cols}]\n"
            multi_ds_section += "\n💡 병합 예시: merged_df = pd.merge(EL_Sensor, EL_Vibration, on='ID', how='inner')\n"
            multi_ds_section += "💡 기본 df는 첫 번째 데이터셋입니다. 병합이 필요하면 위 변수명을 직접 사용하세요.\n"
        
        prompt = f"""{FEW_SHOT_EXAMPLES}

{type_hint}

### CURRENT TASK ###

User Request: "{user_request}"
{multi_ds_section}
⚠️ IMPORTANT: You MUST use these EXACT column names from the dataset:

Dataset Information (Primary — df):
- Shape: {dataset_info['shape'][0]} rows × {dataset_info['shape'][1]} columns

📋 AVAILABLE COLUMNS (use these exact names):
{PromptEngine._format_columns(dataset_info)}

- Detected Date Columns: {dataset_info.get('date_columns', [])}
- Numeric Columns: {dataset_info.get('numeric_columns', [])}
- Categorical Sample Values:
{PromptEngine._format_cat_samples(dataset_info)}

Sample Data (first 5 rows):
{dataset_info['sample_str']}

Statistical Summary:
{dataset_info['describe_str']}

Missing Values:
{dataset_info.get('missing_str', '결측치 없음')}

Correlation Matrix (numeric columns):
{dataset_info.get('corr_str', '상관계수 정보 없음')}

Outliers (IQR method):
{dataset_info.get('outlier_str', '이상치 없음')}

Duplicate Rows: {dataset_info.get('dup_count', 0)}건 ({dataset_info.get('dup_pct', 0)}%)

⚠️ REMINDER: Use ONLY the column names listed above. Do NOT invent column names.
Choose the most suitable chart type for the request. Do NOT default to bar charts.

Generate the Python visualization code now:
"""
        return prompt

    @staticmethod
    def build_error_recovery_prompt(
        user_request: str,
        failed_code: str,
        error_message: str,
        dataset_info: Dict,
    ) -> str:
        return f"""{SYSTEM_PROMPT}

### ERROR RECOVERY TASK ###

The previous code failed. Analyze the error and fix it.

Original User Request: "{user_request}"

Failed Code:
```python
{failed_code}
```

Error Message:
{error_message}

Dataset Columns & Types:
{PromptEngine._format_columns(dataset_info)}

Common fixes to consider:
1. Use exact column names from the schema above
2. Convert date columns: pd.to_datetime(df['col'])
3. Handle missing values: df.dropna(subset=['col'])
4. Ensure 'fig' variable is assigned at the end
5. Check for KeyError — use df.columns to verify names

Generate the CORRECTED Python code:
"""

    # 데이터 품질 알림 임계치 (필요 시 조정 가능)
    QUALITY_THRESHOLDS = {
        'missing_pct': 10,      # 결측치 비율 10% 이상 경고
        'outlier_pct': 2,       # 이상치 비율 2% 이상 경고
        'duplicate_pct': 1,     # 중복 행 비율 1% 이상 경고
    }
    
    @staticmethod
    def build_insight_prompt(user_request: str, dataset_info: Dict, datasets_info: Optional[Dict] = None) -> str:
        # 멀티 데이터셋일 때: 사용자 요청에 언급된 변수가 속한 데이터셋을 찾아 해당 통계 사용
        matched_datasets = {}  # {ds_name: (ds_info, match_count)}

        if datasets_info and len(datasets_info) > 1:
            request_lower = user_request.lower()
            for ds_name, ds_info in datasets_info.items():
                cols = ds_info.get('columns', [])
                match_count = sum(1 for col in cols if col.lower() in request_lower)
                # 데이터셋 이름 자체가 언급된 경우도 매칭
                if ds_name.lower() in request_lower:
                    match_count += 1
                if match_count > 0:
                    matched_datasets[ds_name] = (ds_info, match_count)

        # 매칭된 데이터셋이 여러 개면 모두 포함, 없으면 primary 사용
        if len(matched_datasets) >= 2:
            # 멀티 데이터셋 비교 요청
            all_stats_sections = []
            all_quality_alerts = []
            for ds_name, (ds_info, _) in matched_datasets.items():
                section = f"\n[{ds_name}]\nShape: {ds_info['shape']}\nColumns: {ds_info['columns']}\n"
                section += f"Statistical Summary:\n{ds_info['describe_str']}\n"
                section += f"Missing Values: {ds_info.get('missing_str', '결측치 없음')}\n"
                section += f"Correlation: {ds_info.get('corr_str', '상관계수 정보 없음')}\n"
                section += f"Outliers (IQR): {ds_info.get('outlier_str', '이상치 없음')}\n"
                all_stats_sections.append(section)
                # 품질 알림
                for col, v in ds_info.get('missing_detail', {}).items():
                    if v['pct'] >= PromptEngine.QUALITY_THRESHOLDS['missing_pct']:
                        all_quality_alerts.append(f"[{ds_name}] {col} 결측치 {v['pct']}%")
                total_rows = ds_info['shape'][0]
                for col, cnt in ds_info.get('outlier_detail', {}).items():
                    pct = round(cnt / total_rows * 100, 1) if total_rows > 0 else 0
                    if pct >= PromptEngine.QUALITY_THRESHOLDS['outlier_pct']:
                        all_quality_alerts.append(f"[{ds_name}] {col} 이상치 {cnt}건 ({pct}%)")

            quality_str = "\n".join(all_quality_alerts) if all_quality_alerts else "데이터 품질 양호"
            datasets_section = "\n".join(all_stats_sections)
            ds_names = ", ".join(matched_datasets.keys())

            return f"""You are a sensor data analyst. Provide a concise 2-3 sentence insight.
Use Korean terminology (백분위수, 평균, 중앙값, 표준편차, 상관계수, 이상치).
When statistical significance is available (p-value), ALWAYS mention it explicitly.

CRITICAL: This request involves MULTIPLE datasets: [{ds_names}].
Use statistics from ALL relevant datasets below. Do NOT say data is unavailable when it is provided.

User Request: "{user_request}"

{datasets_section}

Data Quality Alerts:
{quality_str}

Respond in Korean. Use actual numbers from the relevant datasets.
If there are quality alerts, mention their potential impact.
"""
        else:
            # 단일 데이터셋 매칭 또는 매칭 없음
            if matched_datasets:
                active_ds_name, (active_info, _) = max(matched_datasets.items(), key=lambda x: x[1][1])
            else:
                active_info = dataset_info
                active_ds_name = "Primary"

            # 품질 알림 메시지 동적 생성 (active_info 기준)
            quality_alerts = []
            missing_detail = active_info.get('missing_detail', {})
            for col, v in missing_detail.items():
                if v['pct'] >= PromptEngine.QUALITY_THRESHOLDS['missing_pct']:
                    quality_alerts.append(f"{col} 컬럼 결측치 {v['pct']}% — 분석 신뢰도에 영향 가능")
            outlier_detail = active_info.get('outlier_detail', {})
            total_rows = active_info['shape'][0]
            for col, cnt in outlier_detail.items():
                pct = round(cnt / total_rows * 100, 1) if total_rows > 0 else 0
                if pct >= PromptEngine.QUALITY_THRESHOLDS['outlier_pct']:
                    quality_alerts.append(f"{col} 컬럼 이상치 {cnt}건 ({pct}%) — 분포 왜곡 가능")
            dup_pct = active_info.get('dup_pct', 0)
            if dup_pct >= PromptEngine.QUALITY_THRESHOLDS['duplicate_pct']:
                quality_alerts.append(f"중복 행 {active_info.get('dup_count', 0)}건 ({dup_pct}%) — 밀도/빈도 왜곡 가능")
            quality_str = "\n".join(quality_alerts) if quality_alerts else "데이터 품질 양호"

            stat_tests_str = active_info.get('stat_tests_str', '통계 검정 없음')
            return f"""You are a sensor data analyst. Provide a concise 2-3 sentence insight.
Focus on: trends, anomalies, correlations, missing data patterns, and outliers.
Use Korean terminology (백분위수, 평균, 중앙값, 표준편차, 상관계수, 이상치).
When statistical significance is available (p-value), ALWAYS mention it explicitly.

CRITICAL: Base your insight ONLY on variables mentioned in the user request.
Do NOT mention variables from other datasets that are unrelated to the request.
The relevant dataset for this request is: [{active_ds_name}]

User Request: "{user_request}"
Dataset: {active_ds_name}
Shape: {active_info['shape']}
Columns: {active_info['columns']}

Statistical Summary:
{active_info['describe_str']}

Missing Values: {active_info.get('missing_str', '결측치 없음')}
Correlation: {active_info.get('corr_str', '상관계수 정보 없음')}
Outliers (IQR): {active_info.get('outlier_str', '이상치 없음')}
Duplicate Rows: {active_info.get('dup_count', 0)}건 ({active_info.get('dup_pct', 0)}%)

Statistical Significance Tests (Pearson, |r|>0.3 기준):
{stat_tests_str}

Data Quality Alerts:
{quality_str}

Respond in Korean. Be specific with actual numbers from the [{active_ds_name}] dataset only.
If there are quality alerts above, mention them and explain their potential impact on the analysis.
If statistical tests show p<0.05, explicitly state "통계적으로 유의미" in the insight.
"""

    @staticmethod
    def build_recommendation_prompt(user_request: str, dataset_info: Dict, datasets_info: Optional[Dict] = None) -> str:
        # 멀티 데이터셋일 때 사용자 요청에 맞는 데이터셋 선택
        active_info = dataset_info
        if datasets_info and len(datasets_info) > 1:
            request_lower = user_request.lower()
            best_match_count = 0
            for ds_name, ds_info in datasets_info.items():
                cols = ds_info.get('columns', [])
                match_count = sum(1 for col in cols if col.lower() in request_lower)
                if match_count > best_match_count:
                    best_match_count = match_count
                    active_info = ds_info

        return f"""Based on the user's current visualization request and dataset, suggest 3 follow-up analysis questions.
The questions should help the user explore the data more deeply.

Rules:
- Each question must be a natural Korean sentence that can be directly used as a prompt.
- Questions should suggest DIFFERENT chart types (scatter, heatmap, boxplot, line, histogram, etc.)
- Focus on: correlations, outliers, trends, distributions, comparisons.
- Return ONLY 3 lines, one question per line. No numbering, no bullets, no explanation.

User's current request: "{user_request}"
Dataset columns: {active_info['columns']}
Numeric columns: {active_info.get('numeric_columns', [])}
Correlation: {active_info.get('corr_str', 'N/A')}
Outliers: {active_info.get('outlier_str', 'N/A')}
"""

    @staticmethod
    def _format_columns(dataset_info: Dict) -> str:
        return "\n".join(f"  - {col}: {dtype}" for col, dtype in dataset_info["dtypes"].items())

    @staticmethod
    def _format_cat_samples(dataset_info: Dict) -> str:
        samples = dataset_info.get("categorical_samples", {})
        if not samples:
            return "  (없음)"
        return "\n".join(f"  - {col}: {vals}" for col, vals in samples.items())

    @staticmethod
    def _get_type_hint(dataset_info: Dict) -> str:
        hints = []
        if dataset_info.get("date_columns"):
            col = dataset_info["date_columns"][0]
            hints.append(TYPE_HINTS["timeseries"].format(col=col))
        if dataset_info.get("numeric_columns"):
            hints.append(TYPE_HINTS["numerical"])
        if dataset_info.get("categorical_samples"):
            hints.append(TYPE_HINTS["categorical"])
        return "\n".join(hints) if hints else ""


# ================================================================
# CodeValidator
# ================================================================

ALLOWED_MODULES = {
    "pandas", "pd", "numpy", "np", "matplotlib", "mpl",
    "seaborn", "sns", "plotly", "scipy", "statsmodels", "math", "datetime",
    "collections", "itertools", "functools", "string", "re",
    "sklearn",
}

FORBIDDEN_PATTERNS = [
    (r"\bimport\s+os\b",         "os 모듈 사용 금지"),
    (r"\bimport\s+sys\b",        "sys 모듈 사용 금지"),
    (r"\bimport\s+subprocess\b", "subprocess 모듈 사용 금지"),
    (r"\bimport\s+socket\b",     "socket 모듈 사용 금지"),
    (r"\bimport\s+shutil\b",     "shutil 모듈 사용 금지"),
    (r"\bimport\s+requests\b",   "requests 모듈 사용 금지"),
    (r"\bimport\s+urllib\b",     "urllib 모듈 사용 금지"),
    (r"\bimport\s+http\b",       "http 모듈 사용 금지"),
    (r"\bopen\s*\(",             "파일 open() 사용 금지"),
    (r"\beval\s*\(",             "eval() 사용 금지"),
    (r"\bexec\s*\(",             "exec() 사용 금지"),
    (r"__import__\s*\(",         "__import__() 사용 금지"),
    (r"__builtins__",            "__builtins__ 접근 금지"),
    (r"\bgetattr\s*\(",          "getattr() 사용 금지"),
    (r"\bsetattr\s*\(",          "setattr() 사용 금지"),
    (r"\bdelattr\s*\(",          "delattr() 사용 금지"),
    (r"globals\s*\(\s*\)",       "globals() 사용 금지"),
    (r"locals\s*\(\s*\)",        "locals() 사용 금지"),
]


def _check_fig_or_result_assignment(tree: ast.AST) -> bool:
    """fig 또는 result_df 변수 할당 여부 확인"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("fig", "result_df"):
                    return True
                if isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id in ("fig", "result_df"):
                            return True
    return False


class CodeValidator:
    @staticmethod
    def extract_code_block(text: str) -> str:
        match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match2 = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        return text.strip()

    @staticmethod
    def validate(code: str, available_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"보안 위반: {reason}"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"문법 오류: {str(e)}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_MODULES:
                        return False, f"허용되지 않은 모듈: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_MODULES:
                    return False, f"허용되지 않은 모듈: {node.module}"
        
        # 컬럼명 검증 (df['col'] 읽기만 체크, df['new_col'] = ... 할당은 제외)
        if available_columns:
            used_columns = set()
            assigned_columns = set()
            
            # 할당되는 컬럼 찾기 (어떤 DataFrame 변수든 ['new_col'] = ... 형태 모두 탐지)
            _DF_VARS = {'df', 'plot_df', 'merged_df', 'filtered_df', 'data',
                        'high_vib', 'melt_df', 'temp_df', 'result_df', 'group_df'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript):
                            if isinstance(target.value, ast.Name) and target.value.id in _DF_VARS:
                                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                    assigned_columns.add(target.slice.value)
            
            # 사용되는 컬럼 찾기 (df['col'] 구문만 검증 — df.attr은 메서드/컬럼 구분 불가능하므로 제외)
            for node in ast.walk(tree):
                # df['col'] 또는 df[['col1', 'col2']] 형태
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name) and node.value.id in ('df', 'plot_df', 'merged_df', 'data'):
                        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                            used_columns.add(node.slice.value)
                        elif isinstance(node.slice, ast.List):
                            for elt in node.slice.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    used_columns.add(elt.value)
            
            # 할당된 컬럼은 검증에서 제외
            used_columns = used_columns - assigned_columns
            invalid_columns = used_columns - set(available_columns)
            if invalid_columns:
                return False, f"존재하지 않는 컬럼: {', '.join(invalid_columns)}\n\n사용 가능한 컬럼: {', '.join(available_columns)}"
        
        if not _check_fig_or_result_assignment(tree):
            return True, "WARNING: 'fig' 또는 'result_df' 변수가 코드에 없습니다."
        return True, None

    @staticmethod
    def fix_scatter_code(code: str, user_request: str) -> str:
        """산점도 요청인데 plot()이 쓰인 경우 scatter()로 교정 (백업 안전장치)"""
        scatter_keywords = ["산점도", "scatter", "관계", "상관관계", "correlation"]
        if not any(kw in user_request.lower() for kw in scatter_keywords):
            return code
        if "scatter(" in code:
            return code
        # ax.plot() → ax.scatter(), plt.plot() → plt.scatter()
        code = code.replace("ax.plot(", "ax.scatter(")
        code = code.replace("plt.plot(", "plt.scatter(")
        # scatter에 불필요한 line 파라미터 정리
        code = re.sub(r",?\s*linewidth\s*=\s*[\d.]+", "", code)
        code = re.sub(r",?\s*linestyle\s*=\s*['\"][^'\"]*['\"]", "", code)
        code = re.sub(r",?\s*marker\s*=\s*['\"][^'\"]*['\"]", "", code)
        return code

    @staticmethod
    def fix_deprecated_plotly(code: str) -> str:
        """Plotly deprecated 속성을 최신 문법으로 자동 치환"""
        # titlefont=dict(...) → title=dict(font=dict(...))는 복잡하므로
        # 단순히 titlefont → title_font 으로 치환 (update_layout/yaxis dict 내부)
        code = re.sub(r'\btitlefont\b', 'title_font', code)
        return code

    @staticmethod
    def fix_mixed_subplots(code: str) -> str:
        """imshow + histogram 혼합 시 히트맵만 남기도록 변환, .show() 제거"""
        # .show() 호출 제거 (Streamlit에서는 불필요하고 타임아웃 유발)
        code = re.sub(r'\bfig\w*\.show\(\)', '', code)

        has_imshow = 'px.imshow' in code or 'ff.create_annotated_heatmap' in code
        has_histogram = 'px.histogram' in code or 'go.Histogram' in code
        # 두 개의 fig 변수 생성 감지 (fig = ... 와 fig_xxx = ... 패턴)
        multi_fig = len(re.findall(r'\b\w*fig\w*\s*=\s*(?:px\.|go\.)', code)) >= 2

        if has_imshow and has_histogram and (multi_fig or 'make_subplots' in code):
            return """import plotly.express as px
import pandas as pd

numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

fig = px.imshow(corr_matrix, text_auto='.3f', color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, title='전체 변수 간 상관관계 히트맵',
                labels=dict(color='상관계수'))
fig.update_layout(dragmode='zoom')
"""
        return code

    @staticmethod
    def fix_interval_serialization(code: str) -> str:
        """pd.cut/pd.qcut 결과가 Plotly JSON 직렬화 실패하는 문제 방지 — .astype(str) 자동 추가"""
        if 'pd.cut' in code or 'pd.qcut' in code:
            # pd.cut(...) 또는 pd.qcut(...) 결과를 할당하는 라인 뒤에 .astype(str) 추가
            code = re.sub(
                r"((\w+)\[(['\"])([\w가-힣]+)\3\]\s*=\s*pd\.(?:cut|qcut)\([^)]+\))",
                r"\1\n\2[\3\4\3] = \2[\3\4\3].astype(str)",
                code
            )
        return code

    @staticmethod
    def full_check(raw_llm_output: str, available_columns: Optional[List[str]] = None, user_request: str = "") -> Tuple[str, bool, Optional[str]]:
        code = CodeValidator.extract_code_block(raw_llm_output)
        if user_request:
            code = CodeValidator.fix_scatter_code(code, user_request)
        code = CodeValidator.fix_deprecated_plotly(code)
        code = CodeValidator.fix_mixed_subplots(code)
        code = CodeValidator.fix_interval_serialization(code)
        is_safe, msg = CodeValidator.validate(code, available_columns)
        return code, is_safe, msg


# ================================================================
# CodeExecutor
# ================================================================

class CodeExecutor:
    # 실행 결과 캐시: {hash(code + df_shape): (fig, result_df, error)}
    _result_cache: Dict[str, Tuple[Optional[Any], Optional[Any], Optional[str]]] = {}

    @staticmethod
    def _make_cache_key(code: str, df: pd.DataFrame) -> str:
        import hashlib
        content = f"{code}|{df.shape}|{list(df.columns)}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def execute(code: str, df: pd.DataFrame, datasets: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[str]]:
        cache_key = CodeExecutor._make_cache_key(code, df)
        if cache_key in CodeExecutor._result_cache:
            return CodeExecutor._result_cache[cache_key]
        plt.close("all")
        namespace: dict = {
            "df": df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "matplotlib": matplotlib,
        }
        # 멀티 데이터셋 주입: datasets['EL_Sensor'], datasets['EL_Vibration'] 등
        if datasets:
            namespace["datasets"] = datasets
            for ds_name, ds_df in datasets.items():
                # 변수명에 사용할 수 없는 문자를 _로 치환
                safe_name = re.sub(r'[^\w]', '_', ds_name)
                namespace[safe_name] = ds_df
        if _PLOTLY_AVAILABLE:
            namespace["px"] = px
            namespace["go"] = go
            try:
                from plotly.subplots import make_subplots
                namespace["make_subplots"] = make_subplots
            except ImportError:
                pass
        if _SNS_AVAILABLE:
            namespace["sns"] = sns
        try:
            import scipy.stats
            namespace["scipy"] = __import__("scipy")
        except ImportError:
            pass
        try:
            import sklearn
            from sklearn import linear_model, preprocessing, metrics, model_selection, ensemble, svm
            namespace["sklearn"] = sklearn
            namespace["linear_model"] = linear_model
            namespace["preprocessing"] = preprocessing
            namespace["metrics"] = metrics
            namespace["model_selection"] = model_selection
            namespace["ensemble"] = ensemble
            namespace["svm"] = svm
        except ImportError:
            pass
        try:
            import threading
            exec_error_holder = [None]
            
            def _run_code():
                try:
                    exec(code, namespace)  # noqa: S102
                except Exception as e:
                    exec_error_holder[0] = e
            
            thread = threading.Thread(target=_run_code)
            thread.start()
            thread.join(timeout=60)  # 60초 타임아웃 (멀티 데이터셋 병합 + 복합 분석 고려)
            
            if thread.is_alive():
                return None, None, "코드 실행 시간 초과 (60초). 데이터가 너무 크거나 복잡한 연산입니다.\n\n💡 팁: Python for 루프 대신 pandas 벡터 연산(isin, shift, rolling 등)을 사용하면 빨라집니다."
            
            if exec_error_holder[0] is not None:
                raise exec_error_holder[0]
            
            fig = namespace.get("fig")
            result_df = namespace.get("result_df")
            if fig is None:
                current_fig = plt.gcf()
                if current_fig.get_axes():
                    fig = current_fig
            if fig is None and result_df is None:
                return None, None, "'fig' 또는 'result_df' 변수가 생성되지 않았습니다."
            
            # Plotly figure의 Interval 객체를 문자열로 변환 (pd.cut 결과 직렬화 오류 방지)
            if _PLOTLY_AVAILABLE and hasattr(fig, 'data'):
                for trace in fig.data:
                    for attr in ('x', 'y', 'text'):
                        vals = getattr(trace, attr, None)
                        if vals is not None:
                            try:
                                converted = [str(v) if hasattr(v, 'left') else v for v in vals]
                                setattr(trace, attr, converted)
                            except (TypeError, AttributeError):
                                pass

            # Plotly figure의 shape/annotation 개수 제한 (브라우저 렌더링 멈춤 방지)
            if _PLOTLY_AVAILABLE and hasattr(fig, 'layout'):
                n_shapes = len(fig.layout.shapes) if fig.layout.shapes else 0
                n_annotations = len(fig.layout.annotations) if fig.layout.annotations else 0
                if n_shapes > 50:
                    return None, None, (
                        f"⚠️ Shape이 {n_shapes}개 생성되어 브라우저가 멈출 수 있습니다.\n\n"
                        "💡 이상치 구간 표시는 add_vrect 반복 대신, "
                        "Scattergl 마커 트레이스로 표시하세요.\n"
                        "예: outlier 위치에만 빨간 점을 찍는 방식"
                    )
                if n_annotations > 50:
                    return None, None, f"⚠️ Annotation이 {n_annotations}개 생성되어 브라우저가 멈출 수 있습니다. 개수를 줄여주세요."
            
            result = fig, result_df, None
            CodeExecutor._result_cache[cache_key] = result
            return result
        except KeyError as e:
            result = None, None, f"컬럼을 찾을 수 없습니다: {e}\n사용 가능한 컬럼: {df.columns.tolist()}"
            CodeExecutor._result_cache[cache_key] = result
            return result
        except ValueError as e:
            result = None, None, f"값 오류: {str(e)}"
            CodeExecutor._result_cache[cache_key] = result
            return result
        except TypeError as e:
            result = None, None, f"타입 오류: {str(e)}"
            CodeExecutor._result_cache[cache_key] = result
            return result
        except AttributeError as e:
            result = None, None, f"속성 오류: {str(e)}\n(객체에 해당 메서드/속성이 없습니다)"
            CodeExecutor._result_cache[cache_key] = result
            return result
        except NameError as e:
            result = None, None, f"변수명 오류: {str(e)}\n(정의되지 않은 변수를 사용했습니다)"
            CodeExecutor._result_cache[cache_key] = result
            return result
        except Exception as e:
            result = None, None, f"실행 오류 ({type(e).__name__}): {str(e)}\n\n{traceback.format_exc()}"
            CodeExecutor._result_cache[cache_key] = result
            return result

    @staticmethod
    def is_plotly_figure(fig: Any) -> bool:
        if not _PLOTLY_AVAILABLE:
            return False
        try:
            return isinstance(fig, go.Figure)
        except Exception:
            return False

    @staticmethod
    def is_matplotlib_figure(fig: Any) -> bool:
        return isinstance(fig, matplotlib.figure.Figure)


# ================================================================
# Error Retry Guide
# ================================================================

def _build_error_guide(error_msg: str, dataset_info: Optional[Dict] = None) -> str:
    """에러 메시지를 분석해서 사용자에게 재시도 가이드를 생성"""
    guide_lines = []

    if "컬럼을 찾을 수 없습니다" in error_msg or "존재하지 않는 컬럼" in error_msg:
        cols = ", ".join(dataset_info.get("columns", [])) if dataset_info else ""
        guide_lines.append("사용 가능한 컬럼명을 확인하고 정확한 이름으로 다시 요청해보세요.")
        if cols:
            guide_lines.append(f"현재 컬럼: {cols}")

    elif "시간 초과" in error_msg or "timeout" in error_msg.lower():
        guide_lines.append("데이터가 너무 크거나 연산이 복잡합니다.")
        guide_lines.append("'상위 1000개만', '샘플링해서' 같은 조건을 추가해보세요.")

    elif "문법 오류" in error_msg or "SyntaxError" in error_msg:
        guide_lines.append("LLM이 잘못된 코드를 생성했습니다. 요청을 더 구체적으로 바꿔보세요.")
        guide_lines.append("예: '산점도로 x1과 x2의 관계를 보여줘' 처럼 차트 유형을 명시하면 정확도가 올라갑니다.")

    elif "보안 위반" in error_msg or "허용되지 않은 모듈" in error_msg:
        guide_lines.append("허용되지 않은 기능이 포함된 코드가 생성되었습니다.")
        guide_lines.append("시각화 관련 요청으로 다시 시도해보세요.")

    elif "fig" in error_msg and "생성되지 않았습니다" in error_msg:
        guide_lines.append("차트 생성에 실패했습니다. 요청을 더 명확하게 해보세요.")
        guide_lines.append("예: 'x1의 히스토그램을 그려줘', 'ID별 x2 추이를 보여줘'")

    elif "Shape" in error_msg or "Annotation" in error_msg:
        guide_lines.append("차트 요소가 너무 많아 브라우저가 멈출 수 있습니다.")
        guide_lines.append("'상위 10개만', '요약해서' 같은 조건을 추가해보세요.")

    else:
        guide_lines.append("요청을 더 구체적으로 바꿔서 다시 시도해보세요.")
        guide_lines.append("차트 유형(산점도, 히스토그램, 박스플롯 등)과 컬럼명을 명시하면 성공률이 높아집니다.")

    return "\n".join(guide_lines)


# ================================================================
# LLM Client
# ================================================================

class BaseLLMClient(ABC):
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    @abstractmethod
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_call(_self, system_prompt: str, user_prompt: str) -> str:
        """캐싱된 LLM 호출 - 동일 프롬프트 재사용 시 API 비용 절감 (1시간 TTL)"""
        return _self._call_api(system_prompt, user_prompt)
    
    def generate(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return self._cached_call(system_prompt, user_prompt)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(self.RETRY_DELAY * attempt)
        raise RuntimeError(f"LLM API 호출 실패 ({retries}회 재시도 후): {str(last_error)}")


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=2000,
        )
        return response.choices[0].message.content


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.15,
        )
        return response.content[0].text


class GroqClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq 패키지가 필요합니다: pip install groq")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=2000,
        )
        return response.choices[0].message.content


def create_llm_client(provider: str, api_key: str, model: Optional[str] = None) -> BaseLLMClient:
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(api_key=api_key, **( {"model": model} if model else {}))
    if provider in ("claude", "anthropic"):
        return ClaudeClient(api_key=api_key, **( {"model": model} if model else {}))
    if provider == "groq":
        return GroqClient(api_key=api_key, **( {"model": model} if model else {}))
    raise ValueError(f"지원하지 않는 LLM provider: '{provider}'. 지원 목록: 'openai', 'claude', 'groq'")


def _load_api_key(provider: str) -> Optional[str]:
    """
    API Key 로드 우선순위: 하드코딩 → Dataiku Secrets → 환경변수 → None
    """
    # 0순위: 하드코딩된 키
    hardcoded = {
        "openai": _OPENAI_API_KEY,
        "claude": _ANTHROPIC_API_KEY,
        "groq": _GROQ_API_KEY,
    }
    key = hardcoded.get(provider, "")
    if key and not key.startswith("여기에") and not key.endswith("REMOVED") and len(key) > 10:
        return key

    key_names = {
        "openai": ["OPENAI_API_KEY", "openai_api_key"],
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "anthropic_api_key"],
        "groq": ["GROQ_API_KEY", "groq_api_key"],
    }
    names = key_names.get(provider, [])

    # 1순위: Dataiku Managed Secrets
    try:
        import dataiku
        client = dataiku.api_client()
        auth_info = client.get_auth_info(with_secrets=True)
        secrets = {s["key"]: s["value"] for s in auth_info.get("secrets", [])}
        for name in names:
            if secrets.get(name):
                return secrets[name]
    except Exception:
        pass

    # 2순위: 환경변수
    for name in names:
        val = os.environ.get(name)
        if val:
            return val

    return None

# ================================================================
# 페이지 기본 설정
# ================================================================

st.set_page_config(
    page_title="NexusData | LLM Dashboard",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CSS 스타일
# ================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .info-card {
        background: #f0f4f9;
        border-left: 4px solid #2d6a9f;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }

    .stChatMessage { border-radius: 10px; }

    .code-box {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* 차트 컨테이너 */
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background: white;
    }
    
    /* 오른쪽 사이드바 스타일 */
    .right-sidebar {
        position: fixed;
        right: 0;
        top: 0;
        width: 400px;
        height: 100vh;
        background: #f8f9fa;
        border-left: 1px solid #e0e0e0;
        padding: 1rem;
        overflow-y: auto;
        z-index: 999;
    }
    
    /* 메인 컨텐츠 영역 조정 */
    .main-content {
        margin-right: 420px;
    }
    
    .right-sidebar h3 {
        margin-top: 0;
        color: #2d6a9f;
        font-size: 1.2rem;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 0.5rem;
    }

    /* 코드 블록 높이 제한 해제 */
    .stCode, .stCode > div, .stCode pre {
        max-height: none !important;
        overflow: visible !important;
    }

    /* 코드 블록 줄바꿈 */
    .stCode pre, .stCode code {
        white-space: pre-wrap !important;
        word-break: break-all !important;
        overflow-x: hidden !important;
    }


</style>
""", unsafe_allow_html=True)

# ================================================================
# 세션 상태 초기화
# ================================================================

def _init_session():
    defaults = {
        "messages": [],          # 채팅 히스토리
        "df": None,              # 로드된 DataFrame (기본 데이터셋)
        "datasets": {},          # 멀티 데이터셋: {"EL_Sensor": df1, "EL_Vibration": df2}
        "dataset_info": None,    # 데이터셋 메타데이터
        "datasets_info": {},     # 멀티 데이터셋 메타: {"EL_Sensor": info1, ...}
        "llm_client": None,      # LLM 클라이언트 인스턴스
        "selected_dataset": None,
        "error_count": 0,        # 현재 메시지의 오류 횟수
        "history_manager": HistoryManager(),  # 히스토리 관리자
        "user_id": HistoryManager.get_user_id(),  # 현재 사용자 ID
        "flow_exporter": None,   # DataikuFlowExporter 인스턴스
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ── 마지막 대화 자동 복원 (Gemini 스타일) ──
if not st.session_state.get("_history_restored"):
    _hm = st.session_state.history_manager
    _uid = st.session_state.user_id
    _histories = _hm.list_user_histories(_uid)
    if _histories and not st.session_state.messages:
        _last = _histories[0]  # 가장 최근 대화
        _saved = _hm.load_history(_uid, _last["dataset"])
        if _saved:
            st.session_state.messages = _saved
            st.session_state.selected_dataset = _last["dataset"]
    st.session_state["_history_restored"] = True

# ================================================================
# 사이드바
# ================================================================

# ================================================================
# 왼쪽 사이드바: 대화 히스토리 (Gemini 스타일)
# ================================================================

with st.sidebar:
    st.markdown("# 📊 NexusData")

    # 솔루션 선택
    solution_list = ["Dataiku", "솔루션1", "솔루션2"]
    selected_solution = st.selectbox(
        "솔루션",
        solution_list,
        index=0,
        key="sol_select",
    )

    st.markdown("---")
    
    # 대화 히스토리 헤더 + 새 채팅 아이콘 버튼
    st.markdown("### 대화 히스토리")
    if st.button("✏️ 새 채팅", use_container_width=True):
        st.session_state.messages = []
        st.session_state.df = None
        st.session_state.dataset_info = None
        st.session_state.selected_dataset = None
        st.session_state.prompt_key_counter = st.session_state.get('prompt_key_counter', 0) + 1
        st.rerun()
    
    # 히스토리 관리
    history_manager = st.session_state.history_manager
    user_id = st.session_state.user_id
    
    # 저장된 히스토리 목록 (Gemini 스타일)
    histories = history_manager.list_user_histories(user_id)
    if histories:
        for idx, hist in enumerate(histories):
            title = hist.get('title', hist['dataset'])
            col_h, col_del = st.columns([5, 1])
            with col_h:
                if st.button(f"{title}", key=f"hist_{idx}_{hist['dataset'][:20]}", use_container_width=True):
                    try:
                        saved_messages = history_manager.load_history(user_id, hist['dataset'])
                        st.session_state.messages = saved_messages if saved_messages else []
                        st.session_state.selected_dataset = hist['dataset']
                        st.rerun()
                    except Exception:
                        pass
    else:
        st.caption("저장된 대화가 없습니다.")
    
    st.markdown("---")
    
    # 현재 세션 코드 로그 (Session Log)
    if st.session_state.messages:
        code_msgs = [(i, m) for i, m in enumerate(st.session_state.messages) if m.get("role") == "assistant" and m.get("code")]
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
                        if user_prompt:
                            st.session_state.pending_prompt = st.session_state.messages[msg_idx - 1].get("content", "")
                            st.rerun()

# ================================================================
# 메인 영역: 데이터 미리보기 → 대화 → 입력창 → LLM/데이터셋 선택
# ================================================================

# DataikuManager 초기화
data_manager = DataikuManager()

# FlowExporter 초기화
if st.session_state.flow_exporter is None:
    st.session_state.flow_exporter = DataikuFlowExporter(data_manager)

# ── 1) 데이터 미리보기 ──
if st.session_state.df is not None:
    loaded_datasets = st.session_state.get("datasets", {})
    
    if len(loaded_datasets) > 1:
        # 멀티 데이터셋: 탭으로 표시 + 탭별 품질 리포트
        st.markdown("### 데이터 미리보기")
        tabs = st.tabs(list(loaded_datasets.keys()))
        for tab, (ds_name, ds_df) in zip(tabs, loaded_datasets.items()):
            with tab:
                st.dataframe(ds_df.head(10), use_container_width=True, height=200)
                ds_info = st.session_state.datasets_info.get(ds_name, {})
                if ds_info:
                    st.caption(
                        f"**컬럼**: {', '.join(ds_info['columns'])}  |  "
                        f"**행**: {ds_info['shape'][0]:,}  |  **열**: {ds_info['shape'][1]}"
                    )
                    with st.expander("데이터 품질 리포트", expanded=False):
                        q_col1, q_col2, q_col3 = st.columns(3)
                        missing_detail = ds_info.get("missing_detail", {})
                        total_missing = sum(v["count"] for v in missing_detail.values()) if missing_detail else 0
                        with q_col1:
                            if total_missing == 0:
                                st.metric("결측치", "0건", delta="양호", delta_color="normal")
                            else:
                                st.metric("결측치", f"{total_missing:,}건", delta="주의", delta_color="inverse")
                                for col, v in missing_detail.items():
                                    st.caption(f"`{col}`: {v['count']:,}건 ({v['pct']}%)")
                        outlier_detail = ds_info.get("outlier_detail", {})
                        total_outliers = sum(outlier_detail.values()) if outlier_detail else 0
                        with q_col2:
                            if total_outliers == 0:
                                st.metric("이상치 (IQR)", "0건", delta="양호", delta_color="normal")
                            else:
                                st.metric("이상치 (IQR)", f"{total_outliers:,}건", delta="확인 필요", delta_color="inverse")
                                for col, cnt in outlier_detail.items():
                                    st.caption(f"`{col}`: {cnt:,}건")
                        dup_count = ds_info.get("dup_count", 0)
                        dup_pct = ds_info.get("dup_pct", 0)
                        with q_col3:
                            if dup_count == 0:
                                st.metric("중복 행", "0건", delta="양호", delta_color="normal")
                            else:
                                st.metric("중복 행", f"{dup_count:,}건", delta=f"{dup_pct}%", delta_color="inverse")

                        # 멀티 데이터셋 품질 리포트 게시 버튼
                        if _PLOTLY_AVAILABLE:
                            if st.button("📤 품질 리포트 게시", key=f"pub_quality_{ds_name}"):
                                _qr_labels = ["결측치", "이상치 (IQR)", "중복 행"]
                                _qr_values = [total_missing, total_outliers, dup_count]
                                _qr_colors = ['#FF6B6B' if v > 0 else '#51CF66' for v in _qr_values]
                                _fig_qr = go.Figure(data=[go.Bar(
                                    x=_qr_labels, y=_qr_values,
                                    marker_color=_qr_colors, text=_qr_values, textposition='auto',
                                )])
                                _fig_qr.update_layout(
                                    title=f"{ds_name} 데이터 품질 리포트",
                                    yaxis_title="건수", height=350,
                                )
                                _exp = st.session_state.flow_exporter
                                ok, msg = _exp.publish_chart(
                                    user_id=st.session_state.user_id, fig=_fig_qr, code="",
                                    question=f"{ds_name} 품질 리포트", dataset_name=ds_name,
                                )
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(msg)
    else:
        # 단일 데이터셋
        st.markdown("### 데이터 미리보기")
        st.dataframe(
            st.session_state.df.head(10),
            use_container_width=True,
            height=250,
        )
    
    info = st.session_state.dataset_info
    if len(loaded_datasets) <= 1:
        st.caption(
            f"**컬럼**: {', '.join(info['columns'])}  |  "
            f"**행**: {info['shape'][0]:,}  |  **열**: {info['shape'][1]}"
        )

    # ── 데이터 품질 리포트 (단일 데이터셋만) ──
    if len(loaded_datasets) <= 1:
        with st.expander("데이터 품질 리포트", expanded=False):
            q_col1, q_col2, q_col3 = st.columns(3)

            missing_detail = info.get("missing_detail", {})
            total_missing = sum(v["count"] for v in missing_detail.values()) if missing_detail else 0
            with q_col1:
                if total_missing == 0:
                    st.metric("결측치", "0건", delta="양호", delta_color="normal")
                else:
                    st.metric("결측치", f"{total_missing:,}건", delta="주의", delta_color="inverse")
                    for col, v in missing_detail.items():
                        st.caption(f"`{col}`: {v['count']:,}건 ({v['pct']}%)")

            outlier_detail = info.get("outlier_detail", {})
            total_outliers = sum(outlier_detail.values()) if outlier_detail else 0
            with q_col2:
                if total_outliers == 0:
                    st.metric("이상치 (IQR)", "0건", delta="양호", delta_color="normal")
                else:
                    st.metric("이상치 (IQR)", f"{total_outliers:,}건", delta="확인 필요", delta_color="inverse")
                    for col, cnt in outlier_detail.items():
                        st.caption(f"`{col}`: {cnt:,}건")

            dup_count = info.get("dup_count", 0)
            dup_pct = info.get("dup_pct", 0)
            with q_col3:
                if dup_count == 0:
                    st.metric("중복 행", "0건", delta="양호", delta_color="normal")
                else:
                    st.metric("중복 행", f"{dup_count:,}건", delta=f"{dup_pct}%", delta_color="inverse")

            # 품질 리포트 대시보드 게시 버튼
            if _PLOTLY_AVAILABLE:
                if st.button("📤 품질 리포트 게시", key="pub_quality"):
                    _qr_labels = ["결측치", "이상치 (IQR)", "중복 행"]
                    _qr_values = [total_missing, total_outliers, dup_count]
                    _qr_colors = ['#FF6B6B' if v > 0 else '#51CF66' for v in _qr_values]
                    _fig_qr = go.Figure(data=[go.Bar(
                        x=_qr_labels, y=_qr_values,
                        marker_color=_qr_colors, text=_qr_values, textposition='auto',
                    )])
                    _ds_label = st.session_state.selected_dataset or "dataset"
                    _fig_qr.update_layout(
                        title=f"{_ds_label} 데이터 품질 리포트",
                        yaxis_title="건수", height=350,
                    )
                    _exp = st.session_state.flow_exporter
                    ok, msg = _exp.publish_chart(
                        user_id=st.session_state.user_id, fig=_fig_qr, code="",
                        question=f"{_ds_label} 품질 리포트", dataset_name=_ds_label,
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    # ── EDA ──
    with st.expander("EDA", expanded=False):
        _loaded_ds = st.session_state.get("datasets", {})
        _ds_names = list(_loaded_ds.keys()) if _loaded_ds else [st.session_state.selected_dataset or "default"]

        if len(_ds_names) > 1:
            _eda_ds_name = st.selectbox("데이터셋", _ds_names, key="eda_ds_select")
        else:
            _eda_ds_name = _ds_names[0]

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
                    # describe를 Plotly 테이블로 변환
                    _desc_reset = _desc_df.reset_index().rename(columns={"index": "변수"})
                    _fig_desc = go.Figure(data=[go.Table(
                        header=dict(values=list(_desc_reset.columns), fill_color='#4472C4', font=dict(color='white', size=11), align='center'),
                        cells=dict(values=[_desc_reset[c].round(2) if _desc_reset[c].dtype != 'object' else _desc_reset[c] for c in _desc_reset.columns],
                                   fill_color='#D9E2F3', align='center', font=dict(size=10)),
                    )])
                    _fig_desc.update_layout(title=f"{_eda_ds_name} 기술통계", height=max(300, len(_desc_df) * 35 + 80))
                    _exp = st.session_state.flow_exporter
                    ok, msg = _exp.publish_chart(
                        user_id=st.session_state.user_id, fig=_fig_desc, code="",
                        question=f"{_eda_ds_name} 기술통계", dataset_name=_eda_ds_name,
                    )
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
                        _exp = st.session_state.flow_exporter
                        ok, msg = _exp.publish_chart(
                            user_id=st.session_state.user_id, fig=_fig_dist, code="",
                            question=f"[{_eda_ds_name}] {_dist_col} 분포", dataset_name=_eda_ds_name,
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
            else:
                st.info("수치형 컬럼이 없거나 Plotly가 설치되지 않았습니다.")

        with eda_tab3:
            if len(_eda_numeric) >= 2 and _PLOTLY_AVAILABLE:
                _corr_matrix = _eda_df[_eda_numeric].corr(numeric_only=True)
                _fig_corr = px.imshow(
                    _corr_matrix.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="상관관계 히트맵", aspect="auto",
                )
                _fig_corr.update_layout(height=max(400, len(_eda_numeric) * 40))
                st.plotly_chart(_fig_corr, use_container_width=True)
                _stat_tests = _eda_info.get("stat_tests", {})
                if _stat_tests:
                    st.caption("Pearson 상관관계 (|r| > 0.3)")
                    for pair, vals in _stat_tests.items():
                        sig = "✅" if vals["significant"] else "❌"
                        st.caption(f"`{pair}`: r={vals['r']}, p={vals['p']} {sig}")
                if st.button("📤 상관관계 히트맵 게시", key=f"pub_corr_{_eda_ds_name}"):
                    _exp = st.session_state.flow_exporter
                    ok, msg = _exp.publish_chart(
                        user_id=st.session_state.user_id, fig=_fig_corr, code="",
                        question=f"[{_eda_ds_name}] 상관관계 히트맵 ({len(_eda_numeric)}개 변수)", dataset_name=_eda_ds_name,
                    )
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
                    pct = round(cnt / len(_eda_df) * 100, 2)
                    st.caption(f"`{col}`: {cnt:,}건 ({pct}%)")
                if st.button("📤 이상치 차트 게시", key=f"pub_outlier_{_eda_ds_name}"):
                    _exp = st.session_state.flow_exporter
                    ok, msg = _exp.publish_chart(
                        user_id=st.session_state.user_id, fig=_fig_box, code="",
                        question=f"[{_eda_ds_name}] 이상치 분포 (IQR) - {', '.join(_box_cols)}", dataset_name=_eda_ds_name,
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("IQR 기준 이상치가 감지되지 않았습니다.")

    st.markdown("---")

# ── 2) 대화 내용 표시 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("fig") is not None:
                fig = msg["fig"]
                if CodeExecutor.is_plotly_figure(fig):
                    st.plotly_chart(fig, use_container_width=True)
                elif CodeExecutor.is_matplotlib_figure(fig):
                    st.pyplot(fig, use_container_width=True)
            # 결과 데이터프레임 표시
            if msg.get("result_df") is not None:
                _rdf = msg["result_df"]
                st.dataframe(_rdf, use_container_width=True, height=min(400, 35 * len(_rdf) + 38))
                st.caption(f"{_rdf.shape[0]:,}행 × {_rdf.shape[1]}열")
            if msg.get("insight"):
                st.info(f"**인사이트**: {msg['insight']}")
            # 코드 보기
            msg_idx = st.session_state.messages.index(msg)
            _has_fig = msg.get("fig") is not None
            _has_result_df = msg.get("result_df") is not None
            _has_code = bool(msg.get("code"))
            if _has_code:
                with st.expander("코드 보기", expanded=False):
                    st.code(msg["code"], language="python")
            # ── 게시 버튼: 대시보드 + Flow 레시피 ──
            if _has_fig or _has_result_df:
                _btn_col1, _btn_col2 = st.columns(2)
                with _btn_col1:
                    if st.button("📤 대시보드 게시", key=f"publish_{msg_idx}", use_container_width=True):
                        _user_q = st.session_state.messages[msg_idx - 1].get('content', '') if msg_idx > 0 else ''
                        _ds_name = st.session_state.selected_dataset or ''
                        _exporter = st.session_state.flow_exporter
                        _uid = st.session_state.user_id
                        if _has_fig:
                            ok, _pub_msg = _exporter.publish_chart(
                                user_id=_uid, fig=msg["fig"], code=msg.get("code", ""),
                                question=_user_q, insight=msg.get("insight", ""),
                                dataset_name=_ds_name,
                            )
                        else:
                            # result_df → Plotly Table로 변환해서 게시
                            _rdf = msg["result_df"]
                            _tbl_fig = go.Figure(data=[go.Table(
                                header=dict(values=list(_rdf.columns), fill_color='#4472C4',
                                            font=dict(color='white', size=11), align='center'),
                                cells=dict(values=[_rdf[c].head(100) for c in _rdf.columns],
                                           fill_color='#D9E2F3', align='center', font=dict(size=10)),
                            )])
                            _tbl_fig.update_layout(title=f"데이터 처리 결과 ({_rdf.shape[0]:,}행)", height=400)
                            ok, _pub_msg = _exporter.publish_chart(
                                user_id=_uid, fig=_tbl_fig, code=msg.get("code", ""),
                                question=_user_q, dataset_name=_ds_name,
                            )
                        if ok:
                            st.success(_pub_msg)
                        else:
                            st.error(_pub_msg)
                with _btn_col2:
                    if _has_code and st.button("🔧 Flow 레시피 게시", key=f"pub_recipe_{msg_idx}", use_container_width=True):
                        _exporter = st.session_state.flow_exporter
                        _ds_names = list(st.session_state.get("datasets", {}).keys())
                        if not _ds_names:
                            _ds_names = [st.session_state.selected_dataset or "dataset"]
                        _user_q = st.session_state.messages[msg_idx - 1].get('content', '') if msg_idx > 0 else ''
                        ok, _pub_msg = _exporter.publish_recipe(
                            code=msg.get("code", ""),
                            input_datasets=_ds_names,
                            label=_user_q[:40],
                        )
                        if ok:
                            st.success(_pub_msg)
                        else:
                            st.error(_pub_msg)
            if msg.get("error"):
                st.error(msg["error"])
                retry_guide = _build_error_guide(msg["error"], st.session_state.dataset_info)
                st.warning(f"다시 시도하려면:\n{retry_guide}")

st.markdown("---")

# ── 3) 프롬프트 입력 ──
# 입력창 초기화를 위한 카운터 (key를 바꾸면 위젯이 리셋됨)
if "prompt_key_counter" not in st.session_state:
    st.session_state.prompt_key_counter = 0

prompt_col1, prompt_col2 = st.columns([5, 1])
with prompt_col1:
    user_input_text = st.text_input(
        "분석 요청",
        key=f"user_prompt_{st.session_state.prompt_key_counter}",
        label_visibility="collapsed",
        placeholder="요청을 입력하세요."
    )
with prompt_col2:
    send_button = st.button("전송", use_container_width=True)

user_input = user_input_text if send_button and user_input_text else None

# 재실행 클릭 시 자동 입력
if st.session_state.get("pending_prompt"):
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── 4) LLM 선택 + 데이터셋 선택 (맨 아래) ──
col1, col2 = st.columns(2)

with col1:
    st.markdown("**LLM 선택**")
    llm_col1, llm_col2 = st.columns(2)
    with llm_col1:
        llm_provider = st.selectbox(
            "Provider",
            options=["openai", "groq", "claude"],
            format_func=lambda x: "Groq" if x == "groq" else ("OpenAI" if x == "openai" else "Claude"),
            label_visibility="collapsed"
        )
    with llm_col2:
        model_options = {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "claude": ["claude-3-5-sonnet-20241022"],
            "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"],
        }
        selected_model = st.selectbox(
            "Model",
            options=model_options[llm_provider],
            label_visibility="collapsed"
        )
    
    _api_key = _load_api_key(llm_provider)
    _client_key = f"{llm_provider}:{selected_model}"
    if _api_key:
        if st.session_state.llm_client is None or st.session_state.get("_client_key") != _client_key:
            try:
                st.session_state.llm_client = create_llm_client(
                    provider=llm_provider, api_key=_api_key, model=selected_model,
                )
                st.session_state["_client_key"] = _client_key
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
            selected_datasets = st.multiselect(
                "Datasets", options=dataset_list,
                default=[],
                key="dataset_selector", label_visibility="collapsed",
                placeholder="데이터셋을 선택하세요"
            )
        with ds_col2:
            if st.button("로드", use_container_width=True) and selected_datasets:
                try:
                    loaded_datasets = {}
                    loaded_infos = {}
                    for ds_name in selected_datasets:
                        df_loaded = data_manager.load_dataset(ds_name)
                        loaded_datasets[ds_name] = df_loaded
                        loaded_infos[ds_name] = data_manager.build_dataset_info(df_loaded)
                    
                    # 첫 번째 데이터셋을 기본 df로 설정
                    first_ds = selected_datasets[0]
                    st.session_state.df = loaded_datasets[first_ds]
                    st.session_state.dataset_info = loaded_infos[first_ds]
                    st.session_state.datasets = loaded_datasets
                    st.session_state.datasets_info = loaded_infos
                    st.session_state.selected_dataset = first_ds
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"로드 실패: {str(e)}")
        
        # 로드된 데이터셋 표시
        if st.session_state.datasets:
            loaded_names = list(st.session_state.datasets.keys())
            st.caption(f"로드됨: {', '.join(loaded_names)} ({len(loaded_names)}개)")
    else:
        st.warning("데이터셋 없음")

# ================================================================
# 사용자 입력 처리
# ================================================================

# user_input은 왼쪽 컬럼에서 정의됨
prompt = user_input

if prompt:
    # 데이터셋/LLM 미연결 체크
    if st.session_state.df is None:
        st.warning("데이터셋을 선택한 후 로드해주세요.")
        st.stop()
    if st.session_state.llm_client is None:
        st.warning("LLM을 선택해주세요.")
        st.stop()

    # ── 사용자 메시지 추가 ──────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── LLM 호출 + 코드 실행 (직접 렌더링 없이 세션에만 저장) ──
    with st.spinner("코드 생성 및 실행 중..."):
        MAX_AUTO_RETRY = 2

        llm_client    = st.session_state.llm_client
        df            = st.session_state.df
        dataset_info  = st.session_state.dataset_info
        datasets      = st.session_state.get("datasets", {})
        datasets_info = st.session_state.get("datasets_info", {})
        _flow_exporter = st.session_state.flow_exporter

        code      = None
        fig       = None
        insight   = None
        error_msg = None

        try:
            main_prompt = PromptEngine.build_main_prompt(prompt, dataset_info, datasets_info=datasets_info)
            raw_output  = llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=main_prompt,
            )

            # 존재하지 않는 컬럼 요청 감지
            if "ERROR_COLUMN_NOT_FOUND" in raw_output:
                col_name = raw_output.split("ERROR_COLUMN_NOT_FOUND:")[-1].strip()
                available = ", ".join(dataset_info.get('columns', []))
                error_msg = f"'{col_name}' 컬럼이 현재 데이터셋에 존재하지 않습니다.\n\n사용 가능한 컬럼: {available}"
            else:
                # 멀티 데이터셋 컬럼도 허용
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
                        # 자동 재시도 (Self-Correction)
                        for retry in range(1, MAX_AUTO_RETRY + 1):
                            recovery_prompt = PromptEngine.build_error_recovery_prompt(
                                user_request=prompt,
                                failed_code=code,
                                error_message=exec_error,
                                dataset_info=dataset_info,
                            )
                            raw_output2 = llm_client.generate(
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=recovery_prompt,
                            )
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

        # ── 결과를 세션에 저장 (직접 렌더링 하지 않음) ──
        assistant_msg: dict = {"role": "assistant"}

        if error_msg:
            assistant_msg["error"] = error_msg
            if code:
                assistant_msg["code"] = code
        else:
            # 인사이트 생성
            try:
                insight_prompt = PromptEngine.build_insight_prompt(prompt, dataset_info, datasets_info=datasets_info)
                insight = llm_client.generate(
                    system_prompt="You are a sensor data analyst. Answer concisely in Korean.",
                    user_prompt=insight_prompt,
                )
            except Exception:
                insight = None

            if fig is not None:
                assistant_msg["text"] = "차트가 생성되었습니다."
            elif result_df is not None:
                assistant_msg["text"] = f"데이터 처리 완료 ({result_df.shape[0]:,}행 × {result_df.shape[1]}열)"
            else:
                assistant_msg["text"] = "처리가 완료되었습니다."
            assistant_msg["fig"]        = fig
            assistant_msg["result_df"]  = result_df
            assistant_msg["code"]       = code
            assistant_msg["insight"]    = insight

        st.session_state.messages.append(assistant_msg)
        
        # 히스토리 자동 저장
        if st.session_state.selected_dataset:
            history_manager = st.session_state.history_manager
            user_id = st.session_state.user_id
            messages_to_save = []
            for msg in st.session_state.messages:
                msg_copy = {k: v for k, v in msg.items() if k not in ("fig", "result_df")}
                messages_to_save.append(msg_copy)
            # 데이터셋명 목록으로 타이틀 생성
            ds_names = list(st.session_state.get('datasets', {}).keys())
            ds_label = ', '.join(ds_names) if ds_names else st.session_state.selected_dataset
            first_user_msg = next((m.get('content', '') for m in messages_to_save if m.get('role') == 'user'), '')
            short_msg = first_user_msg[:20] + ('...' if len(first_user_msg) > 20 else '')
            hist_title = f"{ds_label} | {short_msg}" if short_msg else ds_label
            history_manager.save_history(user_id, st.session_state.selected_dataset, messages_to_save, title=hist_title)

    # 입력창 초기화 + 페이지 재렌더링
    st.session_state.prompt_key_counter = st.session_state.get('prompt_key_counter', 0) + 1
    st.rerun()

