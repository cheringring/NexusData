# -*- coding: utf-8 -*-
"""Dataiku 연결 및 Flow 내보내기"""
import io
import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# plotly 선택적 import
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    _PLOTLY_AVAILABLE = False


class DataikuManager:
    """Dataiku 데이터셋 관리자"""

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
        """데이터셋 메타정보 생성"""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        describe_str = df[numeric_cols].describe().to_string() if numeric_cols else "수치형 컬럼 없음"
        date_cols = self._detect_date_columns(df)

        cat_info = {}
        for col in df.select_dtypes(include="object").columns:
            cat_info[col] = df[col].dropna().unique()[:10].tolist()

        # 결측치 정보
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

        # 상관관계
        corr_str = "상관계수 정보 없음"
        stat_tests_str = "통계 검정 없음"
        stat_tests = {}
        if len(numeric_cols) >= 2:
            corr_str = df[numeric_cols].corr(numeric_only=True).round(3).to_string()
            try:
                from scipy import stats as _scipy_stats
                sig_pairs = []
                for i, c1 in enumerate(numeric_cols[:8]):
                    for c2 in numeric_cols[i + 1:8]:
                        valid = df[[c1, c2]].dropna()
                        if len(valid) > 10:
                            r, p = _scipy_stats.pearsonr(valid[c1], valid[c2])
                            if abs(r) > 0.3:
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


class DataikuFlowExporter:
    """
    Streamlit 웹앱의 비정형 분석 결과를 Dataiku Flow 자산으로 게시.
    Flow 구조:
    └── 📁 Managed Folder: "nexusdata_charts"
          └── {user_id}/{insight_id}.json  (질의/코드/인사이트 메타)
          + Dataiku Static Insight → 대시보드 타일 자동 추가
    """
    MANAGED_FOLDER_NAME = "nexusdata_charts"
    DASHBOARD_NAME = "NexusData Dashboard"
    RECIPE_PREFIX = "compute_nexusdata_"

    def __init__(self, data_manager: DataikuManager):
        self._dm = data_manager
        self._in_dataiku = data_manager.is_connected
        if self._in_dataiku:
            self._ensure_assets_exist()

    def _ensure_assets_exist(self):
        """Managed Folder 자동 생성"""
        try:
            import dataiku
            client = dataiku.api_client()
            project = client.get_default_project()
            existing_folders = [f["name"] for f in project.list_managed_folders()]
            if self.MANAGED_FOLDER_NAME not in existing_folders:
                project.create_managed_folder(self.MANAGED_FOLDER_NAME)
                print(f"[FlowExporter] Managed Folder '{self.MANAGED_FOLDER_NAME}' 생성 완료")
        except Exception as e:
            print(f"[FlowExporter] 자산 자동 생성 중 오류 (무시됨): {e}")

    def publish_chart(self, user_id: str, fig, code: str, question: str,
                      insight: str = "", dataset_name: str = "",
                      chart_type: str = "plotly") -> Tuple[bool, str]:
        """LLM 생성 차트를 Dataiku Static Insight로 게시"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        insight_id = f"nexusdata_{user_id}_{ts}"

        # 차트 제목 추출
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

        # 라벨 생성
        _ds_tag = f"[{dataset_name}]" if dataset_name else ""
        _title_or_q = chart_title if chart_title else (question[:60] if question else "분석 결과")
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

                # 차트 레이아웃 최적화
                try:
                    if _PLOTLY_AVAILABLE and hasattr(fig, 'update_layout'):
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
                        if hasattr(fig.layout, 'annotations'):
                            for ann in fig.layout.annotations:
                                if hasattr(ann, 'text') and ann.text and len(ann.text) > 20:
                                    ann.text = ann.text[:20] + '...'
                                if hasattr(ann, 'font'):
                                    ann.font.size = 10
                except Exception:
                    pass

                # Static Insight 게시
                if _PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                    dataiku.insights.save_plotly(insight_id, fig, label=label)
                elif hasattr(fig, 'savefig'):
                    dataiku.insights.save_figure(insight_id, fig, label=label)
                else:
                    return False, "지원하지 않는 차트 타입"

                # Managed Folder에 메타 JSON 저장
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

                # 대시보드에 Insight 타일 자동 추가
                try:
                    self._add_insight_to_dashboard(project, insight_id, label)
                except Exception as e:
                    print(f"[FlowExporter] 대시보드 타일 추가 실패: {type(e).__name__}: {e}")

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

    def _add_insight_to_dashboard(self, project, insight_id: str, label: str):
        """NexusData 대시보드에 Insight 타일 자동 추가"""
        import dataiku
        client = dataiku.api_client()

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

        # 타일 위치 계산
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
            "box": {"left": 0, "top": max_row, "width": 12, "height": 15},
            "tileParams": {},
            "clickAction": "DO_NOTHING",
            "resizeImageMode": "FIT_SIZE",
            "displayMode": "INSIGHT",
        }
        tiles.append(new_tile)
        settings.save()
        print(f"[FlowExporter] 대시보드 타일 추가 완료: {label}")

    def _find_nexus_recipe(self, project) -> Optional[str]:
        """기존 nexusdata 레시피 이름 찾기"""
        for r in project.list_recipes():
            if r["name"].startswith(self.RECIPE_PREFIX):
                return r["name"]
        return None

    def publish_recipe(self, code: str, input_datasets: List[str],
                       label: str = "",
                       connection: str = "filesystem_managed") -> Tuple[bool, str]:
        """LLM 생성 코드를 Dataiku Flow의 Python 레시피로 게시"""
        if not self._in_dataiku:
            return False, "Dataiku 연결이 필요합니다 (로컬 모드에서는 사용 불가)"

        try:
            import dataiku
            client = dataiku.api_client()
            project = client.get_default_project()

            slug = self._make_slug(label) if label else datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"nexus_{slug}"
            display_label = label[:50] if label else out_name

            existing_ds = [d["name"] for d in project.list_datasets()]
            if out_name in existing_ds:
                out_name = f"{out_name}_{datetime.now().strftime('%H%M%S')}"

            block_code = self._make_code_block(code, input_datasets, out_name, display_label)
            existing_recipe_name = self._find_nexus_recipe(project)

            if existing_recipe_name:
                recipe = project.get_recipe(existing_recipe_name)
                settings = recipe.get_settings()
                old_code = settings.str_payload or ""
                new_code = old_code + "\n\n" + block_code
                settings.set_payload(new_code)

                try:
                    ds_creator = project.new_managed_dataset(out_name)
                    ds_creator.with_store_into(connection)
                    ds_creator.create()
                except Exception:
                    pass

                settings.add_output("main", out_name)
                settings.save()
                return True, f"✅ 레시피에 추가: {display_label} → {out_name}"
            else:
                header = self._make_header(input_datasets)
                full_code = header + "\n\n" + block_code

                first_out = f"nexusdata_{slug}"
                if first_out in existing_ds:
                    first_out = f"nexusdata_{slug}_{datetime.now().strftime('%H%M%S')}"

                builder = project.new_recipe("python")
                for ds_name in input_datasets:
                    builder.with_input(ds_name)
                builder.with_new_output_dataset(first_out, connection)
                builder.with_script(full_code)
                recipe = builder.create()

                actual_name = recipe.name
                print(f"[FlowExporter] 레시피 생성됨: {actual_name}")
                return True, f"✅ Flow 레시피 생성: {display_label} → {first_out}"
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
        """레시피 최초 생성 시 공통 헤더"""
        lines = ["import dataiku", "import pandas as pd", "import numpy as np", ""]
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
