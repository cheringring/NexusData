# -*- coding: utf-8 -*-
"""코드 검증 및 실행 엔진"""
import ast
import re
import hashlib
import traceback
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils.constants import (
    ALLOWED_MODULES, FORBIDDEN_PATTERNS, DF_VARIABLE_NAMES, CODE_EXECUTION_TIMEOUT
)

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
    """코드 검증기"""

    @staticmethod
    def extract_code_block(text: str) -> str:
        """LLM 응답에서 코드 블록 추출"""
        match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match2 = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        return text.strip()

    @staticmethod
    def validate(code: str, available_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """코드 보안 및 유효성 검증"""
        # 금지 패턴 체크
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"보안 위반: {reason}"

        # 구문 분석
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"문법 오류: {str(e)}"

        # 모듈 검증
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_MODULES:
                        return False, f"허용되지 않은 모듈: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_MODULES:
                    return False, f"허용되지 않은 모듈: {node.module}"

        # 컬럼명 검증
        if available_columns:
            used_columns = set()
            assigned_columns = set()

            # 할당되는 컬럼 찾기
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript):
                            if isinstance(target.value, ast.Name) and target.value.id in DF_VARIABLE_NAMES:
                                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                    assigned_columns.add(target.slice.value)

            # 사용되는 컬럼 찾기
            for node in ast.walk(tree):
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
        """산점도 요청인데 plot()이 쓰인 경우 scatter()로 교정"""
        scatter_keywords = ["산점도", "scatter", "관계", "상관관계", "correlation"]
        if not any(kw in user_request.lower() for kw in scatter_keywords):
            return code
        if "scatter(" in code:
            return code
        code = code.replace("ax.plot(", "ax.scatter(")
        code = code.replace("plt.plot(", "plt.scatter(")
        code = re.sub(r",?\s*linewidth\s*=\s*[\d.]+", "", code)
        code = re.sub(r",?\s*linestyle\s*=\s*['\"][^'\"]*['\"]", "", code)
        code = re.sub(r",?\s*marker\s*=\s*['\"][^'\"]*['\"]", "", code)
        return code

    @staticmethod
    def fix_deprecated_plotly(code: str) -> str:
        """Plotly deprecated 속성을 최신 문법으로 자동 치환"""
        code = re.sub(r'\btitlefont\b', 'title_font', code)
        return code

    @staticmethod
    def fix_mixed_subplots(code: str) -> str:
        """imshow + histogram 혼합 시 히트맵만 남기도록 변환, .show() 제거"""
        code = re.sub(r'\bfig\w*\.show\(\)', '', code)

        has_imshow = 'px.imshow' in code or 'ff.create_annotated_heatmap' in code
        has_histogram = 'px.histogram' in code or 'go.Histogram' in code
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
        """pd.cut/pd.qcut 결과가 Plotly JSON 직렬화 실패하는 문제 방지"""
        if 'pd.cut' in code or 'pd.qcut' in code:
            code = re.sub(
                r"((\w+)\[(['\"])([\w가-힣]+)\3\]\s*=\s*pd\.(?:cut|qcut)\([^)]+\))",
                r"\1\n\2[\3\4\3] = \2[\3\4\3].astype(str)",
                code
            )
        return code

    @staticmethod
    def full_check(raw_llm_output: str, available_columns: Optional[List[str]] = None, 
                   user_request: str = "") -> Tuple[str, bool, Optional[str]]:
        """전체 검증 파이프라인"""
        code = CodeValidator.extract_code_block(raw_llm_output)
        if user_request:
            code = CodeValidator.fix_scatter_code(code, user_request)
        code = CodeValidator.fix_deprecated_plotly(code)
        code = CodeValidator.fix_mixed_subplots(code)
        code = CodeValidator.fix_interval_serialization(code)
        is_safe, msg = CodeValidator.validate(code, available_columns)
        return code, is_safe, msg


class CodeExecutor:
    """코드 실행기"""
    _result_cache: Dict[str, Tuple[Optional[Any], Optional[Any], Optional[str]]] = {}

    @staticmethod
    def _make_cache_key(code: str, df: pd.DataFrame) -> str:
        content = f"{code}|{df.shape}|{list(df.columns)}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def execute(code: str, df: pd.DataFrame, 
                datasets: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[str]]:
        """코드 실행"""
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

        # 멀티 데이터셋 주입
        if datasets:
            namespace["datasets"] = datasets
            for ds_name, ds_df in datasets.items():
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
            exec_error_holder = [None]

            def _run_code():
                try:
                    exec(code, namespace)  # noqa: S102
                except Exception as e:
                    exec_error_holder[0] = e

            thread = threading.Thread(target=_run_code)
            thread.start()
            thread.join(timeout=CODE_EXECUTION_TIMEOUT)

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

            # Plotly figure의 Interval 객체를 문자열로 변환
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

            # Plotly figure의 shape/annotation 개수 제한
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


def build_error_guide(error_msg: str, dataset_info: Optional[Dict] = None) -> str:
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
