# -*- coding: utf-8 -*-
"""상수 및 설정값"""

# 보안: 허용 모듈 목록
ALLOWED_MODULES = {
    "pandas", "pd", "numpy", "np", "matplotlib", "mpl",
    "seaborn", "sns", "plotly", "scipy", "statsmodels", "math", "datetime",
    "collections", "itertools", "functools", "string", "re",
    "sklearn",
}

# 보안: 금지 패턴
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

# DataFrame 변수명 목록 (코드 검증용)
DF_VARIABLE_NAMES = {
    'df', 'plot_df', 'merged_df', 'filtered_df', 'data',
    'high_vib', 'melt_df', 'temp_df', 'result_df', 'group_df'
}

# 데이터 품질 임계치
QUALITY_THRESHOLDS = {
    'missing_pct': 10,      # 결측치 비율 10% 이상 경고
    'outlier_pct': 2,       # 이상치 비율 2% 이상 경고
    'duplicate_pct': 1,     # 중복 행 비율 1% 이상 경고
}

# LLM 설정
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.15

# 코드 실행 타임아웃 (초)
CODE_EXECUTION_TIMEOUT = 60

# 자동 재시도 횟수
MAX_AUTO_RETRY = 2
