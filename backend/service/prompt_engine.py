# -*- coding: utf-8 -*-
"""프롬프트 엔진 (SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, PromptEngine)"""
from typing import Dict, List, Optional

from ..utils.constants import QUALITY_THRESHOLDS

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
fig.update_layout(dragmode='zoom')
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
    """프롬프트 생성 엔진"""

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

    @staticmethod
    def build_insight_prompt(user_request: str, dataset_info: Dict, datasets_info: Optional[Dict] = None,
                             executed_code: Optional[str] = None, analysis_summary: Optional[str] = None) -> str:
        """분석 결과 요약 프롬프트 - 실행된 코드와 결과를 기반으로 인사이트 생성"""
        # 멀티 데이터셋일 때: 사용자 요청에 언급된 변수가 속한 데이터셋을 찾아 해당 통계 사용
        matched_datasets = {}
        if datasets_info and len(datasets_info) > 1:
            request_lower = user_request.lower()
            for ds_name, ds_info in datasets_info.items():
                cols = ds_info.get('columns', [])
                match_count = sum(1 for col in cols if col.lower() in request_lower)
                if ds_name.lower() in request_lower:
                    match_count += 1
                if match_count > 0:
                    matched_datasets[ds_name] = (ds_info, match_count)

        # 코드 섹션
        code_section = ""
        if executed_code:
            code_section = f"""
실행된 분석 코드:
```python
{executed_code}
```
"""

        # 분석 결과 섹션
        result_section = ""
        if analysis_summary:
            result_section = f"""
분석 결과 요약:
{analysis_summary}
"""

        # 매칭된 데이터셋이 여러 개면 모두 포함, 없으면 primary 사용
        if len(matched_datasets) >= 2:
            all_stats_sections = []
            all_quality_alerts = []
            for ds_name, (ds_info, _) in matched_datasets.items():
                section = f"\n[{ds_name}]\nShape: {ds_info['shape']}\nColumns: {ds_info['columns']}\n"
                section += f"Statistical Summary:\n{ds_info['describe_str']}\n"
                section += f"Missing Values: {ds_info.get('missing_str', '결측치 없음')}\n"
                section += f"Correlation: {ds_info.get('corr_str', '상관계수 정보 없음')}\n"
                section += f"Outliers (IQR): {ds_info.get('outlier_str', '이상치 없음')}\n"
                all_stats_sections.append(section)

                for col, v in ds_info.get('missing_detail', {}).items():
                    if v['pct'] >= QUALITY_THRESHOLDS['missing_pct']:
                        all_quality_alerts.append(f"[{ds_name}] {col} 결측치 {v['pct']}%")
                total_rows = ds_info['shape'][0]
                for col, cnt in ds_info.get('outlier_detail', {}).items():
                    pct = round(cnt / total_rows * 100, 1) if total_rows > 0 else 0
                    if pct >= QUALITY_THRESHOLDS['outlier_pct']:
                        all_quality_alerts.append(f"[{ds_name}] {col} 이상치 {cnt}건 ({pct}%)")

            quality_str = "\n".join(all_quality_alerts) if all_quality_alerts else "데이터 품질 양호"
            datasets_section = "\n".join(all_stats_sections)
            ds_names = ", ".join(matched_datasets.keys())

            return f"""You are a sensor data analyst. Provide a concise 2-4 sentence insight.
Use Korean terminology (백분위수, 평균, 중앙값, 표준편차, 상관계수, 이상치).
When statistical significance is available (p-value), ALWAYS mention it explicitly.

CRITICAL: This request involves MULTIPLE datasets: [{ds_names}].
Use statistics from ALL relevant datasets below. Do NOT say data is unavailable when it is provided.

User Request: "{user_request}"
{code_section}{result_section}
{datasets_section}

Data Quality Alerts:
{quality_str}

Respond in Korean. Use actual numbers from the relevant datasets.
If there are quality alerts, mention their potential impact.
- 코드에서 계산된 주요 수치(상관계수, 변화율, 통계값 등)를 구체적으로 언급
- 수치의 의미와 해석을 쉽게 설명
- 다중공선성, 급변 구간, 이상 패턴 등 발견된 특이사항 설명"""
        else:
            # 단일 데이터셋 매칭 또는 매칭 없음
            if matched_datasets:
                active_ds_name, (active_info, _) = max(matched_datasets.items(), key=lambda x: x[1][1])
            else:
                active_info = dataset_info
                active_ds_name = "Primary"

            # 품질 알림 메시지 동적 생성
            quality_alerts = []
            missing_detail = active_info.get('missing_detail', {})
            for col, v in missing_detail.items():
                if v['pct'] >= QUALITY_THRESHOLDS['missing_pct']:
                    quality_alerts.append(f"{col} 컬럼 결측치 {v['pct']}% — 분석 신뢰도에 영향 가능")

            outlier_detail = active_info.get('outlier_detail', {})
            total_rows = active_info['shape'][0]
            for col, cnt in outlier_detail.items():
                pct = round(cnt / total_rows * 100, 1) if total_rows > 0 else 0
                if pct >= QUALITY_THRESHOLDS['outlier_pct']:
                    quality_alerts.append(f"{col} 컬럼 이상치 {cnt}건 ({pct}%) — 분포 왜곡 가능")

            dup_pct = active_info.get('dup_pct', 0)
            if dup_pct >= QUALITY_THRESHOLDS['duplicate_pct']:
                quality_alerts.append(f"중복 행 {active_info.get('dup_count', 0)}건 ({dup_pct}%) — 밀도/빈도 왜곡 가능")

            quality_str = "\n".join(quality_alerts) if quality_alerts else "데이터 품질 양호"
            stat_tests_str = active_info.get('stat_tests_str', '통계 검정 없음')

            return f"""You are a sensor data analyst. Provide a concise 2-4 sentence insight.
Focus on: trends, anomalies, correlations, missing data patterns, and outliers.
Use Korean terminology (백분위수, 평균, 중앙값, 표준편차, 상관계수, 이상치).
When statistical significance is available (p-value), ALWAYS mention it explicitly.

CRITICAL: Base your insight ONLY on variables mentioned in the user request.
Do NOT mention variables from other datasets that are unrelated to the request.
The relevant dataset for this request is: [{active_ds_name}]

User Request: "{user_request}"
{code_section}{result_section}
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
- 코드에서 계산된 주요 수치(상관계수, 변화율, 통계값 등)를 구체적으로 언급
- 수치의 의미와 해석을 쉽게 설명
- 다중공선성(>0.9), 급변 구간, 이상 패턴 등 발견된 특이사항 설명"""

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
