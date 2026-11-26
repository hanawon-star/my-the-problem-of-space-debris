import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import io
import os

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="우주 쓰레기와 군사 위성 경제성 분석",
    page_icon="🛰️",
    layout="wide"
)

# 브라우저 자동 번역기로 인한 React DOM(removeChild) 충돌 방지
st.markdown(
    """
    <meta name="google" content="notranslate">
    <style>
        .stApp { translate: no; }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. 한글 폰트 강제 등록 및 설정 (리눅스/윈도우/맥 완벽 대응)
def get_korean_font():
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return font_prop
    plt.rcParams['axes.unicode_minus'] = False
    return None

kor_font = get_korean_font()

# 3. 403 Forbidden 우회 및 로컬/원격 CSV 겸용 로더
def fetch_csv_with_headers(url):
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
    )
    with urllib.request.urlopen(req) as response:
        csv_text = response.read().decode('utf-8-sig')
    return pd.read_csv(io.StringIO(csv_text))

@st.cache_data(ttl=3600, show_spinner=False)
def load_owid_data():
    if os.path.exists("low-earth-orbits-objects.csv") and os.path.exists("cost-space-launches-low-earth-orbit.csv"):
        df_obj = pd.read_csv("low-earth-orbits-objects.csv", encoding="utf-8-sig")
        df_cost = pd.read_csv("cost-space-launches-low-earth-orbit.csv", encoding="utf-8-sig")
    else:
        url_objects = "https://ourworldindata.org/grapher/low-earth-orbits-objects.csv?v=1&csvType=full&useColumnShortNames=true"
        url_cost = "https://ourworldindata.org/grapher/cost-space-launches-low-earth-orbit.csv?v=1&csvType=full&useColumnShortNames=true"
        df_obj = fetch_csv_with_headers(url_objects)
        df_cost = fetch_csv_with_headers(url_cost)
    return df_obj, df_cost

# 메인 타이틀 및 소개
st.title("🛰️ 우주 쓰레기 증가가 군사 위성의 경제성에 미치는 영향")
st.markdown("""
**Our World in Data (OWID)**의 저궤도 우주물체 데이터와 발사 단가 데이터를 결합하여, 
우주 쓰레기 급증이 군사 위성의 **방호 설계·회피 기동 연료 부담 및 경제적 기회비용**에 미치는 영향을 분석합니다.
""")

# 데이터 불러오기
try:
    df_obj, df_cost = load_owid_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 4. 사이드바: 시뮬레이션 변수 설정
st.sidebar.header("⚙️ 분석 모델 파라미터 설정")
start_year = st.sidebar.slider("분석 시작 연도", min_value=1965, max_value=2010, value=1980, step=5)
base_mass = st.sidebar.slider("기준 군사 정찰위성 질량 (kg)", min_value=500, max_value=5000, value=1500, step=100)
max_penalty_ratio = st.sidebar.slider(
    "최대 혼잡 시 방호·회피 추가 질량 비율 (%)",
    min_value=10, max_value=50, value=30, step=5,
    help="우주 쓰레기가 최대치일 때 방호 장갑 및 회피 연료로 추가되는 질량 비율"
) / 100.0

# 5. 데이터 전처리 (가로형/세로형 CSV 구조 자동 인식)
df_obj.columns = df_obj.columns.str.strip()
year_col_obj = [c for c in df_obj.columns if 'year' in c.lower()][0]

if 'Entity' in df_obj.columns:
    obj_val_col = df_obj.columns[-1]
    debris_types = ['Debris', 'Rocket bodies']
    df_debris = (
        df_obj[df_obj['Entity'].isin(debris_types)]
        .groupby(year_col_obj)[obj_val_col]
        .sum()
        .reset_index()
        .rename(columns={year_col_obj: 'Year', obj_val_col: 'Debris_Count'})
    )
else:
    debris_cols = [c for c in df_obj.columns if 'debris' in c.lower() or 'rocket' in c.lower()]
    if not debris_cols:
        debris_cols = [c for c in df_obj.columns if c != year_col_obj and pd.api.types.is_numeric_dtype(df_obj[c])]
    df_obj['Debris_Count'] = df_obj[debris_cols].fillna(0).sum(axis=1)
    df_debris = df_obj[[year_col_obj, 'Debris_Count']].rename(columns={year_col_obj: 'Year'})

df_cost.columns = df_cost.columns.str.strip()
year_col_cost = [c for c in df_cost.columns if 'year' in c.lower()][0]

if 'Entity' in df_cost.columns:
    cost_val_col = df_cost.columns[-1]
    df_launch_cost = (
        df_cost.groupby(year_col_cost)[cost_val_col]
        .mean()
        .reset_index()
        .rename(columns={year_col_cost: 'Year', cost_val_col: 'Cost_Per_Kg'})
    )
else:
    cost_cols = [c for c in df_cost.columns if c != year_col_cost and pd.api.types.is_numeric_dtype(df_cost[c])]
    df_cost['Cost_Per_Kg'] = df_cost[cost_cols].mean(axis=1)
    df_launch_cost = df_cost[[year_col_cost, 'Cost_Per_Kg']].rename(columns={year_col_cost: 'Year'})

df_debris['Year'] = pd.to_numeric(df_debris['Year'], errors='coerce')
df_launch_cost['Year'] = pd.to_numeric(df_launch_cost['Year'], errors='coerce')

df_merged = pd.merge(df_debris, df_launch_cost, on='Year', how='inner').dropna()
df_merged = df_merged[df_merged['Year'] >= start_year].sort_values('Year').reset_index(drop=True)

if df_merged.empty:
    st.warning("선택한 시작 연도 이후로 두 데이터가 겹치는 연도가 없습니다. 시작 연도를 낮춰보세요.")
    st.stop()

# 경제성 저하 지표 산출
debris_norm = df_merged['Debris_Count'] / df_merged['Debris_Count'].max()
df_merged['Extra_Mass_kg'] = base_mass * (0.05 + max_penalty_ratio * debris_norm)
df_merged['Debris_Penalty_Cost_M'] = (df_merged['Extra_Mass_kg'] * df_merged['Cost_Per_Kg']) / 1_000_000
df_merged['Payload_Efficiency_Pct'] = (base_mass / (base_mass + df_merged['Extra_Mass_kg'])) * 100

# 6. 핵심 요약 지표(KPI) 카드 표시
latest = df_merged.iloc[-1]
earliest = df_merged.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label=f"누적 저궤도 우주 쓰레기 ({int(latest['Year'])}년)",
    value=f"{int(latest['Debris_Count']):,} 개",
    delta=f"{int(latest['Debris_Count'] - earliest['Debris_Count']):,} 개 ({int(earliest['Year'])}년 대비)",
    delta_color="inverse"
)
col2.metric(
    label="평균 저궤도 발사 단가",
    value=f"${latest['Cost_Per_Kg']:,.0f} / kg",
    delta=f"${latest['Cost_Per_Kg'] - earliest['Cost_Per_Kg']:,.0f} / kg",
    delta_color="inverse"
)
col3.metric(
    label="위성 1기당 방호·회피 추가 질량",
    value=f"{latest['Extra_Mass_kg']:.1f} kg",
    delta=f"+{latest['Extra_Mass_kg'] - earliest['Extra_Mass_kg']:.1f} kg 증가",
    delta_color="inverse"
)
col4.metric(
    label="임무 탑재체 질량 효율성",
    value=f"{latest['Payload_Efficiency_Pct']:.1f} %",
    delta=f"{latest['Payload_Efficiency_Pct'] - earliest['Payload_Efficiency_Pct']:.1f} %p 감소"
)

st.divider()

# 7. 시각화 섹션 (폰트 속성 명시적 적용으로 깨짐 원천 차단)
font_kwargs = {'fontproperties': kor_font} if kor_font else {}
tab1, tab2, tab3 = st.tabs(["📊 핵심 비교 그래프", "📈 상관관계 및 효율성 분석", "📋 병합 원시 데이터"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(11, 4.8), dpi=130)
    color_debris = '#D9534F'
    color_cost = '#2B5C8F'

    line1 = ax1.plot(df_merged['Year'], df_merged['Debris_Count'], color=color_debris, linewidth=2.5, marker='o', markersize=4, label='우주 쓰레기 수 (개)')
    ax1.set_xlabel('연도 (Year)', fontsize=11, **font_kwargs)
    ax1.set_ylabel('누적 우주 쓰레기 수 (Debris + Rocket Bodies)', color=color_debris, fontsize=11, fontweight='bold', **font_kwargs)
    ax1.tick_params(axis='y', labelcolor=color_debris)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax1_sub = ax1.twinx()
    line2 = ax1_sub.plot(df_merged['Year'], df_merged['Cost_Per_Kg'], color=color_cost, linewidth=2, linestyle='--', marker='s', markersize=4, label='kg당 발사 단가 ($/kg)')
    ax1_sub.set_ylabel('저궤도 발사 단가 ($/kg)', color=color_cost, fontsize=11, fontweight='bold', **font_kwargs)
    ax1_sub.tick_params(axis='y', labelcolor=color_cost)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, prop=kor_font)
    ax1.set_title('저궤도 우주 쓰레기 급증 vs kg당 발사 단가 추이', fontsize=13, fontweight='bold', pad=12, **font_kwargs)
    
    st.pyplot(fig1)
    plt.close(fig1)
    st.info("💡 **해석 포인트:** 재사용 로켓 등장으로 kg당 발사 단가(파란 점선)는 하락했으나, 우주 쓰레기(빨간 실선)가 폭증하면서 군사 위성은 발사 비용 절감 효과를 온전히 누리지 못하고 방호 및 회피 기동에 추가 자원을 소모하게 됩니다.")

with tab2:
    fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

    sc = ax2_left.scatter(
        df_merged['Debris_Count'], df_merged['Extra_Mass_kg'],
        c=df_merged['Year'], cmap='viridis', s=65, edgecolors='black', alpha=0.85
    )
    z = np.polyfit(df_merged['Debris_Count'], df_merged['Extra_Mass_kg'], 1)
    p = np.poly1d(z)
    ax2_left.plot(df_merged['Debris_Count'], p(df_merged['Debris_Count']), color='#D9534F', linestyle='--', linewidth=2, label='선형 회귀 추세선')
    ax2_left.set_title('우주 쓰레기 수 vs 방호·회피 추가 질량', fontsize=12, fontweight='bold', **font_kwargs)
    ax2_left.set_xlabel('저궤도 우주 쓰레기 수 (개)', **font_kwargs)
    ax2_left.set_ylabel('위성 1기당 추가 질량 (kg)', **font_kwargs)
    ax2_left.grid(True, linestyle='--', alpha=0.4)
    ax2_left.legend(prop=kor_font)
    cbar = plt.colorbar(sc, ax=ax2_left)
    cbar.set_label('연도', **font_kwargs)

    ax2_right.plot(df_merged['Year'], df_merged['Payload_Efficiency_Pct'], color='#2E8B57', linewidth=2.5, marker='d', markersize=4)
    ax2_right.fill_between(df_merged['Year'], df_merged['Payload_Efficiency_Pct'], df_merged['Payload_Efficiency_Pct'].min() - 2, color='#2E8B57', alpha=0.15)
    ax2_right.set_title('연도별 군사 위성 핵심 임무 장비 탑재 효율(%) 변화', fontsize=12, fontweight='bold', **font_kwargs)
    ax2_right.set_xlabel('연도 (Year)', **font_kwargs)
    ax2_right.set_ylabel('핵심 임무 탑재 비율 (%)', **font_kwargs)
    ax2_right.set_ylim(df_merged['Payload_Efficiency_Pct'].min() - 2, 100)
    ax2_right.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with tab3:
    st.subheader("연도별 병합 데이터 테이블")
    display_df = df_merged.rename(columns={
        'Year': '연도',
        'Debris_Count': '우주 쓰레기 수(개)',
        'Cost_Per_Kg': '평균 발사 단가($/kg)',
        'Extra_Mass_kg': '방호·회피 추가 질량(kg)',
        'Debris_Penalty_Cost_M': '추가 발사 비용($M)',
        'Payload_Efficiency_Pct': '임무 탑재 효율(%)'
    })
    st.dataframe(display_df.style.format({
        '연도': '{:.0f}',
        '우주 쓰레기 수(개)': '{:,.0f}',
        '평균 발사 단가($/kg)': '${:,.1f}',
        '방호·회피 추가 질량(kg)': '{:,.1f} kg',
        '추가 발사 비용($M)': '${:,.2f}M',
        '임무 탑재 효율(%)': '{:.1f}%'
    }), width="stretch")

    csv_data = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 분석 결과 CSV 다운로드",
        data=csv_data,
        file_name="space_debris_military_economics_analysis.csv",
        mime="text/csv"
    )
