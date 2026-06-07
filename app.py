import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import time
import tensorflow as tf
import numpy as np

# 모델 로드 (캐싱 처리)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_trash_model.keras",
        safe_mode=False,
        compile=False
    )

model = load_model()

# 카테고리 정의
CLASS_NAMES = ['general', 'glass', 'metal', 'paper', 'plastic', 'vinyl']
CLASS_KOREAN = {
    'paper': '종이류', 'plastic': '플라스틱', 'metal': '캔/금속',
    'glass': '유리', 'vinyl': '비닐', 'general': '일반쓰레기'
}

# 페이지 설정
st.set_page_config(page_title="CleanCycle", layout="wide", initial_sidebar_state="collapsed")

# CSS 스타일 시트 업데이트 (빈 박스 원인 제거 및 컬럼 자체를 카드로 변경)
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    /* 브라우저 강제 다크모드 반전 원천 차단 */
    :root { color-scheme: light !important; }
    
    /* 폰트 강제 적용 시 아이콘 클래스는 예외 처리 */
    *:not(.material-symbols-rounded):not([data-testid="stIconMaterial"]) { 
        font-family: 'Pretendard', sans-serif !important; 
    }

    /* 레이아웃 및 앱 전체 배경색 */
    .stApp, [data-testid="stAppViewContainer"], .main { 
        background-color: #f2f7f3 !important; 
    }
    
    header, .stDeployButton { visibility: hidden !important; height: 0 !important; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 5rem !important; }
    
    .main-title { font-size: 4.0rem !important; font-weight: 900 !important; color: #1b5e20 !important; letter-spacing: -3px !important; line-height: 1.1; margin-bottom: 5px !important; }
    .main-subtitle { font-size: 1.6rem !important; color: #2e7d32 !important; margin-bottom: 2.5rem !important; font-weight: 600; }

    /* ★ 핵심 수정: 빈 박스를 만들던 content-card 클래스 대신, 스트림릿 컬럼 자체를 예쁜 카드로 만듦 ★ */
    [data-testid="column"] {
        background-color: #ffffff !important; 
        padding: 35px !important; 
        border-radius: 30px !important; 
        box-shadow: 0 15px 40px rgba(27, 94, 32, 0.06) !important; 
        border: 1px solid #e8f5e9 !important; 
    }

    /* =========================================
       📸 업로더 UI (직관적인 버튼화)
       ========================================= */
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploadDropzone"] div[data-testid="stMarkdownContainer"],
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] svg { 
        display: none !important; 
    }
    
    [data-testid="stFileUploader"] { 
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploadDropzone"] { 
        border: 2px dashed #4caf50 !important; 
        border-radius: 20px !important; 
        background-color: #fafdfb !important; 
        padding: 50px 20px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #e8f5e9 !important;
        border-color: #2e7d32 !important;
    }

    [data-testid="stFileUploadDropzone"]::before {
        content: "📸";
        font-size: 3.5rem;
        margin-bottom: 15px;
    }
    [data-testid="stFileUploadDropzone"]::after {
        content: "이곳을 터치하여 사진 등록";
        display: block; 
        background-color: #2e7d32;
        color: #ffffff !important;
        font-size: 1.25rem;
        font-weight: 800;
        padding: 14px 28px;
        border-radius: 30px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }
    /* ========================================= */
    
    /* FAQ 섹션 스타일링 */
    .faq-title { font-size: 2.0rem !important; font-weight: 800 !important; color: #1b5e20 !important; margin-top: 3rem !important; margin-bottom: 1.5rem !important; display: flex; align-items: center; gap: 10px; }
    [data-testid="stExpander"] { border: 1px solid #e8f5e9 !important; border-radius: 16px !important; background-color: #ffffff !important; box-shadow: 0 4px 15px rgba(27, 94, 32, 0.05) !important; margin-bottom: 10px !important; overflow: hidden !important; }
    
    [data-testid="stExpander"] summary { padding: 15px !important; list-style: none !important; background-color: #ffffff !important; }
    [data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { font-size: 1.15rem !important; font-weight: 700 !important; color: #2e7d32 !important; -webkit-text-fill-color: #2e7d32 !important; opacity: 1 !important; }
    
    [data-testid="stExpanderDetails"] { background-color: #ffffff !important; }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] p, [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] span { color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-size: 1.05rem !important; line-height: 1.6 !important; font-weight: 500 !important; text-shadow: none !important; }
    
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"], [data-testid="stExpander"] summary .material-symbols-rounded, [data-testid="stExpander"] summary svg { display: none !important; font-size: 0 !important; color: transparent !important; }
    [data-testid="stExpander"] summary:hover { background-color: #f1f8f3 !important; }
    </style>
    """, unsafe_allow_html=True)

# 오리지널 SVG 아이콘 정의
camera_icon = '<svg viewBox="0 0 24 24" width="36" height="36" stroke="#1b5e20" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:12px; vertical-align:middle;"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>'
result_icon = '<svg viewBox="0 0 24 24" width="36" height="36" stroke="#1b5e20" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:12px; vertical-align:middle;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'

# 헤더 출력
st.markdown('<h1 class="main-title">♻️ CleanCycle</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">불필요한 단계 없이 바로 확인하는 스마트 분리배출</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# ─── 왼쪽: 사진 등록 영역 (빈 박스 원인 코드 삭제 완료) ───
with col1:
    st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:1.5rem;">{camera_icon}<span style="font-size:2.4rem; font-weight:900; color:#1b5e20;">사진 등록</span></div>', unsafe_allow_html=True)
    img_file = st.file_uploader(label="", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

# ─── 오른쪽: 분석 결과 영역 (빈 박스 원인 코드 삭제 완료) ───
with col2:
    st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:1.5rem;">{result_icon}<span style="font-size:2.4rem; font-weight:900; color:#1b5e20;">분석 결과</span></div>', unsafe_allow_html=True)
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)
        
        # --- 모델 전처리 및 추론 ---
        img_resized = img.convert("RGB").resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        pred = model.predict(img_array, verbose=0)[0]
        sorted_idx = np.argsort(pred)[::-1]

        top_idx = sorted_idx[0]
        top_class = CLASS_NAMES[top_idx]
        confidence = pred[top_idx] * 100

        st.markdown(
            f'<p style="color:#1b5e20; font-weight:800; font-size:1.2rem;">분석 완료! (신뢰도: {confidence:.1f}%)</p>',
            unsafe_allow_html=True
        )

        st.success(f"분류 결과: {CLASS_KOREAN[top_class]}")

        with st.expander("예측 확률 보기"):
            for idx in sorted_idx:
                st.write(f"{CLASS_NAMES[idx]} : {pred[idx] * 100:.2f}%")
    else:
        st.markdown("""
            <div style="height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #f9fbf9; border: 3px dashed #c8e6c9; border-radius: 30px;">
                <span style="font-size: 5rem; margin-bottom: 25px;">♻️</span>
                <p style="color: #2e7d32; font-size: 1.8rem; font-weight:800;">사진을 등록하면 가이드가 나타납니다</p>
            </div>
        """, unsafe_allow_html=True)

# ─── 하단: FAQ (자주 묻는 질문) 영역 ───
st.markdown('<p class="faq-title">💡 자주 묻는 질문 (FAQ)</p>', unsafe_allow_html=True)

with st.expander("🤔 AI가 어떤 원리로 쓰레기를 분류하나요?"):
    st.write("""
    CleanCycle은 수많은 쓰레기 이미지 데이터를 미리 학습한 **딥러닝(Deep Learning) 모델**을 사용합니다. 
    사진을 업로드하면 AI가 이미지의 특징(질감, 모양, 색상 등)을 추출하여 가장 확률이 높은 재질을 예측하여 알려줍니다.
    """)

with st.expander("📸 사진을 찍을 때 주의할 점이 있나요?"):
    st.write("""
    정확한 분석을 위해 쓰레기가 **화면 중앙**에 오도록 찍어주세요. 
    배경이 너무 복잡하거나 어두운 곳보다는, 밝은 곳에서 쓰레기 하나만 단독으로 찍을 때 AI의 인식률이 가장 높습니다.
    """)

with st.expander("🔄 여러 재질이 섞인 쓰레기는 어떻게 하나요?"):
    st.write("""
    현재 AI는 이미지에서 가장 면적을 많이 차지하는 대표 재질 하나를 우선적으로 판별합니다. 
    예를 들어 페트병의 경우 플라스틱으로 인식되지만, 실제 배출 시에는 **반드시 비닐 라벨을 뜯고 뚜껑과 분리해서 배출**해야 합니다. 추후 복합 재질에 대한 상세 가이드 기능이 업데이트될 예정입니다.
    """)

with st.expander("🔒 제가 올린 사진은 어딘가에 저장되나요?"):
    st.write("""
    아닙니다. 사용자가 업로드하거나 촬영한 사진은 **실시간 분석을 위해서만 일회성으로 사용**되며, 분석이 끝난 즉시 폐기됩니다. 개인정보나 위치 정보는 일절 수집하지 않으니 안심하고 사용하세요.
    """)