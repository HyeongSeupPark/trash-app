import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import time
import tensorflow as tf # 추가
import numpy as np       # 추가

# 모델 로드 (가이드 8페이지 기준) [cite: 285, 287]
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_trash_model.keras",
        safe_mode=False,
        compile=False
    )

model = load_model()

# 카테고리 정의 (가이드 5, 9페이지 기준) [cite: 126, 293]
CLASS_NAMES = ['general', 'glass', 'metal', 'paper', 'plastic', 'vinyl']
CLASS_KOREAN = {
    'paper': '종이류', 'plastic': '플라스틱', 'metal': '캔/금속',
    'glass': '유리', 'vinyl': '비닐', 'general': '일반쓰레기'
}

# 페이지 설정
st.set_page_config(page_title="CleanCycle", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif !important; }

    .main { background-color: #f2f7f3 !important; }
    header, .stDeployButton { visibility: hidden !important; height: 0 !important; }
    .block-container { padding-top: 1.5rem !important; }
    
    .main-title { font-size: 4.0rem !important; font-weight: 900 !important; color: #1b5e20 !important; letter-spacing: -3px !important; line-height: 1.1; margin-bottom: 5px !important; }
    .main-subtitle { font-size: 1.6rem !important; color: #2e7d32 !important; margin-bottom: 2.5rem !important; font-weight: 600; }

    /* 세그먼트 컨트롤 배경 */
    div[role="radiogroup"] {
        background-color: #e0e4e1 !important;
        padding: 5px !important;
        border-radius: 16px !important;
        display: flex !important;
        width: 100% !important;
        gap: 0 !important;
    }

    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"],
    div[role="radiogroup"] label div:first-child { display: none !important; }

    div[role="radiogroup"] label {
        flex: 1 1 0% !important; 
        cursor: pointer !important;
        height: 48px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        border-radius: 12px !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
        padding: 0 20px !important;
    }

    div[role="radiogroup"] label:nth-of-type(1)::after { content: "카메라 촬영"; font-size: 1.3rem; font-weight: 800; color: #88938a; }
    div[role="radiogroup"] label:nth-of-type(2)::after { content: "파일 업로드"; font-size: 1.3rem; font-weight: 800; color: #88938a; }

    /* ★ 흰색 바 이동 해결: 현재 선택된(Checked) 입력이 들어있는 라벨만 배경색 부여 ★ */
    div[role="radiogroup"] label:has(input:checked) {
        background-color: #ffffff !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08) !important;
    }
    div[role="radiogroup"] label:has(input:checked)::after {
        color: #1b5e20 !important;
    }

    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    
    [data-testid="stFileUploader"] { 
        border: 2px dashed #2e7d32 !important; border-radius: 20px !important; background-color: #fafdfb !important; padding: 15px !important;
    }
    [data-testid="stFileUploader"]::after {
        content: "이미지를 여기로 드래그하거나 클릭하세요";
        display: block; text-align: center; color: #2e7d32; font-weight: 700; font-size: 1.2rem; padding: 10px;
    }

    .content-card { background-color: #ffffff; padding: 45px !important; border-radius: 35px !important; box-shadow: 0 20px 60px rgba(27, 94, 32, 0.1) !important; border: 1px solid #e8f5e9 !important; }
    </style>
    """, unsafe_allow_html=True)

# 아이콘 및 레이아웃 생략
camera_icon = '<svg viewBox="0 0 24 24" width="36" height="36" stroke="#1b5e20" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:12px; vertical-align:middle;"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>'
result_icon = '<svg viewBox="0 0 24 24" width="36" height="36" stroke="#1b5e20" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:12px; vertical-align:middle;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'

st.markdown('<h1 class="main-title">♻️ CleanCycle</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">불필요한 단계 없이 바로 확인하는 스마트 분리배출</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:1.5rem;">{camera_icon}<span style="font-size:2.4rem; font-weight:900; color:#1b5e20;">사진 등록</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.35rem; font-weight:800; color:#1b5e20; margin-bottom:12px;">어떤 방식으로 올릴까요?</p>', unsafe_allow_html=True)
    
    # 텍스트 주입용 radio
    mode = st.radio("", ["cam", "file"], horizontal=True, label_visibility="collapsed")
    
    if mode == "cam":
        img_file = st.camera_input(label="", label_visibility="collapsed")
    else:
        img_file = st.file_uploader(label="", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:1.5rem;">{result_icon}<span style="font-size:2.4rem; font-weight:900; color:#1b5e20;">분석 결과</span></div>', unsafe_allow_html=True)
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)
        
        # --- 모델 추론 및 결과 처리 시작 ---
        # 1. 이미지 전처리 (가이드 9페이지 기준) [cite: 317, 318, 319, 320]
        # 이미지 전처리
        img_resized = img.convert("RGB").resize((224, 224))

        img_array = np.array(
            img_resized,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


# 예측
        pred = model.predict(
            img_array,
            verbose=0
        )[0]

        sorted_idx = np.argsort(pred)[::-1]

        top_idx = sorted_idx[0]
        top_class = CLASS_NAMES[top_idx]
        confidence = pred[top_idx] * 100

        st.markdown(
            f'<p style="color:#1b5e20; font-weight:800; font-size:1.2rem;">분석 완료! (신뢰도: {confidence:.1f}%)</p>',
            unsafe_allow_html=True
        )

        st.success(
            f"분류 결과: {CLASS_KOREAN[top_class]}"
        )

# 디버깅용 확률 표시
        with st.expander("예측 확률 보기"):

            for idx in sorted_idx:

                st.write(
                    f"{CLASS_NAMES[idx]} : {pred[idx] * 100:.2f}%"
                )
        
        # 3. 분리수거 가이드 표시 (가이드 10페이지 기준) [cite: 327, 331]
        # (별도의 guides.py가 있다면 import해서 사용)
        # st.markdown(get_guide(top_class)) 
        # --- 모델 추론 끝 ---
    else:
        st.markdown("""
            <div style="height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #f9fbf9; border: 3px dashed #c8e6c9; border-radius: 30px;">
                <span style="font-size: 5rem; margin-bottom: 25px;">♻️</span>
                <p style="color: #2e7d32; font-size: 1.8rem; font-weight:800;">사진을 등록하면 가이드가 나타납니다</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
