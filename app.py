import streamlit as st
from PIL import Image
import numpy as np
from guides import get_guide

# --------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="분리수거 도우미",
    page_icon="♻️",
    layout="centered"
)

# --------------------------------------------------
# 카테고리 정의
# --------------------------------------------------
CLASS_NAMES = ['general', 'glass', 'metal', 'paper', 'plastic', 'vinyl']
CLASS_KOREAN = {
    'paper':   '📄 종이류',
    'plastic': '🧴 플라스틱',
    'metal':   '🥫 캔/금속',
    'glass':   '🍶 유리',
    'vinyl':   '🛍️ 비닐',
    'general': '🗑️ 일반쓰레기'
}

# --------------------------------------------------
# 모델 로드 함수 (지금은 더미, 나중에 실제 모델로 교체)
# --------------------------------------------------
@st.cache_resource
def load_model():
    # TODO: 모델 학습 완료 후 아래 주석 해제
    # import tensorflow as tf
    # return tf.keras.models.load_model('trash_classifier.h5')
    return None  # 지금은 None 반환

model = load_model()

def predict(img: Image.Image):
    """
    이미지를 받아 카테고리와 신뢰도를 반환.
    모델 없을 때는 랜덤 결과 반환 (UI 테스트용).
    """
    if model is None:
        # 모델 없을 때: 랜덤 더미 결과 (UI 확인용)
        idx = np.random.randint(0, len(CLASS_NAMES))
        confidence = np.random.uniform(70, 99)
        return CLASS_NAMES[idx], confidence
    else:
        # 모델 있을 때: 실제 추론
        import tensorflow as tf
        img_resized = img.resize((224, 224)).convert('RGB')
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0]
        idx = np.argmax(pred)
        return CLASS_NAMES[idx], float(pred[idx]) * 100

# --------------------------------------------------
# UI 시작
# --------------------------------------------------
st.title("♻️ 분리수거 도우미")
st.write("쓰레기 사진을 찍거나 업로드하면 **분리수거 방법**을 알려드려요.")

if model is None:
    st.warning("⚠️ 현재 테스트 모드입니다. 모델 없이 랜덤 결과를 보여줍니다.")

st.divider()

# --------------------------------------------------
# 입력 방식 선택
# --------------------------------------------------
option = st.radio(
    "입력 방식을 선택하세요",
    ["📷 카메라로 촬영", "🖼️ 사진 파일 업로드"],
    horizontal=True
)

img = None

if option == "📷 카메라로 촬영":
    img_file = st.camera_input("쓰레기를 화면 가운데에 놓고 촬영하세요")
    if img_file:
        img = Image.open(img_file)

else:
    img_file = st.file_uploader(
        "사진을 업로드하세요",
        type=['jpg', 'jpeg', 'png'],
        help="JPG, PNG 파일 지원"
    )
    if img_file:
        img = Image.open(img_file)

# --------------------------------------------------
# 추론 및 결과 표시
# --------------------------------------------------
if img:
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 입력 이미지")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("🔍 분석 결과")

        with st.spinner("분석 중..."):
            category, confidence = predict(img)

        korean_name = CLASS_KOREAN[category]
        st.markdown(f"### {korean_name}")

        # 신뢰도 바
        st.write(f"신뢰도: **{confidence:.1f}%**")
        st.progress(int(confidence))

        # 신뢰도에 따른 메시지
        if confidence >= 80:
            st.success("높은 확신도로 분류했어요.")
        elif confidence >= 60:
            st.warning("불확실할 수 있어요. 아래 가이드를 참고하세요.")
        else:
            st.error("확신도가 낮아요. 직접 확인해보세요.")

    st.divider()

    # 분리수거 가이드
    st.subheader("📋 분리수거 가이드")
    guide_text = get_guide(category)
    st.markdown(guide_text)

    st.divider()
    st.caption("정보 출처: 환경부 분리배출 가이드 | 이 앱은 참고용이며 지자체마다 다를 수 있습니다.")