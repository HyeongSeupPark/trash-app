import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model('trashnet_model.h5', compile=False)

# 새 형식으로 저장
model.save('trash_classifier.keras')
print("변환 완료!")