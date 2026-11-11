"""
CleanCycle 모델 v2 학습 스크립트

기존 best_trash_model.keras(TrashNet + 웹 비닐로 학습)에서 출발해,
실사진 데이터(RealWaste, TACO, 테이크아웃컵)를 더한 combined 데이터셋으로 추가 학습한다.
모델 구조와 입력 형식(224x224, 0~255 RGB, 모델 안에서 preprocess_input)은 그대로라 app.py 수정 없이 교체 가능.

사용법:
    python model_train_v2.py <combined_dir> <출력.keras> [에폭수]
combined_dir 구조: train/<클래스>/*.jpg, val/<클래스>/*.jpg  (클래스: general glass metal paper plastic vinyl)
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

COMBINED, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
# 출발 모델과 학습률은 환경변수로 바꿀 수 있음 (기본: 기존 배포 모델, 5e-5)
BASE_MODEL = os.environ.get("BASE_MODEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_trash_model.keras")
LR = float(os.environ.get("LR", "5e-5"))
IMG_SIZE, BATCH = (224, 224), 32
CLASSES = ["general", "glass", "metal", "paper", "plastic", "vinyl"]

tf.keras.utils.set_random_seed(42)


def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(COMBINED, split),
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=42,
    )


HAS_VAL = os.path.isdir(os.path.join(COMBINED, "val"))
train_ds = load("train", True)
val_ds = load("val", False) if HAS_VAL else None

labels = np.concatenate([y.numpy() for _, y in train_ds])
counts = np.bincount(labels, minlength=len(CLASSES))
# sklearn의 class_weight="balanced"와 같은 식: 전체 장수 / (클래스 수 * 클래스별 장수)
class_weight = {i: float(len(labels) / (len(CLASSES) * c)) for i, c in enumerate(counts)}
print("train 장수:", dict(zip(CLASSES, np.bincount(labels, minlength=6).tolist())))
print("class weight:", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
if HAS_VAL:
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 기존 모델에서 출발 (같은 구조, MobileNetV2 마지막 30층 + 분류 헤드가 학습 대상)
model = tf.keras.models.load_model(BASE_MODEL, safe_mode=False, compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if HAS_VAL:
    # 검증셋으로 학습 방법(에폭 수, 학습률 감소 시점)을 정하는 단계
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(OUT, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(OUT + ".log.csv"),
    ]
else:
    # 최종 학습: 전체 데이터, 검증셋 없음 -> v2 학습에서 확인한 일정을 그대로 재현
    # (1~10에폭 5e-5, 11에폭부터 1.5e-5, 총 15에폭)
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lambda e, lr: 5e-5 if e < 10 else 1.5e-5, verbose=1),
        tf.keras.callbacks.CSVLogger(OUT + ".log.csv"),
    ]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks)
if not HAS_VAL:
    model.save(OUT)
print("저장:", OUT)
