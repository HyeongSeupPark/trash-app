import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.utils.class_weight import compute_class_weight

# ==================================================
# 경로 설정 (VSCode Windows용)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "trash",
    "trash"
)

print("데이터셋 경로:", DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"데이터셋 폴더를 찾을 수 없음:\n{DATASET_DIR}"
    )

# ==================================================
# 설정
# ==================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ==================================================
# vinyl 증강
# ==================================================

vinyl_dir = os.path.join(DATASET_DIR, "vinyl")

if not os.path.exists(vinyl_dir):
    raise FileNotFoundError(
        f"vinyl 폴더를 찾을 수 없음:\n{vinyl_dir}"
    )

# 이전 증강 이미지 삭제
for file in os.listdir(vinyl_dir):
    if file.startswith("aug_"):
        os.remove(os.path.join(vinyl_dir, file))

vinyl_files = [
    f for f in os.listdir(vinyl_dir)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

print(f"원본 vinyl 수: {len(vinyl_files)}")

TARGET_VINYL = 200

augment_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.2),
])

current_count = len(vinyl_files)
idx = 0

while current_count < TARGET_VINYL:

    img_name = random.choice(vinyl_files)

    img_path = os.path.join(
        vinyl_dir,
        img_name
    )

    img = tf.keras.utils.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img = tf.keras.utils.img_to_array(img)

    img = tf.expand_dims(img, 0)

    aug_img = augment_layer(
        img,
        training=True
    )

    aug_img = tf.squeeze(aug_img, 0)

    save_path = os.path.join(
        vinyl_dir,
        f"aug_{idx}.jpg"
    )

    tf.keras.utils.save_img(
        save_path,
        aug_img
    )

    current_count += 1
    idx += 1

print(f"증강 후 vinyl 수: {current_count}")

# ==================================================
# 데이터셋 로드
# ==================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\n클래스 목록")
print(class_names)

# ==================================================
# class weight
# ==================================================

labels = []

for _, y in train_ds:
    labels.extend(y.numpy())

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = {
    i: w
    for i, w in enumerate(weights)
}

print("\nClass Weight")
print(class_weight_dict)

# ==================================================
# 성능 최적화
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==================================================
# 데이터 증강
# ==================================================

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# ==================================================
# MobileNetV2
# ==================================================

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==================================================
# 모델 생성
# ==================================================

inputs = tf.keras.Input(
    shape=(224,224,3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

# ==================================================
# 컴파일
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# 콜백
# ==================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        "best_trash_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ==================================================
# 학습
# ==================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ==================================================
# 저장
# ==================================================

model.save("trash_classifier.keras")

print("\n학습 완료")
print("모델 저장 완료")