import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pandas as pd
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 로드
data = pd.read_csv('./diabetes.csv', sep=',')
X = data.values[:, 0:8]
y = data.values[:, 8]

# 스케일링
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 모델 구성
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs, output)

# 컴파일 및 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# Keras 모델 저장
model.save('pima_model.keras')
print("모델이 성공적으로 저장되었습니다.")