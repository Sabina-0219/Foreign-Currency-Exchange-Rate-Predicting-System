import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
plt.rcParams["font.sans-serif"] = "mingliu"  #繪圖中文字型
plt.rcParams["axes.unicode_minus"] = False
pd.options.mode.chained_assignment = None  #取消顯示pandas資料重設警告

filename = '8514FC84-4B9F-4D68-9429-AD7BA297AFF8.csv'
df = pd.read_csv(filename, encoding='big5') #以pandas讀取檔案
USD=pd.DataFrame(df['美元／新台幣'])
data_all = np.array(USD).astype(float)    # 轉為浮點型別矩陣

scaler = MinMaxScaler() # 建立 MinMaxScaler 物件
data_all = scaler.fit_transform(data_all)  # 將數據縮放為 0~1之間

TIME_STEPS=7 #讀取後面7天的資料
data = []
# data 資料共有 (2650-7)=2643筆
for i in range(len(data_all) - TIME_STEPS):
    # 每筆 data 資料有 8 欄
    data.append(data_all[i: i + TIME_STEPS + 1])

reshaped_data = np.array(data).astype('float64')
x = reshaped_data[:, :-1] # 第 1至第7個欄位為 特徵
y = reshaped_data[:, -1]  # 第 8個欄位為 標籤

split=0.8
split_boundary = int(reshaped_data.shape[0] * split)
train_x = x[: split_boundary] # 前 80% 為 train 的特徵
test_x = x[split_boundary:]   # 最後 20% 為 test 的特徵

train_y = y[: split_boundary] # 前 80% 為 train 的 label
test_y = y[split_boundary:]   # 最後 20% 為 test 的 label

# 建立 LSTM 模型
model = Sequential()
# 隱藏層：256 個神經元，input_shape：(7,1)
INPUT_SIZE=1
model.add(LSTM(input_shape=(TIME_STEPS,INPUT_SIZE),units=256,unroll=False))
model.add(Dropout(0.2))   #建立拋棄層，拋棄比例為20%
model.add(Dense(units=1)) #輸出層：1 個神經元
# model.summary() #顯示模型

#定義訓練方式
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

#訓練資料保留 10%作驗證,訓練100次、每批次讀取200筆資料，顯示簡易訓練過程
model.fit(train_x, train_y, batch_size=200, epochs=100, validation_split=0.1,verbose=2)

# 以 predict方法預測，返回值是數值
predict = model.predict(test_x)
predict = np.reshape(predict, (predict.size, )) #轉換為1維矩陣
predict_y = scaler.inverse_transform([[i] for i in predict]) # 數據還原
test_y = scaler.inverse_transform(test_y)  # 數據還原

# 以 matplotlib 繪圖
plt.plot(predict_y, 'b:') #預測
plt.plot(test_y, 'r-')  #美金匯率
plt.legend(['預測', '美金匯率'])
plt.show()
