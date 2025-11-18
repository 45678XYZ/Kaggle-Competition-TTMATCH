# Kaggle-Competition-TTMATCH

### 方法與流程

#### 1. 資料與特徵工程

- 讀取 `train.csv` 與 `test.csv`
- 對每筆資料做基本特徵工程
  - 新增 `scoreDiff`（雙方得分差）
  - 新增 `is_server`（此拍是否由該選手發球）

---

#### 2. 特徵編碼與資料切分

- 將欄位分成兩類
  - 分類特徵：`sex`, `strickId`, `handId`, `strengthId`, `spinId`, `pointId`, `actionId`, `positionId`
  - 數值特徵：`strickNumber`, `scoreDiff`, `is_server`
- 對每個分類特徵使用 `LabelEncoder`
  - 用「訓練集 + 測試集」的所有值一起 `fit`，避免測試集中出現未知類別
  - 轉換 `train_df` 與 `test_df`
  - 記錄 `n_action_classes` 與 `n_point_classes`，作為輸出層維度
- 依 `rally_uid` 切訓練 / 驗證：
  - 取得所有 `rally_uid`，前 80% 當訓練集，後 20% 當驗證集

---

#### 3. 建立序列樣本

- 對每個訓練集 rally
  - 依 `strickNumber` 排序，得到一組擊球序列
  - 若該 rally 長度 < 2，略過
  - 只取最後 `K = 6` 個 time steps 來產生樣本，減少高度重複樣本
  - 對每一個可預測位置 `i`
    - 輸入序列 `features_df = rally_df.iloc[:i]`（前 i 拍）。
    - 目標拍 `target_df = rally_df.iloc[i]`（第 i+1 拍）。
    - 將 `features_df` 的各個分類欄位序列加入 `X_train[col]`。
    - 將 `features_df` 的數值特徵矩陣加入 `X_train["numerical_input"]`。
    - 標籤：
      - `y_train["action_output"] = target_df["actionId"]`
      - `y_train["point_output"] = target_df["pointId"]`
      - `y_train["rally_output"] = rally_df.iloc[-1]["serverGetPoint"]`（整條 rally 最終誰得分）
- 驗證集 `X_val`, `y_val` 的處理方式相同，只是使用 `val_rallies`。

---

#### 4. 固定序列長度（Padding）

- 設定 `sequence_length = 64`。
- 對訓練與驗證資料中每條序列做 `pad_sequences`：
  - `maxlen = 64`、`padding = "pre"`、`truncating = "pre"`。
  - 分類特徵用 0 補，數值特徵用 0.0 補。
- 將 `y_train` / `y_val` 的三個輸出轉為 `numpy array`，便於丟入 Keras 模型。

---

#### 5. 模型結構：多輸出 Transformer

- 為每個分類特徵建立：
  - `Input(shape=(sequence_length,), name=col)`
  - 對應的 `Embedding(input_dim=vocab_size, output_dim=embedding_dim, name=f"{col}_embedding")`
- 為數值特徵建立：
  - `Input(shape=(sequence_length, len(numerical_cols)), name="numerical_input")`
- 將所有分類特徵的 embedding 在特徵軸上串接成 `concatenated_embeddings`。
- 再把 `concatenated_embeddings` 與 `numerical_input` 串成 `concatenated_features`。
- 用一層 `Dense(total_dim, activation="relu")` 做 `feature_projection`，統一維度為 `total_dim = embedding_dim * len(categorical_cols)`。

---

#### 6. Transformer Block 與三個任務頭

- 將投影後的序列輸入 Multi-Head Self-Attention：
  - `MultiHeadAttention(num_heads=num_heads, key_dim=total_dim // num_heads)`
  - 加上 Dropout、殘差連接與 Layer Normalization，得到 `x1`。
- 對 `x1` 通過前饋網路 (Feed Forward Network)：
  - `Dense(ff_dim, activation="relu")` → `Dense(total_dim)`
  - 加上 Dropout、殘差連接與 Layer Normalization，得到 `x2`。
- 對 `x2` 做 `GlobalAveragePooling1D()` 得到整條序列的向量 `pooled_output`，再加 Dropout。
- 從 `pooled_output` 接三個輸出頭：
  - `action_output`: `Dense(n_action_classes, activation="softmax")`
  - `point_output`: `Dense(n_point_classes, activation="softmax")`
  - `rally_output`: `Dense(1, activation="sigmoid")`

---

#### 7. Loss、Metric 與訓練

- 建立模型：`Model(inputs=input_layers, outputs=[action_output, point_output, rally_output])`。
- 自訂 F1 metric：
  - 將整數標籤 `y_true` 先用 `tf.one_hot` 轉成 one-hot。
  - 再丟入 `F1Score(average="macro")` 分別得到 `action_f1`、`point_f1`。
- 編譯模型：
  - Optimizer: `Adam(learning_rate=1e-4)`
  - Loss：
    - `action_output`: `sparse_categorical_crossentropy`
    - `point_output`: `sparse_categorical_crossentropy`
    - `rally_output`: `binary_crossentropy`
  - `loss_weights = {"action_output": 0.4, "point_output": 0.4, "rally_output": 0.2}`
  - Metrics：
    - `action_output`: `action_f1`
    - `point_output`: `point_f1`
    - `rally_output`: `AUC`
- 訓練：
  - `model.fit(X_train, y_train, batch_size=64, epochs=70, validation_data=(X_val, y_val))`

---

#### 8. 測試資料預測流程（Inference）

1. 建立 `X_test` 結構：
   - `X_test = {col: [] for col in categorical_cols}`
   - `X_test["numerical_input"] = []`
2. 依 `rally_uid` 建立測試序列：
   - 取 `test_df` 中的每個 `rally_uid`。
   - 對該 rally 依 `strickNumber` 排序成完整序列 `rally_sequence_df`。
   - 設 `features_df = rally_sequence_df`，表示用「目前該 rally 的所有拍」作為模型輸入。
   - 將此序列的各分類特徵與數值特徵依序塞到 `X_test`。
3. 對 `X_test` 做 padding：
   - 與訓練同樣設定：`maxlen = 64`、`padding = "pre"`、`truncating = "pre"`。
4. 模型預測：
   - `action_preds, point_preds, rally_preds = model.predict(X_test)`
   - `action_preds` / `point_preds` 為各類別的機率分布，`rally_preds` 為得分機率。
5. 取機率最大的類別並反編碼：
   - `predicted_action_indices = np.argmax(action_preds, axis=1)`
   - `predicted_point_indices = np.argmax(point_preds, axis=1)`
   - 使用 `feature_encoders["actionId"].inverse_transform(...)` 與 `feature_encoders["pointId"].inverse_transform(...)` 還原成原始 ID。
6. 組成提交檔案：
   - 讀取 `sample_submission.csv` 為模板。
   - 將預測結果填入欄位：
     - `actionId = final_action_preds`
     - `pointId = final_point_preds`
     - `serverGetPoint = rally_preds`
   - 輸出 `submission.csv` 供比賽上傳。

---


### 操作說明
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 43" src="https://github.com/user-attachments/assets/2d4f4c1b-a015-4ecd-85da-6e2da3e09d0c" />

<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 53" src="https://github.com/user-attachments/assets/d2e6b104-afd6-442e-991b-4667c0ebda05" />

<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 58" src="https://github.com/user-attachments/assets/114a564d-28a1-4fbd-a740-16a6bccb0e6b" />

<img width="1440" height="900" alt="截圖 2025-11-18 下午1 31 15" src="https://github.com/user-attachments/assets/fef02f79-39a1-4e18-97cb-802e0952e680" />

<img width="1440" height="900" alt="截圖 2025-11-18 下午1 31 40" src="https://github.com/user-attachments/assets/f0b86438-7adc-484b-a8f5-500e259ffec7" />

<img width="1440" height="900" alt="截圖 2025-11-18 下午1 32 23" src="https://github.com/user-attachments/assets/4a7d721b-90b8-4d24-aadf-0099d6955490" />


執行方法ㄧ
點選 **Run All**
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 41 08" src="https://github.com/user-attachments/assets/b6092b64-34d7-47a8-8fef-7735a6320408" />

執行方法二
點選右上角的 **Save Version**
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 41 08拷貝" src="https://github.com/user-attachments/assets/e70bb15b-ef8a-46f9-b829-e711e15e033f" />
