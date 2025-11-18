# Kaggle-Competition-TTMATCH

### 方法與流程

#### 1. 環境設定與資料載入
- **環境修復**：安裝特定版本的 `protobuf` 以解決 TensorFlow 版本衝突
- **套件載入**：
  - `pandas`、`numpy`：處理數據
  - `tensorflow/keras`：建構模型
  - `sklearn`：進行數據前處理
  - `matplotlib`：繪製圖表
- **讀取資料**：載入訓練集 (`train.csv`) 與測試集 (`test.csv`)

#### 2. 特徵工程
- **衍生特徵**：
    - `scoreDiff`：雙方比分差距
    - `is_server`：當前擊球者是否為發球方
- **特徵類別**：
    - 分類特徵：`sex`, `strickId`, `handId`, `strengthId`, `spinId`, `pointId`, `actionId`, `positionId`
    - 數值特徵：`strickNumber`, `scoreDiff`, `is_server` 
- **資料編碼**：
    - 針對分類特徵欄位進行編碼
    - 使用同一組 `LabelEncoder` 處理訓練與測試資料，確保 ID 對應一致

#### 3. 序列資料製作
- **Sampling**：針對每個 rally，採取 **Latest-K Sampling** 策略，取回合結束前的最後`6`個時間點作為不同訓練樣本
- **滑動視窗**：每個樣本利用過去 i 拍的資訊預測第 i+1 拍
- **Padding**：設定序列長度 `sequence_length = 64`，不足處在前方補 0
- **資料切割**：依據 `rally_uid` 進行 80% 訓練集、20% 驗證集的切割，防止資料洩漏
- **輸入輸出**：
    - **Input**：分類特徵序列 + 數值特徵序列
    - **Output**：同時包含 `actionId`、`pointId`、`serverGetPoint`

#### 4. 模型架構 - Transformer
模型採多輸入、多輸出設計：
- **Embedding Layer**：將分類特徵轉換為向量並與數值特徵串接
- **Transformer Block**：
    - **Multi-Head Attention**：捕捉擊球序列的前後關聯
    - **Feed Forward Network**：增強特徵提取能力
    - **Residual Connection & LayerNorm**：穩定訓練過程
- **Pooling Layer**：將時間序列特徵壓縮為特徵向量
- **Output Layer (Multi-task Heads)**：
    - **Action Head** (`Softmax`)：預測擊球動作
    - **Point Head** (`Softmax`)：預測落點
    - **Rally Head** (`Sigmoid`)：預測回合贏家

#### 5. 模型編譯與訓練
- **Loss Function**：分類任務使用 `sparse_categorical_crossentropy`，勝負預測使用 `binary_crossentropy`
- **權重分配**：`action_output`(0.4) / `point_output`(0.4) / `rally_output`(0.2)
- **Metrics**：使用自定義 **F1 Score** 與 **AUC** 監控效能
- **訓練設定**：Optimizer 使用 `Adam`，訓練 70 個 Epochs

#### 6. 預測與提交
- **測試集處理**：將測試資料轉換為長度 64 的序列格式
- **推論**：模型產出機率分佈，使用 `argmax` 取出最高機率類別
- **還原結果**：利用 `inverse_transform` 將數字 ID 轉回原始文字標籤
- **存檔**：生成 `submission.csv`

#### 7. 視覺化
- 繪製 Loss (訓練/驗證) 變化曲線
- 繪製 F1 Score 與 AUC 指標變化曲線

---

### 操作說明

* 進到 Kaggle 後選擇左側的 `code`，並點選 `New Notebook`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 43" src="https://github.com/user-attachments/assets/2d4f4c1b-a015-4ecd-85da-6e2da3e09d0c" />

<br>
<br>

* 點擊 `File`，使用 `Import Notebook`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 53" src="https://github.com/user-attachments/assets/d2e6b104-afd6-442e-991b-4667c0ebda05" />

<br>
<br>

* 上傳 `.ipynb` 檔，點擊 `Import`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 29 58" src="https://github.com/user-attachments/assets/114a564d-28a1-4fbd-a740-16a6bccb0e6b" />

<br>
<br>

* 接著要載入 dataset，點選右邊的 `Add Input`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 31 15" src="https://github.com/user-attachments/assets/fef02f79-39a1-4e18-97cb-802e0952e680" />

<br>
<br>

* 搜尋 `TTMATCH`，並加入此競賽的 dataset
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 31 40" src="https://github.com/user-attachments/assets/f0b86438-7adc-484b-a8f5-500e259ffec7" />

<br>
<br>

* 將 `Settings` 中的 `Accelerator` 切換為 GPU 可以大幅縮短訓練時間
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 32 23" src="https://github.com/user-attachments/assets/4a7d721b-90b8-4d24-aadf-0099d6955490" />

<br>
<br>

* **執行方法ㄧ**：點選 `Run All`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 41 08" src="https://github.com/user-attachments/assets/b6092b64-34d7-47a8-8fef-7735a6320408" />

<br>
<br>

* **執行方法二**：點選右上角的 `Save Version`
<img width="1440" height="900" alt="截圖 2025-11-18 下午1 41 08拷貝" src="https://github.com/user-attachments/assets/e70bb15b-ef8a-46f9-b829-e711e15e033f" />
