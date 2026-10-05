# Sparse Global Residence-Time Optimizer v1

## 核心改動

這個版本不再逐輪使用 PID 修改 F，也不在每一輪重新建立完整 G-code／TRUE FIFO。

它使用輸入 segmented CSV 內既有的 `t_in`、`t_out`、`tau_s` 作為權威 baseline，固定材料 FIFO 配對後，把相鄰可調整列合併成控制區塊，再使用兩階段全域線性規劃：

1. **Stage 1：最小化最大 residence-time 超界值**  
   如果存在完全可行解，結果會是 0；若無解，這個值代表目前 F/P 限制下理論上最小的最嚴重違規。
2. **Stage 2：保持 Stage 1 最佳值**，同時：
   - 讓 τ 靠近 `tau_ideal`；
   - 減少相對原始速度的改動；
   - 減少相鄰控制區塊的速度跳動；
   - 可選擇縮短總執行時間。

最佳化期間全部在記憶體中進行。只有最後才 patch 一次 G-code，並可選擇執行一次既有 TRUE FIFO builder。

## 重要前提

新版模型假設以下內容不變：

- G-code segment 順序；
- 每段 E／`V_mm3`；
- `V_nozzle`；
- FIFO 材料配對規則。

程式只最佳化時間，也就是 movement feedrate `F` 與選配的 G4 dwell `P`。

## LiQ5 範例指令

```bash
python residence_optimizer_sparse_v1.py \
  --input "LiQ5_sample_RTV-Body_Temp25_2244s_FT0.1_segmented.csv" \
  --gcode_in "LiQ5_sample_RTV-Body_Temp25_2244s_FT0.1_segmented.gcode" \
  --gcode_out "LiQ5_sample_RTV-Body_Temp25_2244s_sparse_optimized.gcode" \
  --out_dir "LiQ5_sparse_optimization" \
  --target_mode alpha \
  --temp_c 25.0 \
  --alpha_mode bounds \
  --alpha_min 0.929034 \
  --alpha_ideal 0.977931 \
  --alpha_max 0.99999 \
  --Fmin 100 \
  --Fmax 3000 \
  --enable_travel_opt \
  --enable_dwell_opt \
  --block_rows 150 \
  --block_duration_sec 12 \
  --max_target_samples 5000 \
  --change_weight 0.10 \
  --smooth_weight 0.05 \
  --target_weight 1.00 \
  --vmax_tolerance_s 0.01 \
  --true_validate \
  --vnozzle 574
```

如果目前資料夾內沒有：

```text
rebuild_csv_and_gcode_fifo_v1_2_precondition_autofill.py
```

請先移除 `--true_validate`。最佳化與 G-code 輸出仍可執行，只是不會做最後 TRUE FIFO 比對。

## 直接輸入 τ 的模式

```bash
python residence_optimizer_sparse_v1.py \
  --input "model_segmented.csv" \
  --target_mode direct \
  --tau_min 1940 \
  --tau_ideal 2000 \
  --tau_max 2060 \
  --Fmin 180 \
  --Fmax 3000 \
  --out_dir "sparse_optimization"
```

## 主要輸出

- `optimized_prediction.csv`  
  每一列的控制區塊、時間倍率、新 F、新 P、預測 τ 與預測 α。
- `control_blocks.csv`  
  每個控制區塊的範圍、原始／最佳化時間、倍率與上下界。
- `summary.json`  
  baseline 與 optimized 指標、solver 狀態、Stage 1 最小違規值、耗時及 TRUE validation 誤差。
- 指定的 `--gcode_out`  
  最終 patch 後的 G-code。
- `true_validation.csv`  
  只有啟用 `--true_validate` 時產生。

## 最重要的調整參數

### `--block_rows`
每個控制區塊最多包含幾個可調整 rows。

- 小：自由度高，但 solver 變大；
- 大：速度快且結果平滑，但可能沒有足夠自由度。

建議先從 `100–300` 測試。

### `--block_duration_sec`
每個控制區塊最多涵蓋多少 baseline 秒數。

建議先從 `5–20 s` 測試。設為 `0` 代表只用 `block_rows` 分割。

### `--max_target_samples`
所有 residence rows 都會進入上下界限制；這個參數只限制有多少 rows 額外參與「靠近 tau_ideal」的品質目標。

- 複雜模型先用 `2000–5000`；
- 想讓全部 τ 更接近 ideal，可提高；
- 設為 `0` 代表全部使用。

### `--change_weight`
越大，越不願意偏離原始 F/P。

### `--smooth_weight`
越大，相鄰同類控制區塊越平滑。

### `--target_weight`
越大，越積極讓 τ 靠近 `tau_ideal`；但不會突破 Stage 1 的最大違規限制。

### `--vmax_tolerance_s`
Stage 2 相對 Stage 1 理論最佳最大違規值，可額外放寬多少秒。建議保持很小，例如 `0.001–0.05 s`。

## 第一輪測試建議

1. 先不使用 `--true_validate`，確認 solver 能完成並輸出 G-code。
2. 查看 `summary.json`：
   - `minimal_worst_violation_s = 0`：代表目前限制下存在完全可行解；
   - 大於 0：代表目前控制區塊與 F/P 範圍下無法讓全部 τ 合格。
3. 再啟用 `--true_validate`，檢查：
   - `true_minus_pred_rmse_s`；
   - `true_minus_pred_abs_max_s`。
4. 若 TRUE 與 predicted 差異大，不應直接使用輸出 G-code；這表示固定 FIFO timing 模型仍需依你的 builder 規則補強。

## 與舊版的關係

目前既有 TRUE FIFO builder 被保留，但用途從「每輪最佳化核心」改成「最後一次物理驗證」。舊 PID optimizer 建議暫時保留，作為新舊結果與速度的 benchmark。
