# MoViNet 流式视频动作识别 · Unity C# 实现说明

本说明文档对应工程内已完成的「把 MoViNet-pytorch 权重导出为 ONNX，并在 Unity 中用 C# + onnxruntime 复刻其流式推理」工作。

核心目标：在 Unity 里逐帧喂入视频（摄像头 / WebCamTexture / 任意 Texture），实时得到 Kinetics-600 的 600 类动作识别结果，且不依赖 Python 运行时。

---

## 1. 交付内容总览

| 路径 | 说明 |
| --- | --- |
| `tools/export_movinet_stream_onnx.py` | Python 导出脚本：把流式权重转成带「流式状态」的 ONNX + 布局 JSON，并做数值校验 |
| `Assets/StreamingAssets/MoViNet/movinet_a0_stream.onnx` | A0 流式 ONNX 模型（单文件内嵌权重，约 16.9 MB） |
| `Assets/StreamingAssets/MoViNet/movinet_a1_stream.onnx` | A1 流式 ONNX 模型（约 28.7 MB） |
| `Assets/StreamingAssets/MoViNet/movinet_a2_stream.onnx` | A2 流式 ONNX 模型（约 32.5 MB） |
| `Assets/StreamingAssets/MoViNet/movinet_a{0,1,2}_stream_layout.json` | 状态布局描述（C# 据此切分/重组 `states` 向量） |
| `Assets/StreamingAssets/MoViNet/kinetics600_labels.txt` | Kinetics-600 类别名（600 行） |
| `Assets/Scripts/MoViNet/MoViNetModelConfig.cs` | 反序列化布局 JSON 的配置类 |
| `Assets/Scripts/MoViNet/MoViNetRunner.cs` | 核心 C# 推理引擎（逐帧流式） |
| `Assets/Scripts/MoViNet/MoViNetStreamExample.cs` | 可直接挂场景的示例 MonoBehaviour |

前置依赖（已由工程 git submodule 引入）：

- `onnxruntime-unity`（提供 `Microsoft.ML.OnnxRuntime` C# API）
- `onnxruntime-unity-cpu`（CPU 执行提供器）

---

## 2. 关键设计：流式状态如何塞进标准 ONNX

MoViNet 的流式（streaming）推理依赖两类跨帧状态：

1. **`ConvBlock3D` 的 stream buffer**：保存最近 `dim_pad` 帧的激活，构成因果 3D 卷积的上下文。
2. **`TemporalCGAvgPool3D` 的累积和（cumsum）+ 帧计数器 `n_cumulated_values`**：实现因果全局均值池化，输出「截至当前帧」的平均 logits。

标准 ONNX 图没有「跨调用记忆」，因此导出脚本把**所有跨帧状态压平成一个一维 `float` 向量 `states`**，作为 ONNX 的额外输入与输出：

- 输入：`input` `[1,3,1,H,W]`（单帧）、`states` `[state_total]`
- 输出：`logits` `[1,600]`（当前累积类别分数）、`states_out` `[state_total]`（更新后的状态，供下一帧使用）

每次喂一帧，把上一帧的 `states_out` 当作这一帧的 `states` 输入，即完整复刻 PyTorch 参考中「逐帧喂入、内部 buffer 自动累积」的流式语义。

> 输出命名为 `states_out` 而非 `states`，是为了避免 ONNX 输入/输出同名冲突（`Duplicate definition of name`）。

---

## 3. 重新导出 ONNX（可选，已本地完成）

只有在你更换了权重、或想新增 A3/A4/A5 时才需要重跑。已导出的 A0/A1/A2 可直接使用。

### 3.1 Python 环境

```bash
# 任意 Python 3.9+，建议新建 venv
python -m venv venv
venv/Scripts/activate        # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy einops onnx onnxscript onnxruntime
```

导出脚本已内置 `fvcore.common.config.CfgNode` 的最小垫片，**无需安装 fvcore**。同时也修复了 `tfAvgPool3D.forward` 中的原地赋值（`x[..., -1] = ...` 在 tracing 下不友好），改用 `torch.cat` 拼接，保证 ONNX 可导出。

### 3.2 执行导出

```bash
# 在工程根目录执行
python tools/export_movinet_stream_onnx.py --model A0
python tools/export_movinet_stream_onnx.py --model A1
python tools/export_movinet_stream_onnx.py --model A2
```

- `--model`：`A0` / `A1` / `A2`（默认 `A0`）。
- `--out`：输出目录，默认 `Assets/StreamingAssets/MoViNet`。
- 脚本执行两步：先 `export`（写出 `.onnx` 与 `_layout.json`），再 `validate`（用 Python `onnxruntime` 逐帧对比 PyTorch 与 ONNX 的 logits，误差在容差内才退出 0）。

> 导出会先把权重存为外置 `.onnx.data`，脚本随后用 `onnx.save(save_as_external_data=False)` 重存为**单文件内嵌权重**，避免 Unity 端还要额外加载 `.data`。

### 3.3 导出产物

| 文件 | 说明 |
| --- | --- |
| `movinet_a{0,1,2}_stream.onnx` | 单文件 ONNX（权重内嵌）。A0 输入 `[1,3,1,172,172]`、A1 `[1,3,1,172,172]`、A2 `[1,3,1,224,224]`；`states`/`states_out` 长度见 layout 的 `state_total`（A0≈674208） |
| `movinet_a{0,1,2}_stream_layout.json` | 状态布局：`model / input_name / states_name / states_out_name / logits_name / input_shape / num_classes / state_total / layout[]` |

`layout[]` 中每条含：`name`（模块名）、`kind`（`conv_buffer`/`tcg_buffer`/`tcg_counter`）、`attr`（`activation`/`n_cumulated_values`）、`shape`、`offset`（在扁平 states 中的起始下标）、`size`（元素个数）。C# 侧只反序列化除 `module` 外的字段。

---

## 4. Unity 接入

### 4.1 放置文件

把以下三个文件放进 `Assets/StreamingAssets/MoViNet/`（示例脚本默认路径）：

- `movinet_a0_stream.onnx`
- `movinet_a0_stream_layout.json`
- `kinetics600_labels.txt`

三个 `.cs` 脚本放在 `Assets/Scripts/MoViNet/`（已就位）。

### 4.2 核心 API：`MoViNetRunner`

```csharp
// 1) 反序列化布局
var cfg = MoViNetModelConfig.FromJson(jsonText);

// 2) 构造推理引擎（modelBytes 来自 .onnx 文件字节）
var runner = new MoViNetRunner(cfg, modelBytes, labels);  // labels 为 600 个类名，可 null

// 3) 开始一段新视频/片段时重置流式状态（等价于 PyTorch 的 clean_activation_buffers）
runner.Reset();

// 4) 每来一帧调用一次，返回 [1,600] 的 logits（未做 softmax）
float[] logits = runner.PredictFrame(pixels);   // pixels: Color32[]，长度 = size*size

// 5) 取 Top-K（按 logits 原始分数排序）
var top = runner.TopK(logits, 5);
foreach (var (index, score, label) in top)
    Debug.Log($"{label} ({index}): {score:F3}");

// 6) 不需要时释放
runner.Dispose();
```

`MoViNetRunner` 的主要成员：

| 成员 | 说明 |
| --- | --- |
| `MoViNetRunner(cfg, modelBytes, labels=null)` | 构造。内部用 `InferenceSession(byte[])` 从内存加载模型，跨平台最稳；CPU 执行提供器线程数设为 0（单帧轻量） |
| `Reset()` | 清空 `_states`，开始新片段 |
| `PredictFrame(Color32[] pixels, bool flipVertically=true)` | 用 Color32 帧喂入；默认垂直翻转（见 §5.3） |
| `PredictFrame(float[] frameCHW)` | 用已归一化到 `[0,1]` 的 NCHW 单帧数组（长度 `3*size*size`）喂入 |
| `TopK(logits, k=5)` | 返回 `(int index, float score, string label)[]` |
| `Softmax(logits)` | 静态方法，把 logits 转成概率 |
| `InputSize / NumClasses / StateTotal` | 只读属性 |

### 4.3 示例组件：`MoViNetStreamExample`

直接把 `MoViNetStreamExample` 挂到场景任意物体上，配置 Inspector：

- `modelFile / layoutFile / labelsFile`：默认指向 A0 的三个文件；
- 输入源二选一（脚本优先用外设摄像头，其次用纹理）：
  - `webcamTexture`（`WebCamTexture`）：外设摄像头。可直接拖一个已存在的 WebCamTexture；留空则由 `webcamDeviceName` 自动创建并 `Play`；
  - `webcamDeviceName`：摄像头设备名，为空则用默认设备（仅在 `webcamTexture` 未手动指定时生效）；
  - `sourceTexture`（`Texture`）：视频纹理 / `RenderTexture` / 图片等作为第二输入源；
- `topK`：默认 5；
- `frameInterval`：每隔几帧推理一次（`0` = 每帧）。流式模型逐帧累积，调大可降低 CPU 占用；
- `flipVertically`：垂直翻转开关（默认 `true`）。外设摄像头与 Unity 从 RT 读像素的行序不同，画面上下颠倒时把它取反；
- `outputText`：可选 `UI.Text`，用于显示 Top-K 结果（留空则输出到 Console）；
- `displayImage`（`RawImage`）：可选，挂在 Canvas 下的 RawImage，用于实时显示相机/纹理画面。`flipVertically` 会同时控制显示的垂直翻转（UV Rect），保证屏幕上所见与喂给模型的输入帧方向一致。留空则只推理、不显示。

`Update()` 每帧解析输入源（外设摄像头优先）→ Blit 到 `size×size` 的 RenderTexture → `ReadPixels` → `PredictFrame(flipVertically)` → `TopK` → 显示。公共方法 `ResetState()` 可在切换视频/片段时清空流式状态；`StartWebcam()` / `StopWebcam()` 控制摄像头开关。

**在 Canvas 下显示相机画面的步骤：**
1. Hierarchy 右键 `UI → Canvas`（若已有则跳过），并确认场景中存在 `EventSystem`；
2. 在 Canvas 下右键 `UI → RawImage`，命名为如 `CameraView`；
3. 选中挂载了 `MoViNetStreamExample` 的物体，把该 RawImage 拖到 Inspector 的 `displayImage` 字段；
4. （推荐）给 RawImage 添加 `AspectRatioFitter` 组件，`Aspect Mode` 选 `Fit In Parent` 或 `Envelope Parent`，避免画面被拉伸变形；
5. 运行后，RawImage 即实时显示外设摄像头画面（与推理输入方向一致），同时打印/显示 Top-K 类别。

> 平台读取：示例对 Android 用 `UnityWebRequest`（StreamingAssets 在 APK 内不可直接用 `File`），其余平台用 `File`，已做分支处理。

---

## 5. 输入规范（务必对齐，否则精度崩）

### 5.1 分辨率

| 模型 | 输入空间分辨率 H=W |
| --- | --- |
| A0 | 172 |
| A1 | 172 |
| A2 | 224 |

帧需缩放到对应尺寸（示例已用 `RenderTexture` 自动缩放到 `size×size`）。

### 5.2 数值与维度

- 维度：`[N=1, C=3, T=1, H, W]`，通道在最外层（NCHW）。
- 归一化：**仅除以 255**，把 `uint8` 像素转到 `[0,1]`。**不做** mean/std 减法（与 PyTorch 参考的预处理一致）。
- 通道顺序：R → G → B（见 `MoViNetRunner.PredictFrame` 中 `p.r/255f`、`p.g/255f`、`p.b/255f` 的赋值顺序）。
- 索引映射：`index = c * (H*W) + h * W + w`。

### 5.3 垂直翻转（重要）

Unity 从 `RenderTexture` 读像素时，第 0 行在**底部**；而训练数据第 0 行在**顶部**。两者行序相反，因此 `PredictFrame(Color32[], flipVertically=true)` 默认会垂直翻转以匹配预训练分布。

- 如果你直接喂来自文件的、行 0 在顶部的图像（如 `Texture2D` 用 `LoadImage` 且未翻转），应传 `flipVertically=false`。
- 用 `WebCamTexture` / 纹理 Blit 到 RT 后读像素，默认 `true` 通常正确；若画面上下颠倒，把示例的 `flipVertically` 开关取反即可。
- 不同外设摄像头/平台可能存在 90°/270° 旋转或水平镜像，本示例未内置旋转校正；如遇旋转，可改用原生插件取帧或在 Blit 前对源纹理做旋转处理。

---

## 6. 流式状态管理

- **`Reset()` = PyTorch 的 `clean_activation_buffers()`**：在每段新视频/新片段开始时调用，清空所有 stream buffer 与累积均值计数器。
- 每个 `MoViNetRunner` 实例内部持有一个 `float[] _states`，跨帧自动更新（每次 `RunModel` 把 `states_out` 拷回 `_states`）。**不要**手动维护 states 向量——直接反复调 `PredictFrame` 即可。
- 多路视频请为每个视频/相机各建一个 `MoViNetRunner` 实例，互相独立。

---

## 7. 性能与精度

### 7.1 数值校验（导出脚本 `validate` 输出）

逐帧喂 50 帧随机视频，对比 PyTorch 参考与 ONNX 输出：

| 模型 | 终帧 logits 最大绝对误差 | 余弦相似度 | 逐帧误差均值 |
| --- | --- | --- | --- |
| A0 | ≈ 3.8e-6 | ≈ 0.9999999 | ≈ 4e-6 |
| A1 | ≈ 4.1e-6 | ≈ 0.9999999 | ≈ 4e-6 |
| A2 | ≈ 4.0e-6 | ≈ 0.9999999 | ≈ 4e-6 |

全部通过 ✅（容差 1e-2）。误差来自 ONNX 图算子融合/精度差异，属正常浮点容差，不影响分类结果。

### 7.2 推理性能

- 模型在 **CPU（onnxruntime-cpu）** 上运行。
- A0 为最轻量（约 16.9 MB、参数量最小），桌面 CPU 单帧仅数毫秒；A2（224 输入）最重，移动端建议优先选 A0/A1。
- 示例用同步 `ReadPixels` 从 RT 取像素，172×172 量级开销很小。若要更低延迟/零拷贝，可改用 `AsyncGPUReadback` 或原生插件取帧（不在本示例范围内）。

### 7.3 注意事项

- **opset**：导出请求 `opset_version=17`，实际导出为 18（Pad 适配器告警为无害回退，可忽略）。
- **线程数**：`SessionOptions.AppendExecutionProvider_CPU(0)` 限制为单线程。若想利用多核可在构造处调整（注意移动端发热）。
- **精度**：`logits` 未做 softmax，`TopK` 直接按原始分数排序；需要概率时先调 `Softmax`。

---

## 8. 常见问题

**Q：Android 上读不到模型？**
A：StreamingAssets 在 APK 内不能用 `File` 直接读。示例已对 `UNITY_ANDROID && !UNITY_EDITOR` 走 `UnityWebRequest.Get`，无需额外处理。

**Q：换模型后报 `states` 长度不匹配？**
A：确保 `.onnx` 与 `_layout.json` 来自同一次导出（同一 `--model`）。`MoViNetRunner` 用 `cfg.state_total` 分配 `_states`，layout 与 onnx 必须一致。

**Q：识别结果明显不对？**
A：优先检查 §5 的输入规范——分辨率、`/255` 而非 mean/std、RGB 顺序、以及垂直翻转开关 `flipVertically` 是否匹配你的取帧方式。

**Q：能上 GPU（NNAPI / CoreML / DirectML）吗？**
A：当前依赖 `onnxruntime-unity-cpu`。若要 GPU 加速，需引入对应的 GPU 执行提供器包（如 `onnxruntime-unity-gpu` 或对应平台 EP），并在 `MoViNetRunner` 构造处改用对应 `AppendExecutionProvider_*`。

**Q：A3/A4/A5 能用吗？**
A：`MoViNet-pytorch/weights/` 中已有 A3/A4 的 `*_stream_statedict_v3`，只需在 `export_movinet_stream_onnx.py` 的 `STREAM_WEIGHTS_LOCAL` 与 `INPUT_SIZE` 字典里补上对应条目，并放开 `--model` 的 `choices`，即可导出。
