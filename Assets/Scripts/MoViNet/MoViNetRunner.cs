// MoViNet 流式推理引擎（纯 C#，依赖 onnxruntime-unity 提供的 Microsoft.ML.OnnxRuntime）。
//
// 关键设计：
//   1) 复刻 MoViNet-pytorch 的“逐帧流式”语义：每次喂入 1 帧，内部 stream buffer
//      与因果全局均值池化的累积状态通过单一 `states` 向量在调用间传递。
//   2) 输入张量维度固定为 [1, 3, 1, H, W]，H=W=InputSize（如 172）。
//      像素按 R/G/B 顺序归一化到 [0,1]（与 PyTorch 参考一致：只除以 255，无 mean/std）。
//   3) 输出 logits 为 [1, 600]（Kinetics-600），未做 softmax；提供 TopK 取前 k 类。
//
// 使用步骤：
//   var cfg  = MoViNetModelConfig.FromJson(jsonText);
//   var run  = new MoViNetRunner(cfg, modelBytes, labels);   // modelBytes 来自 .onnx 文件
//   run.Reset();                                             // 等价于 clean_activation_buffers
//   float[] logits = run.PredictFrame(rgbPixels);             // 每来一帧调用一次
//   var top = run.TopK(logits, 5);
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace MoViNet
{
    public class MoViNetRunner : IDisposable
    {
        private readonly MoViNetModelConfig _cfg;
        private readonly InferenceSession _session;
        private readonly int _size;
        private readonly int _numClasses;
        private readonly string[] _labels;

        private readonly float[] _inputBuffer;   // 复用的单帧输入缓冲 [3*H*W]
        private float[] _states;                 // 流式状态（跨帧保持）
        private bool _disposed;

        public int InputSize => _size;
        public int NumClasses => _numClasses;
        public int StateTotal => _cfg.state_total;

        /// <summary>
        /// 构造推理引擎。
        /// </summary>
        /// <param name="cfg">由导出脚本生成的布局配置。</param>
        /// <param name="modelBytes">.onnx 文件原始字节（可用 InferenceSession(byte[]) 从内存加载，跨平台最稳）。</param>
        /// <param name="labels">可选，600 个 Kinetics-600 类别名；为 null 时 TopK 只返回索引。</param>
        public MoViNetRunner(MoViNetModelConfig cfg, byte[] modelBytes, string[] labels = null)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            if (modelBytes == null || modelBytes.Length == 0)
                throw new ArgumentNullException(nameof(modelBytes));

            _size = cfg.InputSize;
            _numClasses = cfg.num_classes;
            _labels = labels;
            _inputBuffer = new float[3 * _size * _size];
            _states = new float[cfg.state_total];

            var options = new SessionOptions();
            // 单帧推理属轻量计算，限制线程数降低开销（可按需调整）。
            options.AppendExecutionProvider_CPU(0);
            _session = new InferenceSession(modelBytes, options);
        }

        /// <summary>
        /// 清空流式状态，开始一段新视频/新片段的识别（等价于 PyTorch 的 clean_activation_buffers）。
        /// </summary>
        public void Reset()
        {
            if (_states == null) return;
            Array.Clear(_states, 0, _states.Length);
        }

        /// <summary>
        /// 用 Color32 帧（长度 = size*size，行优先）喂入一帧并计算当前累积 logits。
        /// 像素按 R/G/B 归一化到 [0,1]（仅 /255）。
        /// flipVertically=true 时会垂直翻转（Unity 从 RT 读像素时第 0 行在底部，
        /// 而训练数据行 0 在顶部，翻转以匹配预训练分布）。
        /// </summary>
        public float[] PredictFrame(Color32[] pixels, bool flipVertically = true)
        {
            if (pixels == null || pixels.Length != _size * _size)
                throw new ArgumentException($"pixels 长度必须为 size*size = {_size * _size}", nameof(pixels));

            // 填充 NCHW（T=1）：index = c * size*size + h * size + w
            int plane = _size * _size;
            for (int h = 0; h < _size; h++)
            {
                int dstRow = h * _size;
                int srcRow = (flipVertically ? (_size - 1 - h) : h) * _size;
                for (int w = 0; w < _size; w++)
                {
                    int dstI = dstRow + w;
                    Color32 p = pixels[srcRow + w];
                    _inputBuffer[dstI] = p.r / 255f;            // 通道 0 = R
                    _inputBuffer[plane + dstI] = p.g / 255f;    // 通道 1 = G
                    _inputBuffer[2 * plane + dstI] = p.b / 255f; // 通道 2 = B
                }
            }
            return RunModel();
        }

        /// <summary>
        /// 用已归一化到 [0,1] 的 NCHW 单帧数组（长度 3*size*size，通道在最外层）喂入一帧。
        /// </summary>
        public float[] PredictFrame(float[] frameCHW)
        {
            if (frameCHW == null || frameCHW.Length != 3 * _size * _size)
                throw new ArgumentException("frameCHW 长度必须为 3*size*size", nameof(frameCHW));
            Array.Copy(frameCHW, _inputBuffer, frameCHW.Length);
            return RunModel();
        }

        private float[] RunModel()
        {
            var inputDims = new int[] { 1, 3, 1, _size, _size };
            var inputTensor = new DenseTensor<float>(_inputBuffer, inputDims);
            var statesTensor = new DenseTensor<float>(_states, new int[] { _cfg.state_total });

            var inputs = new List<NamedOnnxValue>(2)
            {
                NamedOnnxValue.CreateFromTensor(_cfg.input_name, inputTensor),
                NamedOnnxValue.CreateFromTensor(_cfg.states_name, statesTensor),
            };

            using var results = _session.Run(inputs);

            // 取回更新后的流式状态，供下一帧使用
            // 注意：必须用 GetValue(i) 而不是 tensor[i]。
            // Tensor<T> 的索引器 this[int] 会把单个下标当成多维第一维（rank>=2 时线性下标错位），
            // 而 GetValue(i) 才是按扁平线性下标访问（DenseTensor 即 Buffer.Span[i]）。
            var outStates = results[1].AsTensor<float>();
            int n = (int)outStates.Length;
            if (n == _states.Length)
                for (int i = 0; i < n; i++) _states[i] = outStates.GetValue(i);

            // logits：[1, 600]
            var logitsTensor = results[0].AsTensor<float>();
            int c = (int)logitsTensor.Length;
            var logits = new float[c];
            for (int i = 0; i < c; i++) logits[i] = logitsTensor.GetValue(i);
            return logits;
        }

        /// <summary>
        /// 对 logits 求 Top-K（未做 softmax；如需概率可先 Softmax）。
        /// </summary>
        public (int index, float score, string label)[] TopK(float[] logits, int k = 5)
        {
            if (logits == null || logits.Length != _numClasses)
                throw new ArgumentException("logits 长度必须等于 num_classes", nameof(logits));
            k = Mathf.Min(k, _numClasses);

            var order = new int[_numClasses];
            for (int i = 0; i < _numClasses; i++) order[i] = i;
            // 部分排序取前 k（类别数仅 600，简单冒泡/选择即可）
            for (int i = 0; i < k; i++)
            {
                int maxIdx = i;
                for (int j = i + 1; j < _numClasses; j++)
                    if (logits[order[j]] > logits[order[maxIdx]]) maxIdx = j;
                (order[i], order[maxIdx]) = (order[maxIdx], order[i]);
            }

            var top = new (int, float, string)[k];
            for (int i = 0; i < k; i++)
            {
                int idx = order[i];
                string label = (_labels != null && idx < _labels.Length) ? _labels[idx] : idx.ToString();
                top[i] = (idx, logits[idx], label);
            }
            return top;
        }

        /// <summary>对 logits 做 softmax（可选，用于把分数转成概率）。</summary>
        public static float[] Softmax(float[] logits)
        {
            float max = float.MinValue;
            foreach (var v in logits) if (v > max) max = v;
            float sum = 0f;
            var outp = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                outp[i] = Mathf.Exp(logits[i] - max);
                sum += outp[i];
            }
            for (int i = 0; i < outp.Length; i++) outp[i] /= sum;
            return outp;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _session?.Dispose();
        }
    }
}
