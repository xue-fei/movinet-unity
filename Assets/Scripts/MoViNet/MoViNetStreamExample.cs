// MoViNet 流式动作识别示例（Unity MonoBehaviour）。
//
// 功能：
//   - 从 StreamingAssets/MoViNet 加载 .onnx、布局 JSON 与 Kinetics-600 标签；
//   - 每来一帧（外设摄像头 WebCamTexture 或任意纹理）缩放至模型输入尺寸并推理；
//   - 实时显示 Top-K 动作类别与分数；提供按钮/方法重置流式状态。
//
// 接入说明：
//   1) 把导出的 movinet_a0_stream.onnx / *_layout.json / kinetics600_labels.txt
//      放进 Assets/StreamingAssets/MoViNet/（本示例默认路径）。
//   2) 输入源二选一（脚本优先用外设摄像头，其次用纹理）：
//        - 外设摄像头：在 Inspector 拖一个已存在的 WebCamTexture 到 webcamTexture，
//          或留空并填写 webcamDeviceName 让脚本自动创建并 Play；
//        - 任意纹理：把视频纹理 / RenderTexture / 图片纹理拖到 sourceTexture。
//   3) （可选）拖一个 UI.Text 到 outputText 用于显示结果。
//   4) 若画面上下颠倒，把 flipVertically 取反（外设摄像头与 Unity RT 读像素的行序不同）。
//
// 性能提示：
//   - 本示例用 ReadPixels 从 RenderTexture 同步取像素，仅 172x172 量级开销很小；
//     若要做到零拷贝/更低延迟，建议改用 AsyncGPUReadback 或原生插件取帧。
//   - 推理在 CPU（onnxruntime-cpu），A0 在桌面 CPU 上单帧仅数毫秒。
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace MoViNet
{
    public class MoViNetStreamExample : MonoBehaviour
    {
        [Header("模型文件（位于 StreamingAssets/MoViNet/）")]
        public string modelFile = "movinet_a0_stream.onnx";
        public string layoutFile = "movinet_a0_stream_layout.json";
        public string labelsFile = "kinetics600_labels.txt";

        [Header("输入源（二选一：外设摄像头 或 任意纹理）")]
        [Tooltip("外设摄像头。可直接拖一个已存在的 WebCamTexture；留空则由 webcamDeviceName 自动创建并播放。")]
        public WebCamTexture webcamTexture;
        [Tooltip("摄像头设备名；为空则用默认设备。仅在 webcamTexture 未手动指定时生效。")]
        public string webcamDeviceName;
        [Tooltip("已有的纹理（视频纹理 / RenderTexture / 图片等）作为第二输入源。")]
        public Texture sourceTexture;

        [Header("参数")]
        public int topK = 5;
        [Tooltip("每隔几帧推理一次（0=每帧）。流式模型逐帧累积，调大可降低 CPU 占用。")]
        public int frameInterval = 0;
        [Tooltip("垂直翻转开关。外设摄像头与 Unity RT 读像素的行序不同，画面颠倒时取反。")]
        public bool flipVertically = true;

        [Header("UI")]
        public Text outputText;
        [Tooltip("挂在 Canvas 下的 RawImage，用于实时显示相机/纹理画面。不赋值则仅推理不显示。")]
        public RawImage displayImage;

        private MoViNetRunner _runner;
        private RenderTexture _rt;
        private Texture2D _readbackTex;
        private int _frameCounter;
        private bool _ready;

        private void Start()
        {
            StartCoroutine(LoadAndInit());
        }

        private IEnumerator LoadAndInit()
        {
            string dir = Path.Combine(Application.streamingAssetsPath, "MoViNet");
            byte[] modelBytes = null;
            string layoutJson = null;
            string labelsText = null;

            // StreamingAssets 在不同平台的读取方式不同：Android 必须用 UnityWebRequest
            yield return LoadBytes(Path.Combine(dir, modelFile), b => modelBytes = b);
            yield return LoadText(Path.Combine(dir, layoutFile), t => layoutJson = t);
            yield return LoadText(Path.Combine(dir, labelsFile), t => labelsText = t);

            if (modelBytes == null || layoutJson == null)
            {
                Debug.LogError("[MoViNet] 模型或布局文件加载失败，请检查 StreamingAssets/MoViNet 目录。");
                yield break;
            }

            var cfg = MoViNetModelConfig.FromJson(layoutJson);
            string[] labels = null;
            if (!string.IsNullOrEmpty(labelsText))
                labels = labelsText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            _runner = new MoViNetRunner(cfg, modelBytes, labels);
            _runner.Reset();

            int size = _runner.InputSize;
            _rt = new RenderTexture(size, size, 0, RenderTextureFormat.ARGB32);
            _readbackTex = new Texture2D(size, size, TextureFormat.ARGB32, false);

            // 解析输入源：优先外设摄像头，其次任意纹理
            if (webcamTexture == null && WebCamTexture.devices.Length > 0)
            {
                string dev = string.IsNullOrEmpty(webcamDeviceName)
                    ? WebCamTexture.devices[0].name
                    : webcamDeviceName;
                webcamTexture = new WebCamTexture(dev);
            }
            if (webcamTexture != null && !webcamTexture.isPlaying)
            {
                webcamTexture.Play();
                Debug.Log($"[MoViNet] 已启动外设摄像头: {webcamTexture.deviceName}");
            }
            if (webcamTexture == null && sourceTexture == null)
            {
                Debug.LogError("[MoViNet] 未配置任何输入源（webcamTexture 与 sourceTexture 均为空，且无摄像头设备）。");
            }

            // 把相机/纹理画面同步到 UGUI 的 RawImage（挂在 Canvas 下）
            if (displayImage != null)
            {
                Texture initSrc = ResolveSource() ?? (webcamTexture as Texture) ?? sourceTexture;
                if (initSrc != null)
                {
                    displayImage.texture = initSrc;
                    ApplyDisplayFlip();
                }
            }

            Debug.Log($"[MoViNet] 已就绪：输入尺寸={size}x{size} 状态长度={_runner.StateTotal} 类别={_runner.NumClasses}");
            _ready = true;
        }

        private void Update()
        {
            if (!_ready) return;
            _frameCounter++;
            if (frameInterval > 0 && (_frameCounter % (frameInterval + 1) != 0)) return;

            // 解析当前帧输入源：优先外设摄像头，其次任意纹理
            Texture src = ResolveSource();
            if (src == null) return;

            // 把输入源画到 _rt（size x size，自动缩放）
            Graphics.Blit(src, _rt);

            // 从 RT 读取像素（同步，172x172 开销很小）
            RenderTexture prev = RenderTexture.active;
            RenderTexture.active = _rt;
            _readbackTex.ReadPixels(new Rect(0, 0, _rt.width, _rt.height), 0, 0);
            _readbackTex.Apply();
            RenderTexture.active = prev;

            // 实时把当前帧画面同步到 UGUI：纹理引用变化时更新，并每帧对齐翻转（flipVertically 运行期可调）
            if (displayImage != null)
            {
                if (displayImage.texture != src) displayImage.texture = src;
                ApplyDisplayFlip();
            }

            Color32[] pixels = _readbackTex.GetPixels32();
            float[] logits = _runner.PredictFrame(pixels, flipVertically);
            var top = _runner.TopK(logits, topK);

            ShowResult(top);
        }

        /// <summary>
        /// 用 UV Rect 翻转 RawImage，使 Canvas 上所见与喂给模型的输入帧方向一致。
        /// flipVertically=true 时垂直翻转（与 PredictFrame 的预处理对齐）。
        /// </summary>
        private void ApplyDisplayFlip()
        {
            if (displayImage == null) return;
            displayImage.uvRect = flipVertically
                ? new Rect(0f, 1f, 1f, -1f)
                : new Rect(0f, 0f, 1f, 1f);
        }

        /// <summary>选择当前帧的输入源：外设摄像头优先，其次任意纹理。</summary>
        private Texture ResolveSource()
        {
            if (webcamTexture != null && webcamTexture.isPlaying) return webcamTexture;
            if (sourceTexture != null) return sourceTexture;
            return null;
        }

        private void ShowResult((int index, float score, string label)[] top)
        {
            if (outputText == null)
            {
                foreach (var t in top)
                    Debug.Log($"[MoViNet] {t.label} ({t.index}): {t.score:F3}");
                return;
            }
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("MoViNet Top-K:");
            foreach (var t in top)
                sb.AppendLine($"{t.label}  {t.score:F3}");
            outputText.text = sb.ToString();
        }

        /// <summary>开始新的一段视频/片段识别时调用，清空流式状态。</summary>
        public void ResetState()
        {
            _runner?.Reset();
        }

        /// <summary>启动外设摄像头（若已自动创建则仅 Play）。</summary>
        public void StartWebcam()
        {
            if (webcamTexture != null && !webcamTexture.isPlaying) webcamTexture.Play();
        }

        /// <summary>停止外设摄像头。</summary>
        public void StopWebcam()
        {
            if (webcamTexture != null && webcamTexture.isPlaying) webcamTexture.Stop();
        }

        private void OnDestroy()
        {
            _runner?.Dispose();
            if (_rt != null) _rt.Release();
            StopWebcam();
        }

        // ---- 平台无关的 StreamingAssets 读取助手 ----

        private IEnumerator LoadBytes(string url, Action<byte[]> onLoaded)
        {
            byte[] data = null;
#if UNITY_ANDROID && !UNITY_EDITOR
            using (var req = UnityWebRequest.Get(url))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success) data = req.downloadHandler.data;
                else Debug.LogError("[MoViNet] 读取失败: " + url + " " + req.error);
            }
#else
            if (File.Exists(url)) data = File.ReadAllBytes(url);
            else Debug.LogError("[MoViNet] 文件不存在: " + url);
            yield return null;
#endif
            onLoaded?.Invoke(data);
        }

        private IEnumerator LoadText(string url, Action<string> onLoaded)
        {
            string text = null;
#if UNITY_ANDROID && !UNITY_EDITOR
            using (var req = UnityWebRequest.Get(url))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success) text = req.downloadHandler.text;
                else Debug.LogError("[MoViNet] 读取失败: " + url + " " + req.error);
            }
#else
            if (File.Exists(url)) text = File.ReadAllText(url);
            else Debug.LogError("[MoViNet] 文件不存在: " + url);
            yield return null;
#endif
            onLoaded?.Invoke(text);
        }
    }
}
