// MoViNet 模型配置：从导出脚本生成的 *_layout.json 反序列化得到。
// 该 JSON 描述 ONNX 的输入/输出名、输入分辨率、以及流式状态(states)的总长度与内部结构。
using System;
using UnityEngine;

namespace MoViNet
{
    [Serializable]
    public class MoViNetLayoutEntry
    {
        public string name;   // 模块名（仅用于调试/阅读）
        public string kind;   // conv_buffer | tcg_buffer | tcg_counter
        public string attr;   // 模块上的属性名（activation / n_cumulated_values）
        public int[] shape;   // 该状态段在 ONNX 内部的形状
        public int offset;    // 在扁平 states 向量中的起始下标
        public int size;      // 元素个数
    }

    [Serializable]
    public class MoViNetModelConfig
    {
        public string model;
        public string input_name = "input";
        public string states_name = "states";
        public string states_out_name = "states_out";
        public string logits_name = "logits";
        public int[] input_shape;          // [N, C, T, H, W]
        public int num_classes = 600;
        public int state_total;
        public MoViNetLayoutEntry[] layout;

        /// <summary>输入空间分辨率 H=W（帧会被缩放到该尺寸）。</summary>
        public int InputSize => input_shape != null && input_shape.Length >= 5 ? input_shape[3] : 172;

        /// <summary>从 JSON 文本反序列化。</summary>
        public static MoViNetModelConfig FromJson(string json)
        {
            return JsonUtility.FromJson<MoViNetModelConfig>(json);
        }
    }
}
