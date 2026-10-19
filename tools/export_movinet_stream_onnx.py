# -*- coding: utf-8 -*-
"""
将 MoViNet-pytorch 的 streaming（因果）模型权重导出为带“流式状态”的 ONNX 模型。

核心思路：
  MoViNet 的流式推理依赖两类跨帧状态：
    1) ConvBlock3D 的 stream buffer：保存最近 dim_pad 帧的激活（因果卷积上下文）
    2) TemporalCGAvgPool3D 的累积和(cumsum) + 帧计数器 n（因果全局均值池化）
  本脚本把所有跨帧状态压平成一个一维 float 向量 `states`，作为 ONNX 的额外
  输入与输出。这样单个 ONNX 图即可“逐帧喂入 + 携带状态”，完整复刻
  MoViNet-pytorch 的 streaming 功能（每次喂 1 帧，输出当前累积平均的 600 类 logits）。

用法：
  python export_movinet_stream_onnx.py --model A0
  python export_movinet_stream_onnx.py --model A1
  python export_movinet_stream_onnx.py --model A2

输出（默认写到 ../Assets/StreamingAssets/MoViNet/）：
  movinet_a{0,1,2}_stream.onnx      单帧流式 ONNX 模型
  movinet_a{0,1,2}_stream_layout.json  状态布局（C# 据此拆分/重组 states）
"""

import argparse
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0) 让 movinets 可在不安装 fvcore 的情况下导入（提供一个最小 CfgNode 垫片）
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 工程根
MOVINET_DIR = os.path.join(ROOT, "MoViNet-pytorch")
sys.path.insert(0, MOVINET_DIR)


class _CN:
    """fvcore.common.config.CfgNode 的最小实现，仅覆盖本仓库用到的能力。"""
    def __init__(self, init=None):
        object.__setattr__(self, "_d", {})
        if init:
            self._d.update(init)

    def __getattr__(self, k):
        d = object.__getattribute__(self, "_d")
        if k in d:
            return d[k]
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k == "_d":
            object.__setattr__(self, k, v)
            return
        self._d[k] = v

    def __contains__(self, k):
        return k in self._d

    def get(self, k, default=None):
        return self._d.get(k, default)


_fvcore = types.ModuleType("fvcore")
_fvcore_common = types.ModuleType("fvcore.common")
_fvcore_config = types.ModuleType("fvcore.common.config")
_fvcore_config.CfgNode = _CN
sys.modules["fvcore"] = _fvcore
sys.modules["fvcore.common"] = _fvcore_common
sys.modules["fvcore.common.config"] = _fvcore_config

from movinets.config import _C  # noqa: E402
from movinets import MoViNet  # noqa: E402
from movinets.models import ConvBlock3D, TemporalCGAvgPool3D, tfAvgPool3D  # noqa: E402


# ----------------------------------------------------------------------------
# 1) 修复 tfAvgPool3D 中的原地(in-place)赋值，使其可被 ONNX 导出
#    （原实现 x[..., -1] = ... 在 tracing 下不友好）
# ----------------------------------------------------------------------------
def _safe_tf_avg_pool_forward(self, x):
    if x.shape[-1] != x.shape[-2]:
        raise RuntimeError("tf_like avg 仅支持 h==w")
    f1 = x.shape[-1] % 2 != 0
    if f1:
        padding_pad = (0, 0, 0, 0)
    else:
        padding_pad = (0, 1, 0, 1)
    x = F.pad(x, padding_pad)
    if f1:
        x = F.avg_pool3d(x, (1, 3, 3), stride=(1, 2, 2),
                         count_include_pad=False, padding=(0, 1, 1))
    else:
        x = self.avgf(x)
        # 等价于原实现：把最后一列、最后一行乘以 9/6（=1.5）
        x = torch.cat([x[..., :-1], x[..., -1:] * 1.5], dim=-1)
        x = torch.cat([x[..., :-1, :], x[..., -1:, :] * 1.5], dim=-2)
    return x


tfAvgPool3D.forward = _safe_tf_avg_pool_forward


# ----------------------------------------------------------------------------
# 2) 构建模型并加载本地流式权重
# ----------------------------------------------------------------------------
STREAM_WEIGHTS_LOCAL = {
    "A0": os.path.join(MOVINET_DIR, "weights", "modelA0_stream_statedict_v3"),
    "A1": os.path.join(MOVINET_DIR, "weights", "modelA1_stream_statedict_v3"),
    "A2": os.path.join(MOVINET_DIR, "weights", "modelA2_stream_statedict_v3"),
}

INPUT_SIZE = {"A0": 172, "A1": 172, "A2": 224}  # 各模型官方输入分辨率


def build_model(model_name):
    cfg = getattr(_C.MODEL, f"MoViNet{model_name}")
    model = MoViNet(
        cfg,
        causal=True,
        pretrained=False,
        num_classes=600,
        conv_type="2plus1d",
        tf_like=True,
    )
    wpath = STREAM_WEIGHTS_LOCAL[model_name]
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"未找到本地权重: {wpath}")
    state = torch.load(wpath, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    # 关闭 TemporalCGAvgPool3D 的 detach hook（仅训练时省显存，导出不需要）
    for m in model.modules():
        if isinstance(m, TemporalCGAvgPool3D):
            m._forward_hooks.clear()
    return model


# ----------------------------------------------------------------------------
# 3) 发现全部跨帧状态（stream buffer + 累积和 + 计数器）及其形状
# ----------------------------------------------------------------------------
def discover_layout(model, dummy):
    model.clean_activation_buffers()
    with torch.no_grad():
        model(dummy)
    layout = []
    offset = 0
    for name, m in model.named_modules():
        if isinstance(m, ConvBlock3D):
            if m.dim_pad > 0 and m.conv_2 is not None and m.causal:
                act = m.activation
                shape = list(act.shape)
                size = int(np.prod(shape))
                layout.append({
                    "name": name,
                    "module": m,
                    "kind": "conv_buffer",
                    "attr": "activation",
                    "shape": shape,
                    "offset": offset,
                    "size": size,
                })
                offset += size
        elif isinstance(m, TemporalCGAvgPool3D):
            act = m.activation
            shape = list(act.shape)
            size = int(np.prod(shape))
            layout.append({
                "name": name,
                "module": m,
                "kind": "tcg_buffer",
                "attr": "activation",
                "shape": shape,
                "offset": offset,
                "size": size,
            })
            offset += size
            # 计数器 n（标量）
            layout.append({
                "name": name,
                "module": m,
                "kind": "tcg_counter",
                "attr": "n_cumulated_values",
                "shape": [1],
                "offset": offset,
                "size": 1,
            })
            offset += 1
    return layout, offset


# ----------------------------------------------------------------------------
# 4) 流式包装器：把 states 注入到各模块，跑完再收回
# ----------------------------------------------------------------------------
class StreamMoViNet(torch.nn.Module):
    def __init__(self, model, layout):
        super().__init__()
        self.model = model
        self.layout = layout

    def forward(self, x, states):
        # 注入状态
        for e in self.layout:
            buf = states[e["offset"]:e["offset"] + e["size"]].reshape(e["shape"])
            module = e["module"]
            if e["kind"] == "tcg_counter":
                module.n_cumulated_values = buf.reshape(())  # 标量张量
            else:
                module.activation = buf
        # 前向
        out = self.model(x)
        # 收回状态
        new_states = []
        for e in self.layout:
            module = e["module"]
            if e["kind"] == "tcg_counter":
                new_states.append(module.n_cumulated_values.reshape((1,)))
            else:
                new_states.append(module.activation.reshape(-1))
        return out, torch.cat(new_states)


# ----------------------------------------------------------------------------
# 5) 导出
# ----------------------------------------------------------------------------
def export(model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    size = INPUT_SIZE[model_name]
    model = build_model(model_name)

    dummy = torch.zeros(1, 3, 1, size, size, dtype=torch.float32)
    layout, total = discover_layout(model, dummy)
    print(f"[{model_name}] 状态条目数={len(layout)}  状态总元素数={total}")

    wrapper = StreamMoViNet(model, layout)
    wrapper.eval()

    dummy_states = torch.zeros(total, dtype=torch.float32)
    onnx_path = os.path.join(out_dir, f"movinet_{model_name.lower()}_stream.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy, dummy_states),
            onnx_path,
            input_names=["input", "states"],
            output_names=["logits", "states_out"],
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[{model_name}] 已导出: {onnx_path}")

    # 写出布局 JSON（C# 用）
    layout_path = os.path.join(out_dir,
                               f"movinet_{model_name.lower()}_stream_layout.json")
    meta = {
        "model": model_name,
        "input_name": "input",
        "states_name": "states",
        "states_out_name": "states_out",
        "logits_name": "logits",
        "input_shape": [1, 3, 1, size, size],
        "num_classes": 600,
        "state_total": total,
        "layout": layout,
    }
    layout_json = [{k: v for k, v in e.items() if k != "module"} for e in layout]
    meta["layout"] = layout_json
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[{model_name}] 已写出布局: {layout_path}")

    return model, layout, total, onnx_path, dummy, size


# ----------------------------------------------------------------------------
# 6) 数值校验：PyTorch 逐帧流式 vs ONNX 逐帧流式
# ----------------------------------------------------------------------------
def validate(model_name, model, layout, total, onnx_path, dummy, size, out_dir):
    import onnxruntime as ort

    # 造一段随机“视频”（归一化到 [0,1]），共 50 帧
    T_total = 50
    rng = np.random.RandomState(0)
    video = rng.rand(1, 3, T_total, size, size).astype(np.float32)

    # ---- PyTorch 参考（逐帧喂入，内部 buffer 自动累积）----
    model.clean_activation_buffers()
    pt_logits_seq = []
    with torch.no_grad():
        for t in range(T_total):
            x = torch.from_numpy(video[:, :, t:t + 1])
            out = model(x)
            pt_logits_seq.append(out.detach().numpy().copy())
    pt_final = pt_logits_seq[-1]

    # ---- ONNX 流式（携带 states）----
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    states = np.zeros(total, dtype=np.float32)
    onnx_logits_seq = []
    for t in range(T_total):
        x = video[:, :, t:t + 1]
        res = sess.run(["logits", "states_out"],
                       {"input": x, "states": states})
        logits, states = res[0], res[1]
        onnx_logits_seq.append(logits.copy())
    onnx_final = onnx_logits_seq[-1]

    pt_final = np.asarray(pt_final).reshape(-1)
    onnx_final = np.asarray(onnx_final).reshape(-1)
    max_abs = float(np.max(np.abs(pt_final - onnx_final)))
    cosine = float(np.dot(pt_final, onnx_final) /
                   (np.linalg.norm(pt_final) * np.linalg.norm(onnx_final) + 1e-12))
    print(f"[{model_name}] 终帧 logits 最大绝对误差 = {max_abs:.6e}  余弦相似度 = {cosine:.8f}")

    # 各帧误差统计
    errs = [float(np.max(np.abs(np.asarray(p).reshape(-1) - np.asarray(o).reshape(-1))))
            for p, o in zip(pt_logits_seq, onnx_logits_seq)]
    print(f"[{model_name}] 逐帧最大绝对误差: max={max(errs):.6e} mean={float(np.mean(errs)):.6e}")

    if max_abs < 1e-2:
        print(f"[{model_name}] 校验通过 ✅ (误差在容差内)")
        return True
    else:
        print(f"[{model_name}] 校验失败 ❌ (误差过大，请检查导出)")
        return False


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A0", "A1", "A2"], default="A0")
    ap.add_argument("--out", default=None,
                    help="输出目录，默认 <工程>/Assets/StreamingAssets/MoViNet")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(ROOT, "Assets", "StreamingAssets", "MoViNet")

    model, layout, total, onnx_path, dummy, size = export(args.model, out_dir)
    ok = validate(args.model, model, layout, total, onnx_path, dummy, size, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
