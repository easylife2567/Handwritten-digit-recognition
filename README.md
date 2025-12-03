# MNIST 手写数字识别 DNN 全栈项目技术文档

## 1. 项目概述

本项目实现了一个端到端的深度神经网络（DNN）应用，涵盖了从模型训练、导出到 Web 端部署的全流程。核心目标是展示如何在浏览器环境中高效运行 AI 模型，实现“零后端”推理。

**主要功能：**

- 手写数字识别（0-9）。
- 实时模型推理（基于 WebAssembly）。
- 可解释性可视化（遮挡热力图）。

## 2. 技术栈架构

| 模块         | 技术选型                | 作用                                       |
| :----------- | :---------------------- | :----------------------------------------- |
| **训练端**   | Python, PyTorch         | 构建与训练 MLP 模型，处理 MNIST 数据集。   |
| **模型转换** | ONNX                    | 模型交换格式，连接 PyTorch 与 Web 运行时。 |
| **前端框架** | React, TypeScript, Vite | 构建用户界面，处理 Canvas 交互。           |
| **推理引擎** | ONNX Runtime Web (WASM) | 在浏览器中加载 ONNX 模型并进行推理。       |

---

## 3. 核心模块详解

### 3.1 模型训练 (`train.py`)

- **架构**：多层感知机 (MLP)。
  - 输入：28x28 (784 维) 向量。
  - 隐藏层：256 -> 128，激活函数 ReLU。
  - 正则化：Dropout (0.2), BatchNorm1d。
  - 输出：10 分类 (Softmax logits)。
- **优化器**：Adam (lr=1e-3)，配合 `ReduceLROnPlateau` 学习率调度。
- **性能**：在 MNIST 测试集上准确率达到 98% 以上。
- **产物**：`best.pt` (PyTorch 权重文件)。

### 3.2 模型导出 (`export_onnx.py`)

- **转换**：使用 `torch.onnx.export` 将 PyTorch 模型转换为 ONNX 格式。
- **配置**：
  - `input_names=["input"]`, `output_names=["logits"]`。
  - `dynamic_axes`：支持动态 Batch Size。
  - `opset_version=13`：保证算子兼容性。
- **产物**：`web/public/mnist_dnn.onnx`。

### 3.3 Web 端推理 (`web/src/onnx.ts`)

这是项目的核心难点，解决了浏览器加载 WASM 的诸多限制。

**关键实现：**

1.  **WASM 路径映射**：
    由于 Vite 的打包机制限制，我们手动指定了 WASM 文件的加载路径，避免了自动解析错误。
    ```typescript
    ort.env.wasm.wasmPaths = {
      "ort-wasm.wasm": "/ort-wasm.wasm",
      "ort-wasm-simd.wasm": "/ort-wasm-simd.wasm",
      // ...
    };
    ```
2.  **预处理流水线**：
    - Canvas (280x280) -> 缩放至 28x28。
    - 灰度化与归一化：将像素值映射到 [0, 1] 区间（黑笔白底 -> 白笔黑底）。
    - Tensor 构造：`new ort.Tensor('float32', data, [1, 1, 28, 28])`。
3.  **UMD 加载方案**：
    为了绕过 ESM 动态导入在 Vite `public` 目录下的限制，我们采用 UMD 脚本 (`ort.wasm.min.js`) 通过 `index.html` 直接加载。

### 3.4 可解释性热力图 (`web/src/occlusion.ts`)

- **原理**：遮挡敏感度分析 (Occlusion Sensitivity)。
- **算法**：
  1. 使用滑动窗口（如 4x4）遮挡输入图像的局部区域。
  2. 对遮挡后的图像进行推理，记录目标类别的概率下降幅度。
  3. 概率下降越多，说明被遮挡区域对识别越重要，热力值越高。

---

## 4. 部署与运行

### 环境要求

- Node.js 16+
- Python 3.9+ (仅训练阶段需要)

### 运行步骤

1. **训练模型**（可选，已预置模型）：
   ```bash
   python train.py
   python export_onnx.py
   ```
