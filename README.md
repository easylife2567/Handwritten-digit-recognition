# MNIST 手写数字识别 DNN 全栈项目

这是一个端到端的深度神经网络（DNN）应用，展示了从 PyTorch 模型训练、ONNX 导出到基于 WebAssembly 的浏览器端推理的全流程。

**核心特性：**

- ✍️ **手写数字识别**：支持 0-9 数字的实时识别。
- ⚡ **纯前端推理**：基于 ONNX Runtime Web，零后端延迟，数据不出本地。
- 🔍 **可解释性可视化**：内置遮挡热力图（Occlusion Sensitivity），展示 AI 关注区域。
- 🛠 **全栈架构**：涵盖 Python 训练与 React/Vite 前端工程化。

---

## 1. 技术栈架构

| 模块         | 技术选型                | 作用                                       |
| :----------- | :---------------------- | :----------------------------------------- |
| **训练端**   | Python, PyTorch         | 构建与训练 MLP 模型，处理 MNIST 数据集。   |
| **模型转换** | ONNX                    | 模型交换格式，连接 PyTorch 与 Web 运行时。 |
| **前端框架** | React, TypeScript, Vite | 构建用户界面，处理 Canvas 交互。           |
| **推理引擎** | ONNX Runtime Web (WASM) | 在浏览器中加载 ONNX 模型并进行推理。       |

---

## 2. 快速开始

### 环境要求

- **Node.js**: 16+
- **Python**: 3.9+ (仅当需要重新训练模型时)

### 运行步骤

#### 1. 启动 Web 应用（直接体验）

项目已内置预训练好的模型 (`web/public/mnist_dnn.onnx`)，你可以直接启动前端。

```bash
cd web
npm install
npm run dev
```

浏览器访问终端显示的地址（通常是 `http://localhost:5173`）。

#### 2. 重新训练模型（可选）

如果你想修改模型结构或重新训练：

```bash
# 1. 安装依赖
pip install torch torchvision onnx

# 2. 训练模型 (生成 best.pt)
python train.py

# 3. 导出为 ONNX (生成 web/public/mnist_dnn.onnx)
python export_onnx.py
```

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

### 3.2 Web 端推理 (`web/src/onnx.ts`)

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

### 3.3 可解释性热力图 (`web/src/occlusion.ts`)

- **原理**：遮挡敏感度分析 (Occlusion Sensitivity)。
- **算法**：
  1. 使用滑动窗口（如 4x4）遮挡输入图像的局部区域。
  2. 对遮挡后的图像进行推理，记录目标类别的概率下降幅度。
  3. 概率下降越多，说明被遮挡区域对识别越重要，热力值越高。

---

## 4. 项目目录结构

```
DNN/
├── train.py                # PyTorch 模型训练脚本
├── export_onnx.py          # ONNX 导出脚本
├── best.pt                 # 训练好的 PyTorch 权重
├── web/                    # 前端项目目录
│   ├── public/             # 静态资源
│   │   ├── mnist_dnn.onnx  # 导出的 ONNX 模型
│   │   ├── ort.wasm.min.js # ONNX Runtime UMD 脚本
│   │   └── *.wasm          # WASM 后端文件
│   ├── src/
│   │   ├── App.tsx         # 主 UI 组件
│   │   ├── onnx.ts         # 模型加载与推理逻辑
│   │   └── occlusion.ts    # 热力图生成逻辑
│   ├── index.html          # 入口 HTML
│   └── vite.config.ts      # Vite 配置
└── README.md               # 项目文档
```

---

## 5. 常见问题与解决方案

**Q: 为什么报错 "Failed to load url ... .jsep.mjs"?**
**A:** Vite 在开发模式下对 `public` 目录的文件引用有限制。我们通过切换到 UMD 版本的 ONNX Runtime (`ort.wasm.min.js`) 并显式配置 WASM 路径解决了此问题。

**Q: 为什么报错 "expected magic word 00 61 73 6d"?**
**A:** 这通常是因为浏览器请求 `.wasm` 文件时返回了 HTML（如 404 页面）而非二进制文件。解决方案是确保 `.wasm` 文件真实存在于 `public` 根目录，并且 `onnx.ts` 中的路径配置正确指向根目录 `/`。
