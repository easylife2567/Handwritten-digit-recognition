## 目标
- 以最少文件完成：训练 DNN、导出 ONNX、React 前端纯浏览器推理与创新可视化（遮挡敏感度热力图）。

## 可视化（简洁且创新）
- 片块遮挡敏感度热力图：按 4×4 小片块遮挡输入，概率下降幅度→热力图覆盖，直观展示模型关注区域；纯前端实现，无梯度与后端。

## 代码与依赖（最小集合）
- Python 仅两文件：
  - `train.py`：训练 MLP(784→256→128→10)，Adam(lr=1e-3)，batch=128，5~10 epochs，保存最优；使用 `torchvision.datasets.MNIST` 与标准化。
  - `export_onnx.py`：加载最优权重，`torch.onnx.export` 生成 `public/mnist_dnn.onnx`（opset≥13，批量1）。
- 前端仅一个工程：
  - 依赖：`onnxruntime-web`；不引入 chart.js 等第三方图表库。
  - 主要文件：
    - `src/App.tsx`：单页（画板、Top-3 概率条、热力图开关与覆盖）。
    - `src/onnx.ts`：模型加载与推理。
    - `src/occlusion.ts`：遮挡敏感度计算（在内存中对 28×28 进行片块替换）。
    - `public/mnist_dnn.onnx`：模型文件。
  - 说明：用原生 `<canvas>` + 简易 CSS 绘制概率条，不额外组件库。

## 运行流程
1. 后端（离线产物）：
   - `python -m venv venv && source venv/bin/activate`
   - `pip install torch torchvision onnx`
   - `python train.py && python export_onnx.py`（生成 `mnist_dnn.onnx`）。
2. 前端（React 最小项目）：
   - `npm create vite@latest web -- --template react-ts`
   - `cd web && npm i onnxruntime-web`
   - 将 `mnist_dnn.onnx` 置于 `web/public/`
   - `npm run dev`，在页面手写、查看 Top-3 与热力图。

## 验证标准
- 测试准确率≥97%；ONNX 前端推理与后端一致。
- React 单页包含：画板、Top-3 概率条、热力图开关与覆盖；无多余依赖或组件。

## 约束
- 严控文件数量与依赖，避免冗余；能复现与演示即止。

如确认，我将按照本最小化方案开始实现与交付。