// 声明 ONNX Runtime Web 的全局变量
declare const ort: any

// 显式映射 WASM 文件路径，避免 Vite 打包时路径解析错误
// Vite 在开发模式下对 public 目录的引用有限制，手动指定可以确保加载正确
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': '/ort-wasm.wasm',
  'ort-wasm-simd.wasm': '/ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
}

// 禁用多线程和代理模式，简化在浏览器环境中的配置，避免复杂的跨域隔离（COOP/COEP）问题
ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false

// 全局单例 session，避免重复创建开销
let session: any = null

/**
 * 加载 ONNX 模型
 * @param path 模型文件的路径，默认为 public 目录下的 mnist_dnn.onnx
 */
export async function loadModel(path = '/mnist_dnn.onnx') {
  if (!session) {
    // 创建推理会话，指定使用 'wasm' 后端
    session = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] })
  }
  return session
}

/**
 * 图像归一化处理
 * 将输入图像的像素值标准化，使其分布与训练数据一致 (Mean=0.1307, Std=0.3081)
 * @param img 输入的灰度图像数据 (0-1之间)
 */
export function normalize(img: Float32Array) {
  const mean = 0.1307
  const std = 0.3081
  const out = new Float32Array(img.length)
  for (let i = 0; i < img.length; i++) out[i] = (img[i] - mean) / std
  return out
}

/**
 * 执行推理的核心函数
 * @param img 28x28 的输入图像数据
 * @returns 长度为 10 的概率数组
 */
export async function infer28x28(img: Float32Array) {
  const sess = await loadModel()
  // 1. 预处理：标准化
  const input = normalize(img)
  // 2. 构造 Tensor：形状为 [Batch=1, Channel=1, Height=28, Width=28]
  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28])
  // 3. 准备输入 feeds，key 必须与模型导出时的 input_names 一致
  const feeds: Record<string, any> = { input: tensor }
  // 4. 运行推理
  const results = await sess.run(feeds)
  // 5. 获取输出，key 必须与模型导出时的 output_names 一致
  const logits = results['logits'].data as Float32Array
  // 6. 后处理：Softmax 转换为概率
  const probs = softmax(logits)
  return probs
}

/**
 * Softmax 激活函数
 * 将 Logits (未归一化的得分) 转换为概率分布 (和为 1)
 */
export function softmax(logits: Float32Array) {
  // 数值稳定性优化：减去最大值防止溢出
  const m = Math.max(...logits)
  let sum = 0
  const exps = logits.map(v => Math.exp(v - m))
  for (const e of exps) sum += e
  return Float32Array.from(exps.map(e => e / sum))
}
