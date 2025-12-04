declare const ort: any

// Map exact filenames to public paths to avoid Vite import issues
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': '/ort-wasm.wasm',
  'ort-wasm-simd.wasm': '/ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
}
// Disable multi-threading and proxy to avoid complex setup issues with Vite
ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false

let session: any = null

export async function loadModel(path = '/mnist_dnn.onnx') {
  if (!session) {
    session = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] })
  }
  return session
}

export function normalize(img: Float32Array) {
  const mean = 0.1307
  const std = 0.3081
  const out = new Float32Array(img.length)
  for (let i = 0; i < img.length; i++) out[i] = (img[i] - mean) / std
  return out
}

export async function infer28x28(img: Float32Array) {
  const sess = await loadModel()
  const input = normalize(img)
  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28])
  const feeds: Record<string, any> = { input: tensor }
  const results = await sess.run(feeds)
  const logits = results['logits'].data as Float32Array
  const probs = softmax(logits)
  return probs
}

export function softmax(logits: Float32Array) {
  const m = Math.max(...logits)
  let sum = 0
  const exps = logits.map(v => Math.exp(v - m))
  for (const e of exps) sum += e
  return Float32Array.from(exps.map(e => e / sum))
}
