import React, { useEffect, useRef, useState } from 'react'
import { infer28x28 } from './onnx'
import { occlusionHeatmap, renderHeatmap } from './occlusion'

function App() {
  const drawRef = useRef<HTMLCanvasElement>(null)
  const heatRef = useRef<HTMLCanvasElement>(null)
  const [probs, setProbs] = useState<Float32Array | null>(null)
  const [showHeat, setShowHeat] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const c = drawRef.current!
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, c.width, c.height)
    ctx.lineWidth = 20
    ctx.lineCap = 'round'
    let drawing = false
    const start = (e: PointerEvent) => { drawing = true; draw(e) }
    const stop = () => { drawing = false }
    const draw = (e: PointerEvent) => {
      if (!drawing) return
      const rect = c.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      ctx.strokeStyle = '#000'
      ctx.beginPath()
      ctx.lineTo(x, y)
      ctx.stroke()
    }
    c.addEventListener('pointerdown', start)
    c.addEventListener('pointermove', draw)
    window.addEventListener('pointerup', stop)
    return () => {
      c.removeEventListener('pointerdown', start)
      c.removeEventListener('pointermove', draw)
      window.removeEventListener('pointerup', stop)
    }
  }, [])

  const clear = () => {
    const c = drawRef.current!
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, c.width, c.height)
    setProbs(null)
    const h = heatRef.current!
    h.getContext('2d')!.clearRect(0, 0, h.width, h.height)
  }

  const sample28x28 = () => {
    const src = drawRef.current!
    const tmp = document.createElement('canvas')
    tmp.width = 28; tmp.height = 28
    const tctx = tmp.getContext('2d')!
    tctx.drawImage(src, 0, 0, 28, 28)
    const data = tctx.getImageData(0, 0, 28, 28).data
    const arr = new Float32Array(28 * 28)
    for (let i = 0; i < 28 * 28; i++) {
      const r = data[i * 4]
      const g = data[i * 4 + 1]
      const b = data[i * 4 + 2]
      const a = data[i * 4 + 3]
      const gray = (r + g + b) / 3 / 255
      const val = (1 - gray) // 黑笔在白底，越黑值越大
      arr[i] = val
    }
    return arr
  }

  const predict = async () => {
    setBusy(true)
    const img = sample28x28()
    const p = await infer28x28(img)
    setProbs(p)
    setBusy(false)
    if (showHeat) await genHeat(img)
  }

  const genHeat = async (img: Float32Array) => {
    const hcanvas = heatRef.current!
    const ctx = hcanvas.getContext('2d')!
    const { heat } = await occlusionHeatmap(img, 4)
    ctx.clearRect(0, 0, hcanvas.width, hcanvas.height)
    renderHeatmap(ctx, heat, 4)
  }

  const toggleHeat = async () => {
    const next = !showHeat
    setShowHeat(next)
    if (next && probs) await genHeat(sample28x28())
    if (!next) heatRef.current!.getContext('2d')!.clearRect(0, 0, 280, 280)
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
      <div>
        <div style={{ position: 'relative', width: 280, height: 280, border: '1px solid #ddd' }}>
          <canvas ref={drawRef} width={280} height={280} style={{ position: 'absolute', left: 0, top: 0 }} />
          <canvas ref={heatRef} width={280} height={280} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }} />
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
          <button onClick={predict} disabled={busy}>预测</button>
          <button onClick={clear}>清空</button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={showHeat} onChange={toggleHeat} /> 显示热力图
          </label>
        </div>
      </div>
      <div>
        <h3>Top-3 概率</h3>
        {probs ? <Bar probs={probs} /> : <p>在左侧画板手写一个数字，然后点击“预测”。</p>}
      </div>
    </div>
  )
}

function Bar({ probs }: { probs: Float32Array }) {
  const idxs = Array.from(probs).map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v).slice(0, 3)
  const maxv = idxs[0]?.v ?? 1
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {idxs.map(({ v, i }) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 24, textAlign: 'center' }}>{i}</div>
          <div style={{ background: '#1e90ff', height: 20, width: `${(v / maxv) * 300}px` }}></div>
          <div style={{ width: 60, textAlign: 'right' }}>{(v * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  )
}

export default App

