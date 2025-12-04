import React, { useEffect, useRef, useState } from "react";
import { infer28x28 } from "./onnx";
import { occlusionHeatmap, renderHeatmap } from "./occlusion";
import { Button, Card, Space, Typography, Switch } from "@douyinfe/semi-ui";

function App() {
  const drawRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [showHeat, setShowHeat] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const c = drawRef.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    let drawing = false;
    let px = 0,
      py = 0;
    const start = (e: PointerEvent) => {
      const rect = c.getBoundingClientRect();
      px = e.clientX - rect.left;
      py = e.clientY - rect.top;
      drawing = true;
      ctx.strokeStyle = "#000";
      ctx.beginPath();
      ctx.moveTo(px, py);
    };
    const stop = () => {
      drawing = false;
      ctx.beginPath();
    };
    const draw = (e: PointerEvent) => {
      if (!drawing) return;
      const rect = c.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.moveTo(x, y);
      px = x;
      py = y;
    };
    c.addEventListener("pointerdown", start);
    c.addEventListener("pointermove", draw);
    window.addEventListener("pointerup", stop);
    return () => {
      c.removeEventListener("pointerdown", start);
      c.removeEventListener("pointermove", draw);
      window.removeEventListener("pointerup", stop);
    };
  }, []);

  const clear = () => {
    const c = drawRef.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
    setProbs(null);
    const h = heatRef.current!;
    h.getContext("2d")!.clearRect(0, 0, h.width, h.height);
  };

  const sample28x28 = () => {
    const src = drawRef.current!;
    const sctx = src.getContext("2d")!;
    const sdata = sctx.getImageData(0, 0, src.width, src.height).data;
    const w = src.width,
      h = src.height;
    // 找到前景的包围盒
    let minX = w,
      minY = h,
      maxX = -1,
      maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const gray = (sdata[i] + sdata[i + 1] + sdata[i + 2]) / 3 / 255;
        const val = 1 - gray;
        if (val > 0.2) {
          // 阈值：忽略浅灰背景噪声
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }
    // 目标 28x28，前景缩放至 ~20 像素边，居中
    const tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext("2d")!;
    tctx.imageSmoothingEnabled = true;
    tctx.fillStyle = "#fff";
    tctx.fillRect(0, 0, 28, 28);
    if (maxX >= minX && maxY >= minY) {
      const sw = Math.max(1, maxX - minX + 1);
      const sh = Math.max(1, maxY - minY + 1);
      const target = 20;
      const scale = target / Math.max(sw, sh);
      const dw = Math.max(1, Math.round(sw * scale));
      const dh = Math.max(1, Math.round(sh * scale));
      const dx = Math.floor((28 - dw) / 2);
      const dy = Math.floor((28 - dh) / 2);
      tctx.filter = "blur(0.5px)";
      tctx.drawImage(src, minX, minY, sw, sh, dx, dy, dw, dh);
      tctx.filter = "none";
    } else {
      // 没有前景，直接缩放整图
      tctx.drawImage(src, 0, 0, 28, 28);
    }
    const data = tctx.getImageData(0, 0, 28, 28).data;
    const arr = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const gray = (r + g + b) / 3 / 255;
      let val = 1 - gray;
      if (val < 0.02) val = 0; // 去除极浅噪声
      arr[i] = val;
    }
    return arr;
  };

  const predict = async () => {
    setBusy(true);
    const img = sample28x28();
    const p = await infer28x28(img);
    setProbs(p);
    setBusy(false);
    if (showHeat) await genHeat(img);
  };

  const genHeat = async (img: Float32Array) => {
    const hcanvas = heatRef.current!;
    const ctx = hcanvas.getContext("2d")!;
    const { heat } = await occlusionHeatmap(img, 4);
    ctx.clearRect(0, 0, hcanvas.width, hcanvas.height);
    renderHeatmap(ctx, heat, 4);
  };

  const toggleHeat = async () => {
    const next = !showHeat;
    setShowHeat(next);
    if (next && probs) await genHeat(sample28x28());
    if (!next) heatRef.current!.getContext("2d")!.clearRect(0, 0, 280, 280);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        background: "#f7f8fa",
      }}
    >
      <Card style={{ width: 960 }}>
        <div
          style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 16 }}
        >
          <div>
            <div
              style={{
                position: "relative",
                width: 280,
                height: 280,
                border: "1px solid #E0E0E0",
                borderRadius: 8,
              }}
            >
              <canvas
                ref={drawRef}
                width={280}
                height={280}
                style={{ position: "absolute", left: 0, top: 0 }}
              />
              <canvas
                ref={heatRef}
                width={280}
                height={280}
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  pointerEvents: "none",
                }}
              />
            </div>
            <Space style={{ marginTop: 12 }}>
              <Button
                theme="solid"
                type="primary"
                onClick={predict}
                loading={busy}
              >
                预测
              </Button>
              <Button onClick={clear}>清空</Button>
              <Space>
                <Switch
                  checked={showHeat}
                  onChange={(checked) => {
                    setShowHeat(checked);
                    if (checked && probs) genHeat(sample28x28());
                    if (!checked)
                      heatRef
                        .current!.getContext("2d")!
                        .clearRect(0, 0, 280, 280);
                  }}
                />
                显示热力图
              </Space>
            </Space>
          </div>
          <div>
            <Typography.Title heading={3}>Top-3 概率</Typography.Title>
            {probs ? (
              <Bar probs={probs} />
            ) : (
              <Typography.Text>
                在左侧画板手写一个数字，然后点击“预测”。
              </Typography.Text>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

function Bar({ probs }: { probs: Float32Array }) {
  const idxs = Array.from(probs)
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3);
  const maxv = idxs[0]?.v ?? 1;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {idxs.map(({ v, i }) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 24, textAlign: "center" }}>{i}</div>
          <div
            style={{
              background: "#1e90ff",
              height: 20,
              width: `${(v / maxv) * 300}px`,
            }}
          ></div>
          <div style={{ width: 60, textAlign: "right" }}>
            {(v * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
