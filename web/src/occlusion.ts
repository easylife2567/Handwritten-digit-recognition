import { infer28x28 } from "./onnx";

export async function occlusionHeatmap(base: Float32Array, tile = 4) {
  const H = 28,
    W = 28;
  const baseline = await infer28x28(base);
  const targetIdx = argmax(baseline);
  const heat = new Float32Array((H / tile) * (W / tile));

  let idx = 0;
  for (let y = 0; y < H; y += tile) {
    for (let x = 0; x < W; x += tile) {
      const perturbed = base.slice();
      for (let yy = y; yy < Math.min(y + tile, H); yy++) {
        for (let xx = x; xx < Math.min(x + tile, W); xx++) {
          perturbed[yy * W + xx] = 0;
        }
      }
      const probs = await infer28x28(perturbed);
      const drop = baseline[targetIdx] - probs[targetIdx];
      heat[idx++] = Math.max(0, drop);
    }
  }
  // normalize to [0,1]
  const maxv = Math.max(...heat);
  if (maxv > 0) for (let i = 0; i < heat.length; i++) heat[i] /= maxv;
  return { heat, targetIdx };
}

function argmax(arr: Float32Array) {
  let m = -Infinity,
    mi = -1;
  for (let i = 0; i < arr.length; i++)
    if (arr[i] > m) {
      m = arr[i];
      mi = i;
    }
  return mi;
}

export function renderHeatmap(
  ctx: CanvasRenderingContext2D,
  heat: Float32Array,
  tile = 4
) {
  const W = 28,
    H = 28;
  const cols = W / tile;
  const rows = H / tile;
  const scaleX = ctx.canvas.width / W;
  const scaleY = ctx.canvas.height / H;
  let idx = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = heat[idx++];
      const alpha = v;
      ctx.fillStyle = `rgba(255,0,0,${alpha})`;
      ctx.fillRect(
        c * tile * scaleX,
        r * tile * scaleY,
        tile * scaleX,
        tile * scaleY
      );
    }
  }
}
