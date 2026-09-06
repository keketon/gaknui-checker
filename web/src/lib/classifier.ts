import * as ort from "onnxruntime-web";

export interface ClassificationResult {
  label: string;
  confidence: number;
}

export type Classify = (
  source: CanvasImageSource,
) => Promise<ClassificationResult[]>;

export async function loadClassifier(): Promise<Classify> {
  const classLabels: string[] = await fetch("/models/class_to_idx.json").then(
    (res) => res.json(),
  );

  const inferenceSession = await ort.InferenceSession.create(
    "/models/mobilenet_v3_large_nui.onnx",
  );

  return async (source) => {
    const tensor = preprocessImage(source);
    const inferResult = await inferenceSession.run({ image: tensor });
    // Python側でtorch.onnx.exportしたときにf32を指定していたので、キャストできることが暗に保証されている
    const probabilities = softmax(inferResult["logits"].data as Float32Array);

    return classLabels.map((label, i) => ({
      label: label,
      confidence: probabilities[i],
    }));
  };
}

const MODEL_INPUT_SIZE = 224;

function preprocessImage(source: CanvasImageSource): ort.Tensor {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = MODEL_INPUT_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.error("Failed to get canvas context");
    throw new Error("Failed to get canvas context");
  }

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const { width, height } = getSourceSize(source);

  const scale = MODEL_INPUT_SIZE / Math.max(width, height);
  const drawWidth = width * scale,
    drawHeight = height * scale;
  const offsetX = (MODEL_INPUT_SIZE - drawWidth) / 2,
    offsetY = (MODEL_INPUT_SIZE - drawHeight) / 2;

  ctx.drawImage(source, offsetX, offsetY, drawWidth, drawHeight);
  const pixels = ctx.getImageData(
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  ).data;
  const normalizedPixels = normalizePixelsToChw(pixels, MODEL_INPUT_SIZE);

  return new ort.Tensor("float32", normalizedPixels, [
    1,
    3,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  ]);
}

function getSourceSize(source: CanvasImageSource): {
  width: number;
  height: number;
} {
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  if (source instanceof HTMLCanvasElement) {
    return { width: source.width, height: source.height };
  }
  throw new Error("未対応のsourceの型です");
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

function normalizePixelsToChw(
  pixels: Uint8ClampedArray,
  size: number,
): Float32Array {
  const numPixels = size * size;
  const chw = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;

    chw[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    chw[numPixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    chw[2 * numPixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return chw;
}

function softmax(logits: Float32Array | number[]): number[] {
  // Logitの値が大きいとe^logitの値が発散してしまう(~e^100程度でも顕在化する)ので、各logicから最大値を引く。
  // どのみち e^logit 同士を除算するので最終的な計算結果には影響しない。
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits, (x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((sum, x) => sum + x, 0);
  return exps.map((x) => x / sumExps);
}
