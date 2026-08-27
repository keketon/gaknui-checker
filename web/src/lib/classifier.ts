/**
 * ぬい分類モデルとのインターフェース。
 *
 * ONNX Runtime Webを使った実装はここに書く（モデル連携部分は自前実装のため未実装）。
 *
 * 実装時にやること:
 * 1. 学習済みモデル（models/mobilenet_v3_large_nui.pth）をONNXに変換し、
 *    `public/models/`配下に配置する
 * 2. `npm install onnxruntime-web`
 * 3. 入力画像を224x224にリサイズし、ImageNetのmean/stdで正規化する
 *    （../../../docs/transfer-learning-notes.md の Normalize の項を参照）
 * 4. `ort.InferenceSession.create(...)`でモデルをロードする
 * 5. `classify()`で推論を実行し、confidence降順のClassificationResult[]を返す
 * 6. CLASS_LABELSの並び順を、学習時のclass_to_idx（.pthに保存済み）に合わせる
 */

export interface ClassificationResult {
  label: string;
  confidence: number;
}

export interface Classifier {
  classify(source: CanvasImageSource): Promise<ClassificationResult[]>;
}

// TODO: 学習時のclass_to_idxと順序を合わせる（例: ["kotone", "saki"]）
export const CLASS_LABELS: readonly string[] = [];

export async function loadClassifier(): Promise<Classifier> {
  throw new Error(
    "TODO: src/lib/classifier.ts にONNX Runtime Webでのモデルロード処理を実装してください",
  );
}
