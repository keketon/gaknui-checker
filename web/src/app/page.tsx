"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ChangeEvent } from "react";

import { ResultList } from "@/components/ResultList";
import { useClassifier } from "@/hooks/useClassifier";
import type { ClassificationResult } from "@/lib/classifier";

export default function HomePage() {
  const imageRef = useRef<HTMLImageElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [results, setResults] = useState<ClassificationResult[]>([]);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const {
    isLoading: isModelLoading,
    error: modelError,
    classify,
  } = useClassifier();

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setPreviewUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setIsImageLoaded(false);
    setResults([]);
    setClassifyError(null);
  };

  const handleClassify = async () => {
    const image = imageRef.current;
    if (!image) {
      return;
    }

    setIsClassifying(true);
    setClassifyError(null);

    try {
      const classificationResults = await classify(image);
      setResults(classificationResults);
    } catch (error: unknown) {
      setClassifyError(
        error instanceof Error ? error.message : "分類に失敗しました",
      );
    } finally {
      setIsClassifying(false);
    }
  };

  const isClassifyDisabled =
    !previewUrl || !isImageLoaded || isModelLoading || isClassifying;

  return (
    <main className="flex flex-1 flex-col items-center gap-6 px-4 py-12">
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-semibold">がくぬいチェッカー</h1>
        <p className="text-sm text-gray-500">
          画像をアップロードして、ぬいのキャラクターを判定します
        </p>
      </div>

      <div className="flex w-full max-w-sm flex-col items-center gap-4">
        <div className="flex items-center gap-3">
          <label
            htmlFor="nui-image-input"
            className="cursor-pointer rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            ファイルを選択
          </label>
          <input
            id="nui-image-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          {fileName && (
            <span className="text-sm text-gray-500">{fileName}</span>
          )}
        </div>

        {previewUrl && (
          // eslint-disable-next-line @next/next/no-img-element -- ローカルで選択したファイルのプレビュー表示のため
          <img
            ref={imageRef}
            src={previewUrl}
            alt="アップロードされたぬいの画像"
            onLoad={() => setIsImageLoaded(true)}
            className="h-64 w-64 rounded-lg border border-gray-200 object-cover"
          />
        )}

        <button
          type="button"
          onClick={handleClassify}
          disabled={isClassifyDisabled}
          className="w-full rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white transition-opacity disabled:opacity-40"
        >
          {isModelLoading
            ? "モデルを読み込み中..."
            : isClassifying
              ? "判定中..."
              : "判定する"}
        </button>

        {(modelError || classifyError) && (
          <p className="text-sm text-red-600">{modelError ?? classifyError}</p>
        )}

        <ResultList results={results} />
      </div>

      <Link
        href="/camera"
        className="mt-4 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        カメラで判定する
      </Link>
    </main>
  );
}
