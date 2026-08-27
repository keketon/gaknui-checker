"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { useClassifier } from "@/hooks/useClassifier";
import type { ClassificationResult } from "@/lib/classifier";

const CLASSIFY_INTERVAL_MS = 800;

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [results, setResults] = useState<ClassificationResult[]>([]);

  const {
    isLoading: isModelLoading,
    error: modelError,
    classify,
  } = useClassifier();

  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error: unknown) {
        setCameraError(
          error instanceof Error
            ? error.message
            : "カメラにアクセスできませんでした",
        );
      }
    };

    startCamera();

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (isModelLoading || cameraError) {
      return;
    }

    const intervalId = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < video.HAVE_CURRENT_DATA) {
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      classify(canvas)
        .then(setResults)
        .catch(() => {
          // 1フレーム分の分類失敗は無視して次のフレームで再試行する
        });
    }, CLASSIFY_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [isModelLoading, cameraError, classify]);

  const topResult = results[0];

  return (
    <main className="relative flex flex-1 flex-col bg-zinc-100">
      <Link
        href="/"
        className="absolute top-4 left-4 z-10 rounded-md bg-white/80 px-3 py-1 text-sm font-medium"
      >
        ← 戻る
      </Link>

      <div className="absolute top-4 left-1/2 z-10 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-center text-white">
        {cameraError || modelError ? (
          <span className="text-red-400">{cameraError ?? modelError}</span>
        ) : isModelLoading ? (
          <span>モデルを読み込み中...</span>
        ) : topResult ? (
          <span className="text-lg font-semibold">
            {topResult.label}（{(topResult.confidence * 100).toFixed(0)}%）
          </span>
        ) : (
          <span>ぬいをカメラに映してください</span>
        )}
      </div>

      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="h-full w-full flex-1 object-cover"
      />
      <canvas ref={canvasRef} className="hidden" />
    </main>
  );
}
