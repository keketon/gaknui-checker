"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  loadClassifier,
  type Classify,
  type ClassificationResult,
} from "@/lib/classifier";

interface UseClassifierResult {
  isLoading: boolean;
  error: string | null;
  classify: (source: CanvasImageSource) => Promise<ClassificationResult[]>;
}

export function useClassifier(): UseClassifierResult {
  const classifyRef = useRef<Classify | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;

    loadClassifier()
      .then((classify) => {
        if (isCancelled) {
          return;
        }
        classifyRef.current = classify;
        setIsLoading(false);
      })
      .catch((loadError: unknown) => {
        if (isCancelled) {
          return;
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : "モデルの読み込みに失敗しました",
        );
        setIsLoading(false);
      });

    return () => {
      isCancelled = true;
    };
  }, []);

  const classify = useCallback(
    async (source: CanvasImageSource): Promise<ClassificationResult[]> => {
      if (!classifyRef.current) {
        throw new Error("モデルがまだ読み込まれていません");
      }
      return classifyRef.current(source);
    },
    [],
  );

  return { isLoading, error, classify };
}
