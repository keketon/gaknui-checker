"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  loadClassifier,
  type Classifier,
  type ClassificationResult,
} from "@/lib/classifier";

interface UseClassifierResult {
  isLoading: boolean;
  error: string | null;
  classify: (source: CanvasImageSource) => Promise<ClassificationResult[]>;
}

export function useClassifier(): UseClassifierResult {
  const classifierRef = useRef<Classifier | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;

    loadClassifier()
      .then((classifier) => {
        if (isCancelled) {
          return;
        }
        classifierRef.current = classifier;
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
      if (!classifierRef.current) {
        throw new Error("モデルがまだ読み込まれていません");
      }
      return classifierRef.current.classify(source);
    },
    [],
  );

  return { isLoading, error, classify };
}
