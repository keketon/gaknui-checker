import type { ClassificationResult } from "@/lib/classifier";

interface ResultListProps {
  results: ClassificationResult[];
}

export function ResultList({ results }: ResultListProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <ul className="w-full max-w-sm divide-y divide-gray-200 rounded-lg border border-gray-200">
      {results.map((result) => (
        <li
          key={result.label}
          className="flex items-center justify-between px-4 py-2 text-sm"
        >
          <span className="font-medium">{result.label}</span>
          <span className="text-gray-500">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </li>
      ))}
    </ul>
  );
}
