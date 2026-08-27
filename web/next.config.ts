import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

// GitHub Pages（プロジェクトページ）でのホスティングを想定した静的エクスポート設定。
// https://<user>.github.io/gaknui-checker/ 配下に配置されるため、本番ビルドのみ
// basePath/assetPrefixを付与する（ローカル開発時はルート直下のままにする）。
const nextConfig: NextConfig = {
  output: "export",
  basePath: isProd ? "/gaknui-checker" : "",
  assetPrefix: isProd ? "/gaknui-checker/" : "",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
