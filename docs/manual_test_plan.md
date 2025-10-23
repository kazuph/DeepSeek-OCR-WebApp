## Web UI Manual Validation Checklist

The following smoke tests confirm that both DeepSeek OCR and YomiToku are wired into the
Dockerised Web UI.

### Environment Preparation

1. Build and launch the stack (GPU host with NVIDIA Container Toolkit):

   ```bash
   DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose up -d --build
   ```

2. Wait for the container to download model weights. Verify the FastAPI server is reachable on
   `http://localhost:8080/api/ping` (expect `{ "status": "ok" }`).

### DeepSeek OCR Only

1. In the Web UI, uncheck all models, then enable **DeepSeek OCR** only.
2. Drop a sample image/PDF. Confirm that:
   - The status ribbon completes without errors.
   - “抽出テキスト” shows Markdown in the DeepSeek column.
   - Bounding image (および切り出し画像がある場合)が 1 枚表示され、コピー/ダウンロードボタンが動作する。
3. 履歴一覧に最新エントリが追加され、再選択で同じ結果が読み込める。

### YomiToku のみ

1. モデル選択で **YomiToku Document Analyzer** のみをオンにする。
2. 日本語の文書画像を投入し、以下を確認する:
   - テキスト列に YomiToku の Markdown/プレーンテキストが表示される。
   - バウンディング画像が表示され、コピー/ダウンロードが成功する。
   - （表や図を含む文書の場合）切り出し画像が生成され、カード内で個別にコピー/ダウンロード可能。

### DeepSeek + YomiToku 並列

1. 両モデルにチェックを入れ、同じ入力を解析する。
2. テキスト/切り出し/バウンディング各セクションで 2 枚のカードが横並びになること、および
   モデルごとに独立したコピー・ダウンロード操作ができることを確かめる。
3. 履歴から再読み込みしても同じ構成で比較表示されることを確認する。

### エラーハンドリング

- 不正なファイルや空のアップロードを送信した場合、ステータス領域にエラーメッセージが表示され、
  履歴が追加されないことを確認する。
- サーバが停止中（`docker compose stop` 後など）にリクエストすると、クライアント側でエラーが表示される。

### 後処理

- `docker compose down -v` でコンテナを停止し、必要なら `/workspace/web_history` をクリーンアップする。

