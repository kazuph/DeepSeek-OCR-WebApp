## Web UI Manual Validation Checklist

The following smoke tests confirm that DeepSeek OCR (full precision), DeepSeek OCR (4-bit Quantized),
and YomiToku are wired into the Dockerised Web UI.

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

### DeepSeek OCR (4-bit Quantized) のみ

1. **DeepSeek OCR (4-bit Quantized)** のみをオンにする。
2. 画像を投入し、以下を確認する:
   - テキスト列に 4-bit モデルの結果が表示される。
   - バウンディング画像・切り出し画像が生成され、フル精度版と同様に閲覧できる。
   - 履歴に保存されたバリアントのメタデータに `quantized: true` と `model_repo: Jalea96/DeepSeek-OCR-bnb-4bit-NF4` が含まれ、`elapsed_seconds` が表示される。

### YomiToku のみ

1. モデル選択で **YomiToku Document Analyzer** のみをオンにする。
2. 日本語の文書画像を投入し、以下を確認する:
   - テキスト列に YomiToku の Markdown/プレーンテキストが表示される。
   - バウンディング画像が表示され、コピー/ダウンロードが成功する。
   - （表や図を含む文書の場合）切り出し画像が生成され、カード内で個別にコピー/ダウンロード可能。
   - 「使用モデル」リストで YomiToku が最左に表示され、解析中は「処理中」バッジに更新される。

### モデル進行状況表示

1. DeepSeek OCR（フル精度）と DeepSeek 4-bit、YomiToku の 3 モデルを同時に選択した状態で画像を投入する。
2. 解析開始直後に「使用モデル」欄の各モデルに `待機中` → `処理中` → `完了` の順でステータスバッジが表示されることを確認する。`処理中` バッジの右側に経過秒数が 0.1 秒刻みでカウントアップすること。
3. いずれかのモデルが `完了` になった際、バッジの右側に総処理時間（例: `完了 (5.4秒)`）が表示され続けることを確認する。
4. 処理時間の短い YomiToku が先に完了し、その結果カードが他モデルより先に描画されることを確認する。
5. 各モデルの完了直後に出力カードが追加され、残りのモデルが完了するのを待たずに既に完了した結果を閲覧できる。

### 履歴バーと入力画像プレビュー

1. いずれかのモデルで OCR を完了すると、画面上部の履歴バーにカードが横並びで追加される。溢れたカードは横スクロールで確認できることをチェックする。
2. 履歴カードをクリックすると該当エントリが復元され、左上の「入力画像」グリッドに当時の元画像が `履歴` バッジ付きで表示されることを確認する。
3. 履歴の元画像サムネイルをクリックし、モーダルが全画面プレビューを表示できることを確認する（ズームやコピー/ダウンロードが従来通り動作すること）。
4. 別の履歴を選択した場合にもプレビューが差し替わり、以前の履歴の入力画像が残らないことを確認する。
5. 履歴を復元した直後、「使用モデル」リストの各バッジが完了状態と経過秒数を表示し、当該履歴で走った全モデルが把握できることを確認する。

### マルチモデル比較

1. 3 つすべてのモデルにチェックを入れ、同じ入力を解析する。
2. テキスト/切り出し/バウンディング各セクションで 3 枚のカードが横並びになることを確認する。
   - 処理中は完了済みカードから順次レンダリングされる。
3. DeepSeek / DeepSeek 4-bit / YomiToku の各列でコピー・ダウンロードが独立して動作することを確かめる。
4. 履歴にはモデル単位のエントリが追加されるため、各モデルを個別に選択すると同じ内容が再現されることを確認する。

### エラーハンドリング

- 不正なファイルや空のアップロードを送信した場合、ステータス領域にエラーメッセージが表示され、
  履歴が追加されないことを確認する。
- サーバが停止中（`docker compose stop` 後など）にリクエストすると、クライアント側でエラーが表示される。

### 後処理

- `docker compose down -v` でコンテナを停止し、必要なら `/workspace/web_history` をクリーンアップする。
