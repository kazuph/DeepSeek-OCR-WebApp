<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


## Web UI クイックスタート（Docker）

1. このリポジトリをクローンし、GPU ドライバと Docker + NVIDIA Container Toolkit がセットアップされたホストで作業してください。
2. コンテナをビルドして起動します。

   ```bash
   DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose up -d --build
   # エイリアス dcub を利用している場合:
   DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 dcub -d
   ```

3. 初回起動時はモデルと依存関係のダウンロードに時間が掛かります。完了すると `uvicorn` が `0.0.0.0:8080` で待機します。
4. 同一ネットワーク上のブラウザから `http://<サーバーホスト名または tailscale ホスト名>:8080/` にアクセスしてください。
5. OCR 解析結果や履歴は `/workspace/web_history` に保存されます。必要に応じてバックアップ・クリーンアップを行ってください。


<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>

### Web UI のマルチモデル比較について

- Docker イメージには `yomitoku>=0.8.0` を追加しており、DeepSeek OCR（フル精度）に加えて日本語特化の YomiToku Document Analyzer を利用できます。
- CUDA GPU 環境で `pip install -U bitsandbytes accelerate` を行うと、`Jalea96/DeepSeek-OCR-bnb-4bit-NF4`（BitsAndBytes 4-bit 量子化版）も選択肢に追加され、3 モデル併用による比較が可能です。
- 画面上部の「使用モデル」で各チェックボックスを切り替えると、選択したモデルのみで推論を行います。複数選択時はテキスト / 切り出し画像 / バウンディング画像がモデル単位で並列表示されます。
- 解析結果や履歴はモデル別に保存されるため、履歴からは各モデルを個別に再確認できます。
- 手動の確認手順は [docs/manual_test_plan.md](docs/manual_test_plan.md) にまとめています。

## REST API

Web UI と同じ FastAPI アプリケーションが JSON ベースの API を公開しています。Docker 版では `uvicorn` が `0.0.0.0:8080` で待機し、CORS は `Allow-Origin: *` に設定済みなので外部ホストから直接アクセスできます。認証は現時点でありませんので、外部公開時はリバースプロキシなどで保護してください。

| メソッド | パス | 説明 |
| --- | --- | --- |
| `GET` | `/api/ping` | サーバーの疎通確認。 `{ "status": "ok" }` を返します。 |
| `GET` | `/api/models` | 利用可能な OCR モデルの一覧。`key` と `label` を返します。 |
| `POST` | `/api/ocr` | ファイルをアップロードして OCR を実行。フォームデータで `file`（必須）、`prompt`（任意）、`models`（カンマ区切り／任意）を渡します。複数モデルを指定すると `variants` 配列で集約結果が返ります。 |
| `GET` | `/api/history` | 保存済み履歴の一覧。直近のエントリが降順で返り、プレビュー用のテキスト／画像 URL が含まれます。 |
| `GET` | `/api/history/{entry_id}` | 特定履歴の詳細。`variants` にモデルごとの出力、`input_images` に元入力のダウンロード URL が含まれます。 |
| `DELETE` | `/api/history/{entry_id}` | 指定された履歴を削除します。関連ファイルは `web_history/{entry_id}` から削除されます。 |
| `GET` | `/api/history/{entry_id}/image/bounding` | 保存済みバウンディング画像を取得。`model` クエリでバリアントを指定できます。 |
| `GET` | `/api/history/{entry_id}/image/crop/{path}` | 履歴から個別クロップを取得。`{path}` は `variants[*].crops[].path` と一致させます。 |
| `GET` | `/api/history/{entry_id}/image/input/{path}` | OCR 実行時にアップロードされた元ファイルを取得します。`{path}` は `input_images[].path` に対応します。 |

### cURL での利用例

#### `/api/models`

```bash
curl http://<host>:8080/api/models | jq
```

#### `/api/ocr`

```bash
curl -X POST \
  -F "file=@tests/fixtures/doc.png" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown." \
  -F "models=deepseek,yomitoku" \
  http://<host>:8080/api/ocr | jq
```

レスポンスでは `history_id` に保存済みエントリの ID、`variants` にモデルごとの結果、`input_images` に元ファイルのアクセス URL が含まれます。複数モデルを指定した場合は `variants` に各モデルの枠が順番に入ります。

#### `/api/history` と `/api/history/{id}`

```bash
# 一覧を取得
curl http://<host>:8080/api/history | jq

# ID を指定して詳細を取得
curl http://<host>:8080/api/history/20251029021741-1fb89c21 | jq
```

詳細レスポンスでは `variants[*].text_markdown` や `variants[*].crops[*].url` を使ってモデル別の成果物へアクセスできます。`input_images[*].url` はアップロード時の元ファイルをそのまま返すダウンロード URL です。

#### 画像／入力のダウンロード

```bash
# 特定のクロップ画像を取得
curl -o crop.png \
  "http://<host>:8080/api/history/20251029021741-1fb89c21/image/crop/artifacts/images/0.jpg?model=deepseek"

# 元の入力画像を取得
curl -o original.png \
  http://<host>:8080/api/history/20251029021741-1fb89c21/image/input/image.png
```

### 外部公開時の注意

- コンテナは既定で `0.0.0.0` にバインドし、CORS も許可済みです。社外公開する場合は HTTPS 化や Basic 認証などの追加保護を検討してください。
- `web_history/` ディレクトリに入力画像・出力テキストが平文で保存されます。定期的なクリーンアップや暗号化ストレージの活用を推奨します。
- 大きなファイルを扱う場合、`nginx` や `traefik` などのリバースプロキシでタイムアウトや帯域を調整すると安定します。




<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>



<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href="https://arxiv.org/abs/2510.18234"><b>📄 Arxiv Paper Link</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR: Contexts Optical Compression</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">Explore the boundaries of visual-text compression.</a>       
</p>

## Release
- [2025/10/20]🚀🚀🚀 We release DeepSeek-OCR, a model to investigate the role of vision encoders from an LLM-centric viewpoint.

## Contents
- [Install](#install)
- [vLLM Inference](#vllm-inference)
- [Transformers Inference](#transformers-inference)
  




## Install
>Our environment is cuda11.8+torch2.6.0.
1. Clone this repository and navigate to the DeepSeek-OCR folder
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
```
2. Conda
```Shell
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```
3. Packages

- download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) 
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```
**Note:** if you want vLLM and transformers codes to run in the same environment, you don't need to worry about this installation error like: vllm 0.8.5+cu118 requires transformers>=4.51.1

## vLLM-Inference
- VLLM:
>**Note:** change the INPUT_PATH/OUTPUT_PATH and other settings in the DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```
1. image: streaming output
```Shell
python run_dpsk_ocr_image.py
```
2. pdf: concurrency ~2500tokens/s(an A100-40G)
```Shell
python run_dpsk_ocr_pdf.py
```
3. batch eval for benchmarks
```Shell
python run_dpsk_ocr_eval_batch.py
```
## Transformers-Inference
- Transformers
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
```
or you can
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```
## Support-Modes
The current open-source model supports the following modes:
- Native resolution:
  - Tiny: 512×512 （64 vision tokens）✅
  - Small: 640×640 （100 vision tokens）✅
  - Base: 1024×1024 （256 vision tokens）✅
  - Large: 1280×1280 （400 vision tokens）✅
- Dynamic resolution
  - Gundam: n×640×640 + 1×1024×1024 ✅

## Prompts examples
```python
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
```


## Visualizations
<table>
<tr>
<td><img src="assets/show1.jpg" style="width: 500px"></td>
<td><img src="assets/show2.jpg" style="width: 500px"></td>
</tr>
<tr>
<td><img src="assets/show3.jpg" style="width: 500px"></td>
<td><img src="assets/show4.jpg" style="width: 500px"></td>
</tr>
</table>


## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).

## Citation

```bibtex
@article{wei2024deepseek-ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
