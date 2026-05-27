# 愛知近海 釣り船 釣果予測ツール

愛知県近海（伊勢湾・三河湾・遠州灘）の釣り船向けに、**出船前に分かる情報だけで魚種別の期待釣果を予測**するツール群。

- **実釣果の計数**: GenesisEngine-v6（YOLO）が釣果写真から自動カウント → `count` 列に格納（学習ラベル）
- **予測の入力**: 気象・海象・潮汐・天文・予約メタ（船宿/人数/仕掛け/狙い） — 出船前に確定する量のみ
- **配信**: FastAPI 経由でWeb/アプリへ（MVP は `app/` の Streamlit ダッシュボード）

## Streamlit ダッシュボード

```
app/
├── streamlit_app.py         # トップ（概要ステータス）
├── utils.py                 # 共通: データロード / predictor ラッパー
└── pages/
    ├── 1_🎣_単発予測.py
    ├── 2_⚔️_船宿ランキング.py
    ├── 3_📊_データ詳細.py
    └── 4_🔄_最新エントリ取り込み.py
```

### Colab で起動（ngrok 経由で公開 URL）

```python
# 必要なライブラリ
!pip install -q streamlit pyngrok plotly

from google.colab import drive, userdata
drive.mount('/content/drive', force_remount=False)

# ngrok auth token を Colab userdata に NGROK_AUTH_TOKEN として登録しておく
# 取得: https://dashboard.ngrok.com/get-started/your-authtoken
from pyngrok import ngrok, conf
ngrok.set_auth_token(userdata.get("NGROK_AUTH_TOKEN"))

# LLM API キーも環境変数にエクスポート（utils.py がここから拾う）
import os
for k in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
    try:
        os.environ[k] = userdata.get(k)
    except Exception:
        pass

# 既存トンネルを切る（多重起動防止）
for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)

# Streamlit をバックグラウンドで起動
APP_DIR = "/content/drive/MyDrive/aichi-fishing-analysis/app"
get_ipython().system_raw(
    f"cd {APP_DIR} && streamlit run streamlit_app.py "
    "--server.port 8501 --server.headless true "
    "--server.fileWatcherType none > /content/streamlit.log 2>&1 &"
)

import time; time.sleep(5)
public_url = ngrok.connect(8501).public_url
print("🌐 公開 URL:", public_url)
print("📜 ログ:    !tail -f /content/streamlit.log")
```

### ローカル Windows で起動

```powershell
cd I:\マイドライブ\aichi-fishing-analysis
pip install streamlit plotly
$env:CEREBRAS_API_KEY = "your_key"
$env:GROQ_API_KEY     = "your_key"
streamlit run app\streamlit_app.py
```

## ディレクトリ構成

```
aichi-fishing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── weather/               # Open-Meteo（気象+海象）キャッシュ
│   ├── tide/                  # JMA 潮汐表キャッシュ（年単位）
│   ├── fishing_logs/
│   │   └── catches_template.csv
│   ├── integrated/            # 釣果 × 気象/潮汐/天文 結合済み
│   ├── images/                # 釣果写真（YOLO入力）
│   └── models/                # 学習済みモデル(.joblib)
├── src/
│   ├── config.py
│   ├── font_setup.py          # matplotlib 日本語フォント
│   ├── weather_fetcher.py     # Open-Meteo Forecast/Marine（風・気圧・波・海流・SST・SSH）
│   ├── tide_fetcher.py        # JMA 年次潮汐表
│   ├── astronomy.py           # 日出日没・月齢・潮回り・マヅメ判定
│   ├── fishing_loader.py
│   ├── data_integrator.py     # 釣果に全部結合
│   ├── features.py            # 特徴量行列（学習・推論で同一スキーマ）
│   ├── predictor.py           # 学習・推論
│   ├── visualizer.py
│   └── yolo_predictor.py      # 互換用スタブ（実体は GenesisEngine-v6）
├── notebooks/
└── api/
    └── server.py
```

## 役割分担

```
                              【ラベル供給】
                釣果写真 ──→ GenesisEngine-v6 (YOLO) ──→ count 列
                                                          │
【予測モデルの入力（出船前に確定する量だけ）】              ▼
  ┌─ Open-Meteo Forecast/Marine（気温・風・気圧・波・海流・SST・海面高度）
  ├─ JMA 潮汐表（毎時潮位 cm）
  ├─ astral / 月齢計算（日出日没・月齢・潮回り・マヅメ）
  └─ 予約メタ（船宿・乗船人数・仕掛け・狙い魚種）
                       │
                       ▼
              data_integrator → features → predictor.train / predict
                                                    │
                                                    ▼
                                            FastAPI /predict → Web/アプリ
```

## 入力特徴量（すべて出船前に確定）

| カテゴリ | 列                                                                                     | 出所                |
|----------|----------------------------------------------------------------------------------------|---------------------|
| 時刻     | month / day_of_year / weekday / is_weekend / hour_sin/cos / doy_sin/cos                | datetime            |
| 地点     | site_{shinojima/morozaki/irago/mikawa_bay/chita_tip}                                   | site                |
| 気象     | 気温・湿度・気圧・風速・突風・雲量・降水                                               | Open-Meteo Forecast |
| 風向     | wind_dir_sin/cos                                                                       | Open-Meteo          |
| 波       | 波高・波周期・うねり波高・うねり周期                                                   | Open-Meteo Marine   |
| 海流     | 流速・流向 sin/cos                                                                     | Open-Meteo Marine   |
| 海水温   | sea_surface_temperature                                                                | Open-Meteo Marine   |
| 海面高度 | sea_level_height_msl                                                                   | Open-Meteo Marine   |
| 潮汐     | tide_cm（毎時潮位）                                                                    | JMA                 |
| 天文     | moon_age / moon_phase（新月〜有明 8値）/ tide_phase（大潮〜若潮 5値）                  | astral + 月齢計算   |
| 時刻帯   | sunrise_hour / sunset_hour / is_morning_mazume / is_evening_mazume                     | astral              |
| 予約メタ | anglers / departure_hour / target_match / tackle（サビキ/ジギング等 9値）              | 釣果ログ            |

`count`（釣果数）はラベルとしてのみ使用、入力には含まれない。

## クイックスタート

```bash
pip install -r requirements.txt

# 1. 釣果ログ整備（YOLO計数済み or 空欄）
#    data/fishing_logs/catches.csv に置く

# 2. 統合データセット作成（気象+潮汐+天文を結合）
python -m src.data_integrator --catches data/fishing_logs/catches.csv

# 3. 魚種別モデル学習
python -m src.predictor train --species アジ

# 4. 出船前情報から予測
python -m src.predictor predict \
    --site shinojima --species アジ --date 2026-05-09 \
    --hour 5 --boat 篠島丸 --anglers 8 --tackle サビキ

# 5. APIサーバ
uvicorn api.server:app --reload
# GET /predict?site=shinojima&species=アジ&date=2026-05-09&hour=5&boat=篠島丸&anglers=8&tackle=サビキ
```

## 想定観測地点と JMA 潮汐港

| 地点コード   | 地点名         | 緯度    | 経度     | 潮汐港 |
|--------------|----------------|---------|----------|--------|
| shinojima    | 篠島           | 34.6700 | 136.9700 | IO（伊良湖代用） |
| morozaki     | 師崎           | 34.7100 | 136.9800 | IO |
| irago        | 伊良湖         | 34.5800 | 137.0200 | IO |
| mikawa_bay   | 三河湾中央     | 34.7500 | 137.0500 | IO |
| chita_tip    | 知多半島先端   | 34.6900 | 136.9700 | NA（名古屋） |

JMA 港コード一覧: <https://www.data.jma.go.jp/kaiyou/db/tide/suisan/listall.php>

## YOLO（GenesisEngine-v6）との接続

GenesisEngine-v6 は **自動モデル学習器** として独立運用（別フォルダ: `I:\マイドライブ\GenesisEngine-v6\`）。
そこから書き出された `best.pt` を本リポの `data/models/yolo_unified/` に置いて利用する。

本リポは `count` のラベル供給元として YOLO を扱い、予測モデルの入力には画像由来情報を使わない（推論時に画像が無い前提のため）。

## 釣果ログの seed データを作る（船宿ブログ → catches.csv）

学習データが0からスタートするので、船宿ブログのバックナンバーを YOLO で処理して
最初の `catches.csv` を作る orchestrator を用意してある。

```python
# Colab セルで実行（GPU ランタイム + Drive マウント済み前提）
%cd /content/drive/MyDrive/aichi-fishing-analysis
from src.build_seed_dataset import build
df = build(
    blog_id="maruman2010",   # ameblo の blog id（伊良湖の釣り船）
    site="irago",            # config.SITES のキー
    boat="maruman2010",      # 船宿名（後で正式名に置換可）
    months_back=6,
    conf=0.30,
    limit=None,              # テスト時は limit=3 などで試す
)
df.head()
```

処理の流れ:
1. `blog_scraper.list_entries(blog_id)` で過去エントリ URL を列挙（dedup・cutoff 済み）
2. 各エントリで `predict_from_url.run(url)` を実行 → `data/scraped/<slug>/summary.json`
3. `scrape_to_catches.aggregate()` で全 summary を `data/fishing_logs/catches.csv` に集約

途中で止めてもOK（`skip_existing=True` で再開時に処理済みエントリは飛ばす）。
仕上がった catches.csv をそのまま `data_integrator → predictor.train` に流せる。

## 将来拡張

- **クロロフィルa濃度**: Copernicus Marine。回遊魚の餌場マップ
- **黒潮位置 / SLA**: JCOPE2、Copernicus
- **船宿one-hot**: 学習データが溜まってから `boat` を直接 one-hot に追加
- **時間帯分割予測**: 朝マヅメ/日中/夕マヅメ で別モデル
