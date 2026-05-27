"""釣り船ブログの本文テキストから釣果情報を抽出する。

YOLO の画像カウントとは独立した情報源。Ameblo のエントリページから
本文 (`window.INIT_DATA.entryState.entryMap[id].entry_text`) を取得して、
釣り船ブログ特有の定型表現を正規表現で拾う。

抽出対象:
  - 竿頭 N尾/匹/本/枚  → 個人最大カウント（船全体ではなく1人の最大値）
  - 最大 Nセンチ        → 最大魚体長
  - N匹/N尾（汎用）     → 数字＋単位（フォールバック、サイズと区別必要）

注意:
  - 「日」は「尾」の誤字として許容（maruman2010 のブログでよくある）
  - 「最大Nセンチ」のあとに「竿頭Nセンチ」が来ない限り、サイズと数を分離可能
  - 数字が完全に取れないエントリは多い（"型の良いイサキ" のような定性表現のみ）

使い方:
    from src.blog_text_extractor import process_entry
    info = process_entry("https://ameblo.jp/maruman2010/entry-12960546884.html")
    # → {"title": "イサキ便行ってきました",
    #    "body_text": "...",
    #    "catch_info": {"イサキ": {"top_per_angler": 13, "max_size_cm": 35},
    #                   "石鯛": {"top_per_angler": 2, "max_size_cm": 45}}}

エントリ毎の出力先:
    data/scraped/<slug>/body.txt          プレーンテキスト
    data/scraped/<slug>/text_extracted.json   構造化抽出結果

CLI:
    python -m src.blog_text_extractor --url https://ameblo.jp/maruman2010/entry-XXX.html
    python -m src.blog_text_extractor --batch              # data/scraped/ 全件
    python -m src.blog_text_extractor --batch --blog maruman2010 --months 6
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from . import config
from .title_parser import _TARGET_KEYWORDS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 20
_INIT_DATA_RE = re.compile(r"window\.INIT_DATA\s*=\s*(\{.*?\});", re.S)

# 抽出対象の魚種名（title_parser のキーワードを再利用）
SPECIES_NAMES: tuple[str, ...] = tuple(kw for kw, _ in _TARGET_KEYWORDS if kw not in ("五目", "根魚"))

# 単位（「日」は「尾」の誤字として許容）
COUNT_UNIT = r"(?:尾|匹|本|枚|杯|ハイ|日)"

# よく出る別称・略称マッピング（本文表記 → catches.csv で使う表記）
# 本文で別の名前で書かれている場合、両方マッチさせる
SPECIES_ALIASES: dict[str, list[str]] = {
    "シマアジ":   ["島鯵"],
    "アジ":       ["真鯵", "マアジ"],
    "マダイ":     ["真鯛"],
    "クロダイ":   ["黒鯛"],
    "イシダイ":   ["石鯛"],
    "イサキ":     ["伊佐木"],
    "タチウオ":   ["太刀魚"],
    "アオリイカ": ["障泥烏賊"],
    "オニカサゴ": ["鬼笠子"],
    "シューサイフグ": ["集才ふぐ"],
    "ヒラメ":     ["平目"],
}


def _all_species_to_check() -> list[tuple[str, str]]:
    """(検索用エイリアス, 正規化された種名) のリスト。長い順にソート。"""
    pairs: list[tuple[str, str]] = []
    for sp in SPECIES_NAMES:
        pairs.append((sp, sp))
        for alias in SPECIES_ALIASES.get(sp, []):
            pairs.append((alias, sp))
    # 別途登場するが title_parser に含まれない種
    extras = {
        "イシダイ": ["石鯛", "イシダイ"],
        "シューサイフグ": ["シューサイフグ", "集才ふぐ"],
    }
    for sp, aliases in extras.items():
        for a in aliases:
            pairs.append((a, sp))
    # 長い順（部分マッチ防止）
    pairs.sort(key=lambda x: -len(x[0]))
    # 重複除去
    seen = set()
    out = []
    for k, v in pairs:
        if (k, v) in seen:
            continue
        seen.add((k, v))
        out.append((k, v))
    return out


def fetch_entry_init_data(url: str) -> Optional[dict]:
    """エントリページの window.INIT_DATA を取得。失敗時 None。"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException:
        return None
    m = _INIT_DATA_RE.search(r.text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def fetch_body(url: str) -> Optional[dict]:
    """エントリページからタイトルと本文プレーンテキストを返す。

    Returns:
        {"title": str, "body_text": str, "entry_id": str}
        失敗時 None。
    """
    data = fetch_entry_init_data(url)
    if not data:
        return None
    emap = data.get("entryState", {}).get("entryMap", {})
    if not emap:
        return None
    # エントリページ単独ロード時は1件のみ入ってる想定。
    # 複数あれば URL に一致する entry_id を選ぶ。
    m = re.search(r"/entry-(\d+)\.html", url)
    target_id = m.group(1) if m else None
    info = None
    if target_id and target_id in emap:
        info = emap[target_id]
    else:
        info = next(iter(emap.values()))

    title = (info.get("entry_title") or "").strip()
    raw_html = info.get("entry_text") or ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)  # 余分な空白を1つに

    return {
        "title": title,
        "body_text": text,
        "entry_id": str(info.get("entry_id") or target_id or ""),
    }


def _nearest_species_before(
    text: str, pos: int, max_dist: int,
    species_pairs: list[tuple[str, str]],
) -> Optional[str]:
    """`pos` から遡って max_dist 文字以内で、最も近い魚種名を返す（canonical 名）。

    比較は (end_position, alias_length) の組み合わせ:
      - end_position が大きい（= pos に近い）方を優先
      - end_position が同じなら長い alias を優先
        （"シマアジ" が "アジ" を内包するケースで、より長い名前を採用）
    間に「。」「！」「？」「.」「!」「?」があれば文区切りとみなして対象外。
    """
    start = max(0, pos - max_dist)
    prefix = text[start:pos]
    for sep in "。！？.!?":
        idx = prefix.rfind(sep)
        if idx >= 0:
            prefix = prefix[idx + 1:]
    best: tuple[int, int, str] = (-1, -1, "")  # (end, length, canonical)
    for alias, canonical in species_pairs:
        idx = prefix.rfind(alias)
        if idx < 0:
            continue
        end = idx + len(alias)
        length = len(alias)
        if (end, length) > (best[0], best[1]):
            best = (end, length, canonical)
    return best[2] or None


def _extract_with_regex(text: str) -> dict[str, dict]:
    """正規表現ベースの抽出（無料・高速・maruman2010 系のフォーマットに強い）。

    アンカー先行型: 数字＋単位（竿頭/最大Nセンチ等）を先に見つけて、
    その直前の最も近い魚種名に紐付ける。
    """
    if not text:
        return {}

    species_pairs = _all_species_to_check()
    result: dict[str, dict] = {}

    def _record(sp: Optional[str], key: str, val: int) -> None:
        if not sp:
            return
        info = result.setdefault(sp, {})
        if val > info.get(key, 0):
            info[key] = val

    # 1) 竿頭 N尾 (個人最大カウント)
    for m in re.finditer(rf"竿頭(?:は)?\s*(\d+)\s*{COUNT_UNIT}", text):
        n = int(m.group(1))
        sp = _nearest_species_before(text, m.start(), max_dist=30, species_pairs=species_pairs)
        _record(sp, "top_per_angler", n)

    # 2) 最大 N センチ (魚体長)
    for m in re.finditer(r"最大\s*(\d+)\s*センチ", text):
        n = int(m.group(1))
        sp = _nearest_species_before(text, m.start(), max_dist=20, species_pairs=species_pairs)
        _record(sp, "max_size_cm", n)

    # 3) 単純な "Nセンチ" (魚種名 → 直後の数字＋センチ)
    for m in re.finditer(r"(\d+)\s*センチ", text):
        prefix_check = text[max(0, m.start() - 5):m.start()]
        if "最大" in prefix_check:
            continue
        n = int(m.group(1))
        sp = _nearest_species_before(text, m.start(), max_dist=12, species_pairs=species_pairs)
        _record(sp, "max_size_cm", n)

    # 4) "魚種A は N尾" 連結パターン
    species_alt = "|".join(re.escape(alias) for alias, _ in species_pairs)
    for m in re.finditer(rf"({species_alt})\s*(?:の)?\s*は\s*(\d+)\s*{COUNT_UNIT}", text):
        alias = m.group(1)
        n = int(m.group(2))
        canonical = next((c for a, c in species_pairs if a == alias), alias)
        _record(canonical, "top_per_angler", n)

    return {k: v for k, v in result.items() if v}


# LLM 抽出用: 既知の魚種名一覧（LLM に preferring してもらう表記）
_KNOWN_SPECIES_LIST = [
    "マダイ", "クロダイ", "チダイ", "イシダイ",
    "マダコ", "タコ",
    "イカ", "アオリイカ", "スルメイカ",
    "イサキ", "シマアジ", "アジ", "サバ", "タチウオ",
    "カワハギ", "メバル", "カサゴ", "オニカサゴ",
    "スズキ", "セイゴ", "ヒラメ", "マゴチ",
    "シロギス", "キス",
    "フグ", "シューサイフグ", "トラフグ",
    "ブリ", "ハマチ", "ワラサ", "カツオ", "サワラ",
    "ホウボウ", "シマガツオ", "クロムツ", "キンメダイ", "アコウダイ",
    "オコゼ", "ガンゾウビラメ",
]


def _extract_with_llm(
    text: str,
    provider: str = "groq",
    model: Optional[str] = None,
    fallback_provider: Optional[str] = "gemini",
) -> dict[str, dict]:
    """LLM ベースの釣果情報抽出。船宿ブログのフォーマット非依存。

    各船宿で表現が違う本文（"一人3匹のリミット" "キャッチは1枚のみ" 等）を
    LLM に読ませて構造化抽出する。Groq Llama 3.3 70B なら無料で実用十分。

    Args:
        provider: 一次プロバイダ（デフォルト Groq）
        fallback_provider: 一次プロバイダが rate limit / エラーを返した時の
            自動フォールバック先。None なら例外を投げる。
    """
    if not text or len(text) < 30:
        return {}

    # llm_predictor を遅延 import（循環回避）
    from .llm_predictor import (
        _PROVIDER_CALLERS, _PROVIDER_DEFAULTS, _PROVIDER_API_KEYS, _get_api_key,
    )

    def _resolve(prov: str) -> tuple[str, str]:
        if prov not in _PROVIDER_CALLERS:
            raise ValueError(f"unknown LLM provider: {prov}")
        m = model if (model and prov == provider) else _PROVIDER_DEFAULTS[prov]
        k = _get_api_key(prov)
        if prov != "ollama" and not k:
            tried = ", ".join(_PROVIDER_API_KEYS.get(prov, ()))
            raise RuntimeError(f"{prov} API key 未解決 (試した: {tried})")
        return m, k

    used_model, api_key = _resolve(provider)

    species_list = "、".join(_KNOWN_SPECIES_LIST)
    prompt = f"""あなたは日本の釣り船ブログの本文から釣果情報を抽出する専門家です。

【抽出ルール】
1. 魚種ごとに以下を抽出（該当する数字が明示されていない場合は省略）：
   - top_per_angler: 個人最大釣果（"竿頭 N尾"、"一人 N 匹"、"トップ N"、"N匹/人"）
   - total_catch: 船全体の釣果合計（"全体で N 匹"、"全 N 枚"、"キャッチは N 枚のみ"、"計 N"）
   - max_size_cm: 最大魚体長（"最大Nセンチ"、"NセンチのX"）
   - qualitative: 定性的評価。次のいずれかに分類：
     "絶好調"（爆釣・大漁・ボコボコ）/ "好調"（よく釣れた・絶好調と書かれてる）/
     "普通"（ぼちぼち・なんとか）/ "渋い"（伸び悩み・微妙）/
     "厳しい"（厳しい・1枚のみ等）/ "ボウズ"（釣れなかった・坊主）

2. 釣果に関係ない数字（時刻、電話番号、料金、予約人数）は無視
3. 確実でない数値は省略（推測しない）
4. 釣果報告でないエントリは "catches": {{}} を返す
5. 同じ魚種で複数の記載があれば、より具体的な数字を優先（例: "最大35cm 竿頭13尾"）

【既知の魚種名一覧（一致する場合はこの表記を優先）】
{species_list}
リストにない魚種が登場した場合は本文の表記をそのまま使ってください。

【本文】
{text}

【出力】JSON のみで以下の形式：
{{"catches": {{"魚種名": {{"top_per_angler": int, "total_catch": int, "max_size_cm": int, "qualitative": "好調"}}, ...}}}}
"""

    # 釣果抽出用スキーマ。Gemini の response_schema は additionalProperties 不可なので渡さない。
    # response_mime_type=application/json + prompt 内の構造指示で誘導する。
    # 一次プロバイダ失敗（rate limit 等）→ fallback_provider に自動切替
    raw = None
    last_err: Optional[Exception] = None
    try:
        raw = _PROVIDER_CALLERS[provider](prompt, used_model, api_key, schema=None)
    except Exception as e:
        last_err = e
        msg = str(e)
        rate_limited = ("429" in msg) or ("rate" in msg.lower()) or ("RESOURCE_EXHAUSTED" in msg)
        if fallback_provider and fallback_provider != provider and rate_limited:
            try:
                fb_model, fb_key = _resolve(fallback_provider)
                print(f"  ↳ {provider} rate limit → fallback to {fallback_provider}")
                raw = _PROVIDER_CALLERS[fallback_provider](prompt, fb_model, fb_key, schema=None)
            except Exception as fb_e:
                last_err = fb_e
                raw = None
        if raw is None:
            raise last_err  # type: ignore[misc]

    catches = raw.get("catches", {}) if isinstance(raw, dict) else {}
    if not isinstance(catches, dict):
        return {}
    # 空 dict や無効値を除去
    cleaned: dict[str, dict] = {}
    for sp, info in catches.items():
        if not isinstance(info, dict):
            continue
        valid_info: dict[str, Any] = {}
        for k in ("top_per_angler", "total_catch", "max_size_cm"):
            v = info.get(k)
            if v is not None:
                try:
                    valid_info[k] = int(v)
                except (TypeError, ValueError):
                    pass
        q = info.get("qualitative")
        if q and isinstance(q, str) and q.strip():
            valid_info["qualitative"] = q.strip()
        if valid_info:
            cleaned[sp] = valid_info
    return cleaned


def extract_catch_info(
    text: str,
    use_llm_fallback: bool = False,
    llm_provider: str = "groq",
    llm_model: Optional[str] = None,
    llm_fallback_provider: Optional[str] = "gemini",
) -> dict[str, dict]:
    """本文から魚種別釣果情報を抽出。

    Args:
        text: ブログ本文プレーンテキスト
        use_llm_fallback: True なら正規表現で取れなかった時に LLM へフォールバック
        llm_provider: LLM プロバイダ（"gemini" | "groq" | "cerebras" | "ollama"）
        llm_model: モデル名（None なら provider のデフォルト）
        llm_fallback_provider: 一次 LLM が rate limit 時の自動フォールバック先

    Returns:
        {
            "イサキ":   {"top_per_angler": 13, "max_size_cm": 35, "qualitative": "好調"},
            "ホウボウ": {"qualitative": "ぼちぼち"},
            ...
        }
    """
    info = _extract_with_regex(text)
    if info:
        return info
    if use_llm_fallback:
        try:
            info = _extract_with_llm(
                text, provider=llm_provider, model=llm_model,
                fallback_provider=llm_fallback_provider,
            )
        except Exception as e:
            print(f"  ⚠️ LLM extraction failed: {e}")
            info = {}
    return info


def process_entry(
    url: str,
    out_dir: Path | str | None = None,
    skip_existing: bool = True,
    use_llm_fallback: bool = False,
    llm_provider: str = "groq",
    llm_model: Optional[str] = None,
    llm_fallback_provider: Optional[str] = "gemini",
    entry_posted_at: Optional[Any] = None,
) -> Optional[dict]:
    """1 エントリを処理して本文と抽出結果を保存。

    Args:
        url: エントリ URL
        out_dir: data/scraped/<slug>/ 相当のディレクトリ
        skip_existing: text_extracted.json 既存ならスキップ
        use_llm_fallback: 正規表現で取れなかった時に LLM 抽出
        llm_provider / llm_model: LLM 設定
    """
    from .predict_from_url import _slug_for_url
    slug = _slug_for_url(url)
    out = Path(out_dir) if out_dir else (config.DATA_DIR / "scraped" / slug)
    out.mkdir(parents=True, exist_ok=True)

    body_path = out / "body.txt"
    extracted_path = out / "text_extracted.json"

    if skip_existing and extracted_path.exists():
        try:
            with extracted_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            # 既に何か抽出できてればキャッシュを採用
            if cached.get("catch_info"):
                return cached
            # 空キャッシュ。LLM フォールバックを試す価値がない（=offか過去にLLM試して空）ならそのまま返す
            cached_method = cached.get("extraction_method")
            if not use_llm_fallback:
                return cached
            if cached_method == "llm":
                # 前回 LLM で抽出を試みて空だった → 再処理しても結果同じ
                return cached
            # それ以外（legacy/none/regex）は LLM で再試行する
        except Exception:
            pass

    body = fetch_body(url)
    if not body:
        return None

    catch_info = extract_catch_info(
        body["body_text"],
        use_llm_fallback=use_llm_fallback,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_fallback_provider=llm_fallback_provider,
    )

    result = {
        "url": url,
        "entry_id": body["entry_id"],
        "title": body["title"],
        "body_text": body["body_text"],
        "catch_info": catch_info,
        "extraction_method": "regex" if _extract_with_regex(body["body_text"]) else (
            "llm" if catch_info else "none"
        ),
    }
    # posted_at を永続化（aggregate 単体実行時のフォールバック日付として使う）
    if entry_posted_at is not None:
        try:
            result["posted_at"] = entry_posted_at.isoformat() if hasattr(entry_posted_at, "isoformat") else str(entry_posted_at)
        except Exception:
            pass

    with body_path.open("w", encoding="utf-8") as f:
        f.write(body["body_text"])
    with extracted_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def process_batch(
    urls: list[str],
    skip_existing: bool = True,
    sleep_sec: float = 0.6,
    use_llm_fallback: bool = False,
    llm_provider: str = "groq",
    llm_model: Optional[str] = None,
    llm_fallback_provider: Optional[str] = "gemini",
    posted_at_map: Optional[dict[str, Any]] = None,
) -> list[dict]:
    """複数 URL を順に処理。"""
    results: list[dict] = []
    n = len(urls)
    for i, url in enumerate(urls, 1):
        r = process_entry(
            url, skip_existing=skip_existing,
            use_llm_fallback=use_llm_fallback,
            llm_provider=llm_provider, llm_model=llm_model,
            llm_fallback_provider=llm_fallback_provider,
            entry_posted_at=(posted_at_map or {}).get(url),
        )
        ok = r is not None and r.get("catch_info")
        method = r.get("extraction_method", "?") if r else "?"
        species_summary = (
            ", ".join(f"{k}:{v}" for k, v in (r.get("catch_info") or {}).items())
            if r else "(fetch失敗)"
        )
        mark = "✓" if ok else "·"
        print(f"[{i}/{n}] {url.rsplit('/',1)[-1]:35s} {mark} [{method}] {species_summary}")
        if r is not None:
            results.append(r)
        time.sleep(sleep_sec)
    return results


def process_all_scraped(
    skip_existing: bool = True, sleep_sec: float = 0.6,
    use_llm_fallback: bool = False,
    llm_provider: str = "groq", llm_model: Optional[str] = None,
) -> list[dict]:
    """data/scraped/ 配下の summary.json があるエントリ全てを処理。"""
    scraped_root = config.DATA_DIR / "scraped"
    urls: list[str] = []
    for sp in sorted(scraped_root.glob("*/summary.json")):
        try:
            with sp.open("r", encoding="utf-8") as f:
                s = json.load(f)
            url = s.get("source_url")
            if url:
                urls.append(url)
        except Exception:
            continue
    print(f"📄 {len(urls)} エントリの本文を処理...")
    return process_batch(
        urls, skip_existing=skip_existing, sleep_sec=sleep_sec,
        use_llm_fallback=use_llm_fallback,
        llm_provider=llm_provider, llm_model=llm_model,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ブログ本文から釣果情報を抽出")
    parser.add_argument("--url", help="単一 URL を処理")
    parser.add_argument("--batch", action="store_true", help="data/scraped/ 全件を処理")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.6)
    args = parser.parse_args()

    if args.url:
        r = process_entry(args.url, skip_existing=not args.no_skip_existing)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.batch:
        process_all_scraped(skip_existing=not args.no_skip_existing, sleep_sec=args.sleep)
    else:
        parser.error("--url か --batch のどちらかを指定")


if __name__ == "__main__":
    _cli()
