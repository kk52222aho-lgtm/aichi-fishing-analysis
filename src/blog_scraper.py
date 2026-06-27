"""Ameblo ブログの過去エントリ URL を列挙する。

Ameblo の entrylist ページは <a>/<time> をサーバHTMLには出さず、
`window.INIT_DATA` の JavaScript 変数に全エントリ情報を埋め込んでいる。
本モジュールは正規表現でこの JSON を抜き、`entryState.entryMap` から
  - entry_id
  - entry_title
  - entry_created_datetime (ISO8601 with +09:00)
を取り出して Entry オブジェクトに変換する（1ページに20件）。

URL は screen_name（呼び出し側の `blog_id` 引数）+ entry_id で組み立てる。

これは aichi-fishing-analysis 側の前段：列挙のみ。
YOLO 推論は predict_from_url.run() が担当、本モジュールは画像を触らない。

使い方:
    from src.blog_scraper import list_entries
    entries = list_entries("maruman2010", months_back=6)
    for e in entries:
        print(e.posted_at.date(), e.url)

CLI:
    python -m src.blog_scraper --blog maruman2010 --months 6
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

JST = timezone(timedelta(hours=9))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 20
_INIT_DATA_RE = re.compile(r"window\.INIT_DATA\s*=\s*(\{.*?\});", re.S)


@dataclass
class Entry:
    url: str
    posted_at: datetime
    title: str
    entry_id: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["posted_at"] = self.posted_at.isoformat()
        return d


def _parse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=JST)
        except ValueError:
            continue
    return None


def _extract_init_data(html: str) -> Optional[dict]:
    m = _INIT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _entries_from_html(html: str, blog_screen_name: str) -> list[Entry]:
    """entrylist の HTML から Entry リストを抽出する。"""
    data = _extract_init_data(html)
    if not data:
        return []
    emap = data.get("entryState", {}).get("entryMap", {})
    if not emap:
        return []

    entries: list[Entry] = []
    for entry_id_str, info in emap.items():
        dt = _parse_datetime(info.get("entry_created_datetime") or info.get("ins_datetime"))
        if dt is None:
            continue
        title = (info.get("entry_title") or "").strip()[:160]
        url = f"https://ameblo.jp/{blog_screen_name}/entry-{entry_id_str}.html"
        entries.append(Entry(url=url, posted_at=dt, title=title, entry_id=str(entry_id_str)))
    return entries


def list_entries_daishinmaru(
    months_back: int = 6,
    sleep_sec: float = 0.5,
) -> list[Entry]:
    """daishinmaru.jp/fishing/ から個別記事 URL を列挙。

    URL slug: /YYYY-M-D-<title>/ から日付を解析する。
    """
    base = "https://daishinmaru.jp/fishing/"
    try:
        r = requests.get(base, headers=HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ daishinmaru index: {e}")
        return []

    # href="/2026-6-9-スルメイカコース/" or href="https://daishinmaru.jp/2026-6-9-..."
    pattern = re.compile(
        r'href="(?:https?://daishinmaru\.jp)?/(\d{4}-\d{1,2}-\d{1,2}-[^"/]+)/?"'
    )
    cutoff = datetime.now(tz=JST) - timedelta(days=months_back * 31)
    seen: set[str] = set()
    out: list[Entry] = []

    for slug in pattern.findall(r.text):
        if slug in seen:
            continue
        seen.add(slug)
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})-(.+)", slug)
        if not m:
            continue
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                          hour=5, minute=30, tzinfo=JST)
        except ValueError:
            continue
        if dt < cutoff:
            continue
        title_raw = m.group(4)
        try:
            title = urllib.parse.unquote(title_raw).replace("-", " ").strip()
        except Exception:
            title = title_raw[:160]
        # entry_id は URL デコードしてファイル名に使いやすくする
        try:
            decoded_slug = urllib.parse.unquote(slug)
        except Exception:
            decoded_slug = slug
        out.append(Entry(
            url=f"https://daishinmaru.jp/{slug}/",   # URL は encoded のまま
            posted_at=dt, title=title[:160], entry_id=decoded_slug,
        ))

    out.sort(key=lambda e: e.posted_at, reverse=True)
    return out


def list_entries_ishikawamaru(
    months_back: int = 6,
    sleep_sec: float = 0.5,
    max_pages: int = 500,
) -> list[Entry]:
    """ishikawamaru.jp/blog/ から記事を列挙（新→旧）。

    一覧ページ /blog/page/N/（1ページ目は /blog/）を辿り、各カード(entry-info)の
    日付と記事URL(/blog/<cat>/entry-<id>.html)を取得。cutoff より古くなったら停止。
    """
    cutoff = datetime.now(tz=JST) - timedelta(days=months_back * 31)
    link_re = re.compile(
        r'href="(https://www\.ishikawamaru\.jp/blog/[^"]+/entry-(\d+)\.html)"')
    date_re = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
    out: list[Entry] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        url = ("https://www.ishikawamaru.jp/blog/" if page == 1
               else f"https://www.ishikawamaru.jp/blog/page/{page}/")
        try:
            r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ ishikawamaru p{page}: {e}")
            break
        html = r.content.decode("utf-8", "replace")
        cards = re.split(r'(?=class="entry-info")', html)
        stop = False
        found = 0
        for c in cards[1:]:
            lm = link_re.search(c)
            dm = date_re.search(re.sub(r"<[^>]+>", " ", c[:600]))
            if not lm or not dm:
                continue
            found += 1
            try:
                dt = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)),
                              hour=5, minute=30, tzinfo=JST)
            except ValueError:
                continue
            if dt < cutoff:
                stop = True
                break
            u = lm.group(1)
            if u in seen:
                continue
            seen.add(u)
            out.append(Entry(url=u, posted_at=dt, title="", entry_id=lm.group(2)))
        if stop or found == 0:
            break
        time.sleep(sleep_sec)

    out.sort(key=lambda e: e.posted_at, reverse=True)
    return out


def list_entries_toshikazu(
    months_back: int = 6,
    sleep_sec: float = 0.5,
    max_pages: int = 500,
) -> list[Entry]:
    """としかず釣船（FC2ブログ toshikazumaru.blog72.fc2.com）から記事を列挙（新→旧）。

    一覧 ?page=N を辿り、blog-entry-<id>.html と直後の日付(YYYY-MM-DD)を取得。
    cutoff より古くなったら停止。
    """
    base = "http://toshikazumaru.blog72.fc2.com/"
    cutoff = datetime.now(tz=JST) - timedelta(days=months_back * 31)
    out: list[Entry] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        url = base if page == 1 else base + f"?page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ toshikazu p{page}: {e}")
            break
        html = r.content.decode(r.apparent_encoding or "utf-8", "replace")
        parts = re.split(r"blog-entry-(\d+)\.html", html)
        stop = False
        found = 0
        for i in range(1, len(parts), 2):
            eid = parts[i]
            chunk = parts[i + 1] if i + 1 < len(parts) else ""
            if eid in seen:
                continue
            dm = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", chunk[:400])
            if not dm:
                continue
            seen.add(eid)
            found += 1
            try:
                dt = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)),
                              hour=5, minute=30, tzinfo=JST)
            except ValueError:
                continue
            if dt < cutoff:
                stop = True
                break
            out.append(Entry(url=f"{base}blog-entry-{eid}.html",
                             posted_at=dt, title="", entry_id=eid))
        if stop or found == 0:
            break
        time.sleep(sleep_sec)

    out.sort(key=lambda e: e.posted_at, reverse=True)
    return out


_KYUROKU_DATE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def _kyuroku_page_url(page: int) -> str:
    return "https://tsuri96.com/tyouka/" if page == 1 else f"https://tsuri96.com/tyouka/pageid={page}"


def list_entries_kyuroku(
    months_back: int = 6,
    sleep_sec: float = 0.5,
    max_pages: int = 400,
) -> list[Entry]:
    """久六釣船（tsuri96.com/tyouka/）から日付ブロックを列挙（新→旧）。

    個別記事ページが無く一覧に日付ブロック直書きなので、各日付を1エントリとし
    URL は合成（.../pageid=N#d=YYYY-MM-DD）。全角数字は NFKC で正規化。
    """
    cutoff = datetime.now(tz=JST) - timedelta(days=months_back * 31)
    out: list[Entry] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        try:
            r = requests.get(_kyuroku_page_url(page), headers=HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ kyuroku p{page}: {e}")
            break
        txt = unicodedata.normalize(
            "NFKC", re.sub(r"<[^>]+>", " ", r.content.decode(r.apparent_encoding or "utf-8", "replace")))
        ms = list(_KYUROKU_DATE.finditer(txt))
        if not ms:
            break
        stop = False
        found = 0
        for m in ms:
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                              hour=5, minute=30, tzinfo=JST)
            except ValueError:
                continue
            key = dt.date().isoformat()
            if key in seen:
                continue
            seen.add(key)
            found += 1
            if dt < cutoff:
                stop = True
                break
            out.append(Entry(url=f"{_kyuroku_page_url(page)}#d={key}",
                             posted_at=dt, title="", entry_id=f"kyuroku-{key}"))
        if stop or found == 0:
            break
        time.sleep(sleep_sec)

    out.sort(key=lambda e: e.posted_at, reverse=True)
    return out


def _registry_platform(blog_id: str) -> tuple[str, str]:
    """registry から (platform, blog_url) を返す。fallback は ('ameblo', '')。"""
    try:
        from . import scrape_to_catches as _stc
        reg = _stc.load_blog_registry()
        info = reg.get(blog_id, {}) or {}
        return (
            info.get("blog_platform", "ameblo"),
            info.get("blog_url", ""),
        )
    except Exception:
        return ("ameblo", "")


def list_entries(
    blog_id: str,
    months_back: int = 6,
    max_pages: int = 60,
    sleep_sec: float = 0.8,
) -> list[Entry]:
    """ブログIDから直近 months_back ヶ月のエントリを取得（新→旧でソート）。

    Args:
        blog_id: ameblo の screen_name か、独自サイト用の登録 ID
        months_back: 何ヶ月前まで遡るか
        max_pages: 最大ページ数（安全弁）
        sleep_sec: ページ間スリープ（先方への配慮）
    """
    # registry で blog_platform=custom が設定されてたら dispatch
    platform, blog_url = _registry_platform(blog_id)
    if platform == "custom":
        if "daishinmaru.jp" in blog_url:
            return list_entries_daishinmaru(months_back=months_back, sleep_sec=sleep_sec)
        if "ishikawamaru.jp" in blog_url:
            return list_entries_ishikawamaru(months_back=months_back, sleep_sec=sleep_sec,
                                             max_pages=max(max_pages, 500))
        if "toshikazumaru" in blog_url:
            return list_entries_toshikazu(months_back=months_back, sleep_sec=sleep_sec,
                                          max_pages=max(max_pages, 500))
        if "tsuri96.com" in blog_url:
            return list_entries_kyuroku(months_back=months_back, sleep_sec=sleep_sec,
                                        max_pages=max(max_pages, 400))
        print(f"⚠️ {blog_id}: custom platform だが対応 scraper 未実装 ({blog_url})")
        return []

    cutoff = datetime.now(tz=JST) - timedelta(days=months_back * 31)
    out: list[Entry] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        suffix = "" if page == 1 else f"-{page}"
        url = f"https://ameblo.jp/{blog_id}/entrylist{suffix}.html"
        try:
            r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            print(f"⚠️ page {page}: {exc}")
            break
        if r.status_code != 200:
            break

        page_entries = _entries_from_html(r.text, blog_id)
        if not page_entries:
            # INIT_DATA が無い／変わった可能性
            print(f"⚠️ page {page}: INIT_DATA から entry を抽出できず")
            break

        new_in_range = 0
        for e in page_entries:
            if e.url in seen:
                continue
            seen.add(e.url)
            if e.posted_at < cutoff:
                continue
            out.append(e)
            new_in_range += 1

        # このページの最古エントリが cutoff より古ければ終了
        oldest = min(e.posted_at for e in page_entries)
        if oldest < cutoff:
            break

        if new_in_range == 0:
            # 全部既出 or 全部 cutoff 外
            break

        time.sleep(sleep_sec)

    out.sort(key=lambda e: e.posted_at, reverse=True)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ameblo エントリ列挙")
    parser.add_argument("--blog", required=True, help="ameblo screen_name (例: maruman2010)")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    args = parser.parse_args()

    entries = list_entries(args.blog, months_back=args.months, max_pages=args.max_pages)
    if args.json:
        print(json.dumps([e.to_dict() for e in entries], ensure_ascii=False, indent=2))
    else:
        print(f"{len(entries)} entries (past {args.months} months)")
        for e in entries:
            print(f"  {e.posted_at.date()}  {e.entry_id}  {e.title}")


if __name__ == "__main__":
    _cli()
