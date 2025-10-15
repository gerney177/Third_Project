#!/usr/bin/env python3
"""
Web Ingestor
웹 페이지(기사/게시글) 본문을 수집/정제하여 지식베이스에 추가할 수 있도록 가공합니다.

기능:
- HTML 다운로드 (requests)
- 본문 추출/정제 (BeautifulSoup)
- 메타데이터 수집 (title, url, fetched_at)
"""

import re
import time
from typing import Dict, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup


class WebIngestor:
    """간단한 웹 페이지 본문 수집기."""

    def __init__(self, timeout_seconds: int = 15, user_agent: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )

    def fetch(self, url: str) -> Optional[str]:
        """URL에서 HTML을 다운로드합니다."""
        headers = {"User-Agent": self.user_agent}
        resp = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        if resp.status_code != 200:
            return None
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text

    def extract_main_text(self, html: str) -> Dict[str, str]:
        """HTML에서 제목과 본문 텍스트를 추출합니다."""
        soup = BeautifulSoup(html, "lxml")

        # 불필요한 태그 제거
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = (soup.title.get_text(strip=True) if soup.title else "").strip()

        # 본문 후보: role=main, article, section, divs
        main = soup.find(attrs={"role": "main"}) or soup.find("article") or soup.find("section")
        text = ""
        if main:
            text = main.get_text("\n", strip=True)
        else:
            # fallback: 가장 텍스트가 많은 div 선택
            candidates = soup.find_all("div")
            best = ""
            best_len = 0
            for c in candidates:
                t = c.get_text("\n", strip=True)
                l = len(t)
                if l > best_len:
                    best = t
                    best_len = l
            text = best

        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()
        return {"title": title, "text": text}

    def ingest(self, url: str) -> Optional[Dict]:
        """URL을 수집하여 표준 포맷으로 반환합니다."""
        html = self.fetch(url)
        if not html:
            return None
        extracted = self.extract_main_text(html)
        if not extracted.get("text"):
            return None
        return {
            "url": url,
            "title": extracted.get("title", ""),
            "content": extracted.get("text", ""),
            "fetched_at": datetime.now().isoformat(),
        }


def main():
    print("=== WebIngestor Test ===")
    url = "https://gall.dcinside.com/mgallery/board/view/?id=chartanalysis&no=2023207"
    ingestor = WebIngestor()
    data = ingestor.ingest(url)
    if data:
        print(f"Title: {data['title'][:80]}")
        print(f"Length: {len(data['content'])} chars")
    else:
        print("Failed to ingest")


if __name__ == "__main__":
    main()


