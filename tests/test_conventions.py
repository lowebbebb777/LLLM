"""引き継ぎ規約の機械チェック (dev-conventions README 原則7)。

新規repoへ導入するときは `tests/test_conventions.py` としてコピーするだけでよい。
ROOT はこのファイルから2階層上を repo ルートと見なす(`<repo>/tests/` 配置が前提)。

言語やビルド系に依存しないので、Android / Python / Node いずれのrepoでも動く。
Python が無いrepoでは、同じ5項目を各repoのテスト系で書き直すこと。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_MAX_LINES = 50


MIN_TAGS = 2


def parse_shelf(ideas_md: str) -> list[tuple[str, list[str]]]:
    """IDEAS.md の棚から (行, タグ列) を取り出す。

    行の形: - <詩> — `YYYY-MM-DD` <状態> — `タグ` `タグ`
    `## 棚` より前の説明文と、コメント内の例は拾わない。
    """
    entries: list[tuple[str, list[str]]] = []
    in_shelf = False
    in_comment = False
    for line in ideas_md.splitlines():
        if line.startswith("## "):
            in_shelf = line.startswith("## 棚")
            continue
        if "<!--" in line:
            in_comment = True
        if "-->" in line:
            in_comment = False
            continue
        if not in_shelf or in_comment or not line.startswith("- "):
            continue
        ticks = re.findall(r"`([^`]+)`", line)
        if not ticks or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ticks[0]):
            continue
        entries.append((line, ticks[1:]))
    return entries


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


class HandoffConvention(unittest.TestCase):
    def test_required_files_present(self):
        required = ["AGENTS.md", "docs/STATE.md", "docs/LOG.md", "docs/IDEAS.md"]
        missing = [p for p in required if not (ROOT / p).is_file()]
        self.assertEqual([], missing, f"引き継ぎ規約の必須ファイルが欠けている: {missing}")

    def test_state_within_line_cap(self):
        """STATE は現在地であって履歴ではない (原則4)。"""
        n = len(_read("docs/STATE.md").splitlines())
        self.assertLessEqual(
            n, STATE_MAX_LINES,
            f"docs/STATE.md が {n} 行 (上限 {STATE_MAX_LINES})。"
            "履歴は docs/LOG.md へ、今やらない案は docs/IDEAS.md へ移すこと",
        )

    def test_state_has_last_updated(self):
        self.assertRegex(
            _read("docs/STATE.md"), r"最終更新.*\d{4}-\d{2}-\d{2}",
            "STATE.md に『最終更新: YYYY-MM-DD』が無い。鮮度が判定できない",
        )

    def test_state_points_at_idea_shelf(self):
        """棚は毎回読まれない。STATE の1行だけが唯一の発見経路。"""
        self.assertIn(
            "docs/IDEAS.md", _read("docs/STATE.md"),
            "STATE.md にアイディア棚へのポインタが無い。棚は忘れられる",
        )



class IdeaShelf(unittest.TestCase):
    """詩は grep の取っ手になる名詞を削ぎ落とす形式。

    タグの無い詩は、棚に載っていても見つからない = 捨てたのと同じ。
    """

    def test_every_entry_has_grep_tags(self):
        entries = parse_shelf(_read("docs/IDEAS.md"))
        self.assertTrue(entries, "docs/IDEAS.md の棚から1件も抽出できなかった")
        untagged = [line[:40] for line, tags in entries if len(tags) < MIN_TAGS]
        self.assertEqual(
            [], untagged,
            f"タグが {MIN_TAGS} 語未満の詩がある(grep で見つからない): {untagged}",
        )


if __name__ == "__main__":
    unittest.main()
