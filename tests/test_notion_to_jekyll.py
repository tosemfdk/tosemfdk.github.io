import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.notion_to_jekyll as notion


class NotionToJekyllTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, status_code=200, content=b"video-bytes", chunks=None):
            self.status_code = status_code
            self.content = content
            self._chunks = chunks or [content]

        def iter_content(self, chunk_size):
            del chunk_size
            for chunk in self._chunks:
                yield chunk

    def make_config(self, tempdir, **overrides):
        env = {
            "NOTION_TOKEN": "secret_token",
            "NOTION_PAGE_ID": "24dcbb7d79378091bb83df2ae86685f4",
            "NOTION_IMPORT_MODE": "single",
        }
        env.update(overrides)
        return notion.load_config(env=env, cwd=tempdir)

    def test_load_config_parses_direct_child_settings(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(
                tempdir,
                NOTION_IMPORT_MODE="direct_children",
                NOTION_IMPORT_ROOT_PAGE_ID="24dcbb7d79378091bb83df2ae86685f4",
                NOTION_IMPORT_CATEGORY_OVERRIDE="서울대학교 여름방학 인턴",
                NOTION_DIRECT_CHILD_MAX_GIF_MB="25",
            )
            self.assertEqual(config.import_mode, "direct_children")
            self.assertEqual(config.root_page_id, "24dcbb7d-7937-8091-bb83-df2ae86685f4")
            self.assertEqual(config.import_category_override, "서울대학교 여름방학 인턴")
            self.assertEqual(config.max_gif_bytes, 25 * 1024 * 1024)

    def test_discover_direct_child_pages_filters_non_child_pages(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(tempdir)
            blocks = [
                {"type": "paragraph", "id": "a"},
                {"type": "child_page", "id": "24dcbb7d79378091bb83df2ae86685f4", "child_page": {"title": "A"}},
                {"type": "child_database", "id": "b", "child_database": {"title": "DB"}},
            ]
            with mock.patch.object(notion, "list_block_children", return_value=blocks):
                discovered = notion.discover_direct_child_pages(config, "root")
            self.assertEqual(
                discovered,
                [{"id": "24dcbb7d-7937-8091-bb83-df2ae86685f4", "title": "A"}],
            )

    def test_write_post_if_missing_prefers_notion_source_id(self):
        with tempfile.TemporaryDirectory() as tempdir:
            posts_dir = Path(tempdir) / "_posts"
            posts_dir.mkdir()
            existing = posts_dir / "2026-01-01-old.md"
            existing.write_text(
                "---\nnotion_source_id: \"24dcbb7d-7937-8091-bb83-df2ae86685f4\"\n---\n",
                encoding="utf-8",
            )
            created, path = notion.write_post_if_missing(
                str(posts_dir),
                "2026-01-02-new.md",
                "new content",
                notion_source_id="24dcbb7d-7937-8091-bb83-df2ae86685f4",
            )
            self.assertFalse(created)
            self.assertEqual(path, str(existing))
            self.assertFalse((posts_dir / "2026-01-02-new.md").exists())

    def test_direct_children_skips_existing_source_id_before_render(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(
                tempdir,
                NOTION_IMPORT_MODE="direct_children",
                NOTION_IMPORT_ROOT_PAGE_ID="24dcbb7d79378091bb83df2ae86685f4",
                NOTION_IMPORT_CATEGORY_OVERRIDE="서울대학교 여름방학 인턴",
            )
            existing = Path(config.posts_dir) / "2026-01-01-old.md"
            existing.write_text(
                "---\nnotion_source_id: \"24dcbb7d-7937-8091-bb83-df2ae86685f4\"\n---\n",
                encoding="utf-8",
            )
            page = {
                "created_time": "2026-01-01T00:00:00.000Z",
                "properties": {
                    "Title": {
                        "type": "title",
                        "title": [{"plain_text": "Old"}],
                    }
                },
            }
            with mock.patch.object(
                notion,
                "discover_direct_child_pages",
                return_value=[{"id": "24dcbb7d-7937-8091-bb83-df2ae86685f4", "title": "Old"}],
            ), mock.patch.object(
                notion,
                "resolve_page",
                return_value=(page, "24dcbb7d-7937-8091-bb83-df2ae86685f4"),
            ), mock.patch.object(
                notion,
                "render_page_to_post",
                side_effect=AssertionError("render should not run for skipped posts"),
            ):
                notion.process_direct_children_import(config)

    def test_build_gif_conversion_attempts_prefers_quality_before_trimming(self):
        attempts = notion.build_gif_conversion_attempts(20.0)
        full_duration = attempts[: len(notion.QUALITY_PROFILES)]
        trimmed = attempts[len(notion.QUALITY_PROFILES) :]
        self.assertTrue(all(item["max_duration"] is None for item in full_duration))
        self.assertTrue(trimmed)
        self.assertTrue(all(item["max_duration"] is not None for item in trimmed))
        self.assertLess(trimmed[0]["max_duration"], 20.0)

    def test_select_feature_image_skips_oversized_gif_fallback(self):
        markdown = (
            "![gif](/assets/img/posts/oversized.gif)\n\n"
            "![image](/assets/img/posts/photo.webp)\n"
        )
        with mock.patch.object(notion, "local_media_size", return_value=30 * 1024 * 1024):
            selected = notion.select_feature_image(markdown, 12 * 1024 * 1024)
        self.assertEqual(selected, "/assets/img/posts/photo.webp")

    def test_parse_block_video_never_renders_html_video(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(tempdir)
            block = {
                "type": "video",
                "id": "vid1",
                "video": {
                    "file": {"url": "https://example.com/video.mp4"},
                    "caption": [{"plain_text": "demo"}],
                },
            }
            with mock.patch.object(notion, "download_media", return_value="/assets/img/posts/vid1.gif"):
                rendered = notion.parse_block(config, block)
            self.assertIn("![demo](/assets/img/posts/vid1.gif)", rendered)
            self.assertNotIn("<video", rendered)

    def test_parse_block_video_raises_when_no_gif_is_produced(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(tempdir)
            block = {
                "type": "video",
                "id": "vid2",
                "video": {
                    "file": {"url": "https://example.com/video.mp4"},
                    "caption": [{"plain_text": "demo"}],
                },
            }
            with mock.patch.object(notion, "download_media", return_value="https://example.com/video.mp4"):
                with self.assertRaises(RuntimeError):
                    notion.parse_block(config, block)

    def test_download_media_raises_when_video_download_fails(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(tempdir, NOTION_IMPORT_MODE="direct_children")
            with mock.patch.object(notion.requests, "get", return_value=self.FakeResponse(status_code=500)):
                with self.assertRaises(RuntimeError):
                    notion.download_media(config, "https://example.com/video.mp4", "vid3", "mp4")

    def test_download_media_raises_when_video_cannot_be_converted(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.make_config(tempdir, NOTION_IMPORT_MODE="direct_children")
            with mock.patch.object(
                notion.requests,
                "get",
                return_value=self.FakeResponse(status_code=200, content=b"video", chunks=[b"video"]),
            ), mock.patch.object(notion, "convert_video_to_gif_with_limit", return_value=False):
                with self.assertRaises(RuntimeError):
                    notion.download_media(config, "https://example.com/video.mp4", "vid4", "mp4")


if __name__ == "__main__":
    unittest.main()
