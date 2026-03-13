---
name: notion-jekyll-blog
description: Convert Notion pages or Notion URLs into this repository's Jekyll/Chirpy blog post format. Use when a request mentions Notion page IDs, Notion URLs, `.env` values like `NOTION_PAGE_ID` or `NOTION_HOME_ID`, `tools/notion_to_jekyll.py`, syncing Notion content into `_posts`, or updating the repo's Notion-to-GitHub blog conversion rules.
---

# Notion Jekyll Blog

Use this skill to sync Notion content into this repository's blog without re-deriving the mapping each time. Treat the repo's current converter and existing posts as the canonical implementation of the rules.

## Quick Start

- Read [`tools/notion_to_jekyll.py`](../../../tools/notion_to_jekyll.py) before making assumptions.
- Read `.env` for `NOTION_TOKEN`, `NOTION_PAGE_ID` or `NOTION_DATABASE_ID`, and optional `NOTION_HOME_ID`.
- Prefer refreshing posts by running the script rather than hand-writing Markdown from scratch.
- Preserve unrelated working tree changes in `_posts/` and other files.

## Workflow

1. Resolve the target page.
- Accept a bare UUID, dashed UUID, or full Notion URL.
- Normalize it to the dashed UUID format used by the Notion API.
- Fetch the exact page first.
- Only use the closest accessible search fallback if the exact fetch fails.

2. Resolve the category.
- If the page has a `Project` relation, fetch that related page and use its title as the single Jekyll category.
- Otherwise fall back to the existing converter logic.
- Do not invent a new category if the related project title is already available.

3. Convert the body using repo conventions.
- Ignore Notion `table_of_contents` blocks.
- Convert Notion `bookmark` blocks to Markdown links.
- Keep nested list indentation.
- Render toggles as `<details markdown="1">`.
- Download images and videos into `assets/img/posts/`.

4. Write the post.
- Output to `_posts/YYYY-MM-DD-slug.md`.
- Use page `created_time` for the post date.
- Keep `math: true` in front matter.
- Only write `tags:` if there are real multi-select tags.

5. Verify.
- Run `python3 -m py_compile tools/notion_to_jekyll.py`.
- If local Python is missing dependencies, create a temp virtualenv and install only what is needed there.
- If `bundle exec jekyll b` fails because Bundler is missing, report that as an environment problem instead of changing repo content.

## References

- Read [`references/rules.md`](references/rules.md) when you need the exact field mapping, block conversion rules, output paths, or repo-specific examples.
