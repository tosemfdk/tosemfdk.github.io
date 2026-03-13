# Notion -> Jekyll Rules For This Repo

## Source of truth

- Use [`tools/notion_to_jekyll.py`](../../../tools/notion_to_jekyll.py) as the executable source of truth.
- Use existing posts in `_posts/` to confirm formatting when needed.

## Environment inputs

- `NOTION_TOKEN`: Required.
- `NOTION_PAGE_ID`: Preferred target input.
- `NOTION_DATABASE_ID`: Fallback if `NOTION_PAGE_ID` is absent.
- `NOTION_HOME_ID`: Optional; keep for legacy fallback logic.

## Page ID normalization

- Accept a bare 32-char Notion UUID.
- Accept a dashed Notion UUID.
- Accept a full Notion URL and extract the last UUID-looking token from it.
- Fetch the exact page first.
- Only use search fallback if the exact page fetch fails.

## Front matter rules

- Always write:
  - `layout: post`
  - `title: ...`
  - `date: <page.created_time>`
  - `math: true`
- Add `categories:` only when a category is resolved.
- Add `tags:` from Notion `multi_select` properties only when non-empty.
- Use JSON/YAML-safe quoted scalars for titles, categories, and tags.

## Category resolution order

1. If the page has a `Project` relation, fetch the related page and use that page title as the single Jekyll category.
2. Otherwise use the first populated `select` property.
3. Otherwise, if the page itself is a Notion Projects database row, use its title.
4. Otherwise use the existing parent page / database fallback logic in `tools/notion_to_jekyll.py`.

## Output paths

- Posts go to `_posts/YYYY-MM-DD-slug.md`.
- Media goes to `assets/img/posts/`.
- Images are stored as `.webp`.
- Videos are converted to `.gif` with `ffmpeg`.

## Block mapping

- `paragraph` -> Markdown paragraph.
- `heading_1`, `heading_2`, `heading_3` -> `#`, `##`, `###`.
- `bulleted_list_item` -> `- ...`
- `numbered_list_item` -> `1. ...`
- Nested list children stay indented under the parent item.
- `toggle` -> `<details markdown="1">`.
- `code` -> fenced code block only; do not wrap it in extra details tags.
- `child_page` -> `## <title>`
- `child_database` -> `### <title>`
- `table_of_contents` -> ignore it; Chirpy already renders TOC.
- `bookmark` -> Markdown link using caption if present, otherwise the URL itself.
- `image` -> download and embed local asset path.
- `video` -> download, convert to GIF, embed local asset path.
- `equation` -> inline or block math depending on Notion object type.
- `quote` -> Markdown quote.
- `divider` -> `---`

## Operational rules

- Prefer running the script over hand-writing a post.
- Hand-edit only for cleanup the script cannot express yet.
- Preserve unrelated working tree changes.
- If Python dependencies are missing, use a temp virtualenv instead of changing system Python.
- If Jekyll build fails because Bundler is missing, report it as an environment issue.

## Known repo examples

- Project page category example:
  - `Unitree Go2` project page title becomes category `Unitree Go2`.
- Task page category example:
  - `[Unitree Go2 part 1] Sim2Real에 처음 도전하다.` uses the `Project` relation and still lands in category `Unitree Go2`.
