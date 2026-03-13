## Skills
A skill is a set of local instructions stored in a `SKILL.md` file. Use the local skill below when the request matches.

### Available skills
- notion-jekyll-blog: Convert Notion pages or Notion URLs into this repository's Jekyll/Chirpy blog format, including category resolution from Notion `Project` relations, media download rules, and front matter conventions. Use when a request mentions Notion page IDs, Notion URLs, `.env` values like `NOTION_PAGE_ID` or `NOTION_HOME_ID`, `tools/notion_to_jekyll.py`, or syncing Notion content into `_posts`. (file: `.codex/skills/notion-jekyll-blog/SKILL.md`)

### How to use skills
- Discovery: Read the named skill only when the task matches it.
- Trigger rules: If the user mentions Notion to blog sync, Notion page conversion, or asks to update the conversion rules, use `notion-jekyll-blog`.
- Scope: Keep the skill focused on this repo's Notion -> Jekyll workflow. Do not use it for generic Notion tasks outside this blog.
- Progressive disclosure: Read `SKILL.md` first. Load the referenced `references/` file only if exact mapping details are needed.
- Safety: Preserve unrelated user edits in `_posts` and other files. Never overwrite an existing post unless the user explicitly asks for replacement.

### Repository workflow rules
- If content/config/code was changed in this repository, automatically run `git add -A`, `git commit`, and `git push` at the end of the task unless the user explicitly says not to.
- Use a concise commit message that reflects the actual change.
- Never commit real secrets from `.env`; keep only placeholder values in `.env.example`.
