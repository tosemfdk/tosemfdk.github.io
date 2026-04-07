# Chirpy Starter

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
[![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)][mit]

When installing the [**Chirpy**][chirpy] theme through [RubyGems.org][gem], Jekyll can only read files in the folders
`_data`, `_layouts`, `_includes`, `_sass` and `assets`, as well as a small part of options of the `_config.yml` file
from the theme's gem. If you have ever installed this theme gem, you can use the command
`bundle info --path jekyll-theme-chirpy` to locate these files.

The Jekyll team claims that this is to leave the ball in the user’s court, but this also results in users not being
able to enjoy the out-of-the-box experience when using feature-rich themes.

To fully use all the features of **Chirpy**, you need to copy the other critical files from the theme's gem to your
Jekyll site. The following is a list of targets:

```shell
.
├── _config.yml
├── _plugins
├── _tabs
└── index.html
```

To save you time, and also in case you lose some files while copying, we extract those files/configurations of the
latest version of the **Chirpy** theme and the [CD][CD] workflow to here, so that you can start writing in minutes.

## Usage

Check out the [theme's docs](https://github.com/cotes2020/jekyll-theme-chirpy/wiki).

## Notion import

This repo includes `tools/notion_to_jekyll.py` for importing Notion content into `_posts/`.

- Default mode: `NOTION_IMPORT_MODE=single`
- Direct-child batch mode: `NOTION_IMPORT_MODE=direct_children`

Example direct-child import:

```shell
NOTION_IMPORT_MODE=direct_children \
NOTION_IMPORT_ROOT_PAGE_ID=24dcbb7d79378091bb83df2ae86685f4 \
NOTION_IMPORT_CATEGORY_OVERRIDE='서울대학교 여름방학 인턴' \
NOTION_DIRECT_CHILD_MAX_GIF_MB=25 \
uv run python tools/notion_to_jekyll.py
```

Direct-child mode:

- imports only direct child pages under the root page
- skips database children
- skips existing posts
- writes `notion_source_id` front matter for dedupe
- keeps video output as GIF only

## Contributing

This repository is automatically updated with new releases from the theme repository. If you encounter any issues or want to contribute to its improvement, please visit the [theme repository][chirpy] to provide feedback.

## License

This work is published under [MIT][mit] License.

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE
