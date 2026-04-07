from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image

QUALITY_PROFILES = [
    (10, 720, 256),
    (8, 640, 192),
    (6, 540, 128),
    (5, 480, 96),
    (4, 420, 64),
    (3, 360, 48),
    (2, 320, 32),
]
TRIM_RATIOS = [0.75, 0.5, 0.35, 0.25, 0.15]


@dataclass
class NotionConfig:
    notion_token: str
    page_id: str | None
    home_id: str | None
    import_mode: str
    import_root_page_id: str | None
    import_category_override: str | None
    posts_dir: str
    img_dir: str
    max_gif_bytes: int
    feature_image_max_bytes: int
    search_cache: list[dict[str, Any]] | None = field(default=None)
    generated_assets: set[str] = field(default_factory=set)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    @property
    def root_page_id(self) -> str | None:
        return self.import_root_page_id or self.page_id


def normalize_notion_id(raw_id):
    if not raw_id:
        return None

    raw_id = raw_id.strip()
    matches = re.findall(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32}",
        raw_id,
    )

    if matches:
        compact = re.sub(r"[^0-9a-fA-F]", "", matches[-1])
    else:
        compact = re.sub(r"[^0-9a-fA-F]", "", raw_id)

    if len(compact) != 32:
        return raw_id

    return (
        f"{compact[0:8]}-{compact[8:12]}-{compact[12:16]}-"
        f"{compact[16:20]}-{compact[20:32]}"
    ).lower()


def yaml_scalar(value):
    return json.dumps(value, ensure_ascii=False)


def mb_to_bytes(raw_value: str, default: str) -> int:
    return int(float(raw_value or default) * 1024 * 1024)


def load_config(env: dict[str, str] | None = None, cwd: str | None = None) -> NotionConfig:
    load_dotenv()
    env = env or os.environ
    cwd = cwd or os.getcwd()

    notion_token = env.get("NOTION_TOKEN", "").strip()
    page_id = normalize_notion_id(env.get("NOTION_PAGE_ID") or env.get("NOTION_DATABASE_ID"))
    home_id = normalize_notion_id(env.get("NOTION_HOME_ID"))
    import_mode = (env.get("NOTION_IMPORT_MODE") or "single").strip().lower()
    import_root_page_id = normalize_notion_id(env.get("NOTION_IMPORT_ROOT_PAGE_ID"))
    import_category_override = (env.get("NOTION_IMPORT_CATEGORY_OVERRIDE") or "").strip() or None

    if import_mode not in {"single", "direct_children"}:
        raise ValueError("NOTION_IMPORT_MODE must be one of: single, direct_children")
    if not notion_token:
        raise ValueError("NOTION_TOKEN must be set in .env file.")

    root_page_id = import_root_page_id or page_id
    if import_mode == "single" and not page_id:
        raise ValueError("NOTION_PAGE_ID or NOTION_DATABASE_ID must be set for single import mode.")
    if import_mode == "direct_children" and not root_page_id:
        raise ValueError(
            "NOTION_IMPORT_ROOT_PAGE_ID or NOTION_PAGE_ID must be set for direct_children import mode."
        )

    posts_dir = os.path.join(cwd, "_posts")
    img_dir = os.path.join(cwd, "assets", "img", "posts")
    os.makedirs(posts_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    legacy_gif_bytes = mb_to_bytes(env.get("NOTION_MAX_GIF_MB", "95"), "95")
    direct_child_gif_bytes = mb_to_bytes(env.get("NOTION_DIRECT_CHILD_MAX_GIF_MB", "25"), "25")
    max_gif_bytes = direct_child_gif_bytes if import_mode == "direct_children" else legacy_gif_bytes
    feature_image_max_bytes = mb_to_bytes(env.get("NOTION_FEATURE_IMAGE_MAX_MB", "12"), "12")

    return NotionConfig(
        notion_token=notion_token,
        page_id=page_id,
        home_id=home_id,
        import_mode=import_mode,
        import_root_page_id=root_page_id if import_mode == "direct_children" else import_root_page_id,
        import_category_override=import_category_override,
        posts_dir=posts_dir,
        img_dir=img_dir,
        max_gif_bytes=max_gif_bytes,
        feature_image_max_bytes=feature_image_max_bytes,
    )


def notion_request(config: NotionConfig, path: str, method="GET", params=None, payload=None):
    url = f"https://api.notion.com/v1/{path.lstrip('/')}"
    if method == "POST":
        return requests.post(url, headers=config.headers, json=payload)
    return requests.get(url, headers=config.headers, params=params)


def compact_notion_id(raw_id):
    normalized = normalize_notion_id(raw_id)
    if not normalized:
        return ""
    return normalized.replace("-", "")


def common_prefix_length(left, right):
    count = 0
    for lch, rch in zip(left, right):
        if lch != rch:
            break
        count += 1
    return count


def search_accessible_objects(config: NotionConfig):
    if config.search_cache is not None:
        return config.search_cache

    results = []
    start_cursor = None

    while True:
        payload = {}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        res = notion_request(config, "search", method="POST", payload=payload)
        if res.status_code != 200:
            print(f"Error searching Notion: {res.text}")
            break

        data = res.json()
        results.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        start_cursor = data.get("next_cursor")

    config.search_cache = results
    return config.search_cache


def fetch_page(config: NotionConfig, page_id):
    normalized_id = normalize_notion_id(page_id)
    if not normalized_id:
        return None

    res = notion_request(config, f"pages/{normalized_id}")
    if res.status_code == 200:
        return res.json()
    return None


def fetch_database_title(config: NotionConfig, database_id):
    normalized_id = normalize_notion_id(database_id)
    if not normalized_id:
        return ""

    res = notion_request(config, f"databases/{normalized_id}")
    if res.status_code != 200:
        return ""

    database = res.json()
    return get_rich_text(database.get("title", []))


def find_best_search_match(config: NotionConfig, target_id, object_type="page"):
    target_compact = compact_notion_id(target_id)
    if not target_compact:
        return None

    best_match = None
    best_score = 0

    for item in search_accessible_objects(config):
        if item.get("object") != object_type:
            continue

        score = common_prefix_length(target_compact, compact_notion_id(item.get("id")))
        if score > best_score:
            best_match = item
            best_score = score

    if best_score < 12:
        return None

    return best_match


def resolve_page(config: NotionConfig, page_id):
    page = fetch_page(config, page_id)
    if page:
        return page, normalize_notion_id(page_id)

    fallback = find_best_search_match(config, page_id, object_type="page")
    if not fallback:
        return None, normalize_notion_id(page_id)

    resolved_id = fallback["id"]
    print(f"Direct page lookup failed for {page_id}; using closest accessible page {resolved_id}.")
    return fetch_page(config, resolved_id) or fallback, resolved_id


def extract_title(properties):
    for prop in properties.values():
        if prop.get("type") == "title":
            title_parts = prop.get("title", [])
            title = "".join(part.get("plain_text", "") for part in title_parts).strip()
            if title:
                return title
    return "Untitled"


def infer_category_from_schema(properties):
    property_names = {name.lower() for name in properties}
    property_ids = " ".join(prop.get("id", "") for prop in properties.values()).lower()

    if "project name" in property_names or "projects" in property_ids:
        return extract_title(properties)

    return ""


def resolve_relation_page_title(config: NotionConfig, properties, relation_name="project"):
    for name, prop in properties.items():
        if prop.get("type") != "relation":
            continue

        normalized_name = name.lower()
        normalized_id = prop.get("id", "").lower()
        if relation_name not in normalized_name and relation_name not in normalized_id:
            continue

        relation_items = prop.get("relation", [])
        if not relation_items:
            continue

        related_page = fetch_page(config, relation_items[0]["id"])
        if related_page:
            return extract_title(related_page.get("properties", {}))

    return ""


def resolve_category(config: NotionConfig, page, properties, category_override=None):
    if category_override:
        return category_override

    related_project_title = resolve_relation_page_title(config, properties, relation_name="project")
    if related_project_title:
        return related_project_title

    for prop in properties.values():
        if prop.get("type") == "select" and prop.get("select"):
            return prop["select"]["name"]

    inferred = infer_category_from_schema(properties)
    if inferred:
        return inferred

    parent = page.get("parent", {})
    parent_type = parent.get("type")

    if parent_type == "page_id":
        parent_page, _ = resolve_page(config, parent.get("page_id"))
        if parent_page:
            parent_title = extract_title(parent_page.get("properties", {}))
            if parent_title:
                return parent_title

    if parent_type == "database_id":
        database_title = fetch_database_title(config, parent.get("database_id"))
        if database_title:
            return database_title

    parent_id = parent.get(parent_type) if parent_type else None
    if config.home_id and parent_id and common_prefix_length(compact_notion_id(config.home_id), compact_notion_id(parent_id)) >= 12:
        return extract_title(properties)

    return ""


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def probe_video_duration(input_path: str) -> float | None:
    if not command_exists("ffprobe"):
        print("ffprobe is not available; GIF duration trimming is disabled.")
        return None

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def build_gif_conversion_attempts(duration: float | None = None):
    attempts = [
        {"fps": fps, "width": width, "colors": colors, "max_duration": None}
        for fps, width, colors in QUALITY_PROFILES
    ]

    if duration and duration > 1:
        for ratio in TRIM_RATIOS:
            trimmed_duration = max(1.0, round(duration * ratio, 2))
            if trimmed_duration >= duration:
                continue
            for fps, width, colors in QUALITY_PROFILES:
                attempts.append(
                    {
                        "fps": fps,
                        "width": width,
                        "colors": colors,
                        "max_duration": trimmed_duration,
                    }
                )
    return attempts


def build_gif_filter_chain(fps: int, width: int, colors: int) -> str:
    return (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=max_colors={colors}[p];"
        f"[s1][p]paletteuse"
    )


def convert_video_to_gif_with_limit(input_path, output_path, max_bytes):
    if not command_exists("ffmpeg"):
        print("ffmpeg is not available; cannot convert video to GIF.")
        return False

    duration = probe_video_duration(input_path)
    attempts = build_gif_conversion_attempts(duration)

    for attempt in attempts:
        if os.path.exists(output_path):
            os.remove(output_path)

        command = ["ffmpeg", "-y"]
        if attempt["max_duration"] is not None:
            command.extend(["-t", str(attempt["max_duration"])])
        command.extend(
            [
                "-i",
                input_path,
                "-vf",
                build_gif_filter_chain(attempt["fps"], attempt["width"], attempt["colors"]),
                "-loop",
                "0",
                output_path,
            ]
        )
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) <= max_bytes:
            return True

    return False


def optimize_image(response_content: bytes, filepath: str):
    img = Image.open(BytesIO(response_content))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    max_width = 1920
    if img.width > max_width:
        wpercent = max_width / float(img.width)
        hsize = int(float(img.height) * float(wpercent))
        img = img.resize((max_width, hsize), Image.Resampling.LANCZOS)

    img.save(filepath, "WEBP", quality=85, optimize=True)


def write_stream_to_file(response, filepath: str):
    with open(filepath, "wb") as file_handle:
        for chunk in response.iter_content(1024):
            file_handle.write(chunk)


def ensure_gif_with_limit(config: NotionConfig, response, block_id: str, source_ext: str) -> str:
    filename = f"{block_id}.gif"
    filepath = os.path.join(config.img_dir, filename)
    relative_path = f"/assets/img/posts/{filename}"
    temp_input = os.path.join(config.img_dir, f"temp_{block_id}.{source_ext}")
    write_stream_to_file(response, temp_input)

    if os.path.getsize(temp_input) <= config.max_gif_bytes:
        os.replace(temp_input, filepath)
        config.generated_assets.add(relative_path)
        print(f"Saved to {relative_path}")
        return relative_path

    print(f"Re-encoding GIF to satisfy {config.max_gif_bytes // (1024 * 1024)}MB limit: {filename}...")
    gif_ok = convert_video_to_gif_with_limit(temp_input, filepath, config.max_gif_bytes)
    if os.path.exists(temp_input):
        os.remove(temp_input)
    if not gif_ok:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise RuntimeError(
            f"Unable to convert GIF block {block_id} into a file under "
            f"{config.max_gif_bytes // (1024 * 1024)}MB."
        )
    config.generated_assets.add(relative_path)
    print(f"Saved to {relative_path}")
    return relative_path


def download_media(config: NotionConfig, url, block_id, ext):
    """Download media from Notion, optimize images to WebP, compress videos to GIF, and save to assets."""
    is_image = ext.lower() in ["png", "jpg", "jpeg"]
    is_video = ext.lower() in ["mp4", "mov", "webm"]
    is_gif = ext.lower() == "gif"

    if is_image:
        filename = f"{block_id}.webp"
    elif is_video or is_gif:
        filename = f"{block_id}.gif"
    else:
        filename = f"{block_id}.{ext}"

    filepath = os.path.join(config.img_dir, filename)
    relative_path = f"/assets/img/posts/{filename}"

    oversize_video = (
        (is_video or is_gif)
        and os.path.exists(filepath)
        and filepath.endswith(".gif")
        and os.path.getsize(filepath) > config.max_gif_bytes
    )

    if not os.path.exists(filepath) or oversize_video:
        if oversize_video:
            print(
                f"Existing GIF is too large ({os.path.getsize(filepath) / (1024 * 1024):.2f}MB), re-encoding {filename}..."
            )
        print(f"Downloading media: {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download media from {url}")
            if is_video or is_gif:
                raise RuntimeError(f"Failed to download GIF-constrained media for conversion: {url}")
            return url

        if is_image:
            optimize_image(response.content, filepath)
        elif is_video or is_gif:
            print(f"Converting media to GIF: {filename}...")
            return ensure_gif_with_limit(config, response, block_id, ext)
        else:
            write_stream_to_file(response, filepath)

        print(f"Saved to {relative_path}")

    config.generated_assets.add(relative_path)
    return relative_path


def apply_annotation(content, char):
    if not content:
        return content
    lspaces = len(content) - len(content.lstrip())
    rspaces = len(content) - len(content.rstrip())
    core = content.strip()
    if not core:
        return content
    return (" " * lspaces) + f"{char}{core}{char}" + (" " * rspaces)


def get_rich_text(rich_text_array):
    text = ""
    for rt in rich_text_array:
        content = rt.get("plain_text", "")
        annotations = rt.get("annotations", {})

        if rt.get("type") == "equation":
            content = apply_annotation(content, "$")
        else:
            if annotations.get("bold"):
                content = apply_annotation(content, "**")
            if annotations.get("italic"):
                content = apply_annotation(content, "*")
            if annotations.get("strikethrough"):
                content = apply_annotation(content, "~")
            if annotations.get("code"):
                content = apply_annotation(content, "`")

        link = rt.get("href")
        if link:
            content = f"[{content}]({link})"

        text += content
    return text


def parse_block(config: NotionConfig, block, children_md=""):
    block_type = block.get("type", "")
    md_text = ""

    if block_type == "paragraph":
        md_text = get_rich_text(block["paragraph"]["rich_text"]) + "\n\n" + children_md
    elif block_type == "heading_1":
        md_text = f"# {get_rich_text(block['heading_1']['rich_text'])}\n\n{children_md}"
    elif block_type == "heading_2":
        md_text = f"## {get_rich_text(block['heading_2']['rich_text'])}\n\n{children_md}"
    elif block_type == "heading_3":
        md_text = f"### {get_rich_text(block['heading_3']['rich_text'])}\n\n{children_md}"
    elif block_type == "bulleted_list_item":
        md_text = f"- {get_rich_text(block['bulleted_list_item']['rich_text'])}\n"
        if children_md:
            md_text += "".join([f"  {line}\n" for line in children_md.strip("\n").split("\n")])
    elif block_type == "numbered_list_item":
        md_text = f"1. {get_rich_text(block['numbered_list_item']['rich_text'])}\n"
        if children_md:
            md_text += "".join([f"   {line}\n" for line in children_md.strip("\n").split("\n")])
    elif block_type == "toggle":
        summary_text = get_rich_text(block["toggle"]["rich_text"])
        md_text = f"<details markdown=\"1\">\n<summary>{summary_text}</summary>\n\n{children_md}\n\n</details>\n\n"
    elif block_type == "code":
        language = block["code"].get("language", "")
        code_text = get_rich_text(block["code"]["rich_text"])
        md_text = f"```{language}\n{code_text}\n```\n\n"
    elif block_type == "child_page":
        md_text = f"## {block['child_page']['title']}\n\n{children_md}"
    elif block_type == "child_database":
        md_text = f"### {block['child_database']['title']}\n\n{children_md}"
    elif block_type == "table_of_contents":
        md_text = ""
    elif block_type == "bookmark":
        bookmark = block["bookmark"]
        url = bookmark.get("url", "")
        caption = get_rich_text(bookmark.get("caption", [])).strip()
        label = caption or url
        if url:
            md_text = f"[{label}]({url})\n\n"
    elif block_type == "image":
        image_obj = block["image"]
        url = image_obj.get("file", {}).get("url") or image_obj.get("external", {}).get("url")
        if url:
            caption = get_rich_text(image_obj.get("caption", []))
            ext = url.split("?")[0].split(".")[-1]
            if len(ext) > 4:
                ext = "png"
            local_path = download_media(config, url, block["id"], ext)
            md_text = f"![{caption}]({local_path})\n\n"
    elif block_type == "video":
        video_obj = block["video"]
        url = video_obj.get("file", {}).get("url") or video_obj.get("external", {}).get("url")
        if url:
            caption = get_rich_text(video_obj.get("caption", []))
            ext = url.split("?")[0].split(".")[-1]
            if len(ext) > 4:
                ext = "mp4"
            local_path = download_media(config, url, block["id"], ext)
            if not local_path.lower().endswith(".gif"):
                raise RuntimeError(f"Video block {block['id']} did not produce a GIF asset.")
            md_text = f"![{caption}]({local_path})\n\n"
    elif block_type == "equation":
        expr = block["equation"].get("expression", "")
        md_text = f"$$\n{expr}\n$$\n\n"
    elif block_type == "quote":
        md_text = f"> {get_rich_text(block['quote']['rich_text'])}\n\n"
    elif block_type == "divider":
        md_text = "---\n\n"

    return md_text


def list_block_children(config: NotionConfig, block_id):
    blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        params = {"start_cursor": start_cursor} if start_cursor else {}
        res = notion_request(config, f"blocks/{block_id}/children", params=params)

        if res.status_code != 200:
            print(f"Error fetching blocks: {res.text}")
            break

        data = res.json()
        blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    return blocks


def get_page_blocks(config: NotionConfig, block_id):
    blocks = []
    for block in list_block_children(config, block_id):
        children_md = ""
        if block.get("has_children"):
            children_md = get_page_blocks(config, block["id"])
        md = parse_block(config, block, children_md)
        blocks.append(md)
    return "".join(blocks)


def discover_direct_child_pages(config: NotionConfig, root_page_id: str):
    child_pages = []
    for block in list_block_children(config, root_page_id):
        if block.get("type") != "child_page":
            continue
        child_pages.append(
            {
                "id": normalize_notion_id(block.get("id")),
                "title": block.get("child_page", {}).get("title", "Untitled"),
            }
        )
    return child_pages


def local_media_size(relative_path):
    if not relative_path.startswith("/"):
        return None

    local_path = os.path.join(os.getcwd(), relative_path.lstrip("/"))
    if os.path.exists(local_path):
        return os.path.getsize(local_path)
    return None


def select_feature_image(markdown_body, max_bytes):
    image_paths = re.findall(r"!\[[^\]]*\]\((/assets/img/posts/[^)\s]+)\)", markdown_body)

    gif_candidates = [path for path in image_paths if path.lower().endswith(".gif")]
    for path in gif_candidates:
        size = local_media_size(path)
        if size is None or size <= max_bytes:
            return path

    for path in image_paths:
        if re.search(r"\.(?:webp|png|jpe?g)$", path, re.IGNORECASE):
            return path

    return None


def slugify_title(title: str) -> str:
    slug = re.sub(r"[\W_]+", "-", title.lower()).strip("-")
    return slug or "post"


def collect_tags(properties):
    tags = []
    for prop in properties.values():
        if prop.get("type") == "multi_select":
            tags.extend([tag["name"] for tag in prop["multi_select"]])
    return tags


def build_post_filename(title: str, created_time: str) -> str:
    post_date = created_time.split("T")[0]
    return f"{post_date}-{slugify_title(title)}.md"


def extract_asset_references(markdown_text: str) -> set[str]:
    refs = set(re.findall(r"/assets/img/posts/([^\)\s]+)", markdown_text))
    return {ref.rstrip('"').rstrip("'") for ref in refs}


def build_front_matter(
    *,
    title: str,
    created_time: str,
    featured_image: str | None,
    category: str | None,
    tags: list[str],
    notion_source_id: str | None = None,
):
    front_matter = [
        "---",
        "layout: post",
        f"title: {yaml_scalar(title)}",
        f"date: {created_time}",
        "math: true",
    ]

    if notion_source_id:
        front_matter.append(f"notion_source_id: {yaml_scalar(notion_source_id)}")
    if featured_image:
        front_matter.append("image:")
        front_matter.append(f"  path: {yaml_scalar(featured_image)}")
    if category:
        front_matter.append("categories:")
        front_matter.append(f"  - {yaml_scalar(category)}")
    if tags:
        front_matter.append("tags:")
        for tag in tags:
            front_matter.append(f"  - {yaml_scalar(tag)}")

    front_matter.append("---\n\n")
    return "\n".join(front_matter)


def render_page_to_post(
    config: NotionConfig,
    page: dict[str, Any],
    resolved_page_id: str,
    *,
    category_override: str | None = None,
    include_source_id: bool = False,
):
    properties = page.get("properties", {})
    title = extract_title(properties)
    created_time = page.get("created_time", "")
    tags = collect_tags(properties)
    category = resolve_category(config, page, properties, category_override=category_override)
    body_markdown = get_page_blocks(config, resolved_page_id)
    featured_image = select_feature_image(body_markdown, config.feature_image_max_bytes)
    notion_source_id = resolved_page_id if include_source_id else None

    content = build_front_matter(
        title=title,
        created_time=created_time,
        featured_image=featured_image,
        category=category,
        tags=tags,
        notion_source_id=notion_source_id,
    ) + body_markdown

    return {
        "title": title,
        "filename": build_post_filename(title, created_time),
        "content": content,
        "notion_source_id": notion_source_id,
    }


def extract_notion_source_id(text: str) -> str | None:
    match = re.search(r'^notion_source_id:\s*(?:"([^"]+)"|([^\n]+))$', text, re.MULTILINE)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip().strip('"').strip("'")


def find_existing_post_by_source_id(posts_dir: str, notion_source_id: str | None) -> str | None:
    if not notion_source_id:
        return None

    for filename in os.listdir(posts_dir):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(posts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file_handle:
            existing_source_id = extract_notion_source_id(file_handle.read())
        if existing_source_id == notion_source_id:
            return filepath
    return None


def should_skip_existing_post(posts_dir: str, filename: str, notion_source_id: str | None = None):
    existing_by_source_id = find_existing_post_by_source_id(posts_dir, notion_source_id)
    if existing_by_source_id:
        return True, existing_by_source_id, "source_id"

    filepath = os.path.join(posts_dir, filename)
    if os.path.exists(filepath):
        return True, filepath, "filename"

    return False, filepath, None


def write_post(filepath: str, content: str):
    with open(filepath, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)


def prune_generated_assets(config: NotionConfig, created_filepaths: list[str]):
    referenced_assets = set()
    for filepath in created_filepaths:
        with open(filepath, "r", encoding="utf-8") as file_handle:
            referenced_assets.update(extract_asset_references(file_handle.read()))

    for relative_path in sorted(config.generated_assets):
        basename = os.path.basename(relative_path)
        if basename in referenced_assets:
            continue
        local_path = os.path.join(config.img_dir, basename)
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"Removed unreferenced generated asset: {relative_path}")


def write_post_if_missing(posts_dir: str, filename: str, content: str, notion_source_id: str | None = None):
    should_skip, filepath, reason = should_skip_existing_post(posts_dir, filename, notion_source_id)
    if should_skip:
        if reason == "source_id":
            print(f"Skipping existing post for notion_source_id={notion_source_id}: {filepath}")
        else:
            print(f"Skipping existing post by filename: {filepath}")
        return False, filepath

    write_post(filepath, content)
    print(f"Created: {filepath}")
    return True, filepath


def process_single_page(config: NotionConfig):
    print("Fetching Notion page...")
    page, resolved_page_id = resolve_page(config, config.page_id)
    if not page:
        print(f"Error fetching page: could not resolve {config.page_id}")
        return

    post = render_page_to_post(config, page, resolved_page_id)
    filepath = os.path.join(config.posts_dir, post["filename"])
    write_post(filepath, post["content"])
    print(f"Created/Updated: {filepath}\n")


def process_direct_children_import(config: NotionConfig):
    root_page_id = config.root_page_id
    print(f"Discovering direct child pages for root {root_page_id}...")
    child_pages = discover_direct_child_pages(config, root_page_id)

    discovered_count = len(child_pages)
    skipped_count = 0
    created_count = 0
    created_filepaths: list[str] = []

    for child in child_pages:
        page, resolved_page_id = resolve_page(config, child["id"])
        if not page:
            print(f"Skipping unresolved child page: {child['id']} ({child['title']})")
            skipped_count += 1
            continue

        filename = build_post_filename(
            extract_title(page.get("properties", {})),
            page.get("created_time", ""),
        )
        should_skip, filepath, reason = should_skip_existing_post(
            config.posts_dir,
            filename,
            notion_source_id=resolved_page_id,
        )
        if should_skip:
            if reason == "source_id":
                print(f"Skipping existing post for notion_source_id={resolved_page_id}: {filepath}")
            else:
                print(f"Skipping existing post by filename: {filepath}")
            skipped_count += 1
            continue

        post = render_page_to_post(
            config,
            page,
            resolved_page_id,
            category_override=config.import_category_override,
            include_source_id=True,
        )
        created, _ = write_post_if_missing(
            config.posts_dir,
            post["filename"],
            post["content"],
            notion_source_id=post["notion_source_id"],
        )
        if created:
            created_count += 1
            created_filepaths.append(os.path.join(config.posts_dir, post["filename"]))
        else:
            skipped_count += 1

    prune_generated_assets(config, created_filepaths)
    print(f"Summary: discovered={discovered_count} skipped={skipped_count} created={created_count}")


def main():
    try:
        config = load_config()
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    if config.import_mode == "direct_children":
        process_direct_children_import(config)
        return
    process_single_page(config)


if __name__ == "__main__":
    main()
