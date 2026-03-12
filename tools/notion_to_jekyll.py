import os
import re
import json
import requests
import subprocess
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv


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

# Load environment variables
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = normalize_notion_id(os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_DATABASE_ID"))
HOME_ID = normalize_notion_id(os.getenv("NOTION_HOME_ID"))

if not NOTION_TOKEN or not PAGE_ID:
    print("Error: NOTION_TOKEN and NOTION_PAGE_ID must be set in .env file.")
    exit(1)

# Setup Notion API Headers
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# Paths for Jekyll
POSTS_DIR = os.path.join(os.getcwd(), "_posts")
IMG_DIR = os.path.join(os.getcwd(), "assets", "img", "posts")
os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

SEARCH_CACHE = None


def notion_request(path, method="GET", params=None, payload=None):
    url = f"https://api.notion.com/v1/{path.lstrip('/')}"
    if method == "POST":
        return requests.post(url, headers=HEADERS, json=payload)
    return requests.get(url, headers=HEADERS, params=params)


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


def search_accessible_objects():
    global SEARCH_CACHE
    if SEARCH_CACHE is not None:
        return SEARCH_CACHE

    results = []
    start_cursor = None

    while True:
        payload = {}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        res = notion_request("search", method="POST", payload=payload)
        if res.status_code != 200:
            print(f"Error searching Notion: {res.text}")
            break

        data = res.json()
        results.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        start_cursor = data.get("next_cursor")

    SEARCH_CACHE = results
    return SEARCH_CACHE


def fetch_page(page_id):
    normalized_id = normalize_notion_id(page_id)
    if not normalized_id:
        return None

    res = notion_request(f"pages/{normalized_id}")
    if res.status_code == 200:
        return res.json()
    return None


def fetch_database_title(database_id):
    normalized_id = normalize_notion_id(database_id)
    if not normalized_id:
        return ""

    res = notion_request(f"databases/{normalized_id}")
    if res.status_code != 200:
        return ""

    database = res.json()
    return get_rich_text(database.get("title", []))


def find_best_search_match(target_id, object_type="page"):
    target_compact = compact_notion_id(target_id)
    if not target_compact:
        return None

    best_match = None
    best_score = 0

    for item in search_accessible_objects():
        if item.get("object") != object_type:
            continue

        score = common_prefix_length(target_compact, compact_notion_id(item.get("id")))
        if score > best_score:
            best_match = item
            best_score = score

    if best_score < 12:
        return None

    return best_match


def resolve_page(page_id):
    page = fetch_page(page_id)
    if page:
        return page, normalize_notion_id(page_id)

    fallback = find_best_search_match(page_id, object_type="page")
    if not fallback:
        return None, normalize_notion_id(page_id)

    resolved_id = fallback["id"]
    print(f"Direct page lookup failed for {page_id}; using closest accessible page {resolved_id}.")
    return fetch_page(resolved_id) or fallback, resolved_id


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


def resolve_relation_page_title(properties, relation_name="project"):
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

        related_page = fetch_page(relation_items[0]["id"])
        if related_page:
            return extract_title(related_page.get("properties", {}))

    return ""


def resolve_category(page, properties):
    related_project_title = resolve_relation_page_title(properties, relation_name="project")
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
        parent_page, _ = resolve_page(parent.get("page_id"))
        if parent_page:
            parent_title = extract_title(parent_page.get("properties", {}))
            if parent_title:
                return parent_title

    if parent_type == "database_id":
        database_title = fetch_database_title(parent.get("database_id"))
        if database_title:
            return database_title

    if HOME_ID and common_prefix_length(compact_notion_id(HOME_ID), compact_notion_id(parent.get(parent_type))) >= 12:
        return extract_title(properties)

    return ""


def yaml_scalar(value):
    return json.dumps(value, ensure_ascii=False)

def download_media(url, block_id, ext):
    """Download media from Notion, optimize images to WebP, compress videos to GIF, and save to assets."""
    is_image = ext.lower() in ['png', 'jpg', 'jpeg']
    is_video = ext.lower() in ['mp4', 'mov', 'webm']
    
    if is_image:
        filename = f"{block_id}.webp"
    elif is_video:
        filename = f"{block_id}.gif"
    else:
        filename = f"{block_id}.{ext}"
        
    filepath = os.path.join(IMG_DIR, filename)
    relative_path = f"/assets/img/posts/{filename}"
    
    # Download if not already exists
    if not os.path.exists(filepath):
        print(f"Downloading media: {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            if is_image:
                # Open image from memory
                img = Image.open(BytesIO(response.content))
                # Convert RGBA to RGB (useful for PNG with alpha channel) if target is WebP
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Resize if width > 1920
                max_width = 1920
                if img.width > max_width:
                    wpercent = (max_width / float(img.width))
                    hsize = int((float(img.height) * float(wpercent)))
                    img = img.resize((max_width, hsize), Image.Resampling.LANCZOS)
                
                # Save as optimized WebP
                img.save(filepath, "WEBP", quality=85, optimize=True)
            elif is_video:
                # Save temp mp4
                temp_video = os.path.join(IMG_DIR, f"temp_{block_id}.{ext}")
                with open(temp_video, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                # Convert to gif using ffmpeg
                print(f"Converting video to GIF: {filename}...")
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_video,
                    "-vf", "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    "-loop", "0", filepath
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            else:
                # For other files, just save directly
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                        
            print(f"Saved to {relative_path}")
        else:
            print(f"Failed to download media from {url}")
            return url # Return original url as fallback
            
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
    """Convert Notion rich text objects to Markdown."""
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

def parse_block(block, children_md=""):
    """Parse a single Notion block into Markdown and append recursive children."""
    block_type = block.get("type", "")
    md_text = ""
    
    if block_type == "paragraph":
        md_text = get_rich_text(block["paragraph"]["rich_text"]) + "\n\n" + children_md
        
    elif block_type == "heading_1":
        md_text = f"# {get_rich_text(block['heading_1']['rich_text'])}\n\n"
        
    elif block_type == "heading_2":
        md_text = f"## {get_rich_text(block['heading_2']['rich_text'])}\n\n"
        
    elif block_type == "heading_3":
        md_text = f"### {get_rich_text(block['heading_3']['rich_text'])}\n\n"
        
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
        md_text = f"## {block['child_page']['title']}\n\n"

    elif block_type == "child_database":
        md_text = f"### {block['child_database']['title']}\n\n"

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
            # Extract ext from url, default to png
            ext = url.split("?")[0].split(".")[-1]
            if len(ext) > 4: ext = "png"
            
            local_path = download_media(url, block["id"], ext)
            md_text = f"![{caption}]({local_path})\n\n"
            
    elif block_type == "video":
        video_obj = block["video"]
        url = video_obj.get("file", {}).get("url") or video_obj.get("external", {}).get("url")
        if url:
            caption = get_rich_text(video_obj.get("caption", []))
            ext = url.split("?")[0].split(".")[-1]
            if len(ext) > 4: ext = "mp4"
            local_path = download_media(url, block["id"], ext)
            
            md_text = f"![{caption}]({local_path})\n\n"
                
    elif block_type == "equation":
        expr = block["equation"].get("expression", "")
        md_text = f"$$\n{expr}\n$$\n\n"
                
    elif block_type == "quote":
        md_text = f"> {get_rich_text(block['quote']['rich_text'])}\n\n"
        
    elif block_type == "divider":
        md_text = "---\n\n"

    return md_text

def get_page_blocks(block_id):
    """Recursively fetch and parse blocks from a page."""
    blocks = []
    has_more = True
    start_cursor = None
    
    while has_more:
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        params = {"start_cursor": start_cursor} if start_cursor else {}
        res = requests.get(url, headers=HEADERS, params=params)
        
        if res.status_code != 200:
            print(f"Error fetching blocks: {res.text}")
            break
            
        data = res.json()
        for block in data.get("results", []):
            children_md = ""
            if block.get("has_children"):
                children_md = get_page_blocks(block["id"])
                
            md = parse_block(block, children_md)
            blocks.append(md)
            
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
        
    return "".join(blocks)

def process_notion_page():
    """Fetch a single page and convert it to a Jekyll post."""
    print("Fetching Notion page...")

    page, resolved_page_id = resolve_page(PAGE_ID)
    if not page:
        print(f"Error fetching page: could not resolve {PAGE_ID}")
        return

    properties = page.get("properties", {})

    title = extract_title(properties)

    # Clean title for filename
    slug = re.sub(r'[\W_]+', '-', title.lower()).strip('-')
    if not slug:
        slug = "post"

    created_time = page.get("created_time", "")
    post_date = created_time.split("T")[0]
    matter_date_str = created_time

    tags = []
    for prop in properties.values():
        if prop.get("type") == "multi_select":
            tags.extend([tag["name"] for tag in prop["multi_select"]])

    category = resolve_category(page, properties)

    print(f"Processing: {title} ({post_date})")

    # Fetch body text
    body_markdown = get_page_blocks(resolved_page_id)

    # Construct Front Matter
    front_matter = [
        "---",
        f"layout: post",
        f"title: {yaml_scalar(title)}",
        f"date: {matter_date_str}",
        "math: true"
    ]

    if category:
        front_matter.append("categories:")
        front_matter.append(f"  - {yaml_scalar(category)}")
    if tags:
        front_matter.append("tags:")
        for tag in tags:
            front_matter.append(f"  - {yaml_scalar(tag)}")

    front_matter.append("---\n\n")

    full_content = "\n".join(front_matter) + body_markdown

    # Save File
    filename = f"{post_date}-{slug}.md"
    filepath = os.path.join(POSTS_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"Created/Updated: {filepath}\n")

if __name__ == "__main__":
    process_notion_page()
