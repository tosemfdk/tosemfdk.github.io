import os
import re
import json
import requests
import subprocess
from datetime import datetime
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from notion_client import Client

# Load environment variables
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_DATABASE_ID")

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
    

def get_rich_text(rich_text_array):
    """Convert Notion rich text objects to Markdown."""
    text = ""
    for rt in rich_text_array:
        content = rt.get("plain_text", "")
        annotations = rt.get("annotations", {})
        
        if rt.get("type") == "equation":
            content = f"${content}$"
        else:
            if annotations.get("bold"):
                content = f"**{content}**"
            if annotations.get("italic"):
                content = f"*{content}*"
            if annotations.get("strikethrough"):
                content = f"~{content}~"
            if annotations.get("code"):
                content = f"`{content}`"
            
        link = rt.get("href")
        if link:
            content = f"[{content}]({link})"
            
        text += content
    return text

def parse_block(block):
    """Parse a single Notion block into Markdown."""
    block_type = block.get("type", "")
    md_text = ""
    
    if block_type == "paragraph":
        md_text = get_rich_text(block["paragraph"]["rich_text"]) + "\n\n"
        
    elif block_type == "heading_1":
        md_text = f"# {get_rich_text(block['heading_1']['rich_text'])}\n\n"
        
    elif block_type == "heading_2":
        md_text = f"## {get_rich_text(block['heading_2']['rich_text'])}\n\n"
        
    elif block_type == "heading_3":
        md_text = f"### {get_rich_text(block['heading_3']['rich_text'])}\n\n"
        
    elif block_type == "bulleted_list_item":
        md_text = f"- {get_rich_text(block['bulleted_list_item']['rich_text'])}\n"
        
    elif block_type == "numbered_list_item":
        md_text = f"1. {get_rich_text(block['numbered_list_item']['rich_text'])}\n"
        
    elif block_type == "code":
        language = block["code"].get("language", "")
        code_text = get_rich_text(block["code"]["rich_text"])
        md_text = f"<details>\n<summary>코드 보기 ({language})</summary>\n\n```{language}\n{code_text}\n```\n</details>\n\n"
        
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
            md = parse_block(block)
            blocks.append(md)
            
            # recursive fetch children if exist (like indented lists)
            if block.get("has_children"):
                blocks.append(get_page_blocks(block["id"]))
                
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
        
    return "".join(blocks)

def process_notion_page():
    """Fetch a single page and convert it to a Jekyll post."""
    print("Fetching Notion page...")
    
    url = f"https://api.notion.com/v1/pages/{PAGE_ID}"
    res = requests.get(url, headers=HEADERS)
    
    if res.status_code != 200:
        print(f"Error fetching page: {res.text}")
        return
        
    page = res.json()
    properties = page.get("properties", {})
    
    # Try different property types to find the title
    title = "Untitled"
    for key, prop in properties.items():
         if prop.get("type") == "title" and prop.get("title"):
              title = prop["title"][0]["plain_text"]
              break
    
    # Clean title for filename
    slug = re.sub(r'[\W_]+', '-', title.lower()).strip('-')
    if not slug:
        slug = "post"
        
    created_time = page.get("created_time", "")
    post_date_str = created_time
    post_date = created_time.split("T")[0]
    matter_date_str = created_time
    
    # Assume categories / tags exist as properties if we find any Select / Multi-Select
    tags = []
    category = ""
    for key, prop in properties.items():
        if prop.get("type") == "multi_select":
             tags.extend([t["name"] for t in prop["multi_select"]])
        elif prop.get("type") == "select" and prop.get("select"):
             if not category: category = prop["select"]["name"]
             
    print(f"Processing: {title} ({post_date})")
    
    # Fetch body text
    body_markdown = get_page_blocks(PAGE_ID)
    
    # Construct Front Matter
    front_matter = [
        "---",
        f"layout: post",
        f"title: \"{title}\"",
        f"date: {matter_date_str}"
    ]
    
    if category:
        front_matter.append(f"categories: [{category}]")
    if tags:
        # Format as YAML list
        front_matter.append("tags:")
        for tag in tags:
            front_matter.append(f"  - {tag}")
            
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
