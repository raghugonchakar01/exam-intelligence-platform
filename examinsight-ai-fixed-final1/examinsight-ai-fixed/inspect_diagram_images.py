"""
Inspect actual content images (non-decorative) in Math PDF.
Small images (< 200x200) that repeat on every page are likely decorative borders/logos.
Large images are actual question figures/diagrams.
"""
import fitz, base64, os

doc = fitz.open("pyq/class10/mathematics/2022/KSEAB_SSLC_Class10_Mathematics_2022_Main_81E.pdf")

# Collect unique large images (likely real diagrams, not borders/logos)
# Heuristic: width or height > 150px AND size > 3000 bytes AND not repeated (unique xref)
seen_xrefs = set()
content_images = []

for page_num, page in enumerate(doc):
    imgs = page.get_images(full=True)
    # Get text near each image rect to understand context
    page_text = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,blockno,blocktype)
    
    for img in imgs:
        xref = img[0]
        if xref in seen_xrefs:
            continue
        info = doc.extract_image(xref)
        w, h = info["width"], info["height"]
        size = len(info["image"])
        # Skip tiny decorative images (logos, borders)
        if w < 150 and h < 150:
            continue
        if size < 3000:
            continue
        seen_xrefs.add(xref)
        
        # Get image rect on page
        img_list = page.get_image_rects(xref)
        rect = img_list[0] if img_list else None
        
        # Get nearby text (within 80px above the image)
        nearby_text = ""
        if rect:
            search_rect = fitz.Rect(rect.x0 - 20, rect.y0 - 100, rect.x1 + 20, rect.y0 + 20)
            nearby_text = page.get_text("text", clip=search_rect).strip()[:120]
        
        print(f"Page {page_num+1} | xref={xref} | {w}x{h} | {size}B | nearby: {repr(nearby_text[:80])}")
        content_images.append((page_num+1, xref, w, h, size, info["ext"]))

doc.close()
print(f"\nTotal content images: {len(content_images)}")
