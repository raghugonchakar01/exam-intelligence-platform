import fitz
import os

pdfs = [
    "pyq/class10/mathematics/2022/KSEAB_SSLC_Class10_Mathematics_2022_Main_81E.pdf",
    "pyq/class10/mathematics/2022/KSEAB_SSLC_Class10_Mathematics_2022_Supplementary_81E.pdf",
    "pyq/class10/science/2022/KSEAB_SSLC_Class10_Science_2022_Main_83E.pdf",
    "pyq/class10/social_science/2022/KSEAB_SSLC_Class10_SocialScience_2022_Main_85E.pdf",
    "pyq/class10/english_communicative/2022/KSEAB_SSLC_Class10_English_2022_Main_31E.pdf",
]

for pdf_path in pdfs:
    if not os.path.exists(pdf_path):
        print(f"NOT FOUND: {pdf_path}")
        continue
    doc = fitz.open(pdf_path)
    total_images = 0
    for page_num, page in enumerate(doc):
        imgs = page.get_images(full=True)
        total_images += len(imgs)
        if imgs:
            print(f"{os.path.basename(pdf_path)} p{page_num+1}: {len(imgs)} img(s)")
            for img in imgs[:3]:
                xref = img[0]
                info = doc.extract_image(xref)
                w = info["width"]
                h = info["height"]
                ext = info["ext"]
                size = len(info["image"])
                print(f"  xref={xref} {w}x{h} {ext} {size} bytes")
    print(f"  >> TOTAL for {os.path.basename(pdf_path)}: {total_images} images")
    doc.close()
