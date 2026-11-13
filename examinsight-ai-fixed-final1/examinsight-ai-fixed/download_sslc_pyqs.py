"""Download available catalogued SSLC PDFs into subject/year folders.
Run from the project root: python download_sslc_pyqs.py
The app sync endpoint also registers downloaded papers in the database.
"""
from app.kseab_source import KSEAB_CATALOG, get_pyq_storage_path, download_pdf

ok = failed = skipped = 0
for item in KSEAB_CATALOG:
    path = get_pyq_storage_path(item.get("grade", 10), item["subject"], item["year"], item["file_name"])
    import os
    if os.path.isfile(path) and os.path.getsize(path) > 100 and open(path, "rb").read(4) == b"%PDF":
        skipped += 1
        continue
    if download_pdf(item["source_url"], path):
        ok += 1
        print("DOWNLOADED", path)
    else:
        failed += 1
        print("FAILED", item["year"], item["subject"], item["source_url"])
print(f"\nFinished. Downloaded={ok}, already_present={skipped}, failed={failed}, catalogued={len(KSEAB_CATALOG)}")
print("Review failures and use the source archive links to obtain papers that could not be fetched automatically.")
