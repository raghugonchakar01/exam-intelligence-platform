# Airbnb Listing Page Clone – Playpower Take-Home Project

> **Pixel-perfect desktop clone of the Airbnb listing page ("Romantic Jacuzzi 1BHK Candolim | Mirashya UG10") with full Photo Tour, single-photo Lightbox, production-grade Architecture Diagram, AI Subagent configs, and prompt sequence.**

---

## 🚀 Live Demo & Getting Started

### Prerequisites
- Node.js (v18+ or v20+)
- npm (v9+)

### Installation & Local Run
```bash
# 1. Install dependencies
npm install

# 2. Start the development server
npm run dev

# 3. Build for production
npm run build

# 4. Preview the production build
npm run preview
```

Open `http://localhost:5173` to explore the application.

---

## 🎯 Deliverables & Requirements Fulfillment

| Requirement | Implementation Detail | Location |
| :--- | :--- | :--- |
| **1. Listing Page** | Header, search pill, user avatar, title, Superhost badge, 5-photo hero grid, room specs, guest favorite callout, expandable description, amenities grid, reviews rating bars, location map card, host details, and sticky booking card with dynamic calculation. | [`src/App.tsx`](./src/App.tsx), [`src/components/`](./src/components/) |
| **2. Photo Tour** | Full-screen photo gallery opened from "Show all photos" or hero images, with room categorization (Living room, Jacuzzi & Patio, Bedroom, Bathroom, Exterior & Pool), back button, share and save. | [`src/components/PhotoTourModal.tsx`](./src/components/PhotoTourModal.tsx) |
| **3. Lightbox** | Single-photo viewer with previous/next chevron buttons, image counter (`X / Total`), caption, room badge, and keyboard navigation (`ArrowLeft`, `ArrowRight`, `Escape`). | [`src/components/LightboxModal.tsx`](./src/components/LightboxModal.tsx) |
| **4. Architecture Diagram** | High-level architecture diagram and scaling strategy for an Airbnb-scale vacation-rental marketplace across frontend, edge CDN, API gateway, microservices, distributed SQL, Redis, OpenSearch, Kafka, and Kubernetes GitOps. | [`architecture/system_architecture.svg`](./architecture/system_architecture.svg), [`architecture/ARCHITECTURE.md`](./architecture/ARCHITECTURE.md) |
| **5. AI Subagent Configs** | Sub-agent and skill configurations for code quality, accessibility auditing, and distributed systems architecture review. | [`ai-configs/`](./ai-configs/) |
| **6. Prompt Sequence** | Comprehensive audit trail of all prompts and AI-assisted workflows used to design and execute the task. | [`docs/AI_PROMPT_SEQUENCE.md`](./docs/AI_PROMPT_SEQUENCE.md) |

---

## ⌨️ Keyboard Navigation & Accessibility (a11y)
- **ArrowLeft ($\leftarrow$)**: Previous photo in Lightbox.
- **ArrowRight ($\rightarrow$)**: Next photo in Lightbox.
- **Escape (`Esc`)**: Closes Lightbox or Photo Tour.
- **Accessible ARIA attributes**: All interactive dialogs have `role="dialog"`, `aria-modal="true"`, and descriptive `aria-label` tags.

---

## 🏗️ Project Architecture & Tech Stack
- **Framework**: React 18 + TypeScript
- **Styling**: Tailwind CSS (custom Airbnb color palette and typography)
- **Icons**: Lucide React
- **Build Tool**: Vite 5 (zero bloat, instant HMR, optimized production chunks)

## Karnataka SSLC previous-year paper catalogue

The PYQ source catalogue is in `data/sslc_pyq_sources.json` and is also wired into `app/kseab_source.py`. It includes 2015–2026 links for Mathematics, Science, Social Science, English second language, Hindi third language, and available Kannada first/second language papers, plus 2022 supplementary papers. Subject/year PDFs are stored under `pyq/class10/<subject>/<year>/`.

To populate the PYQ library, run the backend and use the existing **Sync SSLC papers** action (POST `/api/admin/sync-kseab-pyqs`). The sync downloads available PDFs, extracts text, and registers papers in the existing database. Internet access is required, and source links may change or be unavailable. Failed downloads are reported as failures; the app no longer creates fake placeholder PDFs for missing papers. The catalog links point to a third-party archive for many years, so verify each source and its reuse permissions before redistributing the files.

## Auditable PYQ collection and source-backed generation

The collection workflow is deliberately resumable and additive: it does not delete
the existing SQLite library or the original PDFs. Every configured or discovered
candidate URL is kept in `pyq_documents` with a persistent status, attempts,
resolved URL, validation result, checksum, extraction result, and processing
history. This is why a failed source is visible in the report rather than being
silently omitted.

```powershell
# Install runtime requirements in the selected Python environment
python -m pip install -r requirements.txt

# Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Discover, download, validate, OCR when available, and register candidates.
# Re-run safely to resume; use --retry-failed for bounded retry of failures.
python -m app.pyq_pipeline sync
python -m app.pyq_pipeline sync --retry-failed

# Print the current evidence-based collection report without downloading again
python -m app.pyq_pipeline report

# Automated regression suite
python -m pytest -q
```

Useful HTTP endpoints:

- `POST /api/admin/sync-kseab-pyqs?retry_failed=true` runs the same persisted workflow.
- `GET /api/admin/pyq-collection-report` returns counts, statuses, unresolved candidates and the completeness status.
- `GET /api/admin/pyq-documents?status_filter=DOWNLOAD_FAILED` exposes recoverable failure records.

Question-paper generation now retrieves source questions on the server and stores
the generated paper plus source-question IDs in `generated_papers`. A local
fallback contains only retrieved PYQs; it will return a clear 422 response when
the source pool is insufficient instead of inventing questions. The optional
Gemini path receives bounded PYQ/syllabus context and generated questions are
marked as newly generated rather than past-paper questions.

OCR uses Tesseract when it is available. English is supported in the supplied
environment; Kannada OCR requires installing Tesseract's `kan` language data,
after which the collector will use `kan+eng` for Kannada-medium records.
