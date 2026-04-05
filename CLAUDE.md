# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Zync

Zync is an AI-powered video editing app. Users upload videos, the backend extracts keyframes and generates descriptions using a local Vision Language Model (Ollama/moondream), and then a semantic similarity engine matches clips to a natural language prompt. Users arrange matched clips on a timeline, apply effects, and export the final video.

## Running the Project

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
Requires Ollama running locally at `http://localhost:11434` with the `moondream` model pulled (`ollama pull moondream`).

### Frontend
```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:5173
npm run build      # production build
npm run lint       # ESLint
```

### Environment Variables
- `backend/.env`: `SUPABASE_URL`, `SUPABASE_KEY`
- `frontend/.env`: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

## Architecture

### Data Flow
1. User uploads video → stored in Supabase Storage (bucket: `videos`)
2. Frontend calls `POST /videos` → backend downloads video, extracts keyframes (OpenCV), uploads frames to Supabase Storage (bucket: `keyframe`), generates descriptions via Ollama
3. User types a prompt → `POST /videos/process` → `SimilarityFilter` (sentence-transformers `all-mpnet-base-v2`) scores descriptions against prompt using cosine similarity
4. Frontend groups returned frames by `clip_id` into timeline clips
5. User arranges timeline, adjusts effects → `POST /videos/export` → MoviePy + FFmpeg pipeline outputs final video to `backend/outputs/`
6. Frontend downloads result as blob

### Backend (`backend/`)

| File | Role |
|------|------|
| `api.py` | FastAPI entry point. Endpoints: `/health`, `POST /videos`, `POST /videos/process`, `POST /videos/merge`, `POST /videos/export` |
| `vlm_analyzer.py` | Extracts keyframes (scene-change detection via histogram diff), uploads to Supabase, calls Ollama for descriptions |
| `post_processing.py` | MoviePy + FFmpeg pipeline: trims, filters, fades, aspect ratio crop, brightness/contrast, captions, speed, music mixing |
| `my_supabase_helper.py` | Supabase client wrapper for `keyframes` and `descriptions` tables |
| `hanna_rep/src/similarity_filter.py` | Core semantic search: embeds descriptions + prompt, cosine similarity, threshold filtering (0.40), dedup by `clip_id` |
| `hanna_rep/src/config.py` | `EMBED_MODEL`, `SIMILARITY_THRESHOLD`, `TOP_K_PER_CLIP` |
| `hanna_rep/src/db_reader.py` | Reads descriptions from Supabase for similarity input |

**Supabase tables:** `keyframes` (video_name, frame_index, frame_path URL, clip_id), `descriptions` (video_name, frame_index, description, clip_id)

**Export pipeline order:** download clips → trim → apply filter → fade → normalize resolution → concatenate (MoviePy/libx264) → FFmpeg post-pass (aspect ratio, brightness/contrast `eq`, speed `setpts`/`atempo`, captions `drawtext`) → optional music mix

### Frontend (`frontend/src/`)

| File | Role |
|------|------|
| `pages/index.tsx` | Main app UI; all primary state lives here (videos, timelineClips, trimMap, aiScenes, all effect params) |
| `components/VideoUpload.tsx` | Drag-drop upload to Supabase Storage |
| `components/Timeline.tsx` | Ordered clip list; supports delete, click-to-preview |
| `components/TrimControls.tsx` | Canvas-based frame thumbnail strip with draggable trim handles |
| `components/VideoPreview.tsx` | Video player with CSS-based effects preview + draggable caption overlay |
| `components/FadeControls.tsx` | Per-clip fade-in/out sliders |
| `components/AIToolsPanel.tsx` | Filter, aspect ratio, brightness, contrast, music, speed, caption controls |
| `lib/api.ts` | Typed wrappers for all backend API calls |
| `lib/supabase.ts` | Supabase JS client init |

**State management:** Zustand available but most state is local to `index.tsx`. `@tanstack/react-query` available but API calls are mostly direct in handlers.

**CORS:** Backend allows only `http://localhost:5173`.

## Key Technical Details

- `hanna_rep/` is a git submodule containing the semantic search system
- Frame extraction skips every 10 frames by default; scene changes detected via histogram diff
- Similarity match modes: `"any"` (OR), `"all_in_one"` (strict AND), `"all_distributed"` (clauses spread across frames — default)
- MoviePy writes an intermediate file; FFmpeg does a second pass for effects that MoviePy can't handle (speed, captions, eq filter)
- Speed changes use FFmpeg `setpts` + `atempo` for pitch-corrected audio
- Client-side frame thumbnails in TrimControls use `<canvas>` to extract frames from the video element
- `@dnd-kit` is installed for drag-to-reorder on timeline but not yet fully wired up
