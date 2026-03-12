import { useState } from "react";
import { useEffect } from "react";

import {
  Plus,
  Download,
  Layers,
  Zap,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { VideoUploader, type VideoFile } from "@/components/VideoUpload";
import { TrimControls } from "@/components/TrimControls";
import { Timeline, type TimelineClip } from "@/components/Timeline";
import { VideoPreview } from "@/components/VideoPreview";
import { useToast } from "@/hooks/use-toast";
import { sendVideosToBackend } from "../lib/api";
import { FadeControls } from "@/components/FadeControls";

type AIScene = {
  id: string;
  label: string;
  start: number;
  end: number;
  videoUrl?: string;
  videoName?: string;
};

export default function Index() {
  const [mergeProgress, setMergeProgress] = useState(0);
  const [isAssembling, setIsAssembling] = useState(false);
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState("loading...");

  const [assembledVideoUrl, setAssembledVideoUrl] = useState<string | null>(null);
  const [exportedVideoUrl, setExportedVideoUrl] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  /** AI-related state */
  const [prompt, setPrompt] = useState("");
  const [aiStatus, setAiStatus] =
    useState<"idle" | "analyzing" | "matching" | "done">("idle");
  const [aiScenes, setAiScenes] = useState<AIScene[]>([]);

  // ── FIX: Export modal state ───────────────────────────────────────────────
  const [showExportModal, setShowExportModal] = useState(false);

  const { toast } = useToast();

  const selectedVideo = videos.find(v => v.id === selectedVideoId);
  const selectedClip = timelineClips.find(c => c.id === selectedClipId);

  // Per-video trim values — keyed by video id
  const [trimMap, setTrimMap] = useState<Record<string, { start: number; end: number }>>({});
  // Derived trim for currently selected video — now safe to use selectedVideo
  const trimValues = selectedVideoId
    ? (trimMap[selectedVideoId] ?? { start: 0, end: selectedVideo?.duration || 10 })
    : { start: 0, end: 10 };
  const setTrimValues = (val: { start: number; end: number }) => {
    if (selectedVideoId) setTrimMap(prev => ({ ...prev, [selectedVideoId]: val }));
  };

  const MUSIC_OPTIONS = ["None", "Lo-fi", "Cinematic", "Upbeat"];
  const FILTERS = ["None", "Warm", "Cinematic", "Black & White"];
  const [music, setMusic] = useState("None");
  const [musicVolume, setMusicVolume] = useState(0.4);
  const [muteOriginal, setMuteOriginal] = useState(false);
  const [filter, setFilter] = useState("None");
  const [brightness, setBrightness] = useState(100); // percent, 100 = normal
  const [contrast, setContrast] = useState(100);     // percent, 100 = normal
  const [aspectRatio, setAspectRatio] = useState<"16:9" | "9:16" | "1:1" | "original">("original");
  const [cropOffset, setCropOffset] = useState(50); // 0-100, 50 = center
  const [speed, setSpeed] = useState(1);
  const [overlayText, setOverlayText] = useState("");
  // Caption position as x/y percent (0-100) within the video area
  const [captionX, setCaptionX] = useState(50); // percent from left
  const [captionY, setCaptionY] = useState(85); // percent from top (85 = near bottom)

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(res => res.json())
      .then(data => setBackendStatus(data.status))
      .catch(() => setBackendStatus("offline"));
  }, []);

  /* ---------------- AI PIPELINE ---------------- */
  const handleRunAI = async () => {
    if (!prompt || videos.length === 0) {
      toast({
        title: "Missing input",
        description: "Upload a video and enter a prompt.",
        variant: "destructive",
      });
      return;
    }

    setAiStatus("analyzing");
    setAiScenes([]);

    // Send ALL uploaded videos so backend searches across all of them
    const requestBody = {
      prompt,
      videos: videos.map(v => ({
        id: v.id,
        name: v.name,
        url: v.url,
        duration: v.duration,
      })),
    };

    console.log("🔍 Sending to backend:", requestBody);

    try {
      const res = await fetch("http://localhost:8000/videos/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        const errorData = await res.json();
        console.error("❌ Error response:", errorData);
        throw new Error(JSON.stringify(errorData));
      }

      const data = await res.json();
      console.log("✅ Backend AI response:", data);

      if (data.selected_clips && data.selected_clips.length > 0) {
        const FPS = 30;

        // ✅ Group frames by clip_id to create ONE clip per scene
        const clipGroups = new Map<number, any[]>();
        data.selected_clips.forEach((clip: any) => {
          const clipId = clip.clip_id;
          if (!clipGroups.has(clipId)) {
            clipGroups.set(clipId, []);
          }
          clipGroups.get(clipId)!.push(clip);
        });

        console.log(`📊 Found ${clipGroups.size} unique scene(s) from ${data.selected_clips.length} frames`);

        const scenes: AIScene[] = [];

        // ✅ Create ONE SINGLE CLIP per clip_id (from first frame to last frame)
        clipGroups.forEach((frames, clipId) => {
          // Sort frames by index
          frames.sort((a: any, b: any) => a.frame_index - b.frame_index);
          
          const firstFrame = frames[0];
          const lastFrame = frames[frames.length - 1];
          
          const sourceVideo = videos.find(v => v.name === firstFrame.video_name) || videos[0];
          const VIDEO_DURATION = firstFrame.video_duration || sourceVideo?.duration || 11;
          
          // ✅ ONE CLIP: from first frame to last frame (spanning all frames in between)
          const startTime = Math.max(0, firstFrame.frame_index / FPS);
          const endTime = Math.min(VIDEO_DURATION, lastFrame.frame_index / FPS);
          
          scenes.push({
            id: crypto.randomUUID(),
            label: `${sourceVideo?.name ?? "clip"} — Scene ${clipId} (${frames.length} frames, ${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`,
            start: startTime,
            end: endTime,
            videoUrl: firstFrame.video_url || sourceVideo?.url || "",
            videoName: firstFrame.video_name || sourceVideo?.name || "Unknown",
          });
          
          console.log(`✅ Created clip for scene ${clipId}: ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s (${frames.length} frames)`);
        });

        setAiScenes(scenes);

        const newClips: TimelineClip[] = scenes.map(scene => ({
          id: crypto.randomUUID(),
          name: scene.label,
          duration: scene.end - scene.start,
          trimStart: scene.start,
          trimEnd: scene.end,
          videoUrl: scene.videoUrl || "",
          prompt: prompt,
          createdAt: Date.now(),
        }));

        setTimelineClips(prev => [...prev, ...newClips]);
        setAiStatus("done");

        // Count how many unique videos contributed clips
        const uniqueVideos = new Set(data.selected_clips.map((c: any) => c.video_name)).size;
        toast({
          title: "🎬 Clips Generated!",
          description: `${scenes.length} scene(s) found with ${data.selected_clips.length} total frames.`,
          className: "bg-green-600 text-white border-none shadow-xl",
          duration: 4000,
        });
      } else {
        setAiStatus("idle");
        toast({
          title: "No matches found",
          description: "Try a different prompt.",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error(err);
      setAiStatus("idle");
      toast({
        title: "Backend error",
        description: "Failed to process video.",
        variant: "destructive",
      });
    }
  };

  /* ---------------- TIMELINE ---------------- */
  const handleAddToTimeline = () => {
    if (!selectedVideo) return;

    const newClip: TimelineClip = {
      id: crypto.randomUUID(),
      name: selectedVideo.name,
      duration: selectedVideo.duration,
      trimStart: trimValues.start,
      trimEnd: trimValues.end,
      videoUrl: selectedVideo.url,
    };

    setTimelineClips([...timelineClips, newClip]);
  };

  const handleAddAIScene = (scene: AIScene) => {
    // Use the scene's own videoUrl (from multi-video AI search)
    // Fall back to selectedVideo if not set (manual add)
    const videoUrl = scene.videoUrl || selectedVideo?.url;
    if (!videoUrl) return;

    const newClip: TimelineClip = {
      id: crypto.randomUUID(),
      name: scene.label,
      duration: scene.end - scene.start,
      trimStart: scene.start,
      trimEnd: scene.end,
      videoUrl,
    };

    setTimelineClips(prev => [...prev, newClip]);
  };

  const handleAssembleVideo = async () => {
    if (timelineClips.length === 0) return;

    try {
      setIsAssembling(true);
      console.log("🚀 Sending clips to backend:", timelineClips);

      const res = await fetch("http://localhost:8000/videos/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(timelineClips),
      });

      if (!res.ok) throw new Error("Merge failed");

      const data = await res.json();
      console.log("✅ Merge response:", data);

      setAssembledVideoUrl(
        `http://localhost:8000/outputs/${data.output_file}?t=${Date.now()}`
      );

      toast({
        title: "🎬 Video Assembled!",
        description: "Merged video ready for preview.",
      });
    } catch (err) {
      console.error(err);
      toast({
        title: "Merge failed",
        description: "Something went wrong.",
        variant: "destructive",
      });
    } finally {
      setIsAssembling(false);
    }
  };

  // ── Export: calls /videos/export to bake in all effects, then shows modal ──
  const handleExport = async () => {
    if (timelineClips.length === 0) return;
    setIsExporting(true);
    setShowExportModal(true);
    setExportedVideoUrl(null);

    try {
      const res = await fetch("http://localhost:8000/videos/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clips: timelineClips,
          filter: filter,
          overlayText: overlayText,
          captionX: captionX,
          captionY: captionY,
          music: music,
          musicVolume: musicVolume,
          muteOriginal: muteOriginal,
          brightness: brightness,
          contrast: contrast,
          aspectRatio: aspectRatio,
          cropOffset: cropOffset,
          playbackRate: speed,
        }),
      });

      if (!res.ok) throw new Error("Export failed");
      const data = await res.json();
      const url = `http://localhost:8000/outputs/${data.output_file}?t=${Date.now()}`;
      setExportedVideoUrl(url);

      toast({
        title: "✅ Export Ready!",
        description: "Your video with all effects is ready to download.",
        className: "bg-green-600 text-white border-none shadow-xl",
        duration: 4000,
      });
    } catch (err) {
      console.error(err);
      toast({
        title: "Export failed",
        description: "Something went wrong during export.",
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  const handleDirectDownload = async () => {
    const url = exportedVideoUrl || assembledVideoUrl;
    if (!url) return;
    try {
      // Must fetch as blob — direct <a download> won't work cross-origin
      // cache: 'no-store' forces browser to bypass cache and get fresh file
      const res = await fetch(url, { cache: "no-store" });
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = "zync-export.mp4";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch (err) {
      console.error("Download failed:", err);
      // Fallback: open in new tab so user can save manually
      window.open(url, "_blank");
    }
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="min-h-screen flex flex-col">
      {/* HEADER */}
      <header className="h-16 border-b flex justify-between items-center px-6">
        <div className="flex items-center gap-3">
          <Zap />
          <div>
            <h1 className="font-bold">ZYNC</h1>
            <p className="text-xs">AI Video Editor</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleAssembleVideo}
            disabled={timelineClips.length === 0 || isAssembling}
            className="bg-purple-600 hover:bg-purple-700 text-white font-bold px-6 py-3 text-base shadow-xl shadow-purple-400/40 transition-all active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isAssembling ? (
              <>
                🔄 Assembling...
                <span className="animate-spin inline-block">⏳</span>
              </>
            ) : (
              <>
                <Layers className="w-5 h-5 mr-2" />
                Assemble Clips
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={handleExport}
            disabled={timelineClips.length === 0 || isExporting}
          >
            <Download className="w-4 h-4 mr-1" />
            {isExporting ? "Exporting..." : "Export"}
          </Button>
        </div>
      </header>

      <div className="flex flex-1">
        {/* LEFT PANEL */}
        <aside className="w-80 border-r p-4 space-y-6">
          <VideoUploader
            videos={videos}
            onVideosChange={async (v) => {
              console.log("Index received videos:", v);
              setVideos(v);
              if (!selectedVideoId && v.length) setSelectedVideoId(v[0].id);
              // Initialise trim for any new videos
              setTrimMap(prev => {
                const next = { ...prev };
                v.forEach(vid => {
                  if (!next[vid.id]) next[vid.id] = { start: 0, end: vid.duration || 10 };
                });
                return next;
              });
              if (v.length > 0) {
                toast({
                  title: "📹 Video Uploaded!",
                  description: `${v[0].name} added successfully.`,
                  className: "bg-blue-600 text-white border-none shadow-xl",
                  duration: 3000,
                });
              }

              try {
                const res = await sendVideosToBackend(
                  v.map(video => ({
                    id: video.id,
                    name: video.name,
                    url: video.url,
                    duration: video.duration,
                  }))
                );
                console.log("Backend response:", res);
              } catch (err) {
                console.error("Failed to send videos", err);
              }
            }}
          />

          {/* Video selector — click to switch which video to trim */}
          {videos.length > 0 && (
            <div className="space-y-1.5">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                Uploaded Videos ({videos.length})
              </h3>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {videos.map(v => (
                  <button
                    key={v.id}
                    onClick={() => {
                      setSelectedVideoId(v.id);
                      // Only reset trim if this video hasn't been trimmed before
                      setTrimMap(prev => ({
                        ...prev,
                        [v.id]: prev[v.id] ?? { start: 0, end: v.duration || 10 },
                      }));
                    }}
                    className={`w-full text-left px-2 py-1.5 rounded-md border text-xs transition-colors flex items-center gap-2 ${
                      selectedVideoId === v.id
                        ? "bg-purple-600 text-white border-purple-600"
                        : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
                    }`}
                  >
                    <span className="text-base">🎬</span>
                    <span className="truncate flex-1">{v.name}</span>
                    {v.duration && (
                      <span className="opacity-60 shrink-0">
                        {Math.floor(v.duration)}s
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {selectedVideo && (
            <>
              <TrimControls
                duration={selectedVideo.duration || 10}
                trimStart={trimValues.start}
                trimEnd={trimValues.end}
                videoUrl={selectedVideo.url}
                onTrimChange={(s, e) => setTrimValues({ start: s, end: e })}
              />
              <Button onClick={handleAddToTimeline}>
                <Plus className="w-4 h-4" /> Add to Timeline
              </Button>
            </>
          )}

          {/* AI PANEL */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">AI Prompt</h3>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="e.g. classroom setting with a wooden desk"
              className="w-full border rounded p-2 text-sm"
              rows={3}
            />

            <Button
              className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 text-base shadow-xl shadow-purple-400/40 transition-all active:scale-95 disabled:opacity-50"
              onClick={handleRunAI}
              disabled={!videos.length || !prompt || aiStatus === "analyzing"}
            >
              {aiStatus === "analyzing" ? "🔄 Generating Clips..." : "Run AI Editor"}
            </Button>

            {aiStatus === "analyzing" && (
              <p className="text-xs text-muted-foreground">🔍 Analyzing video...</p>
            )}

            {aiStatus === "done" && aiScenes.length === 0 && (
              <p className="text-xs text-yellow-600">⚠️ No matching scenes found</p>
            )}

            {aiStatus === "done" && aiScenes.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-green-600 font-medium">
                  ✅ Found {aiScenes.length} matching scenes
                </p>
                <div className="max-h-64 overflow-y-auto space-y-2">
                  {aiScenes.map(scene => (
                    <div
                      key={scene.id}
                      className="border rounded p-2 flex justify-between items-center text-sm hover:bg-gray-50"
                    >
                      <span className="text-xs">{scene.label}</span>
                      <Button
                        size="sm"
                        onClick={() => handleAddAIScene(scene)}
                        className="text-xs"
                      >
                        Add
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* CENTER */}
        <main className="flex-1">
          <Timeline
            clips={timelineClips}
            selectedClipId={selectedClipId}
            onSelectClip={setSelectedClipId}
            onRemoveClip={(id) =>
              setTimelineClips(prev => prev.filter(c => c.id !== id))
            }
            onReorderClips={setTimelineClips}
          />
        </main>

        {/* RIGHT */}
        <aside className="w-96 border-l flex flex-col p-4 gap-4">
          {/* Source Preview — key forces remount when selected video changes */}
          <VideoPreview
            key={selectedVideoId ?? "no-source"}
            title="Source Preview"
            videoUrl={selectedVideo?.url || null}
            enableFades={false}
          />

          {/* Output Preview — plays the backend-merged MP4 with fade overlays */}
          {/* mergedClipBoundaries tells VideoPreview where each clip starts/ends */}
          {/* within the merged video so it can apply per-clip fades correctly   */}
          <VideoPreview
            key={assembledVideoUrl ?? "no-output"}
            title="Output Preview"
            videoUrl={assembledVideoUrl}
            playbackRate={speed}
            filter={filter}
            overlayText={overlayText}
            captionX={captionX}
            captionY={captionY}
            onCaptionMove={(x, y) => { setCaptionX(x); setCaptionY(y); }}
            music={music}
            musicStart={0}
            enableFades={true}
            muteOriginal={muteOriginal}
            brightness={brightness}
            contrast={contrast}
            aspectRatio={aspectRatio}
            cropOffset={cropOffset}
            mergedClipBoundaries={timelineClips.map((c, i) => {
              // Calculate where this clip starts in the merged video timeline
              const globalStart = timelineClips
                .slice(0, i)
                .reduce((sum, prev) => sum + (prev.trimEnd - prev.trimStart), 0);
              const globalEnd = globalStart + (c.trimEnd - c.trimStart);
              return { globalStart, globalEnd, fadeIn: c.fadeIn ?? 0, fadeOut: c.fadeOut ?? 0 };
            })}
          />

          {/* POST PRODUCTION */}
          <div className="border-t pt-4 space-y-3 overflow-y-auto">
            <h3 className="text-sm font-semibold">Post-Production</h3>

            {/* FADES */}
            <FadeControls clips={timelineClips} onClipsChange={setTimelineClips} />

            {/* MUSIC + VOLUME + MUTE — one compact row */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Background Music</label>
              <div className="flex gap-2 items-center">
                <select
                  value={music}
                  onChange={e => setMusic(e.target.value)}
                  className="flex-1 border rounded p-1.5 text-xs"
                >
                  {MUSIC_OPTIONS.map(m => <option key={m}>{m}</option>)}
                </select>
                {/* Mute original audio toggle */}
                <button
                  onClick={() => setMuteOriginal(m => !m)}
                  title={muteOriginal ? "Unmute original audio" : "Mute original audio"}
                  className={`px-2 py-1.5 rounded border text-xs font-medium transition-colors ${
                    muteOriginal
                      ? "bg-red-100 border-red-300 text-red-700"
                      : "border-border text-muted-foreground hover:border-primary/50"
                  }`}
                >
                  {muteOriginal ? "🔇" : "🔊"}
                </button>
              </div>
              {/* Music volume — only show when music selected */}
              {music !== "None" && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-12">Vol</span>
                  <input type="range" min="0" max="1" step="0.05"
                    value={musicVolume}
                    onChange={e => setMusicVolume(Number(e.target.value))}
                    className="flex-1 h-1.5"
                  />
                  <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">
                    {Math.round(musicVolume * 100)}%
                  </span>
                </div>
              )}
            </div>

            {/* FILTER + BRIGHTNESS/CONTRAST */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Filter & Colour</label>
              <select
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="w-full border rounded p-1.5 text-xs"
              >
                {FILTERS.map(f => <option key={f}>{f}</option>)}
              </select>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground w-12">Bright</span>
                <input type="range" min="50" max="150" step="1"
                  value={brightness}
                  onChange={e => setBrightness(Number(e.target.value))}
                  className="flex-1 h-1.5"
                />
                <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">
                  {brightness}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground w-12">Contrast</span>
                <input type="range" min="50" max="150" step="1"
                  value={contrast}
                  onChange={e => setContrast(Number(e.target.value))}
                  className="flex-1 h-1.5"
                />
                <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">
                  {contrast}%
                </span>
              </div>
            </div>

            {/* ASPECT RATIO — icon button row */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Aspect Ratio</label>
              <div className="flex gap-1.5">
                {(["original", "16:9", "9:16", "1:1"] as const).map(r => (
                  <button
                    key={r}
                    onClick={() => setAspectRatio(r)}
                    className={`flex-1 py-1.5 rounded border text-[10px] font-medium transition-colors ${
                      aspectRatio === r
                        ? "bg-primary text-primary-foreground border-primary"
                        : "border-border text-muted-foreground hover:border-primary/50"
                    }`}
                  >
                    {r === "original" ? "Auto" : r}
                  </button>
                ))}
              </div>

              {/* Crop position slider — shown when any crop ratio active */}
              {aspectRatio !== "original" && (
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-[10px] text-muted-foreground w-14 shrink-0">
                    {aspectRatio === "9:16" ? "← H-pos →" : "↑ V-pos ↓"}
                  </span>
                  <input
                    type="range" min="0" max="100" step="1"
                    value={cropOffset}
                    onChange={e => setCropOffset(Number(e.target.value))}
                    className="flex-1 h-1.5"
                  />
                  <button
                    onClick={() => setCropOffset(50)}
                    className="text-[10px] text-muted-foreground hover:text-primary px-1"
                    title="Reset to center"
                  >↺</button>
                </div>
              )}
            </div>

            {/* SPEED */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-xs font-medium">Speed</label>
                <span className="text-xs font-mono text-muted-foreground">{speed}x</span>
              </div>
              <input type="range" min="0.5" max="2" step="0.25"
                value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
                className="w-full h-1.5"
              />
            </div>

            {/* CAPTION + POSITION */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Caption</label>
              <input
                type="text"
                value={overlayText}
                onChange={e => setOverlayText(e.target.value)}
                placeholder="Enter caption text…"
                className="w-full border rounded p-1.5 text-xs"
              />
              {overlayText && (
                <p className="text-[10px] text-muted-foreground">
                  ✋ Drag the caption in the Output Preview to reposition it
                </p>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Export Modal — shows baked export with all effects */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-background rounded-xl shadow-2xl w-full max-w-xl flex flex-col gap-4 p-6 relative my-auto">
            <button
              onClick={() => setShowExportModal(false)}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
              title="Close"
            >
              <X className="w-5 h-5" />
            </button>

            <h2 className="text-lg font-bold">Export</h2>

            {/* Loading state */}
            {isExporting && (
              <div className="flex flex-col items-center justify-center py-12 gap-3">
                <span className="animate-spin text-4xl">⏳</span>
                <p className="text-sm text-muted-foreground">Baking effects into video...</p>
                <p className="text-xs text-muted-foreground/60">
                  Filter, music, overlays and fades are being rendered
                </p>
              </div>
            )}

            {/* Ready state */}
            {!isExporting && exportedVideoUrl && (
              <>
                {/* Relative wrapper so caption can overlay the video */}
                <div className="relative w-full rounded-lg overflow-hidden bg-black" style={{ maxHeight: "40vh" }}>
                  <video
                    key={exportedVideoUrl}
                    src={exportedVideoUrl}
                    controls
                    className="w-full"
                    style={{
                      maxHeight: "40vh",
                      objectFit: "contain",
                      // Apply CSS effects as preview — they are also baked in the file
                      filter: [
                        FILTER_STYLES_EXPORT[filter] ?? "",
                        brightness !== 100 || contrast !== 100
                          ? `brightness(${brightness}%) contrast(${contrast}%)`
                          : ""
                      ].filter(Boolean).join(" ") || "none",
                    }}
                  />
                  {/* Caption already baked into video file — no overlay needed */}
                </div>

                {/* Effects summary — all baked into the downloaded file */}
                <div className="flex flex-wrap gap-2 text-xs">
                  {filter !== "None" && (
                    <span className="bg-muted rounded px-2 py-1">🎨 Filter: {filter}</span>
                  )}
                  {music !== "None" && (
                    <span className="bg-muted rounded px-2 py-1">🎵 Music: {music}</span>
                  )}
                  {overlayText && (
                    <span className="bg-muted rounded px-2 py-1">💬 Caption: "{overlayText}"</span>
                  )}
                  {speed !== 1 && (
                    <span className="bg-muted rounded px-2 py-1">⚡ Speed: {speed}x</span>
                  )}
                  {timelineClips.some(c => (c.fadeIn ?? 0) > 0 || (c.fadeOut ?? 0) > 0) && (
                    <span className="bg-purple-100 text-purple-700 rounded px-2 py-1">
                      🌅 Fades: {timelineClips.filter(c => (c.fadeIn ?? 0) > 0 || (c.fadeOut ?? 0) > 0).length} clips
                    </span>
                  )}
                </div>

                <div className="flex gap-3 justify-end">
                  <Button variant="outline" onClick={() => setShowExportModal(false)}>
                    Close
                  </Button>
                  <Button onClick={handleDirectDownload} className="bg-green-600 hover:bg-green-700 text-white">
                    <Download className="w-4 h-4 mr-2" />
                    Download Video
                  </Button>
                </div>
              </>
            )}

            {/* Error state */}
            {!isExporting && !exportedVideoUrl && (
              <div className="text-center py-8">
                <p className="text-sm text-destructive">Export failed. Please try again.</p>
                <Button variant="outline" className="mt-4" onClick={() => setShowExportModal(false)}>
                  Close
                </Button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Needed for the export modal video preview filter
const FILTER_STYLES_EXPORT: Record<string, string> = {
  None: "none",
  Warm: "brightness(1.1) saturate(1.2) sepia(0.2)",
  Cinematic: "contrast(1.2) saturate(1.3) brightness(0.9)",
  "Black & White": "grayscale(100%) contrast(1.2)",
};