import { useState } from "react";
import { useEffect } from "react";

import {
  Plus,
  Download,
  Layers,
  Zap,
  X,
  Sparkles,
  Film,
  Wifi,
  WifiOff,
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

  const [showExportModal, setShowExportModal] = useState(false);

  const { toast } = useToast();

  const selectedVideo = videos.find(v => v.id === selectedVideoId);
  const [trimMap, setTrimMap] = useState<Record<string, { start: number; end: number }>>({});
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
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [aspectRatio, setAspectRatio] = useState<"16:9" | "9:16" | "1:1" | "original">("original");
  const [cropOffset, setCropOffset] = useState(50);
  const [speed, setSpeed] = useState(1);
  const [overlayText, setOverlayText] = useState("");
  const [captionX, setCaptionX] = useState(50);
  const [captionY, setCaptionY] = useState(85);

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

    const requestBody = {
      prompt,
      videos: videos.map(v => ({
        id: v.id,
        name: v.name,
        url: v.url,
        duration: v.duration,
      })),
    };

    console.log("Sending to backend:", requestBody);

    try {
      const res = await fetch("http://localhost:8000/videos/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        const errorData = await res.json();
        console.error("Error response:", errorData);
        throw new Error(JSON.stringify(errorData));
      }

      const data = await res.json();
      console.log("Backend AI response:", data);

      if (data.selected_clips && data.selected_clips.length > 0) {
        const FPS = 30;

        const clipGroups = new Map<number, any[]>();
        data.selected_clips.forEach((clip: any) => {
          const clipId = clip.clip_id;
          if (!clipGroups.has(clipId)) {
            clipGroups.set(clipId, []);
          }
          clipGroups.get(clipId)!.push(clip);
        });

        console.log(`Found ${clipGroups.size} unique scene(s) from ${data.selected_clips.length} frames`);

        const scenes: AIScene[] = [];

        clipGroups.forEach((frames, clipId) => {
          frames.sort((a: any, b: any) => a.frame_index - b.frame_index);

          const firstFrame = frames[0];
          const lastFrame = frames[frames.length - 1];

          const sourceVideo = videos.find(v => v.name === firstFrame.video_name) || videos[0];
          const VIDEO_DURATION = firstFrame.video_duration || sourceVideo?.duration || 11;

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

          console.log(`Created clip for scene ${clipId}: ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s (${frames.length} frames)`);
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

        toast({
          title: "Clips Generated",
          description: `${scenes.length} scene(s) found with ${data.selected_clips.length} total frames.`,
          className: "bg-emerald-950 border border-emerald-800 text-emerald-200 shadow-xl",
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
      console.log("Sending clips to backend:", timelineClips);

      const res = await fetch("http://localhost:8000/videos/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(timelineClips),
      });

      if (!res.ok) throw new Error("Merge failed");

      const data = await res.json();
      console.log("Merge response:", data);

      setAssembledVideoUrl(
        `http://localhost:8000/outputs/${data.output_file}?t=${Date.now()}`
      );

      toast({
        title: "Video Assembled",
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
        title: "Export Ready",
        description: "Your video with all effects is ready to download.",
        className: "bg-emerald-950 border border-emerald-800 text-emerald-200 shadow-xl",
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
      window.open(url, "_blank");
    }
  };

  /* ---------------- UI ---------------- */
  return (
    <div className="min-h-screen flex flex-col bg-zinc-950 text-zinc-100">

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <header className="h-14 flex-shrink-0 border-b border-zinc-800/60 bg-zinc-950/95 backdrop-blur-sm flex items-center justify-between px-5 z-40">

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center shadow-lg shadow-violet-500/30 flex-shrink-0">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div className="leading-none">
            <h1 className="text-sm font-bold tracking-[0.2em] bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
              ZYNC
            </h1>
            <p className="text-[10px] text-zinc-500 mt-0.5">AI Video Editor</p>
          </div>

          {/* Backend status pill */}
          <div className={`ml-1 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium border transition-colors ${
            backendStatus === "ok"
              ? "bg-emerald-950/80 border-emerald-800/60 text-emerald-400"
              : backendStatus === "loading..."
              ? "bg-zinc-800/60 border-zinc-700/60 text-zinc-400"
              : "bg-red-950/80 border-red-800/60 text-red-400"
          }`}>
            {backendStatus === "ok"
              ? <Wifi className="w-2.5 h-2.5" />
              : <WifiOff className="w-2.5 h-2.5" />
            }
            {backendStatus === "ok" ? "Connected" : backendStatus === "loading..." ? "Connecting…" : "Offline"}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Button
            onClick={handleAssembleVideo}
            disabled={timelineClips.length === 0 || isAssembling}
            className="h-9 px-4 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700/80 text-zinc-200 text-sm font-medium rounded-lg transition-all disabled:opacity-40"
          >
            <Layers className="w-4 h-4 mr-1.5" />
            {isAssembling ? "Assembling…" : "Assemble"}
          </Button>

          <Button
            onClick={handleExport}
            disabled={timelineClips.length === 0 || isExporting}
            className="h-9 px-4 bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold rounded-lg shadow-lg shadow-violet-500/25 transition-all active:scale-95 disabled:opacity-40"
          >
            <Download className="w-4 h-4 mr-1.5" />
            {isExporting ? "Exporting…" : "Export"}
          </Button>
        </div>
      </header>

      {/* ── BODY ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── LEFT PANEL ─────────────────────────────────────────────────── */}
        <aside className="w-80 flex-shrink-0 border-r border-zinc-800/60 bg-zinc-900/50 flex flex-col overflow-y-auto">

          {/* Upload section */}
          <div className="px-4 pt-4 pb-3">
            <SectionLabel>Upload</SectionLabel>
            <VideoUploader
              videos={videos}
              onVideosChange={async (v) => {
                console.log("Index received videos:", v);
                setVideos(v);
                if (!selectedVideoId && v.length) setSelectedVideoId(v[0].id);
                setTrimMap(prev => {
                  const next = { ...prev };
                  v.forEach(vid => {
                    if (!next[vid.id]) next[vid.id] = { start: 0, end: vid.duration || 10 };
                  });
                  return next;
                });
                if (v.length > 0) {
                  toast({
                    title: "Video Uploaded",
                    description: `${v[v.length - 1].name} added successfully.`,
                    className: "bg-zinc-900 border border-zinc-700 text-zinc-100 shadow-xl",
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
          </div>

          {/* Video selector */}
          {videos.length > 0 && (
            <div className="px-4 pb-3">
              <SectionLabel>Library ({videos.length})</SectionLabel>
              <div className="space-y-1 max-h-44 overflow-y-auto -mx-1 px-1">
                {videos.map(v => (
                  <button
                    key={v.id}
                    onClick={() => {
                      setSelectedVideoId(v.id);
                      setTrimMap(prev => ({
                        ...prev,
                        [v.id]: prev[v.id] ?? { start: 0, end: v.duration || 10 },
                      }));
                    }}
                    className={`w-full text-left px-2.5 py-2 rounded-lg border text-xs transition-all flex items-center gap-2 ${
                      selectedVideoId === v.id
                        ? "bg-violet-600/15 border-violet-500/50 text-violet-300"
                        : "border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 hover:bg-zinc-800/50"
                    }`}
                  >
                    <Film className={`w-3.5 h-3.5 flex-shrink-0 ${selectedVideoId === v.id ? "text-violet-400" : "text-zinc-500"}`} />
                    <span className="truncate flex-1 font-medium">{v.name}</span>
                    {v.duration > 0 && (
                      <span className="opacity-50 shrink-0 font-mono">
                        {Math.floor(v.duration)}s
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Trim + Add to Timeline */}
          {selectedVideo && (
            <div className="px-4 pb-3 space-y-3">
              <SectionLabel>Trim</SectionLabel>
              <TrimControls
                duration={selectedVideo.duration || 10}
                trimStart={trimValues.start}
                trimEnd={trimValues.end}
                videoUrl={selectedVideo.url}
                onTrimChange={(s, e) => setTrimValues({ start: s, end: e })}
              />
              <button
                onClick={handleAddToTimeline}
                className="w-full flex items-center justify-center gap-2 h-9 rounded-lg border border-dashed border-zinc-700 text-zinc-400 text-xs font-medium hover:border-violet-500/60 hover:text-violet-400 hover:bg-violet-500/5 transition-all"
              >
                <Plus className="w-3.5 h-3.5" />
                Add to Timeline
              </button>
            </div>
          )}

          {/* Divider */}
          <div className="mx-4 border-t border-zinc-800/60" />

          {/* AI PANEL */}
          <div className="px-4 py-3 space-y-3 flex-1">
            <SectionLabel icon={<Sparkles className="w-3 h-3 text-violet-400" />}>
              AI Prompt
            </SectionLabel>

            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="e.g. classroom setting with a wooden desk"
              rows={3}
              className="w-full bg-zinc-800/60 border border-zinc-700/60 rounded-lg p-3 text-sm text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none focus:border-violet-500/50 focus:bg-zinc-800 transition-all"
            />

            <button
              onClick={handleRunAI}
              disabled={!videos.length || !prompt || aiStatus === "analyzing"}
              className="w-full h-10 rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-semibold shadow-lg shadow-violet-500/20 transition-all active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none flex items-center justify-center gap-2"
            >
              {aiStatus === "analyzing" ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Generating Clips…
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Run AI Editor
                </>
              )}
            </button>

            {aiStatus === "analyzing" && (
              <p className="text-xs text-zinc-500 text-center">Analyzing video frames…</p>
            )}

            {aiStatus === "done" && aiScenes.length === 0 && (
              <p className="text-xs text-amber-500/80 text-center">No matching scenes found</p>
            )}

            {aiStatus === "done" && aiScenes.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-emerald-400 font-medium">
                  {aiScenes.length} matching scene{aiScenes.length > 1 ? "s" : ""} found
                </p>
                <div className="max-h-60 overflow-y-auto space-y-1.5">
                  {aiScenes.map(scene => (
                    <div
                      key={scene.id}
                      className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-zinc-800/60 border border-zinc-700/50 hover:border-zinc-600/80 transition-colors"
                    >
                      <span className="text-xs text-zinc-300 truncate">{scene.label}</span>
                      <button
                        onClick={() => handleAddAIScene(scene)}
                        className="flex-shrink-0 text-[11px] font-semibold text-violet-400 hover:text-violet-300 border border-violet-500/40 hover:border-violet-400/60 rounded-md px-2 py-0.5 transition-colors hover:bg-violet-500/10"
                      >
                        Add
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* ── CENTER ─────────────────────────────────────────────────────── */}
        <main className="flex-1 bg-zinc-950 min-w-0">
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

        {/* ── RIGHT PANEL ────────────────────────────────────────────────── */}
        <aside className="w-96 flex-shrink-0 border-l border-zinc-800/60 bg-zinc-900/50 flex flex-col overflow-y-auto">

          {/* Source Preview */}
          <div className="p-3 border-b border-zinc-800/60">
            <VideoPreview
              key={selectedVideoId ?? "no-source"}
              title="Source Preview"
              videoUrl={selectedVideo?.url || null}
              enableFades={false}
            />
          </div>

          {/* Output Preview */}
          <div className="p-3 border-b border-zinc-800/60">
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
                const globalStart = timelineClips
                  .slice(0, i)
                  .reduce((sum, prev) => sum + (prev.trimEnd - prev.trimStart), 0);
                const globalEnd = globalStart + (c.trimEnd - c.trimStart);
                return { globalStart, globalEnd, fadeIn: c.fadeIn ?? 0, fadeOut: c.fadeOut ?? 0 };
              })}
            />
          </div>

          {/* Post-Production Controls */}
          <div className="flex-1 overflow-y-auto p-4 space-y-5">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">Post-Production</h3>

            {/* Fades */}
            <FadeControls clips={timelineClips} onClipsChange={setTimelineClips} />

            {/* Music */}
            <div className="space-y-2">
              <ControlLabel>Background Music</ControlLabel>
              <div className="flex gap-2 items-center">
                <select
                  value={music}
                  onChange={e => setMusic(e.target.value)}
                  className="flex-1 bg-zinc-800/60 border border-zinc-700/60 rounded-lg px-2.5 py-2 text-xs text-zinc-200 focus:outline-none focus:border-violet-500/50"
                >
                  {MUSIC_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
                <button
                  onClick={() => setMuteOriginal(m => !m)}
                  title={muteOriginal ? "Unmute original audio" : "Mute original audio"}
                  className={`px-2.5 py-2 rounded-lg border text-xs font-medium transition-colors ${
                    muteOriginal
                      ? "bg-red-950/60 border-red-800/60 text-red-400"
                      : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  {muteOriginal ? "🔇" : "🔊"}
                </button>
              </div>
              {music !== "None" && (
                <SliderRow
                  label="Vol"
                  value={musicVolume}
                  min={0} max={1} step={0.05}
                  onChange={setMusicVolume}
                  display={`${Math.round(musicVolume * 100)}%`}
                />
              )}
            </div>

            {/* Filter & Colour */}
            <div className="space-y-2">
              <ControlLabel>Filter & Colour</ControlLabel>
              <select
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="w-full bg-zinc-800/60 border border-zinc-700/60 rounded-lg px-2.5 py-2 text-xs text-zinc-200 focus:outline-none focus:border-violet-500/50"
              >
                {FILTERS.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
              <SliderRow label="Bright" value={brightness} min={50} max={150} step={1} onChange={setBrightness} display={`${brightness}%`} />
              <SliderRow label="Contrast" value={contrast} min={50} max={150} step={1} onChange={setContrast} display={`${contrast}%`} />
            </div>

            {/* Aspect Ratio */}
            <div className="space-y-2">
              <ControlLabel>Aspect Ratio</ControlLabel>
              <div className="grid grid-cols-4 gap-1.5">
                {(["original", "16:9", "9:16", "1:1"] as const).map(r => (
                  <button
                    key={r}
                    onClick={() => setAspectRatio(r)}
                    className={`py-1.5 rounded-lg border text-[11px] font-medium transition-all ${
                      aspectRatio === r
                        ? "bg-violet-600/20 border-violet-500/60 text-violet-300"
                        : "border-zinc-700/60 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300"
                    }`}
                  >
                    {r === "original" ? "Auto" : r}
                  </button>
                ))}
              </div>
              {aspectRatio !== "original" && (
                <SliderRow
                  label={aspectRatio === "9:16" ? "H-pos" : "V-pos"}
                  value={cropOffset} min={0} max={100} step={1}
                  onChange={setCropOffset}
                  display={
                    <button onClick={() => setCropOffset(50)} className="text-[10px] text-zinc-500 hover:text-violet-400 transition-colors" title="Center">↺</button>
                  }
                />
              )}
            </div>

            {/* Speed */}
            <div className="space-y-2">
              <ControlLabel>Playback Speed <span className="text-zinc-500 font-mono">{speed}x</span></ControlLabel>
              <input type="range" min="0.5" max="2" step="0.25"
                value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
                className="w-full h-1.5 rounded-full"
              />
            </div>

            {/* Caption */}
            <div className="space-y-2">
              <ControlLabel>Caption</ControlLabel>
              <input
                type="text"
                value={overlayText}
                onChange={e => setOverlayText(e.target.value)}
                placeholder="Enter caption text…"
                className="w-full bg-zinc-800/60 border border-zinc-700/60 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-violet-500/50 transition-all"
              />
              {overlayText && (
                <p className="text-[11px] text-zinc-600">
                  Drag the caption in Output Preview to reposition
                </p>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* ── EXPORT MODAL ──────────────────────────────────────────────────── */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl w-full max-w-xl flex flex-col gap-5 p-6 relative">
            <button
              onClick={() => setShowExportModal(false)}
              className="absolute top-4 right-4 text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div>
              <h2 className="text-base font-semibold text-zinc-100">Export</h2>
              <p className="text-xs text-zinc-500 mt-0.5">All effects baked into final file</p>
            </div>

            {/* Loading */}
            {isExporting && (
              <div className="flex flex-col items-center justify-center py-12 gap-4">
                <div className="w-12 h-12 border-2 border-zinc-700 border-t-violet-500 rounded-full animate-spin" />
                <p className="text-sm text-zinc-400">Rendering video…</p>
                <p className="text-xs text-zinc-600">Filter, music, overlays and fades are being baked in</p>
              </div>
            )}

            {/* Ready */}
            {!isExporting && exportedVideoUrl && (
              <>
                <div className="relative w-full rounded-xl overflow-hidden bg-black border border-zinc-800" style={{ maxHeight: "40vh" }}>
                  <video
                    key={exportedVideoUrl}
                    src={exportedVideoUrl}
                    controls
                    className="w-full"
                    style={{
                      maxHeight: "40vh",
                      objectFit: "contain",
                      filter: [
                        FILTER_STYLES_EXPORT[filter] ?? "",
                        brightness !== 100 || contrast !== 100
                          ? `brightness(${brightness}%) contrast(${contrast}%)`
                          : ""
                      ].filter(Boolean).join(" ") || "none",
                    }}
                  />
                </div>

                {/* Effects tags */}
                <div className="flex flex-wrap gap-2 text-xs">
                  {filter !== "None" && (
                    <Tag>Filter: {filter}</Tag>
                  )}
                  {music !== "None" && (
                    <Tag>Music: {music}</Tag>
                  )}
                  {overlayText && (
                    <Tag>Caption: "{overlayText}"</Tag>
                  )}
                  {speed !== 1 && (
                    <Tag>Speed: {speed}x</Tag>
                  )}
                  {timelineClips.some(c => (c.fadeIn ?? 0) > 0 || (c.fadeOut ?? 0) > 0) && (
                    <Tag variant="accent">Fades: {timelineClips.filter(c => (c.fadeIn ?? 0) > 0 || (c.fadeOut ?? 0) > 0).length} clips</Tag>
                  )}
                </div>

                <div className="flex gap-3 justify-end pt-1">
                  <Button variant="outline" onClick={() => setShowExportModal(false)}
                    className="border-zinc-700 text-zinc-400 hover:bg-zinc-800">
                    Close
                  </Button>
                  <Button onClick={handleDirectDownload}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20">
                    <Download className="w-4 h-4 mr-2" />
                    Download Video
                  </Button>
                </div>
              </>
            )}

            {/* Error */}
            {!isExporting && !exportedVideoUrl && (
              <div className="text-center py-8">
                <p className="text-sm text-red-400">Export failed. Please try again.</p>
                <Button variant="outline" className="mt-4 border-zinc-700 text-zinc-400"
                  onClick={() => setShowExportModal(false)}>
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

/* ── Small reusable UI helpers ─────────────────────────────────────────────── */

function SectionLabel({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 mb-2.5">
      {icon}
      <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest">{children}</span>
      <div className="flex-1 h-px bg-zinc-800/80" />
    </div>
  );
}

function ControlLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-1.5 text-xs font-medium text-zinc-400">{children}</label>
  );
}

function SliderRow({
  label, value, min, max, step, onChange, display,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-zinc-500 w-14 flex-shrink-0">{label}</span>
      <input type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-1.5 rounded-full"
      />
      {typeof display === "string"
        ? <span className="text-[11px] font-mono text-zinc-500 w-10 text-right flex-shrink-0">{display}</span>
        : display
      }
    </div>
  );
}

function Tag({ children, variant = "default" }: { children: React.ReactNode; variant?: "default" | "accent" }) {
  return (
    <span className={`rounded-md px-2 py-1 text-[11px] font-medium ${
      variant === "accent"
        ? "bg-violet-950/60 border border-violet-800/40 text-violet-300"
        : "bg-zinc-800 border border-zinc-700/60 text-zinc-400"
    }`}>
      {children}
    </span>
  );
}

// CSS filter map for the export modal video preview
const FILTER_STYLES_EXPORT: Record<string, string> = {
  None: "none",
  Warm: "brightness(1.1) saturate(1.2) sepia(0.2)",
  Cinematic: "contrast(1.2) saturate(1.3) brightness(0.9)",
  "Black & White": "grayscale(100%) contrast(1.2)",
};
