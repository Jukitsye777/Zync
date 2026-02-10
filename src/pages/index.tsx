import { useState } from "react";

import {
  Plus,
  Download,
  Layers,
  Zap,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { VideoUploader, type VideoFile } from "@/components/VideoUpload";
import { TrimControls } from "@/components/TrimControls";
import { Timeline, type TimelineClip } from "@/components/Timeline";
import { VideoPreview } from "@/components/VideoPreview";
import { useToast } from "@/hooks/use-toast";

type AIScene = {
  id: string;
  label: string;
  start: number;
  end: number;
};

export default function Index() {
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);

  const [trimValues, setTrimValues] = useState({ start: 0, end: 10 });
  const [assembledVideoUrl, setAssembledVideoUrl] = useState<string | null>(null);

  /** AI-related state */
  const [prompt, setPrompt] = useState("");
  const [aiStatus, setAiStatus] =
    useState<"idle" | "analyzing" | "matching" | "done">("idle");
  const [aiScenes, setAiScenes] = useState<AIScene[]>([]);

  const { toast } = useToast();

  const selectedVideo = videos.find(v => v.id === selectedVideoId);
  const selectedClip = timelineClips.find(c => c.id === selectedClipId);
  // ADD near top
  const MUSIC_OPTIONS = ["None", "Lo-fi", "Cinematic", "Upbeat"];
  const FILTERS = ["None", "Warm", "Cinematic", "Black & White"];
  const [music, setMusic] = useState("None");
  const [filter, setFilter] = useState("None");
  const [speed, setSpeed] = useState(1);
  const [overlayText, setOverlayText] = useState("");



  /* ---------------- AI PIPELINE (MOCK) ---------------- */

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
    await new Promise(r => setTimeout(r, 1200));

    setAiStatus("matching");
    await new Promise(r => setTimeout(r, 1200));

    setAiScenes([
      { id: "s1", label: "Scene 1", start: 0, end: 8 },
      { id: "s2", label: "Scene 2", start: 14, end: 26 },
    ]);

    setAiStatus("done");

    toast({
      title: "AI Processing Complete",
      description: "Relevant scenes detected.",
    });
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
    if (!selectedVideo) return;

    const newClip: TimelineClip = {
      id: crypto.randomUUID(),
      name: scene.label,
      duration: scene.end - scene.start,
      trimStart: scene.start,
      trimEnd: scene.end,
      videoUrl: selectedVideo.url,
    };

    setTimelineClips(prev => [...prev, newClip]);
  };

  const handleAssembleVideo = async () => {
    if (timelineClips.length === 0) return;

    await new Promise(r => setTimeout(r, 1200));
    setAssembledVideoUrl(timelineClips[0].videoUrl);
  };

  const handleExport = () => {
    if (!assembledVideoUrl) return;
    const a = document.createElement("a");
    a.href = assembledVideoUrl;
    a.download = "zync-output.mp4";
    a.click();
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
            disabled={timelineClips.length === 0}
          >
            <Layers className="w-4 h-4" /> Assemble
          </Button>
          <Button
            variant="outline"
            onClick={handleExport}
            disabled={!assembledVideoUrl}
          >
            <Download className="w-4 h-4" /> Export
          </Button>
        </div>
      </header>

      <div className="flex flex-1">
        {/* LEFT PANEL */}
        <aside className="w-80 border-r p-4 space-y-6">
          <VideoUploader
            videos={videos}
            onVideosChange={(v) => {
              setVideos(v);
              if (!selectedVideoId && v.length) setSelectedVideoId(v[0].id);
            }}
          />

          {selectedVideo && (
            <>
              <TrimControls
                duration={selectedVideo.duration || 10}
                trimStart={trimValues.start}
                trimEnd={trimValues.end}
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
              placeholder="e.g. Extract classroom scenes"
              className="w-full border rounded p-2 text-sm"
            />

            <Button
              className="w-full"
              onClick={handleRunAI}
              disabled={!videos.length || !prompt || aiStatus !== "idle"}
            >
              {aiStatus === "idle" ? "Run AI" : "Processing..."}
            </Button>

            {aiStatus !== "idle" && (
              <p className="text-xs">
                {aiStatus === "analyzing" && "Analyzing video…"}
                {aiStatus === "matching" && "Matching prompt…"}
                {aiStatus === "done" && "AI results ready"}
              </p>
            )}

            {aiScenes.map(scene => (
              <div
                key={scene.id}
                className="border rounded p-2 flex justify-between text-sm"
              >
                <span>{scene.label}</span>
                <Button
                  size="sm"
                  onClick={() => handleAddAIScene(scene)}
                >
                  Add
                </Button>
              </div>
            ))}
          </div>
        </aside>

        {/* CENTER */}
        <main className="flex-1">
          <Timeline
            clips={timelineClips}
            selectedClipId={selectedClipId}
            onSelectClip={setSelectedClipId}
            onRemoveClip={(id) =>
              setTimelineClips(timelineClips.filter(c => c.id !== id))
            }
            onReorderClips={setTimelineClips}
          />
        </main>

        {/* RIGHT */}
        <aside className="w-96 border-l flex flex-col p-4 gap-4">

  <VideoPreview
    title="Source Preview"
    videoUrl={selectedVideo?.url || selectedClip?.videoUrl || null}
    trimStart={selectedClip?.trimStart}
    trimEnd={selectedClip?.trimEnd}
  />

  <VideoPreview
    title="Output Preview"
    videoUrl={assembledVideoUrl}
  />

  {/* POST PRODUCTION */}
  <div className="border-t pt-4 space-y-4">
    <h3 className="text-sm font-semibold">Post-Production</h3>

    {/* MUSIC */}
    <div>
      <label className="text-xs font-medium">Background Music</label>
      <select
        value={music}
        onChange={e => setMusic(e.target.value)}
        className="w-full border rounded p-2 text-sm"
      >
        {MUSIC_OPTIONS.map(m => (
          <option key={m}>{m}</option>
        ))}
      </select>
    </div>

    {/* FILTER */}
    <div>
      <label className="text-xs font-medium">Filter</label>
      <select
        value={filter}
        onChange={e => setFilter(e.target.value)}
        className="w-full border rounded p-2 text-sm"
      >
        {FILTERS.map(f => (
          <option key={f}>{f}</option>
        ))}
      </select>
    </div>

    {/* SPEED */}
    <div>
      <label className="text-xs font-medium">Playback Speed</label>
      <input
        type="range"
        min="0.5"
        max="2"
        step="0.25"
        value={speed}
        onChange={e => setSpeed(Number(e.target.value))}
        className="w-full"
      />
      <p className="text-xs">{speed}x</p>
    </div>

    {/* TEXT OVERLAY */}
    <div>
      <label className="text-xs font-medium">Text Overlay</label>
      <input
        type="text"
        value={overlayText}
        onChange={e => setOverlayText(e.target.value)}
        placeholder="Enter caption text"
        className="w-full border rounded p-2 text-sm"
      />
    </div>
  </div>
</aside>

      </div>
    </div>
  );
}
