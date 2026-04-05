import { useState } from "react";
import { useEffect } from "react";

import {
  Plus, Layers, Zap, Sparkles,
  Film, Wifi, WifiOff, Palette, Clapperboard, Wand2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { VideoUploader, type VideoFile } from "@/components/VideoUpload";
import { TrimControls } from "@/components/TrimControls";
import { Timeline, type TimelineClip } from "@/components/Timeline";
import { VideoPreview } from "@/components/VideoPreview";
import { useToast } from "@/hooks/use-toast";
import { sendVideosToBackend, runEditPrompt } from "../lib/api";

type AIScene = {
  id: string; label: string; start: number; end: number;
  videoUrl?: string; videoName?: string; thumbnailUrl?: string;
};

const MOOD_FILTERS = [
  { value: "None",          label: "None",     description: "No grade",         gradient: "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",            dot: "#555" },
  { value: "Sunset",        label: "Sunset",   description: "Golden hour",      gradient: "linear-gradient(135deg, #c94b0c 0%, #f7931e 50%, #ffcd3c 100%)", dot: "#f7931e" },
  { value: "Happy",         label: "Happy",    description: "Bright & vivid",   gradient: "linear-gradient(135deg, #2980b9 0%, #6dd5fa 50%, #f9f047 100%)", dot: "#6dd5fa" },
  { value: "Sad",           label: "Sad",      description: "Desaturated blue", gradient: "linear-gradient(135deg, #1c2b3a 0%, #3a5068 50%, #7da7c0 100%)", dot: "#4a7a9b" },
  { value: "Dramatic",      label: "Dramatic", description: "High contrast",    gradient: "linear-gradient(135deg, #0a0a0a 0%, #2d0a3e 50%, #1a0a0a 100%)", dot: "#7b2d8b" },
  { value: "Vintage",       label: "Vintage",  description: "Warm sepia",       gradient: "linear-gradient(135deg, #5c3a1e 0%, #a07040 50%, #d4b896 100%)", dot: "#a07040" },
  { value: "Night",         label: "Night",    description: "Deep cool blue",   gradient: "linear-gradient(135deg, #020b18 0%, #0d2137 50%, #1a3a5c 100%)", dot: "#1a3a5c" },
  { value: "Cinematic",     label: "Cinema",   description: "Rich film look",   gradient: "linear-gradient(135deg, #0d0800 0%, #3d1c02 50%, #7a3b10 100%)", dot: "#c4622d" },
  { value: "Black & White", label: "B&W",      description: "Classic mono",     gradient: "linear-gradient(135deg, #000 0%, #555 50%, #ccc 100%)",           dot: "#888" },
  { value: "Teal & Orange", label: "Teal/Org", description: "Hollywood grade",  gradient: "linear-gradient(135deg, #003d3d 0%, #006666 40%, #ff6a00 100%)",  dot: "#009999" },
  { value: "Fade",          label: "Faded",    description: "Lifted shadows",   gradient: "linear-gradient(135deg, #2a2a3a 0%, #5a5a7a 50%, #9a9ab0 100%)", dot: "#7a7a9a" },
  { value: "Neon",          label: "Neon",     description: "Cyberpunk glow",   gradient: "linear-gradient(135deg, #0d0221 0%, #2d0057 40%, #ff00c8 100%)",  dot: "#bf00ff" },
];

const TRANSITION_PRESETS = [
  { value: "none",   label: "Cut",       icon: "✂️",  description: "Instant switch" },
  { value: "fade",   label: "Fade",      icon: "🌫️", description: "Smooth dissolve" },
  { value: "wipe",   label: "Wipe",      icon: "➡️",  description: "Slide across" },
  { value: "zoom",   label: "Zoom",      icon: "🔍",  description: "Push in/out" },
  { value: "flash",  label: "Flash",     icon: "⚡",  description: "White flash" },
  { value: "glitch", label: "Glitch",    icon: "📺",  description: "Digital noise" },
  { value: "blur",   label: "Blur",      icon: "💫",  description: "Motion blur" },
  { value: "dip",    label: "Dip Black", icon: "⬛",  description: "Dip to black" },
];

const MUSIC_OPTIONS = ["None", "Cinematic", "Upbeat", "Sad", "Energetic"];

export const FILTER_STYLES_EXPORT: Record<string, string> = {
  None:            "none",
  Sunset:          "brightness(1.2) saturate(1.8) sepia(0.4) hue-rotate(-15deg) contrast(1.1)",
  Happy:           "brightness(1.25) saturate(1.9) contrast(1.15)",
  Sad:             "brightness(0.82) saturate(0.35) hue-rotate(200deg) contrast(1.1)",
  Dramatic:        "contrast(1.7) saturate(1.2) brightness(0.75)",
  Vintage:         "brightness(1.1) saturate(1.3) sepia(0.55) contrast(1.15)",
  Night:           "brightness(0.6) saturate(0.6) hue-rotate(210deg) contrast(1.35)",
  Cinematic:       "contrast(1.4) saturate(1.6) brightness(0.85) sepia(0.12)",
  "Black & White": "grayscale(100%) contrast(1.45) brightness(0.95)",
  "Teal & Orange": "saturate(1.6) hue-rotate(8deg) contrast(1.2) brightness(0.92)",
  Fade:            "brightness(1.12) contrast(0.8) saturate(0.7)",
  Neon:            "saturate(2.5) contrast(1.4) brightness(0.8) hue-rotate(270deg)",
};

const FILTER_DEFAULT_MUSIC: Record<string, string> = {
  Cinematic:       "Cinematic",
  Dramatic:        "Cinematic",
  Night:           "Cinematic",
  Sad:             "Sad",
  Sunset:          "Cinematic",
  Vintage:         "Cinematic",
  Fade:            "Cinematic",
  Happy:           "Upbeat",
  Neon:            "Upbeat",
  "Teal & Orange": "Cinematic",
  "Black & White": "Cinematic",
};

function generateCaptionFromPrompt(prompt: string): string {
  const p = prompt.toLowerCase().trim();

  if (p.includes("sad") && (p.includes("bus") || p.includes("road") || p.includes("journey"))) return "Miles From Home";
  if (p.includes("sad") && (p.includes("plant") || p.includes("green") || p.includes("lush"))) return "Quietly Blooming";
  if (p.includes("sad") && (p.includes("stair") || p.includes("hallway"))) return "Heavy Steps";
  if (p.includes("sad") && (p.includes("rain") || p.includes("cloud"))) return "After the Storm";
  if ((p.includes("bus") || p.includes("road")) && (p.includes("plant") || p.includes("green"))) return "The Long Way Home 🌿";
  if ((p.includes("stair") || p.includes("hallway")) && (p.includes("plant") || p.includes("green"))) return "Growing Through It 🌿";

  if (p.includes("bus") || p.includes("train") || p.includes("road")) return "Every Road Taken";
  if (p.includes("stair") || p.includes("steps") || p.includes("hallway") || p.includes("corridor")) return "Echoes of Time";
  if (p.includes("run") || p.includes("chase") || p.includes("speed")) return "Never Look Back";
  if (p.includes("walk") || p.includes("path") || p.includes("journey")) return "One Step at a Time";

  if (p.includes("plant") || p.includes("lush") || p.includes("garden") || p.includes("flower")) return "Still Growing 🌿";
  if (p.includes("forest") || p.includes("tree") || p.includes("wood")) return "Lost in the Wild 🌲";
  if (p.includes("beach") || p.includes("ocean") || p.includes("sea")) return "Chasing the Tide 🌊";
  if (p.includes("mountain") || p.includes("hill")) return "Above It All";
  if (p.includes("rain") || p.includes("storm")) return "After the Rain";
  if (p.includes("sun") || p.includes("dawn") || p.includes("sunrise")) return "Chasing the Light ✨";

  if (p.includes("city") || p.includes("urban") || p.includes("street")) return "City Never Sleeps 🌃";
  if (p.includes("building") || p.includes("roof") || p.includes("window")) return "From the Heights";
  if (p.includes("market") || p.includes("shop")) return "Everyday Stories";

  if (p.includes("classroom") || p.includes("school") || p.includes("student")) return "Every Day Counts 📚";
  if (p.includes("friend") || p.includes("together") || p.includes("group")) return "Better Together ❤️";
  if (p.includes("alone") || p.includes("solitude") || p.includes("empty")) return "In the Silence";
  if (p.includes("dance") || p.includes("concert")) return "Feel the Beat 🎵";
  if (p.includes("sport") || p.includes("game")) return "In the Zone 🔥";
  if (p.includes("food") || p.includes("eat") || p.includes("cook")) return "Good Food Good Life 🍽️";

  if (p.includes("night") || p.includes("dark")) return "Night Vibes 🌙";
  if (p.includes("sunset") || p.includes("golden")) return "Golden Hour ✨";
  if (p.includes("neon") || p.includes("cyber")) return "Neon Dreams ⚡";
  if (p.includes("vintage") || p.includes("retro")) return "Back in Time 📼";
  if (p.includes("sad") || p.includes("emotional") || p.includes("melancholy")) return "Feel It All";
  if (p.includes("happy") || p.includes("joy") || p.includes("bright")) return "Good Vibes ☀️";
  if (p.includes("cinematic") || p.includes("dramatic")) return "Raw Emotion";

  return "My Story";
}

function inferSettingsFromPrompt(prompt: string): {
  filter?: string; transition?: string; music?: string;
  brightness?: number; contrast?: number; saturation?: number; caption?: string;
} {
  const p = prompt.toLowerCase();
  const s: ReturnType<typeof inferSettingsFromPrompt> = {};

  if (p.includes("cinematic") || p.includes("film look") || p.includes("film grain")) s.filter = "Cinematic";
  else if (p.includes("sunset") || p.includes("golden hour") || p.includes("warm look") || p.includes("warm tone") || p.includes("golden")) s.filter = "Sunset";
  else if (p.includes("sad") || p.includes("melancholy") || p.includes("grief") || p.includes("lonely") || p.includes("depressed") || p.includes("heartbreak") || p.includes("gloomy") || p.includes("sorrow")) s.filter = "Sad";
  else if (p.includes("happy") || p.includes("bright") || p.includes("vibrant") || p.includes("joyful") || p.includes("joy") || p.includes("excited") || p.includes("cheerful") || p.includes("fun") || p.includes("lively") || p.includes("energetic") || p.includes("playful")) s.filter = "Happy";
  else if (p.includes("dramatic") || p.includes("moody") || p.includes("dark look") || p.includes("intense") || p.includes("gritty") || p.includes("raw") || p.includes("powerful") || p.includes("bold")) s.filter = "Dramatic";
  else if (p.includes("vintage") || p.includes("retro") || p.includes("old school") || p.includes("nostalgic") || p.includes("nostalgia") || p.includes("throwback") || p.includes("classic")) s.filter = "Vintage";
  else if (p.includes("night") || p.includes("cool tone") || p.includes("midnight") || p.includes("late night") || p.includes("dark blue") || p.includes("cold")) s.filter = "Night";
  else if (p.includes("black and white") || p.includes("monochrome") || p.includes("b&w") || p.includes("bw") || p.includes("grayscale") || p.includes("black & white")) s.filter = "Black & White";
  else if (p.includes("teal") || p.includes("hollywood") || p.includes("teal and orange") || p.includes("blockbuster")) s.filter = "Teal & Orange";
  else if (p.includes("neon") || p.includes("cyberpunk") || p.includes("futuristic") || p.includes("synthwave") || p.includes("glowing")) s.filter = "Neon";
  else if (p.includes("faded") || p.includes("matte look") || p.includes("matte") || p.includes("soft") || p.includes("dreamy") || p.includes("hazy")) s.filter = "Fade";

  if (p.includes("fade transition") || p.includes("dissolve") || p.includes("crossfade") || p.includes("fade cut") || p.includes("smooth transition")) s.transition = "fade";
  else if (p.includes("wipe")) s.transition = "wipe";
  else if (p.includes("zoom transition") || p.includes("zoom cut") || p.includes("push in") || p.includes("zoom in")) s.transition = "zoom";
  else if (p.includes("white flash") || p.includes("flash cut") || p.includes("flash transition") || p.includes("flash")) s.transition = "flash";
  else if (p.includes("glitch")) s.transition = "glitch";
  else if (p.includes("blur transition") || p.includes("blur cut")) s.transition = "blur";
  else if (p.includes("dip to black") || p.includes("dip black") || p.includes("dip")) s.transition = "dip";
  else if (p.includes("hard cut") || p.includes("no transition") || p.includes("cut only") || p.includes("straight cut")) s.transition = "none";

  if (p.includes("cinematic music") || p.includes("orchestral") || p.includes("epic") || p.includes("dramatic music") || p.includes("intense music")) s.music = "Cinematic";
  else if (p.includes("upbeat") || p.includes("hype") || p.includes("energetic music") || p.includes("happy music") || p.includes("fun music") || p.includes("cheerful music")) s.music = "Upbeat";
  else if (p.includes("sad music") || p.includes("emotional music") || p.includes("melancholic music") || p.includes("slow music")) s.music = "Sad";
  else if (p.includes("energetic")) s.music = "Energetic";
  else if (s.filter && FILTER_DEFAULT_MUSIC[s.filter]) {
    s.music = FILTER_DEFAULT_MUSIC[s.filter];
  }

  if (p.includes("overexposed")) s.brightness = 140;
  else if (p.includes("very dark")) s.brightness = 65;
  if (p.includes("desaturated") || p.includes("muted colors")) s.saturation = 50;
  else if (p.includes("oversaturated")) s.saturation = 160;
  if (p.includes("high contrast")) s.contrast = 140;
  else if (p.includes("low contrast") || p.includes("flat look")) s.contrast = 75;

  s.caption = generateCaptionFromPrompt(prompt);
  return s;
}

export default function Index() {
  const [isAssembling, setIsAssembling] = useState(false);
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState("loading...");
  const [assembledVideoUrl, setAssembledVideoUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [aiStatus, setAiStatus] = useState<"idle" | "analyzing" | "matching" | "done">("idle");
  const [aiScenes, setAiScenes] = useState<AIScene[]>([]);
  const { toast } = useToast();

  const selectedVideo = videos.find(v => v.id === selectedVideoId);
  const [trimMap, setTrimMap] = useState<Record<string, { start: number; end: number }>>({});
  const trimValues = selectedVideoId
    ? (trimMap[selectedVideoId] ?? { start: 0, end: selectedVideo?.duration || 10 })
    : { start: 0, end: 10 };
  const setTrimValues = (val: { start: number; end: number }) => {
    if (selectedVideoId) setTrimMap(prev => ({ ...prev, [selectedVideoId]: val }));
  };

  const [music, setMusic] = useState("None");
  const [musicVolume, setMusicVolume] = useState(0.4);
  const [muteOriginal, setMuteOriginal] = useState(false);
  const [filter, setFilter] = useState("None");
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [saturation, setSaturation] = useState(100);
  const [aspectRatio, setAspectRatio] = useState<"16:9" | "9:16" | "1:1" | "original">("original");
  const [cropOffset, setCropOffset] = useState(50);
  const [overlayText, setOverlayText] = useState("");
  const [captionX, setCaptionX] = useState(50);
  const [captionY, setCaptionY] = useState(85);
  const [transition, setTransition] = useState("fade");
  const [hasAssembled, setHasAssembled] = useState(false);

  const handleTransitionChange = (val: string) => {
    setTransition(val);
    if (hasAssembled) {
      toast({
        title: "Transition changed",
        description: "Click Assemble & Preview to apply the new transition.",
        className: "bg-zinc-900 border border-fuchsia-700/60 text-zinc-200 shadow-xl",
        duration: 3000,
      });
    }
  };

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(res => res.json())
      .then(data => setBackendStatus(data.status))
      .catch(() => setBackendStatus("offline"));
  }, []);

  const applySettings = (
    s: ReturnType<typeof inferSettingsFromPrompt>,
    source: "ai" | "inferred",
    summary?: string
  ) => {
    const applied: string[] = [];
    if (s.filter)     { setFilter(s.filter);        applied.push(s.filter); }
    if (s.transition) { setTransition(s.transition); applied.push(`${s.transition} transition`); }
    if (s.music)      { setMusic(s.music);           applied.push(`${s.music} music`); }
    if (s.brightness !== undefined) setBrightness(s.brightness);
    if (s.contrast   !== undefined) setContrast(s.contrast);
    if (s.saturation !== undefined) setSaturation(s.saturation);
    if (s.caption)    { setOverlayText(s.caption);   applied.push(`"${s.caption}"`); }
    if (applied.length > 0 || summary) {
      toast({
        title: source === "ai" ? "AI applied effects" : "✨ Auto-detected",
        description: summary || applied.join(" · "),
        className: "bg-violet-950 border border-violet-700 text-violet-200 shadow-xl",
        duration: 4000,
      });
    }
  };

  const handleRunAI = async () => {
    if (!prompt || videos.length === 0) {
      toast({ title: "Missing input", description: "Upload a video and enter a prompt.", variant: "destructive" });
      return;
    }
    setAiStatus("analyzing");
    try {
      const data = await runEditPrompt(prompt, videos.map(v => ({
        id: v.id, name: v.name, url: v.url, duration: v.duration,
      })));

      const s = data.edit_settings;
      const inferred = inferSettingsFromPrompt(prompt);
      const merged: ReturnType<typeof inferSettingsFromPrompt> = { ...inferred };

      const VALID_MUSIC = ["None", "Cinematic", "Upbeat", "Sad", "Energetic"];

      if (s) {
        if (s.filter)     merged.filter     = s.filter;
        if (s.music && VALID_MUSIC.includes(s.music)) merged.music = s.music;
        if (s.transition) merged.transition  = s.transition;
        if (s.brightness !== undefined) merged.brightness = s.brightness;
        if (s.contrast   !== undefined) merged.contrast   = s.contrast;
        if (s.saturation !== undefined) merged.saturation = s.saturation;
        if (s.musicVolume  !== undefined) setMusicVolume(s.musicVolume);
        if (s.muteOriginal !== undefined) setMuteOriginal(s.muteOriginal);
        if (s.aspectRatio) setAspectRatio(s.aspectRatio);
      }

      // ✅ Always update caption on every prompt run
      // Ollama caption (from backend) wins, otherwise frontend generates from new prompt
      if (s?.overlayText) {
        merged.caption = s.overlayText;
      } else {
        merged.caption = generateCaptionFromPrompt(prompt);
      }

      // Fill music from filter if still missing
      if (!merged.music && merged.filter && FILTER_DEFAULT_MUSIC[merged.filter]) {
        merged.music = FILTER_DEFAULT_MUSIC[merged.filter];
      }

      applySettings(merged, "inferred", data.settings_summary);

      if (data.selected_clips && data.selected_clips.length > 0) {
        const FPS = 30;
        const clipGroups = new Map<number, any[]>();
        data.selected_clips.forEach((clip: any) => {
          const id = clip.clip_id;
          if (!clipGroups.has(id)) clipGroups.set(id, []);
          clipGroups.get(id)!.push(clip);
        });

        const scenes: AIScene[] = [];
        clipGroups.forEach((frames, clipId) => {
          frames.sort((a: any, b: any) => a.frame_index - b.frame_index);
          const first = frames[0];
          const last = frames[frames.length - 1];
          const sourceVideo = videos.find(v => v.name === first.video_name) || videos[0];
          const totalDur = first.video_duration || sourceVideo?.duration || 11;
          const clampedStart = Math.max(0, first.frame_index / FPS);
          const clampedEnd = Math.min(totalDur, (last.frame_index / FPS) + (1 / FPS));
          scenes.push({
            id: crypto.randomUUID(),
            label: `${sourceVideo?.name ?? "clip"} — Scene ${clipId} (${clampedStart.toFixed(1)}s–${clampedEnd.toFixed(1)}s)`,
            start: clampedStart, end: clampedEnd,
            videoUrl: first.video_url || sourceVideo?.url || "",
            videoName: first.video_name || sourceVideo?.name || "Unknown",
          });
        });

        setAiScenes(scenes);
        setTimelineClips(prev => [
          ...prev,
          ...scenes.map(scene => ({
            id: crypto.randomUUID(),
            name: scene.label,
            duration: scene.end - scene.start,
            trimStart: scene.start,
            trimEnd: scene.end,
            videoUrl: scene.videoUrl || "",
            prompt,
            createdAt: Date.now(),
          })),
        ]);
        setAiStatus("done");
        toast({
          title: "Clips Generated",
          description: `${scenes.length} scene(s) added to timeline.`,
          className: "bg-emerald-950 border border-emerald-800 text-emerald-200 shadow-xl",
          duration: 4000,
        });
      } else {
        setAiStatus("done");
        setAiScenes([]);
        toast({ title: "No matches found", description: "Try describing what you see in the video.", variant: "destructive" });
      }
    } catch (err) {
      console.error(err);
      setAiStatus("idle");
      toast({ title: "Backend error", description: "Failed to process prompt.", variant: "destructive" });
    }
  };

  const handleAddToTimeline = () => {
    if (!selectedVideo) return;
    setTimelineClips([...timelineClips, {
      id: crypto.randomUUID(), name: selectedVideo.name,
      duration: selectedVideo.duration, trimStart: trimValues.start,
      trimEnd: trimValues.end, videoUrl: selectedVideo.url,
    }]);
  };

  const handleAddAIScene = (scene: AIScene) => {
    const videoUrl = scene.videoUrl || selectedVideo?.url;
    if (!videoUrl) return;
    setTimelineClips(prev => [...prev, {
      id: crypto.randomUUID(), name: scene.label,
      duration: scene.end - scene.start, trimStart: scene.start,
      trimEnd: scene.end, videoUrl,
    }]);
  };

  const handleAssembleVideo = async () => {
    if (timelineClips.length === 0) return;
    try {
      setIsAssembling(true);
      const res = await fetch("http://localhost:8000/videos/merge", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clips: timelineClips, transition }),
      });
      if (!res.ok) throw new Error("Assemble failed");
      const data = await res.json();
      setAssembledVideoUrl(`http://localhost:8000/outputs/${data.output_file}?t=${Date.now()}`);
      setHasAssembled(true);
      toast({
        title: "✅ Video Assembled",
        description: `${transition} transition applied`,
        className: "bg-emerald-950 border border-emerald-800 text-emerald-200 shadow-xl",
        duration: 3000,
      });
    } catch {
      toast({ title: "Assemble failed", description: "Something went wrong.", variant: "destructive" });
    } finally {
      setIsAssembling(false);
    }
  };



  const activeFilter = MOOD_FILTERS.find(f => f.value === filter);
  const activeTransition = TRANSITION_PRESETS.find(t => t.value === transition);
  const liveInferred = prompt.trim().length > 3 ? inferSettingsFromPrompt(prompt) : null;
  const livePreviewItems = liveInferred ? [
    liveInferred.filter && liveInferred.filter !== "None" && liveInferred.filter,
    liveInferred.transition && `${liveInferred.transition} cut`,
    liveInferred.music,
    // ✅ Show "AI caption" instead of frontend-generated text
    // since Ollama will generate a different caption when Run is clicked
    prompt.trim().length > 3 && "AI caption ✨",
  ].filter(Boolean) as string[] : [];

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950 text-zinc-100">
      <header className="h-14 flex-shrink-0 border-b border-zinc-800/60 bg-zinc-950/95 backdrop-blur-sm flex items-center justify-between px-5 z-40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center shadow-lg shadow-violet-500/30 flex-shrink-0">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div className="leading-none">
            <h1 className="text-sm font-bold tracking-[0.2em] bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">ZYNC</h1>
            <p className="text-[10px] text-zinc-500 mt-0.5">AI Video Editor</p>
          </div>
          <div className={`ml-1 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium border transition-colors ${
            backendStatus === "ok" ? "bg-emerald-950/80 border-emerald-800/60 text-emerald-400"
            : backendStatus === "loading..." ? "bg-zinc-800/60 border-zinc-700/60 text-zinc-400"
            : "bg-red-950/80 border-red-800/60 text-red-400"
          }`}>
            {backendStatus === "ok" ? <Wifi className="w-2.5 h-2.5" /> : <WifiOff className="w-2.5 h-2.5" />}
            {backendStatus === "ok" ? "Connected" : backendStatus === "loading..." ? "Connecting…" : "Offline"}
          </div>
        </div>
        <div className="flex items-center gap-2">

          <Button onClick={handleAssembleVideo} disabled={timelineClips.length === 0 || isAssembling}
            className="h-9 px-4 bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold rounded-lg shadow-lg shadow-violet-500/25 transition-all active:scale-95 disabled:opacity-40">
            <Layers className="w-4 h-4 mr-1.5" />{isAssembling ? "Processing…" : "Assemble & Preview"}
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 flex-shrink-0 border-r border-zinc-800/60 bg-zinc-900/50 flex flex-col overflow-y-auto">
          <div className="px-4 pt-4 pb-3">
            <SectionLabel>Upload</SectionLabel>
            <VideoUploader videos={videos} onVideosChange={async (v) => {
              setVideos(v);
              if (!selectedVideoId && v.length) setSelectedVideoId(v[0].id);
              setTrimMap(prev => {
                const next = { ...prev };
                v.forEach(vid => { if (!next[vid.id]) next[vid.id] = { start: 0, end: vid.duration || 10 }; });
                return next;
              });
              if (v.length > 0) toast({ title: "Video Uploaded", description: `${v[v.length - 1].name} added.`, className: "bg-zinc-900 border border-zinc-700 text-zinc-100 shadow-xl", duration: 3000 });
              try { await sendVideosToBackend(v.map(video => ({ id: video.id, name: video.name, url: video.url, duration: video.duration }))); }
              catch (err) { console.error("Failed to send videos", err); }
            }} />
          </div>

          {videos.length > 0 && (
            <div className="px-4 pb-3">
              <SectionLabel>Library ({videos.length})</SectionLabel>
              <div className="space-y-1 max-h-44 overflow-y-auto -mx-1 px-1">
                {videos.map(v => (
                  <button key={v.id} onClick={() => { setSelectedVideoId(v.id); setTrimMap(prev => ({ ...prev, [v.id]: prev[v.id] ?? { start: 0, end: v.duration || 10 } })); }}
                    className={`w-full text-left px-2.5 py-2 rounded-lg border text-xs transition-all flex items-center gap-2 ${
                      selectedVideoId === v.id ? "bg-violet-600/15 border-violet-500/50 text-violet-300" : "border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 hover:bg-zinc-800/50"
                    }`}>
                    <Film className={`w-3.5 h-3.5 flex-shrink-0 ${selectedVideoId === v.id ? "text-violet-400" : "text-zinc-500"}`} />
                    <span className="truncate flex-1 font-medium">{v.name}</span>
                    {v.duration > 0 && <span className="opacity-50 shrink-0 font-mono">{Math.floor(v.duration)}s</span>}
                  </button>
                ))}
              </div>
            </div>
          )}

          {selectedVideo && (
            <div className="px-4 pb-3 space-y-3">
              <SectionLabel>Trim</SectionLabel>
              <TrimControls duration={selectedVideo.duration || 10} trimStart={trimValues.start} trimEnd={trimValues.end} videoUrl={selectedVideo.url} onTrimChange={(s, e) => setTrimValues({ start: s, end: e })} />
              <button onClick={handleAddToTimeline}
                className="w-full flex items-center justify-center gap-2 h-9 rounded-lg border border-dashed border-zinc-700 text-zinc-400 text-xs font-medium hover:border-violet-500/60 hover:text-violet-400 hover:bg-violet-500/5 transition-all">
                <Plus className="w-3.5 h-3.5" />Add to Timeline
              </button>
            </div>
          )}

          <div className="mx-4 border-t border-zinc-800/60" />

          <div className="px-4 py-3 space-y-3 flex-1">
            <SectionLabel icon={<Sparkles className="w-3 h-3 text-violet-400" />}>AI Prompt</SectionLabel>
            <div className="flex flex-wrap gap-1">
              {["cinematic fade", "golden hour wipe", "moody dramatic", "b&w glitch", "neon cyberpunk"].map(hint => (
                <button key={hint} onClick={() => setPrompt(hint)}
                  className="text-[10px] px-2 py-0.5 rounded-full border border-zinc-700/60 text-zinc-500 hover:border-violet-500/50 hover:text-violet-400 hover:bg-violet-500/5 transition-all">
                  {hint}
                </button>
              ))}
            </div>
            <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
              placeholder={"Describe scene + look:\n\"sad scene with yellow bus\"\n\"cinematic hallway and stairs\""}
              rows={4}
              className="w-full bg-zinc-800/60 border border-zinc-700/60 rounded-lg p-3 text-sm text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none focus:border-violet-500/50 focus:bg-zinc-800 transition-all" />
            {livePreviewItems.length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap bg-zinc-800/40 rounded-lg px-2.5 py-2 border border-zinc-700/40">
                <Wand2 className="w-3 h-3 text-fuchsia-400 flex-shrink-0" />
                <span className="text-[10px] text-zinc-500">Will apply:</span>
                {livePreviewItems.map(item => (
                  <span key={item} className="text-[10px] px-1.5 py-0.5 rounded-md bg-fuchsia-950/60 border border-fuchsia-800/40 text-fuchsia-300">{item}</span>
                ))}
              </div>
            )}
            <button onClick={handleRunAI} disabled={!videos.length || !prompt || aiStatus === "analyzing"}
              className="w-full h-10 rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-semibold shadow-lg shadow-violet-500/20 transition-all active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none flex items-center justify-center gap-2">
              {aiStatus === "analyzing"
                ? (<><span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Generating Clips…</>)
                : (<><Sparkles className="w-3.5 h-3.5" />Run AI Editor</>)}
            </button>
            {aiStatus === "analyzing" && <p className="text-xs text-zinc-500 text-center">Analyzing frames & applying effects…</p>}
            {aiStatus === "done" && aiScenes.length === 0 && <p className="text-xs text-amber-500/80 text-center">No matching scenes found</p>}
            {aiStatus === "done" && aiScenes.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-emerald-400 font-medium">{aiScenes.length} scene(s) added to timeline</p>
                <div className="max-h-52 overflow-y-auto space-y-1.5">
                  {aiScenes.map(scene => (
                    <div key={scene.id} className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-zinc-800/60 border border-zinc-700/50 hover:border-zinc-600/80 transition-colors">
                      <span className="text-xs text-zinc-300 truncate">{scene.label}</span>
                      <button onClick={() => handleAddAIScene(scene)}
                        className="flex-shrink-0 text-[11px] font-semibold text-violet-400 hover:text-violet-300 border border-violet-500/40 hover:border-violet-400/60 rounded-md px-2 py-0.5 transition-colors hover:bg-violet-500/10">
                        +Add
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>

        <main className="flex-1 bg-zinc-950 min-w-0">
          <Timeline clips={timelineClips} selectedClipId={selectedClipId} onSelectClip={setSelectedClipId}
            onRemoveClip={(id) => setTimelineClips(prev => prev.filter(c => c.id !== id))}
            onReorderClips={setTimelineClips} />
        </main>

        <aside className="w-96 flex-shrink-0 border-l border-zinc-800/60 bg-zinc-900/50 flex flex-col overflow-y-auto">
          <div className="p-3 border-b border-zinc-800/60">
            <VideoPreview key={selectedVideoId ?? "no-source"} title="Source Preview" videoUrl={selectedVideo?.url || null} enableFades={false} />
          </div>
          <div className="p-3 border-b border-zinc-800/60">
            <VideoPreview
              key={assembledVideoUrl ?? "no-output"}
              title="Output Preview"
              videoUrl={assembledVideoUrl}
              filter={filter} overlayText={overlayText}
              captionX={captionX} captionY={captionY}
              onCaptionMove={(x, y) => { setCaptionX(x); setCaptionY(y); }}
              music={music} musicStart={0}
              enableFades={true} muteOriginal={music !== "None" ? true : muteOriginal}
              brightness={brightness} contrast={contrast} saturation={saturation}
              transition={transition}
              aspectRatio={aspectRatio} cropOffset={cropOffset}
              mergedClipBoundaries={timelineClips.map((c, i) => {
                const globalStart = timelineClips.slice(0, i).reduce((sum, prev) => sum + (prev.trimEnd - prev.trimStart), 0);
                const globalEnd = globalStart + (c.trimEnd - c.trimStart);
                return { globalStart, globalEnd, fadeIn: c.fadeIn ?? 0, fadeOut: c.fadeOut ?? 0 };
              })}
            />
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">Post-Production</h3>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Clapperboard className="w-3.5 h-3.5 text-fuchsia-400" />
                <ControlLabel>Clip Transitions</ControlLabel>
                {transition !== "none" && (
                  <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full border border-fuchsia-500/40 text-fuchsia-400 bg-fuchsia-500/10">
                    {activeTransition?.icon} {activeTransition?.label}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {TRANSITION_PRESETS.map(t => (
                  <button key={t.value} onClick={() => handleTransitionChange(t.value)}
                    className={`flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-left transition-all duration-150 ${
                      transition === t.value ? "bg-fuchsia-600/15 border-fuchsia-500/60 shadow-md shadow-fuchsia-500/10" : "border-zinc-800/80 hover:border-zinc-600 hover:bg-zinc-800/40"
                    }`}>
                    <span className="text-base">{t.icon}</span>
                    <div>
                      <p className={`text-[11px] font-semibold leading-tight ${transition === t.value ? "text-fuchsia-300" : "text-zinc-300"}`}>{t.label}</p>
                      <p className="text-[9px] text-zinc-600 leading-tight">{t.description}</p>
                    </div>
                    {transition === t.value && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-fuchsia-400 flex-shrink-0" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Palette className="w-3.5 h-3.5 text-violet-400" />
                <ControlLabel>Cinematic Look</ControlLabel>
                {filter !== "None" && (
                  <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full border border-violet-500/40 text-violet-400 bg-violet-500/10 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full inline-block" style={{ background: activeFilter?.dot }} />
                    {activeFilter?.label}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2">
                {MOOD_FILTERS.map(mood => (
                  <button key={mood.value} onClick={() => setFilter(mood.value)}
                    className={`group relative overflow-hidden rounded-xl border text-left transition-all duration-200 ${
                      filter === mood.value ? "border-violet-400/70 shadow-lg shadow-violet-500/20 scale-[1.03]" : "border-zinc-800/80 hover:border-zinc-600 hover:scale-[1.02]"
                    }`}>
                    <div className="h-9 w-full" style={{ background: mood.gradient }} />
                    <div className={`px-2 py-1.5 ${filter === mood.value ? "bg-violet-950/70" : "bg-zinc-900/90"}`}>
                      <p className={`text-[10px] font-semibold truncate ${filter === mood.value ? "text-violet-300" : "text-zinc-300"}`}>{mood.label}</p>
                      <p className="text-[8px] text-zinc-600 leading-tight truncate">{mood.description}</p>
                    </div>
                    {filter === mood.value && <div className="absolute top-1.5 right-1.5 w-2.5 h-2.5 rounded-full bg-violet-400 shadow shadow-violet-400/60" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2.5">
              <ControlLabel>Color Grading</ControlLabel>
              <div className="space-y-2 bg-zinc-900/60 rounded-xl p-3 border border-zinc-800/60">
                <SliderRow label="Brightness" value={brightness} min={50} max={150} step={1} onChange={setBrightness} display={`${brightness}%`} accentColor="#facc15" />
                <SliderRow label="Contrast"   value={contrast}   min={50} max={150} step={1} onChange={setContrast}   display={`${contrast}%`}   accentColor="#60a5fa" />
                <SliderRow label="Saturation" value={saturation} min={0}  max={200} step={1} onChange={setSaturation} display={`${saturation}%`} accentColor="#f472b6" />
                <div className="pt-1.5 border-t border-zinc-800/60 flex items-center justify-between">
                  <span className="text-[10px] text-zinc-600">Manual overrides</span>
                  <button onClick={() => { setBrightness(100); setContrast(100); setSaturation(100); }} className="text-[10px] text-zinc-600 hover:text-violet-400 transition-colors">↺ Reset</button>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <ControlLabel>Background Music</ControlLabel>
              <div className="flex gap-2 items-center">
                <select value={music} onChange={e => setMusic(e.target.value)}
                  className="flex-1 bg-zinc-800/60 border border-zinc-700/60 rounded-lg px-2.5 py-2 text-xs text-zinc-200 focus:outline-none focus:border-violet-500/50">
                  {MUSIC_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
                <button onClick={() => setMuteOriginal(m => !m)}
                  title={muteOriginal ? "Original audio muted" : "Mute original audio"}
                  className={`px-2.5 py-2 rounded-lg border text-xs font-medium transition-colors ${muteOriginal ? "bg-red-950/60 border-red-800/60 text-red-400" : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:border-zinc-600"}`}>
                  {muteOriginal ? "🔇 Muted" : "🔊 Original"}
                </button>
              </div>
              {music !== "None" && (
                <SliderRow label="Vol" value={musicVolume} min={0} max={1} step={0.05} onChange={setMusicVolume} display={`${Math.round(musicVolume * 100)}%`} />
              )}
            </div>

            <div className="space-y-2">
              <ControlLabel>Aspect Ratio</ControlLabel>
              <div className="grid grid-cols-4 gap-1.5">
                {(["original", "16:9", "9:16", "1:1"] as const).map(r => (
                  <button key={r} onClick={() => setAspectRatio(r)}
                    className={`py-1.5 rounded-lg border text-[11px] font-medium transition-all ${
                      aspectRatio === r ? "bg-violet-600/20 border-violet-500/60 text-violet-300" : "border-zinc-700/60 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300"
                    }`}>
                    {r === "original" ? "Auto" : r}
                  </button>
                ))}
              </div>
              {aspectRatio !== "original" && (
                <SliderRow label={aspectRatio === "9:16" ? "H-pos" : "V-pos"} value={cropOffset} min={0} max={100} step={1} onChange={setCropOffset}
                  display={<button onClick={() => setCropOffset(50)} className="text-[10px] text-zinc-500 hover:text-violet-400 transition-colors">↺</button>} />
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <ControlLabel>Caption</ControlLabel>
                {prompt.trim().length > 3 && (
                  <button onClick={() => setOverlayText(generateCaptionFromPrompt(prompt))}
                    className="text-[10px] text-fuchsia-400 hover:text-fuchsia-300 border border-fuchsia-700/40 rounded-md px-2 py-0.5 hover:bg-fuchsia-500/10 transition-all flex items-center gap-1">
                    <Wand2 className="w-2.5 h-2.5" />Auto
                  </button>
                )}
              </div>
              <input type="text" value={overlayText} onChange={e => setOverlayText(e.target.value)}
                placeholder="Enter caption or click Auto…"
                className="w-full bg-zinc-800/60 border border-zinc-700/60 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-violet-500/50 transition-all" />
              {overlayText && (
                <div className="flex items-center justify-between">
                  <p className="text-[11px] text-zinc-600">Drag caption in Output Preview to reposition</p>
                  <button onClick={() => setOverlayText("")} className="text-[10px] text-zinc-600 hover:text-red-400 transition-colors">✕ Clear</button>
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

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
  return <label className="flex items-center gap-1.5 text-xs font-medium text-zinc-400">{children}</label>;
}
function SliderRow({ label, value, min, max, step, onChange, display, accentColor }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; display?: React.ReactNode; accentColor?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-zinc-500 w-20 flex-shrink-0">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-1.5 rounded-full"
        style={accentColor ? { accentColor } : undefined} />
      {typeof display === "string"
        ? <span className="text-[11px] font-mono text-zinc-500 w-10 text-right flex-shrink-0">{display}</span>
        : display}
    </div>
  );
}
function Tag({ children, variant = "default" }: { children: React.ReactNode; variant?: "default" | "accent" }) {
  return (
    <span className={`rounded-md px-2 py-1 text-[11px] font-medium flex items-center ${
      variant === "accent" ? "bg-violet-950/60 border border-violet-800/40 text-violet-300" : "bg-zinc-800 border border-zinc-700/60 text-zinc-400"
    }`}>{children}</span>
  );
}
