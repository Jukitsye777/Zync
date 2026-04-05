import { useRef, useState, useEffect, useCallback } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize, Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

export interface Clip {
  url: string;
  trimStart?: number;
  trimEnd?: number;
  title?: string;
  fadeIn?: number;
  fadeOut?: number;
}

export interface MergedClipBoundary {
  globalStart: number;
  globalEnd: number;
  fadeIn: number;
  fadeOut: number;
}

interface VideoPreviewProps {
  videoUrl?: string | null;
  clips?: Clip[];
  trimStart?: number;
  trimEnd?: number;
  title?: string;
  playbackRate?: number;
  filter?: string;
  overlayText?: string;
  music?: string;
  musicStart?: number;
  musicEnd?: number;
  enableFades?: boolean;
  fadeDuration?: number;
  mergedClipBoundaries?: MergedClipBoundary[];
  muteOriginal?: boolean;
  captionX?: number;
  captionY?: number;
  onCaptionMove?: (x: number, y: number) => void;
  brightness?: number;
  contrast?: number;
  saturation?: number;
  transition?: string;
  aspectRatio?: "16:9" | "9:16" | "1:1" | "original";
  cropOffset?: number;
  exportMode?: boolean;
  onClose?: () => void;
}

// ── Caption colors + fonts per filter/mood ───────────────────────────────────
const CAPTION_STYLES: Record<string, {
  bg: string; border: string; text: string; shadow: string;
  font: string; weight: string; style: string; size: string; letterSpacing: string;
}> = {
  Happy:           { bg: "rgba(250,200,0,0.25)",   border: "rgba(250,200,0,0.5)",   text: "#fff9d6", shadow: "0 2px 12px rgba(250,200,0,0.5)",
                     font: "Georgia, serif",         weight: "800", style: "normal",  size: "13px",  letterSpacing: "0.02em" },
  Sad:             { bg: "rgba(59,130,246,0.25)",   border: "rgba(59,130,246,0.5)",  text: "#dbeafe", shadow: "0 2px 12px rgba(59,130,246,0.5)",
                     font: "Georgia, serif",         weight: "300", style: "italic",  size: "12px",  letterSpacing: "0.04em" },
  Dramatic:        { bg: "rgba(120,0,0,0.40)",      border: "rgba(200,0,0,0.5)",     text: "#ffe4e4", shadow: "0 2px 12px rgba(200,0,0,0.5)",
                     font: "Impact, sans-serif",     weight: "900", style: "normal",  size: "13px",  letterSpacing: "0.08em" },
  Cinematic:       { bg: "rgba(160,80,0,0.35)",     border: "rgba(220,120,0,0.5)",   text: "#fff0dc", shadow: "0 2px 12px rgba(180,90,0,0.5)",
                     font: "Georgia, serif",         weight: "400", style: "italic",  size: "12px",  letterSpacing: "0.06em" },
  Vintage:         { bg: "rgba(140,100,50,0.35)",   border: "rgba(190,140,80,0.5)",  text: "#fdf0dc", shadow: "0 2px 12px rgba(160,112,64,0.4)",
                     font: "Georgia, serif",         weight: "400", style: "italic",  size: "12px",  letterSpacing: "0.05em" },
  Night:           { bg: "rgba(20,50,90,0.50)",     border: "rgba(56,100,160,0.5)",  text: "#bfdbfe", shadow: "0 2px 12px rgba(30,80,150,0.6)",
                     font: "Arial, sans-serif",      weight: "300", style: "normal",  size: "11px",  letterSpacing: "0.12em" },
  Sunset:          { bg: "rgba(200,80,10,0.35)",    border: "rgba(250,140,40,0.5)",  text: "#ffedd5", shadow: "0 2px 12px rgba(220,100,20,0.5)",
                     font: "Georgia, serif",         weight: "600", style: "normal",  size: "12px",  letterSpacing: "0.03em" },
  Neon:            { bg: "rgba(120,0,220,0.35)",    border: "rgba(200,0,255,0.6)",   text: "#f5d0fe", shadow: "0 2px 16px rgba(200,0,255,0.7)",
                     font: "Arial, sans-serif",      weight: "700", style: "normal",  size: "12px",  letterSpacing: "0.15em" },
  "Black & White": { bg: "rgba(60,60,60,0.50)",     border: "rgba(180,180,180,0.4)", text: "#f0f0f0", shadow: "0 2px 12px rgba(0,0,0,0.7)",
                     font: "Georgia, serif",         weight: "400", style: "normal",  size: "12px",  letterSpacing: "0.08em" },
  "Teal & Orange": { bg: "rgba(0,90,90,0.35)",      border: "rgba(0,180,180,0.5)",   text: "#ccfbf1", shadow: "0 2px 12px rgba(0,150,150,0.5)",
                     font: "Arial, sans-serif",      weight: "600", style: "normal",  size: "12px",  letterSpacing: "0.06em" },
  Fade:            { bg: "rgba(70,70,110,0.35)",    border: "rgba(140,140,180,0.4)", text: "#e8e8f4", shadow: "0 2px 12px rgba(80,80,120,0.4)",
                     font: "Georgia, serif",         weight: "300", style: "italic",  size: "12px",  letterSpacing: "0.04em" },
  None:            { bg: "rgba(0,0,0,0.6)",         border: "rgba(255,255,255,0.15)",text: "#ffffff", shadow: "0 1px 6px rgba(0,0,0,1)",
                     font: "Arial, sans-serif",      weight: "600", style: "normal",  size: "12px",  letterSpacing: "0.02em" },
};

function getCaptionStyle(filter: string) {
  return CAPTION_STYLES[filter] ?? CAPTION_STYLES["None"];
}

const FILTER_STYLES: Record<string, string> = {
  None:            "none",
  Warm:            "brightness(1.15) saturate(1.5) sepia(0.25)",
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

type TransitionState = "idle" | "out" | "in";

function getTransitionOverlayStyle(
  transitionType: string,
  state: TransitionState
): React.CSSProperties {
  const base: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    zIndex: 25,
    pointerEvents: "none",
  };

  if (state === "idle") return { ...base, opacity: 0 };

  switch (transitionType) {
    case "fade":
      return {
        ...base,
        backgroundColor: "black",
        opacity: state === "out" ? 1 : 0,
        transition: state === "out" ? "opacity 0.3s ease-in" : "opacity 0.3s ease-out",
      };
    case "flash":
      return {
        ...base,
        backgroundColor: "white",
        opacity: state === "out" ? 1 : 0,
        transition: state === "out" ? "opacity 0.1s ease-in" : "opacity 0.15s ease-out",
      };
    case "dip":
      return {
        ...base,
        backgroundColor: "black",
        opacity: state === "out" ? 1 : 0,
        transition: state === "out" ? "opacity 0.4s ease-in" : "opacity 0.4s ease-out",
      };
    case "wipe":
      return {
        ...base,
        backgroundColor: "black",
        transform: state === "out" ? "translateX(0%)" : "translateX(100%)",
        transition: "transform 0.35s ease-in-out",
        opacity: 1,
      };
    case "zoom":
      return {
        ...base,
        backgroundColor: "black",
        opacity: state === "out" ? 0.85 : 0,
        transform: state === "out" ? "scale(1.08)" : "scale(1)",
        transition: "opacity 0.3s ease, transform 0.3s ease",
      };
    case "blur":
      return {
        ...base,
        backgroundColor: "rgba(0,0,0,0.6)",
        backdropFilter: state === "out" ? "blur(20px)" : "blur(0px)",
        opacity: state === "out" ? 1 : 0,
        transition: "opacity 0.3s ease, backdrop-filter 0.3s ease",
      };
    case "glitch":
      return {
        ...base,
        background: state === "out"
          ? "repeating-linear-gradient(0deg, rgba(0,255,255,0.15) 0px, rgba(0,255,255,0.15) 2px, transparent 2px, transparent 8px), repeating-linear-gradient(90deg, rgba(255,0,255,0.1) 0px, rgba(255,0,255,0.1) 1px, transparent 1px, transparent 12px), black"
          : "black",
        opacity: state === "out" ? 1 : 0,
        transition: state === "out" ? "opacity 0.05s steps(3)" : "opacity 0.15s ease-out",
        filter: state === "out" ? "hue-rotate(90deg)" : "none",
      };
    default:
      return { ...base, opacity: 0 };
  }
}

// ── Draggable caption ─────────────────────────────────────────────────────────
function DraggableCaption({ text, x, y, onMove, filter = "None" }: {
  text: string; x: number; y: number; onMove: (x: number, y: number) => void; filter?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const toPercent = (clientX: number, clientY: number) => {
    const parent = containerRef.current?.parentElement;
    if (!parent) return { x, y };
    const rect = parent.getBoundingClientRect();
    return {
      x: Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)),
      y: Math.min(100, Math.max(0, ((clientY - rect.top) / rect.height) * 100)),
    };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation();
    dragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    const { x: nx, y: ny } = toPercent(e.clientX, e.clientY);
    onMove(nx, ny);
  };
  const onPointerUp = () => { dragging.current = false; };

  return (
    <div ref={containerRef}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
      className="absolute z-30 cursor-grab active:cursor-grabbing select-none"
      style={{ left: `${x}%`, top: `${y}%`, transform: "translate(-50%, -50%)" }}>
      <div className="px-3 py-1 rounded-lg max-w-[240px] text-center"
        style={{
          background: getCaptionStyle(filter).bg,
          backdropFilter: "blur(10px)",
          border: `1px solid ${getCaptionStyle(filter).border}`,
          boxShadow: getCaptionStyle(filter).shadow,
        }}>
        <span style={{
            color: getCaptionStyle(filter).text,
            textShadow: "0 1px 4px rgba(0,0,0,0.8)",
            fontFamily: getCaptionStyle(filter).font,
            fontWeight: getCaptionStyle(filter).weight,
            fontStyle: getCaptionStyle(filter).style,
            fontSize: getCaptionStyle(filter).size,
            letterSpacing: getCaptionStyle(filter).letterSpacing,
            lineHeight: "1.3",
          }}>
          {text}
        </span>
      </div>
    </div>
  );
}

export function VideoPreview({
  videoUrl,
  clips = [],
  trimStart = 0,
  trimEnd,
  title = "Preview",
  playbackRate = 1,
  filter = "None",
  overlayText = "",
  music = "None",
  musicStart = 0,
  musicEnd,
  enableFades = false,
  fadeDuration = 0,
  mergedClipBoundaries = [],
  muteOriginal = false,
  captionX = 50,
  captionY = 85,
  onCaptionMove,
  brightness = 100,
  contrast = 100,
  saturation = 100,
  transition = "fade",
  aspectRatio = "original",
  cropOffset = 50,
  exportMode = false,
  onClose,
}: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const isPlayingRef = useRef<boolean>(false);
  const transitionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // ✅ Use ref for transition state in timeUpdate to avoid stale closure
  const transitionActiveRef = useRef(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(trimStart);
  const [duration, setDuration] = useState(0);
  const [currentClipIndex, setCurrentClipIndex] = useState(0);
  const [fadeOpacity, setFadeOpacity] = useState(0);
  const [transitionState, setTransitionState] = useState<TransitionState>("idle");

  const isClipsMode = clips.length > 0;
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);

  const ensureAudioContext = useCallback(() => {
    if (!audioRef.current || audioCtxRef.current) return;
    try {
      const ctx = new AudioContext();
      const track = ctx.createMediaElementSource(audioRef.current);
      const gain = ctx.createGain();
      gain.gain.value = 1;
      track.connect(gain).connect(ctx.destination);
      audioCtxRef.current = ctx;
      gainNodeRef.current = gain;
    } catch (e) { console.warn("AudioContext setup failed:", e); }
  }, []);

  useEffect(() => {
    if (isClipsMode || !videoRef.current) return;
    const v = videoRef.current;
    setIsPlaying(false); isPlayingRef.current = false;
    setCurrentTime(trimStart); setFadeOpacity(0);
    setTransitionState("idle"); transitionActiveRef.current = false;
    if (videoUrl) { v.src = videoUrl; v.load(); v.currentTime = trimStart; }
    else { v.src = ""; v.load(); }
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0; }
  }, [videoUrl, trimStart, isClipsMode]);

  const MAX_FADE_SEC = 0.5;
  const updateVisualFade = useCallback((t: number, clipStart: number, clipEnd: number, clipFadeIn?: number, clipFadeOut?: number) => {
    if (!enableFades) { setFadeOpacity(0); return; }
    const fi = Math.min(clipFadeIn ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC);
    const fo = Math.min(clipFadeOut ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC);
    if (fi === 0 && fo === 0) { setFadeOpacity(0); return; }
    const localT = t - clipStart;
    const localRemaining = clipEnd - t;
    let opacity = 0;
    if (fi > 0 && localT >= 0 && localT < fi) opacity = Math.max(opacity, 1 - (localT / fi));
    if (fo > 0 && localRemaining >= 0 && localRemaining < fo) opacity = Math.max(opacity, 1 - (localRemaining / fo));
    setFadeOpacity(Math.min(1, Math.max(0, opacity)));
  }, [enableFades, fadeDuration]);

  const applyAudioFade = useCallback((t: number, clipStart: number, clipEnd: number, clipFadeIn?: number, clipFadeOut?: number) => {
    if (!enableFades || !gainNodeRef.current || !audioCtxRef.current) return;
    const gain = gainNodeRef.current;
    const ctx = audioCtxRef.current;
    const fi = Math.min(clipFadeIn ?? (fadeDuration > 0 ? fadeDuration : 0), 0.5);
    const fo = Math.min(clipFadeOut ?? (fadeDuration > 0 ? fadeDuration : 0), 0.5);
    if (fi === 0 && fo === 0) { gain.gain.setTargetAtTime(1, ctx.currentTime, 0.05); return; }
    const localT = t - clipStart;
    const localRemaining = clipEnd - t;
    const now = ctx.currentTime;
    if (fi > 0 && localT >= 0 && localT < fi) gain.gain.setTargetAtTime(Math.max(0, localT / fi), now, 0.05);
    else if (fo > 0 && localRemaining >= 0 && localRemaining < fo) gain.gain.setTargetAtTime(Math.max(0, localRemaining / fo), now, 0.05);
    else gain.gain.setTargetAtTime(1, now, 0.05);
  }, [enableFades, fadeDuration]);

  const syncMusic = useCallback((videoT: number, playing: boolean) => {
    const audio = audioRef.current;
    if (!audio || music === "None") return;
    const start = musicStart ?? 0;
    const inWindow = videoT >= start && (musicEnd === undefined || videoT < musicEnd);
    if (!inWindow || !playing) { if (!audio.paused) audio.pause(); return; }
    const offset = videoT - start;
    const audioDur = audio.duration;
    if (audioDur && audioDur > 0) {
      const expectedPos = offset % audioDur;
      if (Math.abs(audio.currentTime - expectedPos) > 0.3) audio.currentTime = expectedPos;
    }
    if (audio.paused) audio.play().catch(() => {});
  }, [music, musicStart, musicEnd]);

  useEffect(() => {
    const audio = audioRef.current;
    const video = videoRef.current;
    if (!audio || !video || music === "None") return;
    const offset = video.currentTime - (musicStart ?? 0);
    const audioDur = audio.duration;
    if (audioDur > 0 && offset >= 0) audio.currentTime = offset % audioDur;
  }, [playbackRate, music, musicStart]);

  // ── Fire a transition (used for clip-sequencing mode) ────────────────────
  const fireTransition = useCallback((onMidpoint: () => void) => {
    if (transition === "none") { onMidpoint(); return; }
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    const outDur = transition === "flash" ? 80 : transition === "glitch" ? 120 : 300;
    transitionActiveRef.current = true;
    setTransitionState("out");
    transitionTimerRef.current = setTimeout(() => {
      onMidpoint();
      setTransitionState("in");
      transitionTimerRef.current = setTimeout(() => {
        setTransitionState("idle");
        transitionActiveRef.current = false;
      }, outDur);
    }, outDur);
  }, [transition]);

  const playClip = useCallback((index: number) => {
    if (!clips[index] || !videoRef.current) return;
    const clip = clips[index];
    const v = videoRef.current;
    fireTransition(() => {
      v.pause(); v.src = clip.url; v.load();
      const handleCanPlay = () => {
        v.removeEventListener("canplay", handleCanPlay);
        v.currentTime = clip.trimStart ?? 0;
        v.play().catch(() => {});
        setIsPlaying(true); isPlayingRef.current = true;
      };
      v.addEventListener("canplay", handleCanPlay);
      setCurrentClipIndex(index); setFadeOpacity(0);
    });
  }, [clips, fireTransition]);

  // ── Fire a transition in merged-video mode at clip boundaries ────────────
  const fireMergedTransition = useCallback(() => {
    if (transition === "none" || transitionActiveRef.current) return;
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    const outDur = transition === "flash" ? 80 : transition === "glitch" ? 120 : 300;
    transitionActiveRef.current = true;
    setTransitionState("out");
    transitionTimerRef.current = setTimeout(() => {
      setTransitionState("in");
      transitionTimerRef.current = setTimeout(() => {
        setTransitionState("idle");
        transitionActiveRef.current = false;
      }, outDur);
    }, outDur);
  }, [transition]);

  const handleTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    setCurrentTime(t);

    if (isClipsMode) {
      const clip = clips[currentClipIndex];
      if (!clip) return;
      const clipStart = clip.trimStart ?? 0;
      const clipEnd = clip.trimEnd ?? v.duration;
      updateVisualFade(t, clipStart, clipEnd, clip.fadeIn, clip.fadeOut);
      applyAudioFade(t, clipStart, clipEnd, clip.fadeIn, clip.fadeOut);
      syncMusic(t, isPlayingRef.current);
      if (t >= clipEnd - 0.05) {
        const next = currentClipIndex + 1;
        if (next < clips.length) playClip(next);
        else {
          v.pause(); setIsPlaying(false); isPlayingRef.current = false;
          if (audioRef.current) audioRef.current.pause();
          setFadeOpacity(0); setTransitionState("idle"); transitionActiveRef.current = false;
        }
      }
    } else {
      if (mergedClipBoundaries.length > 0) {
        const boundary = mergedClipBoundaries.find(b => t >= b.globalStart && t < b.globalEnd)
          ?? mergedClipBoundaries[mergedClipBoundaries.length - 1];
        updateVisualFade(t, boundary.globalStart, boundary.globalEnd, boundary.fadeIn, boundary.fadeOut);
        applyAudioFade(t, boundary.globalStart, boundary.globalEnd, boundary.fadeIn, boundary.fadeOut);

        // ✅ FIXED: use ref instead of state to avoid stale closure
        // Trigger transition ~100ms before each clip boundary
        const nearBoundary = mergedClipBoundaries.find(
          b => b.globalEnd > 0 && t >= b.globalEnd - 0.12 && t < b.globalEnd
        );
        if (nearBoundary) {
          fireMergedTransition();
        }
      } else {
        updateVisualFade(t, trimStart, trimEnd ?? v.duration, undefined, undefined);
        applyAudioFade(t, trimStart, trimEnd ?? v.duration, undefined, undefined);
      }
      syncMusic(t, isPlayingRef.current);
      if (trimEnd !== undefined && t >= trimEnd) {
        v.pause(); setIsPlaying(false); isPlayingRef.current = false;
        if (audioRef.current) audioRef.current.pause();
        setFadeOpacity(0); setTransitionState("idle"); transitionActiveRef.current = false;
      }
    }
  }, [
    isClipsMode, clips, currentClipIndex, trimStart, trimEnd, mergedClipBoundaries,
    updateVisualFade, applyAudioFade, syncMusic, playClip, fireMergedTransition,
  ]);

  const handleEnded = useCallback(() => {
    if (isClipsMode) {
      const next = currentClipIndex + 1;
      if (next < clips.length) playClip(next);
      else {
        setIsPlaying(false); isPlayingRef.current = false;
        if (audioRef.current) audioRef.current.pause();
        setFadeOpacity(0); setTransitionState("idle"); transitionActiveRef.current = false;
      }
    } else {
      setIsPlaying(false); isPlayingRef.current = false;
      if (audioRef.current) audioRef.current.pause();
      setFadeOpacity(0); setTransitionState("idle"); transitionActiveRef.current = false;
    }
  }, [isClipsMode, clips.length, currentClipIndex, playClip]);

  const handleLoadedMetadata = useCallback(() => {
    if (!videoRef.current) return;
    setDuration(videoRef.current.duration);
    if (trimStart > 0) videoRef.current.currentTime = trimStart;
  }, [trimStart]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (audioCtxRef.current?.state === "suspended") audioCtxRef.current.resume();
    if (isPlayingRef.current) {
      v.pause(); if (audioRef.current) audioRef.current.pause();
      setIsPlaying(false); isPlayingRef.current = false;
    } else {
      ensureAudioContext();
      if (isClipsMode && clips.length > 0 && !v.src) playClip(0);
      else { v.play().catch(() => {}); setIsPlaying(true); isPlayingRef.current = true; }
    }
  }, [isClipsMode, clips, ensureAudioContext, playClip]);

  const toggleMute = useCallback(() => {
    setIsMuted(m => {
      const next = !m;
      // Only mute the original video audio — background music is unaffected
      if (videoRef.current) videoRef.current.muted = next;
      return next;
    });
  }, []);

  const handleSeek = useCallback((value: number[]) => {
    const v = videoRef.current;
    if (!v) return;
    const seekTo = value[0];
    v.currentTime = seekTo; setCurrentTime(seekTo);
    const audio = audioRef.current;
    if (audio && music !== "None") {
      const start = musicStart ?? 0;
      const inWindow = seekTo >= start && (musicEnd === undefined || seekTo < musicEnd);
      if (!inWindow) audio.pause();
      else {
        const offset = seekTo - start;
        const audioDur = audio.duration;
        audio.currentTime = audioDur > 0 ? offset % audioDur : offset;
        if (isPlayingRef.current) audio.play().catch(() => {});
      }
    }
  }, [music, musicStart, musicEnd]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
    if (audioRef.current) audioRef.current.playbackRate = 1.0;
  }, [playbackRate]);

  const handleDownload = useCallback(() => {
    const src = videoUrl || videoRef.current?.src;
    if (!src) return;
    const a = document.createElement("a");
    a.href = src; a.download = "zync-output.mp4";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }, [videoUrl]);

  useEffect(() => () => {
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
  }, []);

  const formatTime = (s: number) =>
    `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`;

  const effectiveDuration = trimEnd ?? duration;
  const baseFilter = FILTER_STYLES[filter] ?? "none";
  const bcFilter = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturation}%)`;
  const filterCss = baseFilter === "none" ? bcFilter : `${baseFilter} ${bcFilter}`;

  const isRatio = aspectRatio !== "original";
  const aspectWrapStyle: React.CSSProperties = isRatio ? {
    position: "relative", width: "100%", overflow: "hidden",
    ...(aspectRatio === "16:9" ? { aspectRatio: "16/9" }
      : aspectRatio === "9:16" ? { aspectRatio: "9/16" }
      : { aspectRatio: "1/1" }),
  } : {};
  const objectPos =
    aspectRatio === "16:9" ? `50% ${cropOffset}%`
    : aspectRatio === "9:16" ? `${cropOffset}% 50%`
    : `${cropOffset}% ${cropOffset}%`;
  const aspectStyle: React.CSSProperties = isRatio
    ? { width: "100%", height: "100%", objectFit: "cover", objectPosition: objectPos }
    : { maxWidth: "100%", maxHeight: "100%" };
  const hasVideo = !!(videoUrl || (isClipsMode && clips.length > 0));
  const transitionOverlayStyle = getTransitionOverlayStyle(transition, transitionState);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between px-4 py-3 border-b border-border items-center">
        <h2 className="font-semibold text-foreground">{title}</h2>
        {exportMode && (
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" onClick={handleDownload}><Download className="w-5 h-5" /></Button>
            <Button variant="ghost" size="icon" onClick={onClose}><X className="w-5 h-5" /></Button>
          </div>
        )}
      </div>

      <div className="flex-1 bg-background/50 flex items-center justify-center overflow-hidden">
        {hasVideo ? (
          <div className="relative w-full h-full flex items-center justify-center">

            {/* Fade overlay */}
            <div className="absolute inset-0 bg-black pointer-events-none z-10"
              style={{ opacity: fadeOpacity, transition: "opacity 0.08s linear" }} />

            {/* ✅ Transition overlay */}
            <div style={transitionOverlayStyle} />

            <div style={aspectWrapStyle}>
              <video ref={videoRef}
                className={isRatio ? "" : "max-w-full max-h-full object-contain"}
                style={{ filter: filterCss, ...aspectStyle }}
                muted={muteOriginal}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                onClick={togglePlay}
                playsInline preload="auto" />
            </div>

            {music !== "None" && (
              <audio ref={audioRef}
                src={`/music/${music.toLowerCase().replace(/ /g, "_")}.mp3`}
                preload="auto" loop={false} />
            )}

            {!isPlaying && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/30 cursor-pointer z-20"
                onClick={togglePlay}>
                <div className="w-16 h-16 rounded-full bg-primary/90 flex items-center justify-center shadow-lg hover:scale-110 transition-transform">
                  <Play className="w-8 h-8 text-primary-foreground ml-1" />
                </div>
              </div>
            )}

            {overlayText && onCaptionMove && (
              <DraggableCaption text={overlayText} x={captionX} y={captionY} onMove={onCaptionMove} filter={filter} />
            )}
            {overlayText && !onCaptionMove && (
              <div className="absolute z-30 pointer-events-none"
                style={{ left: `${captionX}%`, top: `${captionY}%`, transform: "translate(-50%, -50%)" }}>
                <div className="px-3 py-1 rounded-lg max-w-[240px] text-center"
                  style={{
                    background: getCaptionStyle(filter).bg,
                    backdropFilter: "blur(10px)",
                    border: `1px solid ${getCaptionStyle(filter).border}`,
                    boxShadow: getCaptionStyle(filter).shadow,
                  }}>
                  <span style={{
                      color: getCaptionStyle(filter).text,
                      textShadow: "0 1px 4px rgba(0,0,0,0.8)",
                      fontFamily: getCaptionStyle(filter).font,
                      fontWeight: getCaptionStyle(filter).weight,
                      fontStyle: getCaptionStyle(filter).style,
                      fontSize: getCaptionStyle(filter).size,
                      letterSpacing: getCaptionStyle(filter).letterSpacing,
                      lineHeight: "1.3",
                    }}>
                    {overlayText}
                  </span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center">
            <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
              <Play className="w-10 h-10 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground">No video selected</p>
          </div>
        )}
      </div>

      {hasVideo && (
        <div className="p-4 border-t border-border space-y-3">
          <Slider value={[currentTime]} onValueChange={handleSeek}
            min={trimStart} max={effectiveDuration || 1} step={0.1} className="w-full" />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={togglePlay} className="hover:bg-primary/10 hover:text-primary">
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </Button>
              <Button variant="ghost" size="icon" onClick={toggleMute} className="hover:bg-primary/10 hover:text-primary">
                {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </Button>
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              {formatTime(currentTime)} / {formatTime(effectiveDuration)}
            </span>
            <Button variant="ghost" size="icon" className="hover:bg-primary/10 hover:text-primary">
              <Maximize className="w-5 h-5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
