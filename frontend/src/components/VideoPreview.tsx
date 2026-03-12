import { useRef, useState, useEffect, useCallback } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize, Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

export interface Clip {
  url: string;
  trimStart?: number;
  trimEnd?: number;
  title?: string;
  fadeIn?: number;   // seconds, only used when enableFades=true
  fadeOut?: number;  // seconds, only used when enableFades=true
}

/** One entry per clip in the merged video — used for fade timing */
export interface MergedClipBoundary {
  globalStart: number;  // seconds into merged video where this clip starts
  globalEnd: number;    // seconds into merged video where this clip ends
  fadeIn: number;       // seconds of fade-in at clip start (0 = none)
  fadeOut: number;      // seconds of fade-out at clip end  (0 = none)
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
  /** For merged-video mode: tells where each original clip sits in the merged file */
  mergedClipBoundaries?: MergedClipBoundary[];
  muteOriginal?: boolean;
  captionX?: number;         // 0-100 percent from left
  captionY?: number;         // 0-100 percent from top
  onCaptionMove?: (x: number, y: number) => void;
  brightness?: number;   // percent 50-150, 100 = normal
  contrast?: number;     // percent 50-150, 100 = normal
  aspectRatio?: "16:9" | "9:16" | "1:1" | "original";
  cropOffset?: number; // 0-100, 50 = center
  exportMode?: boolean;
  onClose?: () => void;
}

const FILTER_STYLES: Record<string, string> = {
  None: "none",
  Warm: "brightness(1.1) saturate(1.2) sepia(0.2)",
  Cinematic: "contrast(1.2) saturate(1.3) brightness(0.9)",
  "Black & White": "grayscale(100%) contrast(1.2)",
};

// ── Draggable caption ────────────────────────────────────────────────────────
function DraggableCaption({
  text, x, y, onMove,
}: { text: string; x: number; y: number; onMove: (x: number, y: number) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const getParent = () => containerRef.current?.parentElement;

  const toPercent = (clientX: number, clientY: number) => {
    const parent = getParent();
    if (!parent) return { x, y };
    const rect = parent.getBoundingClientRect();
    return {
      x: Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width)  * 100)),
      y: Math.min(100, Math.max(0, ((clientY - rect.top)  / rect.height) * 100)),
    };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
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
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      className="absolute z-30 cursor-grab active:cursor-grabbing select-none px-2"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        transform: "translate(-50%, -50%)",
        textShadow: "0 1px 4px rgba(0,0,0,0.9), 0 0 8px rgba(0,0,0,0.6)",
      }}
    >
      <span className="text-white text-base font-bold">{text}</span>
      {/* Subtle drag handle indicator */}
      <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-[9px] text-white/60 whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none">
        drag
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

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(trimStart);
  const [duration, setDuration] = useState(0);
  const [currentClipIndex, setCurrentClipIndex] = useState(0);
  const [fadeOpacity, setFadeOpacity] = useState(0);

  const isClipsMode = clips.length > 0;

  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);

  // ─── Audio context — created once on first play ───────────────────────────
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
    } catch (e) {
      console.warn("AudioContext setup failed:", e);
    }
  }, []);

  // ─── Apply videoUrl in single-video mode ──────────────────────────────────
  useEffect(() => {
    if (isClipsMode || !videoRef.current) return;
    const v = videoRef.current;
    setIsPlaying(false);
    isPlayingRef.current = false;
    setCurrentTime(trimStart);
    setFadeOpacity(0);
    if (videoUrl) {
      v.src = videoUrl;
      v.load();
      v.currentTime = trimStart;
    } else {
      v.src = "";
      v.load();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [videoUrl, trimStart, isClipsMode]);

  // ─── Visual fade ──────────────────────────────────────────────────────────
  // t = raw video currentTime. clipStart/clipEnd = trimStart/trimEnd.
  // localT = seconds elapsed since clip start (from trimStart).
  // localRemaining = seconds left until clip end (to trimEnd).
  // Cap fade at 0.5s max so 3s clips aren't mostly fading.
  const MAX_FADE_SEC = 0.5;
  const updateVisualFade = useCallback((
    t: number,
    clipStart: number,
    clipEnd: number,
    clipFadeIn?: number,
    clipFadeOut?: number,
  ) => {
    if (!enableFades) { setFadeOpacity(0); return; }
    // Use per-clip value, or fallback, capped at 0.5s
    const fi = Math.min(clipFadeIn ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC);
    const fo = Math.min(clipFadeOut ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC);
    if (fi === 0 && fo === 0) { setFadeOpacity(0); return; }
    // localT counts from trimStart (not from 0)
    const localT = t - clipStart;
    const localRemaining = clipEnd - t;
    let opacity = 0;
    if (fi > 0 && localT >= 0 && localT < fi)
      opacity = Math.max(opacity, 1 - (localT / fi));
    if (fo > 0 && localRemaining >= 0 && localRemaining < fo)
      opacity = Math.max(opacity, 1 - (localRemaining / fo));
    setFadeOpacity(Math.min(1, Math.max(0, opacity)));
  }, [enableFades, fadeDuration]);

  // ─── Audio gain fade ──────────────────────────────────────────────────────
  const MAX_FADE_SEC_AUDIO = 0.5;
  const applyAudioFade = useCallback((
    t: number,
    clipStart: number,
    clipEnd: number,
    clipFadeIn?: number,
    clipFadeOut?: number,
  ) => {
    if (!enableFades || !gainNodeRef.current || !audioCtxRef.current) return;
    const gain = gainNodeRef.current;
    const ctx = audioCtxRef.current;
    const fi = Math.min(clipFadeIn ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC_AUDIO);
    const fo = Math.min(clipFadeOut ?? (fadeDuration > 0 ? fadeDuration : 0), MAX_FADE_SEC_AUDIO);
    if (fi === 0 && fo === 0) { gain.gain.setTargetAtTime(1, ctx.currentTime, 0.05); return; }
    const localT = t - clipStart;
    const localRemaining = clipEnd - t;
    const now = ctx.currentTime;
    if (fi > 0 && localT >= 0 && localT < fi) {
      gain.gain.setTargetAtTime(Math.max(0, localT / fi), now, 0.05);
    } else if (fo > 0 && localRemaining >= 0 && localRemaining < fo) {
      gain.gain.setTargetAtTime(Math.max(0, localRemaining / fo), now, 0.05);
    } else {
      gain.gain.setTargetAtTime(1, now, 0.05);
    }
  }, [enableFades, fadeDuration]);

  // ─── Music sync ───────────────────────────────────────────────────────────
  // Called every timeUpdate. Keeps music locked to video position.
  // Music position = videoTime - musicStart (mod audio duration).
  // Music stops when: video pauses, video ends, or videoTime >= musicEnd.
  const syncMusic = useCallback((videoT: number, playing: boolean) => {
    const audio = audioRef.current;
    if (!audio || music === "None") return;

    const start = musicStart ?? 0;
    const inWindow = videoT >= start && (musicEnd === undefined || videoT < musicEnd);

    if (!inWindow || !playing) {
      if (!audio.paused) audio.pause();
      return;
    }

    // Expected position within music file
    const offset = videoT - start;
    const audioDur = audio.duration;
    if (audioDur && audioDur > 0) {
      const expectedPos = offset % audioDur;
      const drift = Math.abs(audio.currentTime - expectedPos);
      if (drift > 0.3) audio.currentTime = expectedPos;
    }

    if (audio.paused) audio.play().catch(() => {});
  }, [music, musicStart, musicEnd]);

  // ─── Clip sequencing ─────────────────────────────────────────────────────
  const playClip = useCallback((index: number) => {
    if (!clips[index] || !videoRef.current) return;
    const clip = clips[index];
    const v = videoRef.current;
    v.pause();
    v.src = clip.url;
    v.load();
    const handleCanPlay = () => {
      v.removeEventListener("canplay", handleCanPlay);
      v.currentTime = clip.trimStart ?? 0;
      v.play().catch(() => {});
      setIsPlaying(true);
      isPlayingRef.current = true;
      // Music continues uninterrupted — syncMusic handles position
    };
    v.addEventListener("canplay", handleCanPlay);
    setCurrentClipIndex(index);
    setFadeOpacity(0);
  }, [clips]);

  // ─── Time update ──────────────────────────────────────────────────────────
  const handleTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    setCurrentTime(t);

    if (isClipsMode) {
      const clip = clips[currentClipIndex];
      if (!clip) return;
      const clipStart = clip.trimStart ?? 0;
      const clipEnd   = clip.trimEnd   ?? v.duration;

      updateVisualFade(t, clipStart, clipEnd, clip.fadeIn, clip.fadeOut);
      applyAudioFade(t, clipStart, clipEnd, clip.fadeIn, clip.fadeOut);
      syncMusic(t, isPlayingRef.current);

      if (t >= clipEnd - 0.05) {
        const next = currentClipIndex + 1;
        if (next < clips.length) {
          playClip(next);
        } else {
          v.pause();
          setIsPlaying(false);
          isPlayingRef.current = false;
          if (audioRef.current) audioRef.current.pause();
          setFadeOpacity(0);
        }
      }
    } else {
      // Single/merged video mode
      // If mergedClipBoundaries are provided, find which clip boundary we're in
      // and apply its fade values. Otherwise fall back to whole-video fade.
      if (mergedClipBoundaries.length > 0) {
        const boundary = mergedClipBoundaries.find(
          b => t >= b.globalStart && t < b.globalEnd
        ) ?? mergedClipBoundaries[mergedClipBoundaries.length - 1];
        updateVisualFade(t, boundary.globalStart, boundary.globalEnd, boundary.fadeIn, boundary.fadeOut);
        applyAudioFade(t, boundary.globalStart, boundary.globalEnd, boundary.fadeIn, boundary.fadeOut);
      } else {
        const clipStart = trimStart;
        const clipEnd   = trimEnd ?? v.duration;
        updateVisualFade(t, clipStart, clipEnd, undefined, undefined);
        applyAudioFade(t, clipStart, clipEnd, undefined, undefined);
      }
      syncMusic(t, isPlayingRef.current);

      if (trimEnd !== undefined && t >= trimEnd) {
        v.pause();
        setIsPlaying(false);
        isPlayingRef.current = false;
        if (audioRef.current) audioRef.current.pause();
        setFadeOpacity(0);
      }
    }
  }, [
    isClipsMode, clips, currentClipIndex,
    trimStart, trimEnd, mergedClipBoundaries,
    updateVisualFade, applyAudioFade, syncMusic, playClip,
  ]);

  // ─── Video ended naturally ────────────────────────────────────────────────
  const handleEnded = useCallback(() => {
    if (isClipsMode) {
      const next = currentClipIndex + 1;
      if (next < clips.length) {
        playClip(next);
      } else {
        setIsPlaying(false);
        isPlayingRef.current = false;
        if (audioRef.current) audioRef.current.pause();
        setFadeOpacity(0);
      }
    } else {
      setIsPlaying(false);
      isPlayingRef.current = false;
      if (audioRef.current) audioRef.current.pause();
      setFadeOpacity(0);
    }
  }, [isClipsMode, clips.length, currentClipIndex, playClip]);

  // ─── Metadata ────────────────────────────────────────────────────────────
  const handleLoadedMetadata = useCallback(() => {
    if (!videoRef.current) return;
    setDuration(videoRef.current.duration);
    if (trimStart > 0) videoRef.current.currentTime = trimStart;
  }, [trimStart]);

  // ─── Play / Pause ─────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (audioCtxRef.current?.state === "suspended") audioCtxRef.current.resume();

    if (isPlayingRef.current) {
      v.pause();
      if (audioRef.current) audioRef.current.pause();
      setIsPlaying(false);
      isPlayingRef.current = false;
    } else {
      ensureAudioContext();
      if (isClipsMode && clips.length > 0 && !v.src) {
        playClip(0);
      } else {
        v.play().catch(() => {});
        setIsPlaying(true);
        isPlayingRef.current = true;
        // syncMusic will fire on next timeUpdate
      }
    }
  }, [isClipsMode, clips, ensureAudioContext, playClip]);

  // ─── Mute ────────────────────────────────────────────────────────────────
  const toggleMute = useCallback(() => {
    setIsMuted(m => {
      const next = !m;
      if (videoRef.current) videoRef.current.muted = next;
      if (audioRef.current) audioRef.current.muted = next;
      return next;
    });
  }, []);

  // ─── Seek ────────────────────────────────────────────────────────────────
  const handleSeek = useCallback((value: number[]) => {
    const v = videoRef.current;
    if (!v) return;
    const seekTo = value[0];
    v.currentTime = seekTo;
    setCurrentTime(seekTo);

    const audio = audioRef.current;
    if (audio && music !== "None") {
      const start = musicStart ?? 0;
      const inWindow = seekTo >= start && (musicEnd === undefined || seekTo < musicEnd);
      if (!inWindow) {
        audio.pause();
      } else {
        const offset = seekTo - start;
        const audioDur = audio.duration;
        audio.currentTime = audioDur > 0 ? offset % audioDur : offset;
        if (isPlayingRef.current) audio.play().catch(() => {});
      }
    }
  }, [music, musicStart, musicEnd]);

  // ─── Playback rate — video only, music always stays at 1x ──────────────
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
    // Never change music playback rate — it should always play at normal speed
    if (audioRef.current) audioRef.current.playbackRate = 1.0;
  }, [playbackRate]);

  // ─── Download ────────────────────────────────────────────────────────────
  const handleDownload = useCallback(() => {
    const src = videoUrl || videoRef.current?.src;
    if (!src) return;
    const a = document.createElement("a");
    a.href = src;
    a.download = "zync-output.mp4";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [videoUrl]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const effectiveDuration = trimEnd ?? duration;
  const baseFilter = FILTER_STYLES[filter] ?? "none";
  // Compose brightness/contrast on top of named filter
  const bcFilter = `brightness(${brightness}%) contrast(${contrast}%)`;
  const filterCss = baseFilter === "none" ? bcFilter : `${baseFilter} ${bcFilter}`;

  // Aspect ratio crop preview
  // We wrap the video in a container with the target ratio + overflow:hidden,
  // then size the video LARGER than the container and shift it via margin/transform
  // so the visible portion matches the crop offset.
  const isRatio = aspectRatio !== "original";
  const aspectWrapStyle: React.CSSProperties = isRatio ? {
    position: "relative",
    width: "100%",
    overflow: "hidden",
    ...(aspectRatio === "16:9" ? { aspectRatio: "16/9" } :
        aspectRatio === "9:16" ? { aspectRatio: "9/16"  } :
        aspectRatio === "1:1"  ? { aspectRatio: "1/1"   } : {}),
  } : {};
  // objectPosition: for 16:9 pan vertically, for 9:16 pan horizontally
  const objectPos =
    aspectRatio === "16:9" ? `50% ${cropOffset}%` :
    aspectRatio === "9:16" ? `${cropOffset}% 50%` :
    aspectRatio === "1:1"  ? `${cropOffset}% ${cropOffset}%` : "50% 50%";
  const aspectStyle: React.CSSProperties = isRatio
    ? { width: "100%", height: "100%", objectFit: "cover", objectPosition: objectPos }
    : { maxWidth: "100%", maxHeight: "100%" };
  const hasVideo = !!(videoUrl || (isClipsMode && clips.length > 0));

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex justify-between px-4 py-3 border-b border-border items-center">
        <h2 className="font-semibold text-foreground">{title}</h2>
        {exportMode && (
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" onClick={handleDownload} title="Download">
              <Download className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose} title="Close">
              <X className="w-5 h-5" />
            </Button>
          </div>
        )}
      </div>

      {/* Video area */}
      <div className="flex-1 bg-background/50 flex items-center justify-center overflow-hidden">
        {hasVideo ? (
          <>
            {/* Inner wrapper — position:relative so caption absolute coords work correctly */}
            <div className="relative w-full h-full flex items-center justify-center">
              {/* Fade overlay */}
              <div
                className="absolute inset-0 bg-black pointer-events-none z-10"
                style={{ opacity: fadeOpacity, transition: "opacity 0.08s linear" }}
              />

              <div style={aspectWrapStyle}>
                <video
                  ref={videoRef}
                  className={isRatio ? "" : "max-w-full max-h-full object-contain"}
                  style={{ filter: filterCss, ...aspectStyle }}
                  muted={muteOriginal}
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onEnded={handleEnded}
                  onClick={togglePlay}
                  playsInline
                  preload="auto"
                />
              </div>

              {music !== "None" && (
                <audio
                  ref={audioRef}
                  src={`/music/${music.toLowerCase().replace(/ /g, "_")}.mp3`}
                  preload="auto"
                  loop={false}
                />
              )}

              {!isPlaying && (
                <div
                  className="absolute inset-0 flex items-center justify-center bg-background/30 cursor-pointer z-20"
                  onClick={togglePlay}
                >
                  <div className="w-16 h-16 rounded-full bg-primary/90 flex items-center justify-center shadow-lg hover:scale-110 transition-transform">
                    <Play className="w-8 h-8 text-primary-foreground ml-1" />
                  </div>
                </div>
              )}

              {/* Draggable caption overlay */}
              {overlayText && onCaptionMove && (
                <DraggableCaption
                  text={overlayText}
                  x={captionX}
                  y={captionY}
                  onMove={onCaptionMove}
                />
              )}
              {/* Read-only caption (export modal / source preview) */}
              {overlayText && !onCaptionMove && (
                <div
                  className="absolute text-white text-base font-bold z-30 pointer-events-none px-2"
                  style={{
                    left: `${captionX}%`,
                    top: `${captionY}%`,
                    transform: "translate(-50%, -50%)",
                    textShadow: "0 1px 4px rgba(0,0,0,0.9)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {overlayText}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="text-center">
            <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
              <Play className="w-10 h-10 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground">No video selected</p>
          </div>
        )}
      </div>

      {/* Controls */}
      {hasVideo && (
        <div className="p-4 border-t border-border space-y-3">
          <Slider
            value={[currentTime]}
            onValueChange={handleSeek}
            min={trimStart}
            max={effectiveDuration || 1}
            step={0.1}
            className="w-full"
          />
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
