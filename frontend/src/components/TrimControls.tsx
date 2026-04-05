import { useState, useEffect, useRef, useCallback } from "react";
import { Scissors } from "lucide-react";

interface TrimControlsProps {
  duration: number;
  trimStart: number;
  trimEnd: number;
  videoUrl?: string | null;
  onTrimChange: (start: number, end: number) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

const THUMB_COUNT = 10;
const THUMB_W = 48;
const THUMB_H = 36;
const MIN_GAP = 0.1;

async function extractFrames(
  url: string,
  duration: number,
  count: number,
  thumbW: number,
  thumbH: number,
  signal: AbortSignal
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.src = url;
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.preload = "metadata";
    const canvas = document.createElement("canvas");
    canvas.width = thumbW;
    canvas.height = thumbH;
    const ctx = canvas.getContext("2d")!;
    const frames: string[] = [];
    let index = 0;
    const seekNext = () => {
      if (signal.aborted) { reject(new DOMException("Aborted")); return; }
      if (index >= count) { resolve(frames); return; }
      const t = (duration / count) * index + (duration / count) * 0.5;
      video.currentTime = Math.min(t, duration - 0.05);
    };
    video.addEventListener("seeked", () => {
      if (signal.aborted) { reject(new DOMException("Aborted")); return; }
      ctx.drawImage(video, 0, 0, thumbW, thumbH);
      frames.push(canvas.toDataURL("image/jpeg", 0.6));
      index++;
      seekNext();
    }, { passive: true });
    video.addEventListener("error", () => reject(new Error("Video load error")));
    video.addEventListener("loadedmetadata", seekNext);
    video.load();
  });
}

export function TrimControls({
  duration,
  trimStart,
  trimEnd,
  videoUrl,
  onTrimChange,
}: TrimControlsProps) {
  const [start, setStart] = useState(trimStart);
  const [end, setEnd]     = useState(trimEnd);
  const [frames, setFrames]       = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const trackRef   = useRef<HTMLDivElement>(null);
  const abortRef   = useRef<AbortController | null>(null);
  const dragging   = useRef<"start" | "end" | "range" | null>(null);
  const dragStartX = useRef(0);
  const dragStartVal = useRef({ start: 0, end: 0 });

  useEffect(() => { setStart(trimStart); setEnd(trimEnd); }, [trimStart, trimEnd]);

  useEffect(() => {
    if (!videoUrl || duration <= 0) { setFrames([]); return; }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsLoading(true);
    setFrames([]);
    extractFrames(videoUrl, duration, THUMB_COUNT, THUMB_W, THUMB_H, controller.signal)
      .then(f => { if (!controller.signal.aborted) { setFrames(f); setIsLoading(false); } })
      .catch(err => { if (err.name !== "AbortError") setIsLoading(false); });
    return () => controller.abort();
  }, [videoUrl, duration]);

  const pxToSec = useCallback((px: number) => {
    if (!trackRef.current) return 0;
    return (px / trackRef.current.clientWidth) * duration;
  }, [duration]);

  const onPointerDown = useCallback((
    e: React.PointerEvent,
    handle: "start" | "end" | "range"
  ) => {
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragging.current = handle;
    dragStartX.current = e.clientX;
    dragStartVal.current = { start, end };
  }, [start, end]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const delta = pxToSec(e.clientX - dragStartX.current);
    const { start: s0, end: e0 } = dragStartVal.current;
    let ns = s0, ne = e0;
    if (dragging.current === "start") {
      ns = Math.max(0, Math.min(s0 + delta, e0 - MIN_GAP));
    } else if (dragging.current === "end") {
      ne = Math.min(duration, Math.max(e0 + delta, s0 + MIN_GAP));
    } else {
      const span = e0 - s0;
      ns = Math.max(0, Math.min(s0 + delta, duration - span));
      ne = ns + span;
    }
    setStart(ns);
    setEnd(ne);
    onTrimChange(ns, ne);
  }, [pxToSec, duration, onTrimChange]);

  const onPointerUp = useCallback(() => { dragging.current = null; }, []);

  const leftPct  = duration > 0 ? (start / duration) * 100 : 0;
  const rightPct = duration > 0 ? (end   / duration) * 100 : 100;
  const trimmedDuration = end - start;
  const showStrip = !!(videoUrl && (frames.length > 0 || isLoading));

  return (
    <div className="space-y-3 p-4 rounded-xl bg-card border border-border select-none">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Scissors className="w-4 h-4 text-primary" />
          Trim Controls
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          {formatTime(trimmedDuration)}
          {isLoading && <span className="ml-2 opacity-50">extracting…</span>}
        </span>
      </div>

      {/* Track */}
      <div className="pt-5 pb-2">
        <div
          ref={trackRef}
          className="relative w-full"
          style={{ height: THUMB_H }}
        >
          {/* Thumbnail strip / fallback */}
          <div className="absolute inset-0 rounded-md overflow-hidden border border-border">
            {showStrip ? (
              <div className="absolute inset-0 flex">
                {isLoading
                  ? Array.from({ length: THUMB_COUNT }).map((_, i) => (
                      <div key={i} className="flex-1 bg-muted animate-pulse border-r border-black/10 last:border-0" />
                    ))
                  : frames.map((src, i) => (
                      <img key={i} src={src} alt=""
                        className="flex-1 object-cover border-r border-black/20 last:border-0"
                        style={{ minWidth: 0, height: THUMB_H }}
                        draggable={false}
                      />
                    ))
                }
              </div>
            ) : (
              <div className="absolute inset-0 bg-muted" />
            )}
            {/* Dim outside selection */}
            <div className="absolute inset-y-0 left-0 bg-black/55 pointer-events-none"
              style={{ width: `${leftPct}%` }} />
            <div className="absolute inset-y-0 right-0 bg-black/55 pointer-events-none"
              style={{ width: `${100 - rightPct}%` }} />
          </div>

          {/* Selected region — drag to move whole range */}
          <div
            className="absolute inset-y-0 border-2 border-primary rounded-sm cursor-grab active:cursor-grabbing z-10"
            style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
            onPointerDown={e => onPointerDown(e, "range")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          />

          {/* Left handle */}
          <div
            className="absolute z-20 cursor-ew-resize"
            style={{
              left: `${leftPct}%`,
              top: "50%",
              transform: "translate(-50%, -50%)",
            }}
            onPointerDown={e => onPointerDown(e, "start")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {/* Strip nub */}
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-primary"
              style={{ width: 6, height: THUMB_H, borderRadius: "3px 0 0 3px" }}
            />
            {/* Drag circle */}
            <div className="relative z-10 w-5 h-5 rounded-full bg-primary border-2 border-background shadow-lg flex items-center justify-center">
              <div className="flex gap-px">
                <div className="w-px h-2.5 bg-white/80 rounded-full" />
                <div className="w-px h-2.5 bg-white/80 rounded-full" />
              </div>
            </div>
          </div>

          {/* Right handle */}
          <div
            className="absolute z-20 cursor-ew-resize"
            style={{
              left: `${rightPct}%`,
              top: "50%",
              transform: "translate(-50%, -50%)",
            }}
            onPointerDown={e => onPointerDown(e, "end")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-primary"
              style={{ width: 6, height: THUMB_H, borderRadius: "0 3px 3px 0" }}
            />
            <div className="relative z-10 w-5 h-5 rounded-full bg-primary border-2 border-background shadow-lg flex items-center justify-center">
              <div className="flex gap-px">
                <div className="w-px h-2.5 bg-white/80 rounded-full" />
                <div className="w-px h-2.5 bg-white/80 rounded-full" />
              </div>
            </div>
          </div>

          {/* Timecode tooltips */}
          <div
            className="absolute -top-5 text-[10px] font-mono text-primary -translate-x-1/2 pointer-events-none whitespace-nowrap"
            style={{ left: `${leftPct}%` }}
          >
            {formatTime(start)}
          </div>
          <div
            className="absolute -top-5 text-[10px] font-mono text-primary -translate-x-1/2 pointer-events-none whitespace-nowrap"
            style={{ left: `${rightPct}%` }}
          >
            {formatTime(end)}
          </div>
        </div>
      </div>

      {/* Timecodes row */}
      <div className="flex justify-between text-xs font-mono">
        <span className="text-primary">{formatTime(start)}</span>
        <span className="text-muted-foreground/40">{formatTime(duration)}</span>
        <span className="text-primary">{formatTime(end)}</span>
      </div>
    </div>
  );
}
