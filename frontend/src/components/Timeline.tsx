import { GripVertical, Trash2, Clock, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState, useRef, useEffect } from "react";

export interface TimelineClip {
  id: string;
  name: string;
  duration: number;
  trimStart: number;
  trimEnd: number;
  videoUrl: string;
  prompt?: string;
  createdAt?: number;
  fadeIn?: number;
  fadeOut?: number;
  thumbnailUrl?: string;
}

interface TimelineProps {
  clips: TimelineClip[];
  selectedClipId: string | null;
  onSelectClip: (id: string) => void;
  onRemoveClip: (id: string) => void;
  onReorderClips: (clips: TimelineClip[]) => void;
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function Timeline({
  clips,
  selectedClipId,
  onRemoveClip,
}: TimelineProps) {
  const totalDuration = clips.reduce(
    (sum, clip) => sum + (clip.trimEnd - clip.trimStart),
    0
  );

  const [modalClip, setModalClip] = useState<TimelineClip | null>(null);
  const modalVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const videoEl = modalVideoRef.current;
    if (!videoEl || !modalClip) return;

    videoEl.currentTime = modalClip.trimStart;
    videoEl.play();

    const interval = setInterval(() => {
      if (videoEl.currentTime >= modalClip.trimEnd) {
        videoEl.currentTime = modalClip.trimStart;
      }
    }, 100);

    return () => {
      clearInterval(interval);
      videoEl.pause();
    };
  }, [modalClip]);

  const groupedClips = clips.reduce((acc, clip) => {
    const key = clip.prompt || "Manual Clips";
    if (!acc[key]) acc[key] = [];
    acc[key].push(clip);
    return acc;
  }, {} as Record<string, TimelineClip[]>);

  return (
    <div className="h-full flex flex-col bg-zinc-950">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/60">
        <h2 className="text-sm font-semibold text-zinc-200">Timeline</h2>
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-500 font-mono">
          <Clock className="w-3 h-3" />
          {formatDuration(totalDuration)}
          {clips.length > 0 && (
            <span className="ml-1 text-zinc-600">· {clips.length} clip{clips.length !== 1 ? "s" : ""}</span>
          )}
        </div>
      </div>

      {/* Clips list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {clips.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-zinc-800/60 border border-zinc-700/40 flex items-center justify-center mb-4">
              <Clock className="w-6 h-6 text-zinc-600" />
            </div>
            <p className="text-sm text-zinc-500 font-medium">No clips yet</p>
            <p className="text-xs text-zinc-600 mt-1">
              Upload videos and add them here, or use AI to find matching scenes
            </p>
          </div>
        ) : (
          Object.entries(groupedClips).map(([promptGroup, groupClips]) => (
            <div key={promptGroup} className="space-y-2">
              {/* Group header */}
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-violet-500 flex-shrink-0" />
                <span className="text-[10px] font-semibold text-violet-400 uppercase tracking-wider truncate">
                  {promptGroup}
                </span>
                <div className="flex-1 h-px bg-zinc-800/80" />
                <span className="text-[10px] text-zinc-600 flex-shrink-0">{groupClips.length}</span>
              </div>

              {/* Clips in group */}
              <div className="space-y-1.5">
                {groupClips.map((clip, index) => (
                  <div
                    key={clip.id}
                    onClick={() => setModalClip(clip)}
                    className={cn(
                      "group relative flex items-center gap-3 p-2.5 rounded-xl border transition-all duration-150 cursor-pointer",
                      selectedClipId === clip.id
                        ? "border-violet-500/40 bg-violet-500/8 shadow-[0_0_0_1px_rgba(139,92,246,0.15)]"
                        : "border-zinc-800/60 bg-zinc-900/60 hover:border-zinc-700/80 hover:bg-zinc-800/60"
                    )}
                  >
                    {/* Drag handle */}
                    <div className="text-zinc-700 hover:text-zinc-500 cursor-grab flex-shrink-0">
                      <GripVertical className="w-3.5 h-3.5" />
                    </div>

                    {/* Clip number */}
                    <div className={cn(
                      "w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold flex-shrink-0",
                      selectedClipId === clip.id
                        ? "bg-violet-600 text-white"
                        : "bg-zinc-700/60 text-zinc-400"
                    )}>
                      {index + 1}
                    </div>

                    {/* Mini video preview */}
                    <div className="w-20 h-12 rounded-lg overflow-hidden flex-shrink-0 bg-zinc-800 border border-zinc-700/40">
                      <div className="w-20 h-12 rounded-lg overflow-hidden flex-shrink-0 bg-zinc-800 border border-zinc-700/40">
  <video
    key={`${clip.id}-${clip.trimStart}-${clip.trimEnd}`}
    src={clip.videoUrl}
    muted
    preload="metadata"
    className="w-full h-full object-cover"
    style={{ visibility: "hidden" }}
    onLoadedMetadata={(e) => {
      const video = e.currentTarget;
      video.currentTime = clip.trimStart+0.1;
    }}
    onSeeked={(e) => {
      const video = e.currentTarget;
      video.pause();
      video.style.visibility = "visible";
    }}
  />
</div>
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0 space-y-0.5">
                      <p className="text-xs font-medium text-zinc-200 truncate leading-tight">
                        {clip.name.replace(/\(\d+% match\)/g, "").trim()}
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-zinc-500">
                          {formatDuration(clip.trimEnd - clip.trimStart)}
                        </span>
                        {((clip.fadeIn ?? 0) > 0 || (clip.fadeOut ?? 0) > 0) && (
                          <span className="text-[9px] bg-violet-950/60 border border-violet-800/40 text-violet-400 rounded px-1.5 py-0.5">
                            fades
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Delete button */}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-zinc-600 hover:text-red-400 hover:bg-red-900/20 rounded-lg"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveClip(clip.id);
                      }}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Timeline ruler */}
      {clips.length > 0 && (
        <div className="px-3 py-2.5 border-t border-zinc-800/60 bg-zinc-900/40">
          <div className="h-6 rounded-lg overflow-hidden flex gap-px">
            {clips.map((clip, index) => {
              const clipDuration = clip.trimEnd - clip.trimStart;
              const widthPercent = (clipDuration / totalDuration) * 100;
              return (
                <div
                  key={clip.id}
                  className={cn(
                    "h-full flex items-center justify-center text-[9px] font-mono transition-all rounded-sm",
                    selectedClipId === clip.id
                      ? "bg-violet-600 text-white"
                      : index % 2 === 0
                      ? "bg-violet-900/40 text-violet-400"
                      : "bg-zinc-800 text-zinc-500"
                  )}
                  style={{ width: `${widthPercent}%`, minWidth: "2px" }}
                >
                  {widthPercent > 8 ? index + 1 : ""}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Clip preview modal */}
      {modalClip && (
        <div className="fixed inset-0 bg-black/85 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="relative w-full max-w-3xl bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden shadow-2xl">
            <button
              className="absolute top-3 right-3 z-10 w-8 h-8 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 flex items-center justify-center transition-colors"
              onClick={() => setModalClip(null)}
            >
              <X className="w-4 h-4" />
            </button>

            <video
              ref={modalVideoRef}
              controls
              className="w-full max-h-[75vh] object-contain bg-black"
              src={modalClip.videoUrl}
            />

            <div className="px-4 py-3 flex items-center justify-between">
              <p className="text-sm text-zinc-300 font-medium truncate">
                {modalClip.name.replace(/\(\d+% match\)/g, "").trim()}
              </p>
              <span className="text-xs font-mono text-zinc-500 flex-shrink-0 ml-3">
                {formatDuration(modalClip.trimEnd - modalClip.trimStart)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
