import { GripVertical, Trash2, Clock, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState, useRef, useEffect } from "react";

export interface TimelineClip  {
  id: string;
  name: string;
  duration: number;
  trimStart: number;
  trimEnd: number;
  videoUrl: string;
  prompt?: string;
  createdAt?: number;
  fadeIn?: number;   // ← ADD
  fadeOut?: number; 
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
  onSelectClip,
  onRemoveClip,
}: TimelineProps) {
  const totalDuration = clips.reduce(
    (sum, clip) => sum + (clip.trimEnd - clip.trimStart),
    0
  );

  const [modalClip, setModalClip] = useState<TimelineClip | null>(null);
  const modalVideoRef = useRef<HTMLVideoElement>(null);

  // Handle trimmed playback in modal
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

  // Group clips by prompt
  const groupedClips = clips.reduce((acc, clip) => {
    const key = clip.prompt || "Manual Clips";
    if (!acc[key]) acc[key] = [];
    acc[key].push(clip);
    return acc;
  }, {} as Record<string, TimelineClip[]>);

  return (
    <div className="h-full flex flex-col">
      {/* Timeline header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="font-semibold text-foreground">Timeline</h2>
        <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
          <Clock className="w-3 h-3" />
          Total: {formatDuration(totalDuration)}
        </div>
      </div>

      {/* Clips list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {clips.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <Clock className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground text-sm">
              Add clips to your timeline
            </p>
            <p className="text-muted-foreground/60 text-xs mt-1">
              Upload videos and add them here
            </p>
          </div>
        ) : (
          Object.entries(groupedClips).map(([promptGroup, groupClips]) => (
            <div key={promptGroup} className="space-y-2">
              {/* Group header */}
              <div className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">
                {promptGroup}
              </div>

              {/* Clips in group */}
              <div className="space-y-2">
                {groupClips.map((clip, index) => (
                  <div
                    key={clip.id}
                    onClick={() => setModalClip(clip)}
                    className={cn(
                      "group relative flex items-center gap-3 p-3 rounded-lg border transition-all duration-200 cursor-pointer",
                      selectedClipId === clip.id
                        ? "border-primary bg-primary/10 shadow-[0_0_15px_hsl(var(--primary)/0.2)]"
                        : "border-border bg-card hover:border-primary/30 hover:bg-card/80"
                    )}
                  >
                    {/* Drag handle */}
                    <div className="text-muted-foreground hover:text-foreground cursor-grab">
                      <GripVertical className="w-4 h-4" />
                    </div>

                    {/* Clip number */}
                    <div
                      className={cn(
                        "w-6 h-6 rounded flex items-center justify-center text-xs font-bold",
                        selectedClipId === clip.id
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      )}
                    >
                      {index + 1}
                    </div>

                    {/* Mini Preview */}
                    <div className="w-24 h-16 rounded overflow-hidden flex-shrink-0 bg-black">
                      <video
                        src={clip.videoUrl}
                        muted
                        autoPlay
                        className="w-full h-full object-cover"
                        onLoadedMetadata={(e) => {
                          e.currentTarget.currentTime = clip.trimStart;
                          e.currentTarget.play();
                        }}
                        onTimeUpdate={(e) => {
                          if (e.currentTarget.currentTime >= clip.trimEnd) {
                            e.currentTarget.currentTime = clip.trimStart;
                          }
                        }}
                      />
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {clip.name.replace(/\(\d+% match\)/g, "").trim()}
                      </p>
                      <p className="text-xs text-muted-foreground font-mono">
                        {formatDuration(clip.trimEnd - clip.trimStart)}
                      </p>
                    </div>

                    {/* Delete button */}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveClip(clip.id);
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Timeline ruler visualization */}
      {clips.length > 0 && (
        <div className="px-4 py-3 border-t border-border">
          <div className="h-8 rounded-lg bg-muted overflow-hidden flex">
            {clips.map((clip, index) => {
              const clipDuration = clip.trimEnd - clip.trimStart;
              const widthPercent = (clipDuration / totalDuration) * 100;
              return (
                <div
                  key={clip.id}
                  className={cn(
                    "h-full flex items-center justify-center text-xs font-mono transition-all",
                    selectedClipId === clip.id
                      ? "bg-primary text-primary-foreground"
                      : index % 2 === 0
                      ? "bg-primary/20 text-primary"
                      : "bg-primary/10 text-primary/80"
                  )}
                  style={{ width: `${widthPercent}%` }}
                >
                  {widthPercent > 10 && index + 1}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Modal for full clip */}
      {modalClip && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="relative w-[80%] max-w-4xl bg-card rounded-lg p-4 flex flex-col items-center">
            {/* Close Button */}
            <button
              className="absolute top-2 right-2 p-2 bg-gray-800/70 text-white rounded-full hover:bg-gray-800/90 z-50"
              onClick={() => setModalClip(null)}
            >
              ✕
            </button>

            {/* Full clip video */}
            <video
              ref={modalVideoRef}
              controls
              className="w-full max-h-[80vh] object-contain"
              src={modalClip.videoUrl}
            />
            <p className="mt-2 text-sm text-muted-foreground">
              {modalClip.name} |{" "}
              {formatDuration(modalClip.trimEnd - modalClip.trimStart)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
  
}



