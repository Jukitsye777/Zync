import { GripVertical, Trash2, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface TimelineClip {
  id: string;
  name: string;
  duration: number;
  trimStart: number;
  trimEnd: number;
  videoUrl: string;
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
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
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
          clips.map((clip, index) => (
            <div
              key={clip.id}
              onClick={() => onSelectClip(clip.id)}
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

              {/* Thumbnail */}
              <div className="w-16 h-10 rounded overflow-hidden bg-muted flex-shrink-0">
                <video
                  src={clip.videoUrl}
                  className="w-full h-full object-cover"
                  muted
                />
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {clip.name}
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
    </div>
  );
}

export type { TimelineClip };
