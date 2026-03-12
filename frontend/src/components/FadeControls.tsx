/**
 * FadeControls.tsx
 * ────────────────
 * UI for adding / editing fade-in, fade-out, and crossfade between clips.
 * Emits updated clip array back to parent via onClipsChange.
 *
 * Usage in index.tsx:
 *   <FadeControls clips={timelineClips} onClipsChange={setTimelineClips} />
 */

import { useState } from "react";
import { Sunset, Sunrise, ArrowLeftRight, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { type TimelineClip } from "@/components/Timeline";

interface FadeControlsProps {
  clips: TimelineClip[];
  onClipsChange: (clips: TimelineClip[]) => void;
}

// How the fade data is stored on each clip
export interface ClipFade {
  fadeIn: number;    // seconds, 0 = none
  fadeOut: number;   // seconds, 0 = none
}

const MAX_FADE = 3; // max seconds allowed per fade
const STEP = 0.1;

// ─── Small badge shown on a clip in the timeline strip ───────────────────────
function FadeBadge({ label, seconds }: { label: string; seconds: number }) {
  if (seconds === 0) return null;
  return (
    <span className="inline-flex items-center gap-1 text-[10px] bg-purple-100 text-purple-700 rounded px-1.5 py-0.5 font-medium">
      {label} {seconds.toFixed(1)}s
    </span>
  );
}

// ─── Single clip fade editor row ─────────────────────────────────────────────
function ClipFadeRow({
  clip,
  index,
  isLast,
  onChange,
}: {
  clip: TimelineClip;
  index: number;
  isLast: boolean;
  onChange: (id: string, field: "fadeIn" | "fadeOut", val: number) => void;
}) {
  const fadeIn = clip.fadeIn ?? 0;
  const fadeOut = clip.fadeOut ?? 0;
  const clipDuration = clip.trimEnd - clip.trimStart;
  const maxFade = Math.min(MAX_FADE, clipDuration / 2);

  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-3">
      {/* Clip header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex-shrink-0 w-5 h-5 rounded-full bg-purple-600 text-white text-[10px] font-bold flex items-center justify-center">
            {index + 1}
          </span>
          <span className="text-xs font-medium text-foreground truncate max-w-[140px]">
            {clip.name}
          </span>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <FadeBadge label="↑" seconds={fadeIn} />
          <FadeBadge label="↓" seconds={fadeOut} />
        </div>
      </div>

      {/* Fade-in row */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
            
            Fade In
          </label>
          <div className="flex items-center gap-1">
            <span className="text-xs font-mono w-8 text-right text-foreground">
              {fadeIn === 0 ? "off" : `${fadeIn.toFixed(1)}s`}
            </span>
            {fadeIn > 0 && (
              <button
                onClick={() => onChange(clip.id, "fadeIn", 0)}
                className="text-[10px] text-muted-foreground hover:text-destructive transition-colors ml-1"
                title="Remove fade-in"
              >
                ✕
              </button>
            )}
          </div>
        </div>
        <Slider
          value={[fadeIn]}
          onValueChange={([v]) => onChange(clip.id, "fadeIn", v)}
          min={0}
          max={maxFade}
          step={STEP}
          className="w-full"
        />
      </div>

      {/* Fade-out row */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
            Fade Out
          </label>
          <div className="flex items-center gap-1">
            <span className="text-xs font-mono w-8 text-right text-foreground">
              {fadeOut === 0 ? "off" : `${fadeOut.toFixed(1)}s`}
            </span>
            {fadeOut > 0 && (
              <button
                onClick={() => onChange(clip.id, "fadeOut", 0)}
                className="text-[10px] text-muted-foreground hover:text-destructive transition-colors ml-1"
                title="Remove fade-out"
              >
                ✕
              </button>
            )}
          </div>
        </div>
        <Slider
          value={[fadeOut]}
          onValueChange={([v]) => onChange(clip.id, "fadeOut", v)}
          min={0}
          max={maxFade}
          step={STEP}
          className="w-full"
        />
      </div>

      {/* Between-clip crossfade hint */}
      {!isLast && (fadeOut > 0) && (
        <div className="flex items-center gap-1.5 text-[10px] text-purple-600 bg-purple-50 rounded px-2 py-1">
          <ArrowLeftRight className="w-3 h-3" />
          <span>Crossfade with next clip via fade-out → fade-in overlap</span>
        </div>
      )}
    </div>
  );
}

// ─── Quick-apply presets ─────────────────────────────────────────────────────
const PRESETS: { label: string; fadeIn: number; fadeOut: number }[] = [
  { label: "Subtle (0.5s)", fadeIn: 0.5, fadeOut: 0.5 },
  { label: "Standard (1s)", fadeIn: 1, fadeOut: 1 },
  { label: "Cinematic (2s)", fadeIn: 2, fadeOut: 2 },
  { label: "Fade-in only (1s)", fadeIn: 1, fadeOut: 0 },
  { label: "Fade-out only (1s)", fadeIn: 0, fadeOut: 1 },
];

// ─── Main FadeControls ────────────────────────────────────────────────────────
export function FadeControls({ clips, onClipsChange }: FadeControlsProps) {
  const [isOpen, setIsOpen] = useState(true);

  const handleChange = (
    id: string,
    field: "fadeIn" | "fadeOut",
    val: number
  ) => {
    onClipsChange(
      clips.map((c) => (c.id === id ? { ...c, [field]: val } : c))
    );
  };

  const applyPreset = (fadeIn: number, fadeOut: number) => {
    onClipsChange(clips.map((c) => ({ ...c, fadeIn, fadeOut })));
  };

  const clearAll = () => {
    onClipsChange(clips.map((c) => ({ ...c, fadeIn: 0, fadeOut: 0 })));
  };

  const hasAnyFade = clips.some((c) => (c.fadeIn ?? 0) > 0 || (c.fadeOut ?? 0) > 0);

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Header / toggle */}
      <button
        onClick={() => setIsOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-muted/50 hover:bg-muted/80 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Fade Effects</span>
          {hasAnyFade && (
            <span className="text-[10px] bg-purple-600 text-white rounded-full px-2 py-0.5 font-medium">
              Active
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        )}
      </button>

      {isOpen && (
        <div className="p-3 space-y-3">
          {clips.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-4">
              Add clips to the timeline to set fades.
            </p>
          ) : (
            <>
              {/* Presets */}
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5 font-semibold">
                  Quick Apply to All
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {PRESETS.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => applyPreset(p.fadeIn, p.fadeOut)}
                      className="text-[10px] border rounded px-2 py-1 hover:bg-purple-50 hover:border-purple-400 hover:text-purple-700 transition-colors"
                    >
                      {p.label}
                    </button>
                  ))}
                  {hasAnyFade && (
                    <button
                      onClick={clearAll}
                      className="text-[10px] border border-red-200 rounded px-2 py-1 hover:bg-red-50 hover:text-red-600 transition-colors"
                    >
                      Clear All
                    </button>
                  )}
                </div>
              </div>

              {/* Per-clip editors */}
              <div className="space-y-2 max-h-72 overflow-y-auto pr-0.5">
                {clips.map((clip, i) => (
                  <ClipFadeRow
                    key={clip.id}
                    clip={clip}
                    index={i}
                    isLast={i === clips.length - 1}
                    onChange={handleChange}
                  />
                ))}
              </div>

              {/* Legend */}
              <p className="text-[10px] text-muted-foreground leading-relaxed">
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
