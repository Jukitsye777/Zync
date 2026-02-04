import { useState, useEffect } from "react";
import { Slider } from "@/components/ui/slider";
import { Scissors } from "lucide-react";

interface TrimControlsProps {
  duration: number;
  trimStart: number;
  trimEnd: number;
  onTrimChange: (start: number, end: number) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

export function TrimControls({
  duration,
  trimStart,
  trimEnd,
  onTrimChange,
}: TrimControlsProps) {
  const [values, setValues] = useState([trimStart, trimEnd]);

  useEffect(() => {
    setValues([trimStart, trimEnd]);
  }, [trimStart, trimEnd]);

  const handleChange = (newValues: number[]) => {
    setValues(newValues);
    onTrimChange(newValues[0], newValues[1]);
  };

  const trimmedDuration = values[1] - values[0];

  return (
    <div className="space-y-4 p-4 rounded-xl bg-card border border-border">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Scissors className="w-4 h-4 text-primary" />
          Trim Controls
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          Duration: {formatTime(trimmedDuration)}
        </span>
      </div>

      {/* Custom dual-handle slider visualization */}
      <div className="space-y-3">
        <div className="relative h-2 bg-muted rounded-full overflow-hidden">
          {/* Trimmed region highlight */}
          <div
            className="absolute h-full bg-primary/30"
            style={{
              left: `${(values[0] / duration) * 100}%`,
              width: `${((values[1] - values[0]) / duration) * 100}%`,
            }}
          />
        </div>

        <Slider
          value={values}
          onValueChange={handleChange}
          min={0}
          max={duration}
          step={0.1}
          className="relative"
        />

        {/* Time labels */}
        <div className="flex justify-between text-xs font-mono text-muted-foreground">
          <span className="text-primary">{formatTime(values[0])}</span>
          <span className="text-primary">{formatTime(values[1])}</span>
        </div>
      </div>
    </div>
  );
}
