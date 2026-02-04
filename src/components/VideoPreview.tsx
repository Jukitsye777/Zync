import { useRef, useState, useEffect } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

interface VideoPreviewProps {
  videoUrl: string | null;
  trimStart?: number;
  trimEnd?: number;
  title?: string;
}

export function VideoPreview({
  videoUrl,
  trimStart = 0,
  trimEnd,
  title = "Preview",
}: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, [videoUrl]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      
      // Loop within trim bounds
      if (trimEnd && videoRef.current.currentTime >= trimEnd) {
        videoRef.current.currentTime = trimStart;
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      if (trimStart > 0) {
        videoRef.current.currentTime = trimStart;
      }
    }
  };

  const handleSeek = (value: number[]) => {
    if (videoRef.current) {
      videoRef.current.currentTime = value[0];
      setCurrentTime(value[0]);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const effectiveDuration = trimEnd || duration;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border">
        <h2 className="font-semibold text-foreground">{title}</h2>
      </div>

      {/* Video container */}
      <div className="flex-1 relative bg-background/50 flex items-center justify-center">
        {videoUrl ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              className="max-w-full max-h-full object-contain"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              onClick={togglePlay}
            />
            {/* Play overlay */}
            {!isPlaying && (
              <div
                className="absolute inset-0 flex items-center justify-center bg-background/30 cursor-pointer"
                onClick={togglePlay}
              >
                <div className="w-16 h-16 rounded-full bg-primary/90 flex items-center justify-center shadow-lg hover:scale-110 transition-transform">
                  <Play className="w-8 h-8 text-primary-foreground ml-1" />
                </div>
              </div>
            )}
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
      {videoUrl && (
        <div className="p-4 border-t border-border space-y-3">
          {/* Progress bar */}
          <Slider
            value={[currentTime]}
            onValueChange={handleSeek}
            min={trimStart}
            max={effectiveDuration}
            step={0.1}
            className="w-full"
          />

          {/* Control buttons */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={togglePlay}
                className="hover:bg-primary/10 hover:text-primary"
              >
                {isPlaying ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                className="hover:bg-primary/10 hover:text-primary"
              >
                {isMuted ? (
                  <VolumeX className="w-5 h-5" />
                ) : (
                  <Volume2 className="w-5 h-5" />
                )}
              </Button>
            </div>

            <span className="text-xs font-mono text-muted-foreground">
              {formatTime(currentTime)} / {formatTime(effectiveDuration)}
            </span>

            <Button
              variant="ghost"
              size="icon"
              className="hover:bg-primary/10 hover:text-primary"
            >
              <Maximize className="w-5 h-5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
