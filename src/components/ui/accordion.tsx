import { useState } from "react";
import { 
  Plus, 
  Play, 
  Download,
  Layers,
  Zap
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { VideoUploader, type VideoFile } from "@/components/VideoUpload";
import { TrimControls } from "@/components/TrimControls";
import { Timeline, type TimelineClip } from "@/components/Timeline";
import { VideoPreview } from "@/components/VideoPreview";
import { AIToolsPanel } from "@/components/AIToolsPanel";
import { useToast } from "@/hooks/use-toast";

export default function Index() {
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [timelineClips, setTimelineClips] = useState<TimelineClip[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [trimValues, setTrimValues] = useState<{ start: number; end: number }>({
    start: 0,
    end: 10,
  });
  const [assembledVideoUrl, setAssembledVideoUrl] = useState<string | null>(null);
  const { toast } = useToast();

  // Get currently selected video for preview
  const selectedVideo = videos.find((v) => v.id === selectedVideoId);
  const selectedClip = timelineClips.find((c) => c.id === selectedClipId);

  // Add video to timeline
  const handleAddToTimeline = () => {
    if (!selectedVideo) {
      toast({
        title: "Select a Video",
        description: "Click on an uploaded video to select it first.",
        variant: "destructive",
      });
      return;
    }

    const newClip: TimelineClip = {
      id: crypto.randomUUID(),
      name: selectedVideo.name,
      duration: selectedVideo.duration,
      trimStart: trimValues.start,
      trimEnd: trimValues.end,
      videoUrl: selectedVideo.url,
    };

    setTimelineClips([...timelineClips, newClip]);
    toast({
      title: "Clip Added",
      description: `"${selectedVideo.name}" added to timeline.`,
    });
  };

  // Remove clip from timeline
  const handleRemoveClip = (id: string) => {
    setTimelineClips(timelineClips.filter((c) => c.id !== id));
    if (selectedClipId === id) {
      setSelectedClipId(null);
    }
  };

  /**
   * Assembles trimmed clips into one video
   * TODO: Replace with actual video assembly logic (FFmpeg.wasm or backend)
   */
  const handleAssembleVideo = async () => {
    if (timelineClips.length === 0) {
      toast({
        title: "No Clips",
        description: "Add clips to the timeline first.",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Assembling Video...",
      description: "This may take a moment. (Placeholder)",
    });

    // Placeholder: In real implementation, use FFmpeg.wasm or send to backend
    // For now, just show the first clip as "assembled" output
    await new Promise((resolve) => setTimeout(resolve, 1500));
    
    setAssembledVideoUrl(timelineClips[0].videoUrl);
    
    toast({
      title: "Video Assembled",
      description: "Your video is ready for preview. (Placeholder)",
    });
  };

  /**
   * Exports the final assembled video
   * TODO: Implement actual video export/download
   */
  const handleExport = () => {
    if (!assembledVideoUrl) {
      toast({
        title: "Assemble First",
        description: "Assemble your clips before exporting.",
        variant: "destructive",
      });
      return;
    }

    // Placeholder: Trigger download
    const link = document.createElement("a");
    link.href = assembledVideoUrl;
    link.download = "zync-export.mp4";
    link.click();

    toast({
      title: "Export Started",
      description: "Your video is being downloaded.",
    });
  };

  // When a video thumbnail is clicked
  const handleVideoSelect = (id: string) => {
    setSelectedVideoId(id);
    const video = videos.find((v) => v.id === id);
    if (video) {
      setTrimValues({ start: 0, end: video.duration || 10 });
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="h-16 border-b border-border flex items-center justify-between px-6 bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary/50 flex items-center justify-center shadow-lg glow-primary">
            <Zap className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">ZYNC</h1>
            <p className="text-xs text-muted-foreground">AI Video Editor</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="glow"
            onClick={handleAssembleVideo}
            disabled={timelineClips.length === 0}
          >
            <Layers className="w-4 h-4" />
            Assemble
          </Button>
          <Button
            variant="outline"
            onClick={handleExport}
            disabled={!assembledVideoUrl}
          >
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </header>

      {/* Main content - 3 column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left column - Upload & AI Tools */}
        <aside className="w-80 border-r border-border flex flex-col bg-card/30">
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Upload section */}
            <section>
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Upload
              </h2>
              <VideoUploader
                videos={videos}
                onVideosChange={(newVideos) => {
                  setVideos(newVideos);
                  // Auto-select first video if none selected
                  if (!selectedVideoId && newVideos.length > 0) {
                    handleVideoSelect(newVideos[0].id);
                  }
                }}
              />
              
              {/* Click to select hint */}
              {videos.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div className="flex flex-wrap gap-1">
                    {videos.map((v) => (
                      <button
                        key={v.id}
                        onClick={() => handleVideoSelect(v.id)}
                        className={`px-2 py-1 text-xs rounded transition-all ${
                          selectedVideoId === v.id
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted text-muted-foreground hover:bg-muted/80"
                        }`}
                      >
                        {v.name.slice(0, 10)}...
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Trim controls */}
            {selectedVideo && (
              <section className="animate-slide-up">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                  Trim
                </h2>
                <TrimControls
                  duration={selectedVideo.duration || 10}
                  trimStart={trimValues.start}
                  trimEnd={trimValues.end}
                  onTrimChange={(start, end) => setTrimValues({ start, end })}
                />
                <Button
                  variant="glow"
                  className="w-full mt-3"
                  onClick={handleAddToTimeline}
                >
                  <Plus className="w-4 h-4" />
                  Add to Timeline
                </Button>
              </section>
            )}

            {/* AI Tools */}
            <section>
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                AI Features
              </h2>
              <AIToolsPanel hasVideo={videos.length > 0} />
            </section>
          </div>
        </aside>

        {/* Center column - Timeline */}
        <main className="flex-1 flex flex-col bg-surface/30">
          <Timeline
            clips={timelineClips}
            selectedClipId={selectedClipId}
            onSelectClip={setSelectedClipId}
            onRemoveClip={handleRemoveClip}
            onReorderClips={setTimelineClips}
          />
        </main>

        {/* Right column - Preview */}
        <aside className="w-96 border-l border-border flex flex-col bg-card/30">
          {/* Source preview */}
          <div className="flex-1 border-b border-border">
            <VideoPreview
              videoUrl={selectedVideo?.url || selectedClip?.videoUrl || null}
              trimStart={selectedClip?.trimStart || trimValues.start}
              trimEnd={selectedClip?.trimEnd || trimValues.end}
              title="Source Preview"
            />
          </div>

          {/* Output preview */}
          <div className="flex-1">
            <VideoPreview
              videoUrl={assembledVideoUrl}
              title="Output Preview"
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
