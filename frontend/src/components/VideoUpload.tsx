import { useCallback, useState } from "react";
import { Upload, Film, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabase";

/* ---------------- TYPES ---------------- */

interface VideoFile {
  id: string;
  url: string;
  name: string;
  duration: number;
}

interface VideoUploaderProps {
  onVideosChange: (videos: VideoFile[]) => void;
  videos: VideoFile[];
}

/* ---------------- COMPONENT ---------------- */

export function VideoUploader({ onVideosChange, videos }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  /* ---------------- UPLOAD LOGIC ---------------- */

  const processFiles = async (files: File[]) => {
    setIsUploading(true);
    const uploadedVideos: VideoFile[] = [];

    for (const file of files) {
      try {
        const ext = file.name.split(".").pop();
        const fileName = `${crypto.randomUUID()}.${ext}`;
        const filePath = `inputs/${fileName}`;

        const { error: uploadError } = await supabase.storage
          .from("videos")
          .upload(filePath, file);

        if (uploadError) throw uploadError;

        const { data: urlData } = supabase.storage
          .from("videos")
          .getPublicUrl(filePath);

        const { data: videoRow, error: dbError } = await supabase
          .from("videos")
          .insert({
            video_name: file.name,
            storage_path: filePath,
            public_url: urlData.publicUrl,
          })
          .select()
          .single();

        if (dbError) throw dbError;

        uploadedVideos.push({
          id: videoRow.id,
          name: videoRow.video_name,
          url: videoRow.public_url,
          duration: 0,
        });
      } catch (err) {
        console.error("Video upload failed:", err);
      }
    }

    onVideosChange([...videos, ...uploadedVideos]);
    setIsUploading(false);
  };

  /* ---------------- HANDLERS ---------------- */

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files).filter(
        (f) => f.type === "video/mp4" || f.type === "video/quicktime"
      );
      await processFiles(files);
    },
    [videos]
  );

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      await processFiles(Array.from(e.target.files));
    }
  };

  const removeVideo = (id: string) => {
    onVideosChange(videos.filter((v) => v.id !== id));
  };

  /* ---------------- UI ---------------- */

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "relative rounded-xl border-2 border-dashed p-6 text-center transition-all duration-200 cursor-pointer",
          isDragging
            ? "border-violet-500/70 bg-violet-500/8 scale-[1.01]"
            : "border-zinc-700/60 hover:border-zinc-600/80 bg-zinc-800/30 hover:bg-zinc-800/50"
        )}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime"
          multiple
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        <div className="flex flex-col items-center gap-2.5">
          <div className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center transition-colors",
            isDragging ? "bg-violet-500/20" : "bg-zinc-700/60"
          )}>
            {isUploading
              ? <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />
              : <Upload className={cn("w-5 h-5 transition-colors", isDragging ? "text-violet-400" : "text-zinc-400")} />
            }
          </div>
          <div>
            <p className={cn("text-sm font-medium transition-colors", isDragging ? "text-violet-300" : "text-zinc-300")}>
              {isUploading ? "Uploading…" : "Drop videos here"}
            </p>
            <p className="text-xs text-zinc-600 mt-0.5">MP4 or MOV files</p>
          </div>
        </div>
      </div>

      {/* Video thumbnails */}
      {videos.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {videos.map((video) => (
            <div
              key={video.id}
              className="relative group rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700/50"
            >
              <video
                src={video.url}
                className="w-full h-20 object-cover"
                muted
                onLoadedMetadata={(e) => {
                  const target = e.target as HTMLVideoElement;
                  const updated = videos.map((v) =>
                    v.id === video.id ? { ...v, duration: target.duration } : v
                  );
                  onVideosChange(updated);
                }}
              />

              {/* Gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />

              {/* Label */}
              <div className="absolute bottom-1.5 left-2 right-6 flex items-center gap-1">
                <Film className="w-2.5 h-2.5 text-zinc-300 flex-shrink-0" />
                <span className="text-[10px] text-zinc-200 truncate font-medium">{video.name}</span>
              </div>

              {/* Duration badge */}
              {video.duration > 0 && (
                <div className="absolute top-1.5 left-1.5 bg-black/60 rounded px-1.5 py-0.5 text-[9px] font-mono text-zinc-300">
                  {Math.floor(video.duration)}s
                </div>
              )}

              {/* Remove button */}
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-1 right-1 h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 hover:bg-red-900/80 text-zinc-300 hover:text-white rounded-md"
                onClick={() => removeVideo(video.id)}
              >
                <X className="w-2.5 h-2.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export type { VideoFile };
