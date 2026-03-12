import { useCallback, useState } from "react";
import { Upload, Film, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabase";
import { processVideo } from "@/lib/api";


/* ---------------- TYPES ---------------- */

interface VideoFile {
  id: string;        // videos.id from DB
  url: string;       // public_url from Supabase
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

        // 1️⃣ Upload to Supabase Storage
        const { error: uploadError } = await supabase.storage
          .from("videos")
          .upload(filePath, file);

        if (uploadError) throw uploadError;

        // 2️⃣ Get public URL
        const { data: urlData } = supabase.storage
          .from("videos")
          .getPublicUrl(filePath);

        // 3️⃣ Insert into videos table
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
        (file) =>
          file.type === "video/mp4" || file.type === "video/quicktime"
      );

      await processFiles(files);
    },
    [videos]
  );

  const handleFileInput = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      await processFiles(files);
    }
  };

  const removeVideo = (id: string) => {
    onVideosChange(videos.filter((v) => v.id !== id));
  };

  /* ---------------- UI ---------------- */

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 text-center",
          isDragging
            ? "border-primary bg-primary/10 scale-[1.02]"
            : "border-border hover:border-primary/50 bg-card/50"
        )}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime"
          multiple
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        <div className="flex flex-col items-center gap-3">
          <div
            className={cn(
              "p-4 rounded-full transition-colors",
              isDragging ? "bg-primary/20" : "bg-muted"
            )}
          >
            <Upload
              className={cn(
                "w-8 h-8 transition-colors",
                isDragging ? "text-primary" : "text-muted-foreground"
              )}
            />
          </div>

          <div>
            <p className="font-medium text-foreground">
              {isUploading ? "Uploading..." : "Drop videos here"}
            </p>
            <p className="text-sm text-muted-foreground">
              MP4 or MOV files
            </p>
          </div>
        </div>
      </div>

      {/* Video thumbnails */}
      {videos.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground font-medium">
            Uploaded ({videos.length})
          </p>

          <div className="grid grid-cols-2 gap-2">
            {videos.map((video) => (
              <div
                key={video.id}
                className="relative group rounded-lg overflow-hidden bg-muted border border-border"
              >
                <video
                  src={video.url}
                  className="w-full h-20 object-cover"
                  muted
                  onLoadedMetadata={(e) => {
                    const target = e.target as HTMLVideoElement;
                    const updated = videos.map((v) =>
                      v.id === video.id
                        ? { ...v, duration: target.duration }
                        : v
                    );
                    onVideosChange(updated);
                  }}
                />

                <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />

                <div className="absolute bottom-1 left-2 right-2 flex items-center justify-between">
                  <div className="flex items-center gap-1 text-xs text-foreground">
                    <Film className="w-3 h-3" />
                    <span className="truncate max-w-[80px]">
                      {video.name}
                    </span>
                  </div>
                </div>

                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-1 right-1 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity bg-background/80 hover:bg-destructive"
                  onClick={() => removeVideo(video.id)}
                >
                  <X className="w-3 h-3" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export type { VideoFile };
