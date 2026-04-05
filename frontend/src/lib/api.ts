// src/lib/api.ts

export const getHealthStatus = async (): Promise<string> => {
  try {
    const response = await fetch("http://127.0.0.1:8000/health");
    if (!response.ok) throw new Error("Network error");
    const data = await response.json();
    return data.status;
  } catch (err) {
    console.error(err);
    return "error";
  }
};

export interface VideoPayload {
  id: string;
  name: string;
  url: string;
  duration?: number;
}

export const sendVideosToBackend = async (videos: VideoPayload[]) => {
  const res = await fetch("http://127.0.0.1:8000/videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ videos }),
  });
  if (!res.ok) throw new Error("Failed to send videos");
  return res.json();
};

export const processVideo = async (prompt: string, videoUrl: string) => {
  const res = await fetch("http://127.0.0.1:8000/videos/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, video_url: videoUrl }),
  });
  if (!res.ok) throw new Error("Processing failed");
  return res.json();
};

export type FilterValue =
  | "None" | "Sunset" | "Happy" | "Sad" | "Dramatic"
  | "Vintage" | "Night" | "Cinematic" | "Black & White"
  | "Teal & Orange" | "Fade" | "Neon" | "Warm";

export type TransitionValue =
  | "none" | "fade" | "wipe" | "zoom" | "flash" | "glitch" | "blur" | "dip";

export type MusicValue = "None" | "Lo-fi" | "Cinematic" | "Upbeat" | "Sad" | "Energetic";

export interface EditSettings {
  filter?: FilterValue;
  music?: MusicValue;
  musicVolume?: number;
  muteOriginal?: boolean;
  brightness?: number;
  contrast?: number;
  saturation?: number;
  aspectRatio?: "16:9" | "9:16" | "1:1" | "original";
  overlayText?: string;
  transition?: TransitionValue;
}

export interface EditPromptResponse {
  status: string;
  selected_clips: any[];
  edit_settings: EditSettings;
  settings_summary: string;
}

export const runEditPrompt = async (
  prompt: string,
  videos: VideoPayload[]
): Promise<EditPromptResponse> => {
  const res = await fetch("http://localhost:8000/videos/edit-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, videos }),
  });
  if (!res.ok) throw new Error("Edit prompt failed");
  return res.json();
};