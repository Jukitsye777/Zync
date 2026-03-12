// src/hooks/api.ts
export const getHealthStatus = async (): Promise<string> => {
  try {
    const response = await fetch("http://127.0.0.1:8000/health");
    if (!response.ok) throw new Error("Network error");
    const data = await response.json();
    return data.status; // "ok"
  } catch (err) {
    console.error(err);
    return "error";
  }
};
// src/hooks/api.ts

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

  if (!res.ok) {
    throw new Error("Failed to send videos");
  }

  return res.json();
};

export const processVideo = async (
  prompt: string,
  videoUrl: string
) => {
  const res = await fetch("http://127.0.0.1:8000/videos/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      video_url: videoUrl,
    }),
  });

  if (!res.ok) {
    throw new Error("Processing failed");
  }

  return res.json();
};


