import { 
  Mic, 
  Subtitles, 
  Wand2, 
  Sparkles,
  Loader2 
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

interface AIToolsPanelProps {
  hasVideo: boolean;
}

export function AIToolsPanel({ hasVideo }: AIToolsPanelProps) {
  const [stylePrompt, setStylePrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState<string | null>(null);
  const { toast } = useToast();

  // ============================================
  // PLACEHOLDER FUNCTIONS - Replace with AI logic
  // ============================================

  /**
   * Transcribes audio from the video
   * TODO: Integrate with speech-to-text API (e.g., OpenAI Whisper)
   */
  const handleTranscribeAudio = async () => {
    setIsProcessing("transcribe");
    
    // Placeholder: Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    toast({
      title: "Transcription Complete",
      description: "Audio has been transcribed. (Placeholder)",
    });
    
    setIsProcessing(null);
  };

  /**
   * Generates subtitles from transcription
   * TODO: Use transcription to create SRT/VTT subtitle file
   */
  const handleGenerateSubtitles = async () => {
    setIsProcessing("subtitles");
    
    // Placeholder: Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    toast({
      title: "Subtitles Generated",
      description: "Subtitles have been created. (Placeholder)",
    });
    
    setIsProcessing(null);
  };

  /**
   * Applies a style transformation based on text prompt
   * TODO: Integrate with video style transfer model
   * @param prompt - Text description of desired style
   */
  const handleApplyStyle = async () => {
    if (!stylePrompt.trim()) {
      toast({
        title: "Enter a Style Prompt",
        description: "Describe the style you want to apply.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing("style");
    
    // Placeholder: Simulate API call
    console.log("Applying style:", stylePrompt);
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    toast({
      title: "Style Applied",
      description: `Applied style: "${stylePrompt}" (Placeholder)`,
    });
    
    setStylePrompt("");
    setIsProcessing(null);
  };

  // ============================================

  const aiTools = [
    {
      id: "transcribe",
      label: "Transcribe Audio",
      icon: Mic,
      onClick: handleTranscribeAudio,
      description: "Convert speech to text",
    },
    {
      id: "subtitles",
      label: "Generate Subtitles",
      icon: Subtitles,
      onClick: handleGenerateSubtitles,
      description: "Create captions automatically",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-secondary" />
        <h3 className="font-semibold text-foreground">AI Tools</h3>
      </div>

      {/* AI Tool buttons */}
      <div className="space-y-2">
        {aiTools.map((tool) => (
          <Button
            key={tool.id}
            variant="surface"
            className="w-full justify-start gap-3 h-auto py-3"
            disabled={!hasVideo || isProcessing !== null}
            onClick={tool.onClick}
          >
            {isProcessing === tool.id ? (
              <Loader2 className="w-4 h-4 animate-spin text-secondary" />
            ) : (
              <tool.icon className="w-4 h-4 text-secondary" />
            )}
            <div className="text-left">
              <p className="font-medium">{tool.label}</p>
              <p className="text-xs text-muted-foreground">{tool.description}</p>
            </div>
          </Button>
        ))}
      </div>

      {/* Style prompt */}
      <div className="space-y-2 pt-2 border-t border-border">
        <label className="text-sm font-medium text-foreground flex items-center gap-2">
          <Wand2 className="w-4 h-4 text-secondary" />
          Apply Style
        </label>
        <Input
          placeholder="e.g., cinematic, vintage film..."
          value={stylePrompt}
          onChange={(e) => setStylePrompt(e.target.value)}
          className="bg-muted border-border"
          disabled={!hasVideo || isProcessing !== null}
        />
        <Button
          variant="ai"
          className="w-full"
          disabled={!hasVideo || !stylePrompt.trim() || isProcessing !== null}
          onClick={handleApplyStyle}
        >
          {isProcessing === "style" ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Applying...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              Apply Style
            </>
          )}
        </Button>
      </div>

      {!hasVideo && (
        <p className="text-xs text-muted-foreground text-center py-2">
          Upload a video to use AI tools
        </p>
      )}
    </div>
  );
}
