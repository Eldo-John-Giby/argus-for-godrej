import { useRef, useEffect, useCallback } from "react";

interface Detection {
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  class_name: string;
  confidence: number;
  track_id?: number;
  keypoints?: Record<string, [number, number]>;
}

interface OverlayCanvasProps {
  width: number;
  height: number;
  detections: Detection[];
  showSkeleton?: boolean;
  showLabels?: boolean;
}

const CLASS_COLORS: Record<string, string> = {
  person: "#3b82f6",
  box: "#f59e0b",
  forklift: "#ef4444",
  pallet: "#22c55e",
  trolley: "#8b5cf6",
};

const KEYPOINT_CONNECTIONS: [string, string][] = [
  ["nose", "left_eye"], ["nose", "right_eye"],
  ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
];

export default function OverlayCanvas({
  width,
  height,
  detections,
  showSkeleton = true,
  showLabels = true,
}: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    for (const det of detections) {
      const color = CLASS_COLORS[det.class_name] || "#ffffff";
      const [x1, y1, x2, y2] = det.bbox;

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Confidence fill
      ctx.fillStyle = color + "20";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

      // Label
      if (showLabels) {
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%${det.track_id ? ` #${det.track_id}` : ""}`;
        ctx.font = "12px monospace";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 18, textWidth + 8, 18);
        ctx.fillStyle = "#000";
        ctx.fillText(label, x1 + 4, y1 - 5);
      }

      // Pose keypoints
      if (showSkeleton && det.keypoints) {
        // Draw connections
        ctx.strokeStyle = color + "80";
        ctx.lineWidth = 1.5;
        for (const [start, end] of KEYPOINT_CONNECTIONS) {
          const kp1 = det.keypoints[start];
          const kp2 = det.keypoints[end];
          if (kp1 && kp2) {
            ctx.beginPath();
            ctx.moveTo(kp1[0], kp1[1]);
            ctx.lineTo(kp2[0], kp2[1]);
            ctx.stroke();
          }
        }

        // Draw keypoints
        for (const [name, point] of Object.entries(det.keypoints)) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(point[0], point[1], 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, [width, height, detections, showSkeleton, showLabels]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}
