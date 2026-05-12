"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  // Start camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      setStream(mediaStream);
      setCameraOn(true);
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setCameraOn(false);
  };

  // Capture + send to backend
  const captureImage = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    // Show preview
    const image = canvas.toDataURL("image/png");
    setCapturedImage(image);

    // 🔥 Send to backend
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      console.log("Sending image to backend...");

      try {
        const res = await fetch("http://localhost:5000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        console.log("Backend response:", data);

        setResult(data);
      } catch (err) {
        console.error("Error sending to backend:", err);
      }
    }, "image/jpeg");
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (stream) stopCamera();
    };
  }, [stream]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-gray-800 text-white p-6">
      <h1 className="text-3xl font-bold mb-6">
        Smart Grocery Tracker Dashboard
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Camera */}
        <div className="bg-white/10 p-4 rounded-2xl shadow-lg">
          <h2 className="text-lg font-semibold mb-3">Live Camera</h2>

          <div className="flex gap-3 mb-3">
            {!cameraOn ? (
              <button
                onClick={startCamera}
                className="bg-green-500 px-4 py-2 rounded-lg"
              >
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={captureImage}
                  className="bg-blue-500 px-4 py-2 rounded-lg"
                >
                  Capture
                </button>

                <button
                  onClick={stopCamera}
                  className="bg-red-500 px-4 py-2 rounded-lg"
                >
                  Stop
                </button>
              </>
            )}
          </div>

          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-64 object-cover rounded-xl"
          />

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Preview */}
        <div className="bg-white/10 p-4 rounded-2xl shadow-lg">
          <h2 className="text-lg font-semibold mb-3">Captured Image</h2>

          {capturedImage ? (
            <img
              src={capturedImage}
              alt="Captured"
              className="w-full h-64 object-cover rounded-xl"
            />
          ) : (
            <p className="text-gray-400">No image captured yet</p>
          )}
        </div>
      </div>

      {/* Results */}
      {result && result.success && (
        <div className="bg-white/10 p-4 rounded-2xl shadow-lg mt-6">
          <h2 className="text-lg font-semibold mb-3">Detection Results</h2>

          {result.results.map((item: any, i: number) => (
            <div key={i}>
              <p><b>Item:</b> {item.item}</p>
              <p><b>Freshness:</b> {item.freshness}</p>
              <p><b>Days Remaining:</b> {item.days_remaining}</p>
              <hr className="my-2"/>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}