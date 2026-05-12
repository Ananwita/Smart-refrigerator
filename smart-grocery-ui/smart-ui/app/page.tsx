"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("");

  const [alerts, setAlerts] = useState<string[]>([]);

  // 🔹 Load Cameras
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((allDevices) => {
      const cams = allDevices.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      if (cams.length > 0) setSelectedDevice(cams[0].deviceId);
    });
  }, []);

  // 🔹 Alerts
  const addAlert = (msg: string) => {
    setAlerts((prev) => [msg, ...prev]);
  };

  // 🔹 Start Camera
  const startCamera = async () => {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: selectedDevice },
    });

    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
    }

    setStream(mediaStream);
    setCameraOn(true);
    addAlert("Camera started 🎥");
  };

  // 🔹 Stop Camera
  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setCameraOn(false);
    addAlert("Camera stopped ❌");
  };

  // 🔹 Capture
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");

    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;

    ctx?.drawImage(videoRef.current, 0, 0);

    const img = canvasRef.current.toDataURL("image/png");
    setCapturedImage(img);

    addAlert("Image captured 📸");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white px-6 py-10">
      
      {/* 🔥 Centered Heading */}
      <h1 className="text-4xl font-bold text-center mb-10 tracking-wide">
        Smart Grocery Tracker
      </h1>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Camera */}
        <div className="bg-white/10 backdrop-blur-lg p-5 rounded-2xl shadow-xl border border-white/10 hover:scale-[1.02] transition">
          <h2 className="text-lg font-semibold mb-3">📷 Live Camera</h2>

          <select
            value={selectedDevice}
            onChange={(e) => setSelectedDevice(e.target.value)}
            className="w-full p-2 mb-3 rounded bg-gray-800 text-white"
          >
            {devices.map((d, i) => (
              <option key={i} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>

          <div className="flex gap-2 mb-3">
            {!cameraOn ? (
              <button
                onClick={startCamera}
                className="bg-green-500 hover:bg-green-600 px-4 py-2 rounded-lg"
              >
                Start
              </button>
            ) : (
              <>
                <button
                  onClick={captureImage}
                  className="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded-lg"
                >
                  Capture
                </button>
                <button
                  onClick={stopCamera}
                  className="bg-red-500 hover:bg-red-600 px-4 py-2 rounded-lg"
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
            className="w-full h-48 rounded-xl object-cover"
          />

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Captured */}
        <div className="bg-white/10 backdrop-blur-lg p-5 rounded-2xl shadow-xl border border-white/10 hover:scale-[1.02] transition">
          <h2 className="text-lg font-semibold mb-3">🖼 Captured Image</h2>

          {capturedImage ? (
            <img
              src={capturedImage}
              className="w-full h-48 object-cover rounded-xl"
            />
          ) : (
            <p className="text-gray-400">No image yet</p>
          )}
        </div>

        {/* Alerts */}
        <div className="bg-white/10 backdrop-blur-lg p-5 rounded-2xl shadow-xl border border-white/10 hover:scale-[1.02] transition">
          <h2 className="text-lg font-semibold mb-3">🚨 Alerts</h2>

          {alerts.length ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {alerts.map((a, i) => (
                <div
                  key={i}
                  className="bg-yellow-500/20 border border-yellow-500 p-2 rounded text-sm"
                >
                  ⚡ {a}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400">No alerts</p>
          )}
        </div>

      </div>
    </div>
  );
}