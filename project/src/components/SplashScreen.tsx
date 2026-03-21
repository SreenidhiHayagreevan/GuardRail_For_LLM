import { useEffect, useState, useRef } from 'react';

interface SplashScreenProps {
  onComplete: () => void;
}

function SplashScreen({ onComplete }: SplashScreenProps) {
  const [phase, setPhase] = useState<'video' | 'fadeOut'>('video');
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.play().catch(() => {
      // If autoplay is blocked, skip straight to app
      onComplete();
    });

    const handleVideoEnd = () => {
      setPhase('fadeOut');
      setTimeout(() => {
        onComplete();
      }, 800);
    };

    video.addEventListener('ended', handleVideoEnd);
    return () => video.removeEventListener('ended', handleVideoEnd);
  }, [onComplete]);

  return (
    <div
      className={`fixed inset-0 z-50 bg-black transition-opacity duration-700 ${
        phase === 'fadeOut' ? 'opacity-0 pointer-events-none' : 'opacity-100'
      }`}
    >
      <video
        ref={videoRef}
        className="w-full h-full object-cover"
        muted
        playsInline
      >
        <source src="/gatekeeper.mp4" type="video/mp4" />
      </video>
    </div>
  );
}

export default SplashScreen;
