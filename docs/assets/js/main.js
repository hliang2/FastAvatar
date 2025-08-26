// main.js — lightweight video-only behavior

// Swap the MP4 shown in the demo player and update button styles
function swapVideo(src, btn) {
  const video = document.getElementById('demoVideo');
  if (!video) return;

  // Update button styles (primary for active)
  const btns = document.querySelectorAll('#demo .btn');
  btns.forEach(b => {
    b.classList.remove('btn-primary');
    b.classList.add('btn-outline-primary');
  });
  if (btn) {
    btn.classList.remove('btn-outline-primary');
    btn.classList.add('btn-primary');
  }

  // Swap source and (re)play
  // Restart from 0 for consistency
  video.pause();
  video.src = src;
  video.currentTime = 0;
  // iOS/Safari prefers load() before play()
  video.load();
  const p = video.play();
  if (p && typeof p.catch === 'function') {
    p.catch(() => {
      // Autoplay may be blocked if not muted, but your video is muted.
      // No-op just in case.
    });
  }
}

// Expose for inline onclick handlers
window.swapVideo = swapVideo;

document.addEventListener('DOMContentLoaded', () => {
  // Ensure the default video autoplays
  const video = document.getElementById('demoVideo');
  if (video) {
    const p = video.play();
    if (p && typeof p.catch === 'function') {
      p.catch(() => {/* ignore autoplay block if any */});
    }
  }

  // Optional: pause the demo video when it’s off-screen to save battery/CPU
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!video) return;
        if (entry.isIntersecting) {
          if (video.paused) video.play().catch(() => {});
        } else {
          if (!video.paused) video.pause();
        }
      });
    }, { threshold: 0.1 });
    const demoSection = document.getElementById('demo');
    if (demoSection) io.observe(demoSection);
  }
});