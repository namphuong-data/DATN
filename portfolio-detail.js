function init() {
  const urlParams = new URLSearchParams(window.location.search);
  const embedUrl = urlParams.get("url");
  if (embedUrl) {
    document.getElementById("pbi-url-input").value = embedUrl;
    loadIframe(embedUrl);
  }
}

function loadIframe(url) {
  const iframe = document.getElementById("pbi-iframe");
  const loading = document.getElementById("loading-overlay");
  const placeholder = document.getElementById("embed-placeholder");

  placeholder.classList.add("hidden");
  loading.style.display = "flex";
  iframe.style.display = "none";

  iframe.src = url;
  iframe.onload = () => {
    loading.style.display = "none";
    iframe.style.display = "block";
  };

  // Fallback timeout
  setTimeout(() => {
    if (loading.style.display !== "none") {
      loading.style.display = "none";
      iframe.style.display = "block";
    }
  }, 8000);
}

function loadDashboard() {
  const url = document.getElementById("pbi-url-input").value.trim();
  if (!url) return alert("Vui lòng nhập URL Power BI embed.");
  if (!url.startsWith("http"))
    return alert("URL không hợp lệ. Hãy dán URL đầy đủ từ Power BI.");
  document.getElementById("open-pbi-btn").href = url;
  loadIframe(url);
}

// Enter key on input
document.addEventListener("DOMContentLoaded", () => {
  const inp = document.getElementById("pbi-url-input");
  if (inp)
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadDashboard();
    });
  init();
});

function toggleFullscreen() {
  const container = document.getElementById("embed-container");
  if (!document.fullscreenElement) {
    container.requestFullscreen?.() ||
      container.mozRequestFullScreen?.() ||
      container.webkitRequestFullscreen?.() ||
      container.msRequestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}
