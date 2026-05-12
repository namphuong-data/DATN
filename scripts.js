// Hiệu ứng xuất hiện khi cuộn trang (Scroll animation)
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.style.opacity = "1";
        e.target.style.transform = "translateY(0)";
      }
    });
  },
  {
    threshold: 0.1,
  },
);

// Áp dụng hiệu ứng cho các phần tử có class .fade-in
document.querySelectorAll(".fade-in").forEach((el) => {
  el.style.opacity = "0";
  el.style.transform = "translateY(20px)";
  el.style.transition = "all 0.8s ease-out";
  observer.observe(el);
});
