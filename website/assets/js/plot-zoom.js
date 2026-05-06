document.addEventListener("DOMContentLoaded", () => {
  const figures = Array.from(document.querySelectorAll("main img"));
  if (!figures.length) return;

  const overlay = document.createElement("div");
  overlay.className = "plot-lightbox";
  overlay.innerHTML = `
    <button class="plot-lightbox-close" type="button" aria-label="Close enlarged figure">Close</button>
    <img alt="">
  `;
  document.body.appendChild(overlay);

  const overlayImage = overlay.querySelector("img");
  const close = () => {
    overlay.classList.remove("open");
    overlayImage.removeAttribute("src");
    overlayImage.removeAttribute("alt");
  };

  figures.forEach((image) => {
    image.classList.add("zoomable-plot");
    image.setAttribute("title", "Click to enlarge");
    image.addEventListener("click", () => {
      overlayImage.src = image.currentSrc || image.src;
      overlayImage.alt = image.alt || "OceanLens figure";
      overlay.classList.add("open");
    });
  });

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay || event.target.classList.contains("plot-lightbox-close")) {
      close();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && overlay.classList.contains("open")) {
      close();
    }
  });
});
