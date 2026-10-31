const gallery = document.querySelector(".gallery-section");
const viewer = document.querySelector(".gallery-viewer");

if (gallery && viewer) {
  const cards = [...gallery.querySelectorAll("[data-gallery-category]")];
  const filters = gallery.querySelectorAll("[data-gallery-filter]");
  const count = gallery.querySelector(".gallery-result-count");
  const stage = viewer.querySelector(".gallery-viewer-stage");
  let opener;

  filters.forEach((filter) => {
    filter.addEventListener("click", () => {
      const category = filter.dataset.galleryFilter;
      filters.forEach((button) => button.setAttribute("aria-pressed", String(button === filter)));
      cards.forEach((card) => {
        card.hidden = category !== "All" && card.dataset.galleryCategory !== category;
      });
      const visible = cards.filter((card) => !card.hidden).length;
      count.textContent = `${visible} ${visible === 1 ? "result" : "results"}`;
    });
  });

  cards.forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || typeof viewer.showModal !== "function") return;
      event.preventDefault();
      opener = card;
      const title = card.querySelector("h2").textContent;
      const isVideo = card.dataset.mediaType === "video";
      const media = document.createElement(isVideo ? "video" : "img");
      if (isVideo) {
        media.controls = true;
        media.autoplay = true;
        media.muted = true;
        media.loop = true;
        media.playsInline = true;
        media.preload = "metadata";
        media.poster = card.querySelector("img").src;
        media.setAttribute("aria-label", title);
      } else {
        media.alt = title;
      }
      media.src = card.href;
      stage.replaceChildren(media);
      viewer.querySelector("#gallery-viewer-title").textContent = title;
      viewer.querySelector("#gallery-viewer-category").textContent = card.dataset.galleryCategory;
      viewer.querySelector("#gallery-viewer-description").textContent = card.querySelector(".gallery-card-body > p:not(.eyebrow)").textContent;
      viewer.querySelector("#gallery-viewer-file").href = card.href;
      viewer.showModal();
      document.body.classList.add("gallery-viewer-open");
      if (isVideo) media.play().catch(() => {});
    });
  });

  viewer.querySelector(".gallery-viewer-close").addEventListener("click", () => viewer.close());
  viewer.addEventListener("click", (event) => {
    if (event.target === viewer) viewer.close();
  });
  viewer.addEventListener("close", () => {
    const video = stage.querySelector("video");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    stage.replaceChildren();
    document.body.classList.remove("gallery-viewer-open");
    opener?.focus();
  });
}
