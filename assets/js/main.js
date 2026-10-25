const menuButton = document.querySelector(".menu-button");

function closeMobileMenu() {
  document.body.classList.remove("nav-open");
  menuButton?.setAttribute("aria-expanded", "false");
}

menuButton?.addEventListener("click", () => {
  const isOpen = document.body.classList.toggle("nav-open");
  menuButton.setAttribute("aria-expanded", String(isOpen));
});

document.querySelectorAll(".top-links a").forEach((link) => {
  link.addEventListener("click", closeMobileMenu);
});

function findTabAttribute(button) {
  return button.getAttributeNames().find((name) => name.startsWith("data-") && name.endsWith("-tab"));
}

function showTab(button) {
  const tabAttr = findTabAttribute(button);
  if (!tabAttr) return;
  const panelAttr = tabAttr.replace(/-tab$/, "-panel");
  const tabName = button.getAttribute(tabAttr);
  const scope = button.closest(".plasticity-document-panel, .file-tabs, .solid-method-document-panel, section, main") || document;
  const buttons = scope.querySelectorAll(`[${tabAttr}]`);
  const panels = scope.querySelectorAll(`[${panelAttr}]`);
  buttons.forEach((b) => {
    const active = b.getAttribute(tabAttr) === tabName;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  panels.forEach((panel) => {
    const active = panel.getAttribute(panelAttr) === tabName;
    panel.classList.toggle("active", active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => console.warn("MathJax typeset failed:", err));
    }
  });
}

document.querySelectorAll('[role="tab"]').forEach((button) => {
  button.addEventListener("click", () => showTab(button));
  button.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showTab(button);
    }
  });
});

const moduleControls = document.querySelectorAll("[data-module]");
const moduleDetails = document.querySelectorAll("[data-module-detail]");
function showModule(moduleName = "input") {
  moduleControls.forEach((control) => control.classList.toggle("active", control.dataset.module === moduleName));
  moduleDetails.forEach((detail) => detail.classList.toggle("active", detail.dataset.moduleDetail === moduleName));
}
moduleControls.forEach((control) => {
  control.addEventListener("click", () => showModule(control.dataset.module));
  control.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showModule(control.dataset.module);
    }
  });
});

const applicationPreviewControls = document.querySelectorAll("[data-preview]");
function setApplicationPreview(previewName = "solid", scope = document) {
  const boxes = scope.querySelectorAll ? scope.querySelectorAll("[data-preview-box]") : [];
  boxes.forEach((box) => {
    box.dataset.activePreview = previewName;
    box.querySelectorAll("[data-preview-panel]").forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.previewPanel === previewName);
    });
  });
  applicationPreviewControls.forEach((control) => {
    control.classList.toggle("preview-active", control.dataset.preview === previewName);
  });
}
applicationPreviewControls.forEach((control) => {
  const activate = () => setApplicationPreview(control.dataset.preview || "solid", control.closest("section") || document);
  control.addEventListener("mouseenter", activate);
  control.addEventListener("focus", activate);
  control.addEventListener("touchstart", activate, { passive: true });
});
setApplicationPreview("solid");

const heroVideo = document.querySelector(".video-hero video");
if (heroVideo) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) {
    heroVideo.pause();
  } else {
    const heroVideoObserver = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        heroVideo.play().catch(() => {});
      } else {
        heroVideo.pause();
      }
    }, { threshold: 0.2 });
    heroVideoObserver.observe(heroVideo);
  }
}

if (window.location.hash) {
  const target = document.querySelector(window.location.hash);
  if (target) setTimeout(() => target.scrollIntoView({ behavior: "smooth" }), 80);
}

// Format configuration-file previews without changing their copyable text.
document.querySelectorAll(".example-code code").forEach((code) => {
  const lines = code.textContent.replace(/\r\n/g, "\n").split("\n");
  const fragment = document.createDocumentFragment();
  lines.forEach((line) => {
    const lineElement = document.createElement("span");
    lineElement.className = "config-line";
    const trimmed = line.trimStart();
    if (trimmed.startsWith("#")) {
      const comment = document.createElement("span");
      comment.className = "config-comment";
      comment.textContent = line;
      lineElement.append(comment);
    } else {
      const match = line.match(/^(\s*)(\S+)(.*)$/);
      if (match) {
        lineElement.append(document.createTextNode(match[1]));
        const key = document.createElement("span");
        key.className = "config-key";
        key.textContent = match[2];
        lineElement.append(key, document.createTextNode(match[3]));
      }
    }
    fragment.append(lineElement);
  });
  code.replaceChildren(fragment);
});
