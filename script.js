const menuButton = document.querySelector(".menu-button");
const pages = document.querySelectorAll("[data-page]");
const routeButtons = document.querySelectorAll("[data-route]");
routeButtons.forEach((button) => { button.dataset.routeHandled = "true"; });

function closeMobileMenu() {
  document.body.classList.remove("nav-open");
  menuButton?.setAttribute("aria-expanded", "false");
}

function parseHashRoute() {
  const raw = window.location.hash.replace("#", "");
  if (!raw) return { route: "home", anchor: "" };
  const [route, anchor = ""] = raw.split(":");
  return { route: route || "home", anchor };
}

function showPage(route = "home", anchor = "", updateHistory = true) {
  const targetRoute = document.querySelector(`[data-page="${route}"]`) ? route : "home";

  pages.forEach((page) => {
    page.classList.toggle("active", page.dataset.page === targetRoute);
  });

  const activePage = document.querySelector(`[data-page="${targetRoute}"]`);
  if (window.MathJax?.typesetPromise && activePage) {
    window.MathJax.typesetPromise([activePage]).catch((err) => {
      console.warn("MathJax typeset failed:", err);
    });
  }

  document.body.dataset.currentPage = targetRoute;
  closeMobileMenu();

  if (anchor) {
    setTimeout(() => {
      document.querySelector(`#${anchor}`)?.scrollIntoView({ behavior: "smooth" });
    }, 40);
  } else {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const nextHash = anchor ? `${targetRoute}:${anchor}` : targetRoute;
  if (updateHistory && window.location.hash.replace("#", "") !== nextHash) {
    history.pushState({ route: targetRoute, anchor }, "", `#${nextHash}`);
  }
}

function showPageFromCurrentHash() {
  const { route, anchor } = parseHashRoute();
  showPage(route, anchor, false);
}

menuButton?.addEventListener("click", () => {
  const isOpen = document.body.classList.toggle("nav-open");
  menuButton.setAttribute("aria-expanded", String(isOpen));
});

routeButtons.forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    showPage(button.dataset.route || "home", button.dataset.anchor || "", true);
  });

  button.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showPage(button.dataset.route || "home", button.dataset.anchor || "", true);
    }
  });
});

// Delegated route handling keeps later-added function links clickable too.
document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-route]");
  if (!target) return;
  if (target.dataset.routeHandled === "true") return;
  event.preventDefault();
  showPage(target.dataset.route || "home", target.dataset.anchor || "", true);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const target = event.target.closest("[data-route]");
  if (!target || target.dataset.routeHandled === "true") return;
  event.preventDefault();
  showPage(target.dataset.route || "home", target.dataset.anchor || "", true);
});

window.addEventListener("popstate", showPageFromCurrentHash);
window.addEventListener("hashchange", showPageFromCurrentHash);

showPageFromCurrentHash();

const moduleControls = document.querySelectorAll('[data-module]');
const moduleDetails = document.querySelectorAll('[data-module-detail]');

function showModule(moduleName = 'input') {
  moduleControls.forEach((control) => {
    control.classList.toggle('active', control.dataset.module === moduleName);
  });
  moduleDetails.forEach((detail) => {
    detail.classList.toggle('active', detail.dataset.moduleDetail === moduleName);
  });
}

moduleControls.forEach((control) => {
  control.addEventListener('click', () => showModule(control.dataset.module));
  control.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showModule(control.dataset.module);
    }
  });
});

const plasticityTabButtons = document.querySelectorAll('[data-plasticity-tab]');
const plasticityTabPanels = document.querySelectorAll('[data-plasticity-panel]');

function showPlasticityTab(tabName = 'theory') {
  plasticityTabButtons.forEach((button) => {
    const active = button.dataset.plasticityTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  plasticityTabPanels.forEach((panel) => {
    const active = panel.dataset.plasticityPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

plasticityTabButtons.forEach((button) => {
  button.addEventListener('click', () => showPlasticityTab(button.dataset.plasticityTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showPlasticityTab(button.dataset.plasticityTab);
    }
  });
});


const hyperelasticTabButtons = document.querySelectorAll('[data-hyperelastic-tab]');
const hyperelasticTabPanels = document.querySelectorAll('[data-hyperelastic-panel]');

function showHyperelasticTab(tabName = 'theory') {
  hyperelasticTabButtons.forEach((button) => {
    const active = button.dataset.hyperelasticTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  hyperelasticTabPanels.forEach((panel) => {
    const active = panel.dataset.hyperelasticPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

hyperelasticTabButtons.forEach((button) => {
  button.addEventListener('click', () => showHyperelasticTab(button.dataset.hyperelasticTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showHyperelasticTab(button.dataset.hyperelasticTab);
    }
  });
});



const viscoelasticTabButtons = document.querySelectorAll('[data-viscoelastic-tab]');
const viscoelasticTabPanels = document.querySelectorAll('[data-viscoelastic-panel]');

function showViscoelasticTab(tabName = 'theory') {
  viscoelasticTabButtons.forEach((button) => {
    const active = button.dataset.viscoelasticTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  viscoelasticTabPanels.forEach((panel) => {
    const active = panel.dataset.viscoelasticPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

viscoelasticTabButtons.forEach((button) => {
  button.addEventListener('click', () => showViscoelasticTab(button.dataset.viscoelasticTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showViscoelasticTab(button.dataset.viscoelasticTab);
    }
  });
});


const newmarkTabButtons = document.querySelectorAll('[data-newmark-tab]');
const newmarkTabPanels = document.querySelectorAll('[data-newmark-panel]');

function showNewmarkTab(tabName = 'theory') {
  newmarkTabButtons.forEach((button) => {
    const active = button.dataset.newmarkTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  newmarkTabPanels.forEach((panel) => {
    const active = panel.dataset.newmarkPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

newmarkTabButtons.forEach((button) => {
  button.addEventListener('click', () => showNewmarkTab(button.dataset.newmarkTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showNewmarkTab(button.dataset.newmarkTab);
    }
  });
});


const fluidTabButtons = document.querySelectorAll('[data-fluid-tab]');
const fluidTabPanels = document.querySelectorAll('[data-fluid-panel]');

function showFluidTab(tabName = 'theory') {
  fluidTabButtons.forEach((button) => {
    const active = button.dataset.fluidTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  fluidTabPanels.forEach((panel) => {
    const active = panel.dataset.fluidPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

fluidTabButtons.forEach((button) => {
  button.addEventListener('click', () => showFluidTab(button.dataset.fluidTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showFluidTab(button.dataset.fluidTab);
    }
  });
});

const fileTabButtons = document.querySelectorAll('[data-file-tab]');
const fileTabPanels = document.querySelectorAll('[data-file-panel]');

function showFileTab(tabName = 'data') {
  fileTabButtons.forEach((button) => {
    const active = button.dataset.fileTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
  fileTabPanels.forEach((panel) => {
    panel.classList.toggle('active', panel.dataset.filePanel === tabName);
  });
}

fileTabButtons.forEach((button) => {
  button.addEventListener('click', () => showFileTab(button.dataset.fileTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showFileTab(button.dataset.fileTab);
    }
  });
});

const vemLinearTabButtons = document.querySelectorAll('[data-vem-linear-tab]');
const vemLinearTabPanels = document.querySelectorAll('[data-vem-linear-panel]');

function showVemLinearTab(tabName = 'theory') {
  vemLinearTabButtons.forEach((button) => {
    const active = button.dataset.vemLinearTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  vemLinearTabPanels.forEach((panel) => {
    const active = panel.dataset.vemLinearPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

vemLinearTabButtons.forEach((button) => {
  button.addEventListener('click', () => showVemLinearTab(button.dataset.vemLinearTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showVemLinearTab(button.dataset.vemLinearTab);
    }
  });
});

const solidMethodTabButtons = document.querySelectorAll('[data-solid-method-tab]');
const solidMethodTabPanels = document.querySelectorAll('[data-solid-method-panel]');

function showSolidMethodTab(tabName = 'fem') {
  solidMethodTabButtons.forEach((button) => {
    const active = button.dataset.solidMethodTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  solidMethodTabPanels.forEach((panel) => {
    const active = panel.dataset.solidMethodPanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

solidMethodTabButtons.forEach((button) => {
  button.addEventListener('click', () => showSolidMethodTab(button.dataset.solidMethodTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showSolidMethodTab(button.dataset.solidMethodTab);
    }
  });
});


// Application hover preview: keep the real images optional.
// Later, replace the CSS placeholder graphics or add background images per preview type.
const applicationPreviewControls = document.querySelectorAll('[data-preview]');
const applicationPreviewBoxes = document.querySelectorAll('[data-preview-box]');

function setApplicationPreview(previewName = 'solid', scope = document) {
  const boxes = scope.querySelectorAll ? scope.querySelectorAll('[data-preview-box]') : applicationPreviewBoxes;
  boxes.forEach((box) => {
    box.dataset.activePreview = previewName;
    box.querySelectorAll('[data-preview-panel]').forEach((panel) => {
      panel.classList.toggle('active', panel.dataset.previewPanel === previewName);
    });
  });

  applicationPreviewControls.forEach((control) => {
    control.classList.toggle('preview-active', control.dataset.preview === previewName);
  });
}

applicationPreviewControls.forEach((control) => {
  const activate = () => {
    const previewName = control.dataset.preview || 'solid';
    const localScope = control.closest('section') || document;
    setApplicationPreview(previewName, localScope);
  };

  control.addEventListener('mouseenter', activate);
  control.addEventListener('focus', activate);
  control.addEventListener('touchstart', activate, { passive: true });
});

setApplicationPreview('solid');

const rveTabButtons = document.querySelectorAll('[data-rve-tab]');
const rveTabPanels = document.querySelectorAll('[data-rve-panel]');

function showRveTab(tabName = 'theory') {
  rveTabButtons.forEach((button) => {
    const active = button.dataset.rveTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  rveTabPanels.forEach((panel) => {
    const active = panel.dataset.rvePanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

rveTabButtons.forEach((button) => {
  button.addEventListener('click', () => showRveTab(button.dataset.rveTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showRveTab(button.dataset.rveTab);
    }
  });
});

const macroMultiscaleTabButtons = document.querySelectorAll('[data-macro-multiscale-tab]');
const macroMultiscaleTabPanels = document.querySelectorAll('[data-macro-multiscale-panel]');

function showMacroMultiscaleTab(tabName = 'theory') {
  macroMultiscaleTabButtons.forEach((button) => {
    const active = button.dataset.macroMultiscaleTab === tabName;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  macroMultiscaleTabPanels.forEach((panel) => {
    const active = panel.dataset.macroMultiscalePanel === tabName;
    panel.classList.toggle('active', active);
    if (active && window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((err) => {
        console.warn('MathJax typeset failed:', err);
      });
    }
  });
}

macroMultiscaleTabButtons.forEach((button) => {
  button.addEventListener('click', () => showMacroMultiscaleTab(button.dataset.macroMultiscaleTab));
  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      showMacroMultiscaleTab(button.dataset.macroMultiscaleTab);
    }
  });
});
