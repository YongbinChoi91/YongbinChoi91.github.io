const explorer = document.querySelector("[data-code-explorer]");

if (explorer) {
  const data = JSON.parse(document.querySelector("#code-explorer-data").textContent);
  const fileList = explorer.querySelector("[data-file-list]");
  const functionList = explorer.querySelector("[data-function-list]");
  const functionCount = explorer.querySelector("[data-function-count]");
  const search = explorer.querySelector("[data-function-search]");
  const petscOverview = explorer.querySelector("[data-petsc-overview]");
  const params = new URLSearchParams(window.location.search);
  let selectedFile = null;

  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);

  const updateUrl = () => {
    const next = new URL(window.location.href);
    if (selectedFile) next.searchParams.set("file", selectedFile.name);
    else next.searchParams.delete("file");
    if (search.value) next.searchParams.set("search", search.value);
    else next.searchParams.delete("search");
    history.replaceState({}, "", next);
  };

  const renderFunctions = () => {
    const needle = search.value.trim().toLowerCase();
    const files = selectedFile ? [selectedFile] : data.files;
    const total = files.reduce((count, file) => count + file.functions.length, 0);
    const matches = files.map((file) => ({
      file,
      functions: file.functions.filter((fn) => `${fn.name} ${fn.signature}`.toLowerCase().includes(needle)),
    }));
    const visibleCount = matches.reduce((count, match) => count + match.functions.length, 0);
    functionCount.textContent = `${visibleCount} of ${total}`;
    functionList.innerHTML = "";

    const appendFunctionLink = (list, fn) => {
      const link = document.createElement("a");
      link.href = fn.detailUrl;
      link.innerHTML = `<span><strong>${escapeHtml(fn.name)}()</strong><small>${escapeHtml(fn.signature)}</small></span><em>line ${fn.line}</em>`;
      list.append(link);
    };

    if (!selectedFile) {
      matches.forEach(({ file, functions }) => {
        if (!functions.length) return;
        const section = document.createElement("section");
        section.className = "function-file-group";
        section.innerHTML = `<h3><span>${escapeHtml(file.name)}</span><small>${functions.length}</small></h3><div class="function-group-list"></div>`;
        const list = section.querySelector(".function-group-list");
        functions.forEach((fn) => appendFunctionLink(list, fn));
        functionList.append(section);
      });
      if (!visibleCount) functionList.innerHTML = '<p class="explorer-empty">No functions match this search.</p>';
      updateUrl();
      return;
    }

    const visible = matches.flatMap(({ functions }) => functions);
    const groups = [
      ["public", "Public API", true],
      ["internal", "Internal", true],
      ["special", "Special members", false],
    ];
    groups.forEach(([key, label, open]) => {
      const functions = visible.filter((fn) => fn.group === key);
      if (!functions.length) return;
      const section = document.createElement("details");
      section.className = "function-group";
      section.open = open;
      section.innerHTML = `<summary><span>${label}</span><small>${functions.length}</small></summary><div class="function-group-list"></div>`;
      const list = section.querySelector(".function-group-list");
      functions.forEach((fn) => appendFunctionLink(list, fn));
      functionList.append(section);
    });
    if (!visible.length) functionList.innerHTML = '<p class="explorer-empty">No functions match this search.</p>';
    updateUrl();
  };

  const showFile = (file) => {
    selectedFile = file;
    fileList.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.file === (file ? file.name : "")));
    const isPetsc = file && data.folder === "solver" && file.name === "linear_solver_petsc.cpp";
    if (petscOverview) petscOverview.hidden = !isPetsc;
    renderFunctions();
  };

  const allFilesButton = document.createElement("button");
  allFilesButton.type = "button";
  allFilesButton.dataset.file = "";
  allFilesButton.innerHTML = `<strong>All files</strong><small>${data.files.reduce((count, file) => count + file.functions.length, 0)}</small>`;
  allFilesButton.addEventListener("click", () => showFile(null));
  fileList.append(allFilesButton);

  data.files.forEach((file) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.file = file.name;
    button.innerHTML = `<strong>${escapeHtml(file.name)}</strong><small>${file.functions.length}</small>`;
    button.addEventListener("click", () => showFile(file));
    fileList.append(button);
  });

  search.value = params.get("search") || "";
  search.addEventListener("input", renderFunctions);

  explorer.querySelectorAll("[data-petsc-tab]").forEach((button) => button.addEventListener("click", () => {
    explorer.querySelectorAll("[data-petsc-tab]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    explorer.querySelectorAll("[data-petsc-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.petscPanel === button.dataset.petscTab));
  }));

  const requested = data.files.find((file) => file.name === params.get("file"));
  showFile(requested || null);
}
