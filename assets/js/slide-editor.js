(() => {
  "use strict";

  const script = document.currentScript;
  const deckSlug = script?.dataset.deckSlug;
  const deck = document.querySelector("[data-deck]");
  const slides = Array.from(deck?.querySelectorAll("[data-slide]") || []);
  if (!deckSlug || !deck || slides.length === 0) return;

  const API_ROOT = "/slide-editor-api";
  const TOKEN_KEY = `slide-editor-token:${deckSlug}`;
  const EDITABLE_SELECTOR =
    "h1, h2, h3, p, li, td, th, figcaption, img, article, figure";
  const TEXT_EDITABLE_SELECTOR = "h1, h2, h3, p, li, td, th, figcaption";
  const FONT_OPTIONS = {
    Pretendard: "Pretendard, Inter, sans-serif",
    Inter: "Inter, Pretendard, sans-serif",
    Avenir: '"Avenir Next", Avenir, sans-serif',
    명조: '"Nanum Myeongjo", "Noto Serif KR", serif',
    고딕: '"Noto Sans KR", Pretendard, sans-serif',
    Mono: '"SFMono-Regular", Consolas, monospace',
  };
  const ANIMATION_LABELS = {
    none: "없음",
    "fade-in": "서서히 나타나기",
    "slide-in-right": "오른쪽에서",
    "slide-in-left": "왼쪽에서",
    "slide-in-up": "아래에서",
    "slide-in-down": "위에서",
    "zoom-in": "확대하며 나타나기",
  };
  const STYLE_KEYS = [
    "fontFamily",
    "fontSize",
    "fontWeight",
    "color",
    "backgroundColor",
    "textAlign",
    "opacity",
    "left",
    "top",
    "width",
    "height",
    "borderRadius",
  ];

  let state = emptyState();
  let editorActive = false;
  let selectedId = null;
  let shell = null;
  let saveTimer = null;
  let saving = false;
  let saveQueued = false;
  let activeSave = Promise.resolve();
  let undoStack = [];
  let redoStack = [];
  let textEditBaseline = null;
  const baseSnapshots = new Map();

  function emptyState() {
    return {
      schemaVersion: 1,
      deckSlug,
      revision: 0,
      frozen: false,
      updatedAt: new Date().toISOString(),
      objectPatches: {},
      additions: [],
      comments: [],
    };
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function assignEditorIds() {
    slides.forEach((slide, slideIndex) => {
      slide.dataset.editorSlideId = `slide-${slideIndex + 1}`;
      const counts = {};
      slide.querySelectorAll(EDITABLE_SELECTOR).forEach((element) => {
        if (element.closest("[data-editor-ui]")) return;
        const tag = element.tagName.toLowerCase();
        counts[tag] = (counts[tag] || 0) + 1;
        element.dataset.editorId ||= `slide-${slideIndex + 1}-${tag}-${counts[tag]}`;
        if (!baseSnapshots.has(element.dataset.editorId)) {
          baseSnapshots.set(element.dataset.editorId, {
            text: element.tagName === "IMG" ? null : element.textContent,
            src: element.tagName === "IMG" ? element.getAttribute("src") : null,
            alt: element.tagName === "IMG" ? element.getAttribute("alt") : null,
            style: element.getAttribute("style") || "",
            animation: element.dataset.editorAnimation || "",
            hidden: element.hidden,
          });
        }
      });
    });
  }

  function elementFor(id) {
    return deck.querySelector(`[data-editor-id="${CSS.escape(id)}"]`);
  }

  function restoreBaseObjects() {
    for (const [id, snapshot] of baseSnapshots) {
      const element = elementFor(id);
      if (!element) continue;
      if (snapshot.text !== null) element.textContent = snapshot.text;
      if (snapshot.src !== null) element.setAttribute("src", snapshot.src);
      if (snapshot.alt !== null) element.setAttribute("alt", snapshot.alt || "");
      if (snapshot.style) element.setAttribute("style", snapshot.style);
      else element.removeAttribute("style");
      if (snapshot.animation) element.dataset.editorAnimation = snapshot.animation;
      else delete element.dataset.editorAnimation;
      element.hidden = snapshot.hidden;
      delete element.dataset.commentCount;
      element.classList.remove("editor-object-selected", "editor-translated");
      element.style.removeProperty("--editor-x");
      element.style.removeProperty("--editor-y");
      element.removeAttribute("contenteditable");
    }
    deck.querySelectorAll(".editor-added-object").forEach((element) => element.remove());
  }

  function applyStyles(element, styles = {}) {
    for (const key of STYLE_KEYS) {
      if (styles[key] !== undefined && styles[key] !== "") {
        element.style[key] = styles[key];
      }
    }

    if (styles.translateX || styles.translateY) {
      element.classList.add("editor-translated");
      element.style.setProperty("--editor-x", styles.translateX || "0px");
      element.style.setProperty("--editor-y", styles.translateY || "0px");
    }
  }

  function applyPatch(element, patch) {
    if (!element || !patch) return;
    if (patch.text !== undefined && element.tagName !== "IMG") {
      element.textContent = patch.text;
    }
    if (patch.src && element.tagName === "IMG") element.setAttribute("src", patch.src);
    if (patch.alt !== undefined && element.tagName === "IMG") {
      element.setAttribute("alt", patch.alt);
    }
    applyStyles(element, patch.styles);
    if (patch.animation && patch.animation !== "none") {
      element.dataset.editorAnimation = patch.animation;
    }
    element.hidden = Boolean(patch.hidden);
  }

  function createAddition(addition) {
    const slide = slides[addition.slideIndex];
    if (!slide || addition.hidden) return null;

    let element;
    if (addition.type === "image") {
      element = document.createElement("img");
      element.src = addition.src;
      element.alt = addition.alt || "추가한 이미지";
    } else {
      element = document.createElement(addition.type === "shape" ? "div" : "p");
      element.textContent = addition.text || (addition.type === "shape" ? "" : "새 텍스트");
    }

    element.className = `editor-added-object editor-added-object--${addition.type}`;
    element.dataset.editorId = addition.id;
    element.dataset.editorAddition = "true";
    element.style.left = addition.styles?.left || "50%";
    element.style.top = addition.styles?.top || "50%";
    applyStyles(element, addition.styles);
    if (addition.animation && addition.animation !== "none") {
      element.dataset.editorAnimation = addition.animation;
    }
    slide.append(element);
    return element;
  }

  function applyState(nextState) {
    state = {
      ...emptyState(),
      ...nextState,
      objectPatches: nextState?.objectPatches || {},
      additions: nextState?.additions || [],
      comments: nextState?.comments || [],
    };
    restoreBaseObjects();
    for (const [id, patch] of Object.entries(state.objectPatches)) {
      applyPatch(elementFor(id), patch);
    }
    state.additions.forEach(createAddition);
    annotateComments();
    renderSelection();
    refreshInspector();
    refreshStatus();
  }

  function annotateComments() {
    const counts = new Map();
    state.comments
      .filter((comment) => comment.status !== "resolved")
      .forEach((comment) => counts.set(comment.objectId, (counts.get(comment.objectId) || 0) + 1));
    for (const [id, count] of counts) {
      const element = elementFor(id);
      if (element) element.dataset.commentCount = String(count);
    }
  }

  async function loadPublishedState() {
    try {
      const response = await fetch(`/assets/data/slides/${deckSlug}.json`, {
        cache: "no-store",
      });
      if (!response.ok) return emptyState();
      return await response.json();
    } catch {
      return emptyState();
    }
  }

  function editorToken() {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  }

  async function api(path, options = {}) {
    const response = await fetch(`${API_ROOT}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${editorToken()}`,
        ...(options.headers || {}),
      },
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 401) {
      sessionStorage.removeItem(TOKEN_KEY);
      throw new Error("AUTH_REQUIRED");
    }
    if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
    return body;
  }

  function showLogin() {
    let login = document.querySelector("[data-editor-login]");
    if (!login) {
      login = document.createElement("dialog");
      login.dataset.editorLogin = "";
      login.dataset.editorUi = "";
      login.className = "slide-editor-login";
      login.innerHTML = `
        <form method="dialog">
          <span class="editor-brand">LOE SLIDE STUDIO</span>
          <h2>편집 모드 잠금 해제</h2>
          <p>이 Mac에 저장된 소유자용 Editor Key를 입력하세요.</p>
          <label>Editor Key<input type="password" name="token" autocomplete="off" required></label>
          <p class="editor-login-error" aria-live="polite"></p>
          <div><a href="${location.pathname}">취소</a><button value="login">편집 시작</button></div>
        </form>`;
      document.body.append(login);
      login.addEventListener("close", async () => {
        if (login.returnValue !== "login") return;
        const token = new FormData(login.querySelector("form")).get("token")?.trim();
        sessionStorage.setItem(TOKEN_KEY, token || "");
        try {
          await activateEditor();
        } catch (error) {
          login.querySelector(".editor-login-error").textContent =
            error.message === "AUTH_REQUIRED" ? "Editor Key가 올바르지 않습니다." : error.message;
          login.showModal();
        }
      });
    }
    login.showModal();
    login.querySelector("input").focus();
  }

  function shellMarkup() {
    const fontOptions = Object.keys(FONT_OPTIONS)
      .map((font) => `<option value="${font}">${font}</option>`)
      .join("");
    const animationOptions = Object.entries(ANIMATION_LABELS)
      .map(([value, label]) => `<option value="${value}">${label}</option>`)
      .join("");

    return `
      <header class="slide-editor-topbar" data-editor-ui>
        <span class="editor-brand">LOE SLIDE STUDIO</span>
        <span class="editor-save-status" data-editor-status>Draft</span>
        <nav>
          <button type="button" data-editor-action="previous" title="이전 슬라이드">←</button>
          <button type="button" data-editor-action="next" title="다음 슬라이드">→</button>
          <button type="button" data-editor-action="undo" title="실행 취소">↶</button>
          <button type="button" data-editor-action="redo" title="다시 실행">↷</button>
        </nav>
        <div class="editor-topbar-actions">
          <button type="button" data-editor-action="save">저장</button>
          <button type="button" data-editor-action="preview">미리보기</button>
          <button type="button" class="editor-publish" data-editor-action="publish">Freeze &amp; Publish</button>
          <button type="button" data-editor-action="close" aria-label="편집기 닫기">×</button>
        </div>
      </header>

      <aside class="slide-editor-insert" data-editor-ui aria-label="콘텐츠 추가">
        <button type="button" data-editor-action="add-text"><b>T</b><span>텍스트</span></button>
        <button type="button" data-editor-action="add-image"><b>▧</b><span>이미지</span></button>
        <button type="button" data-editor-action="add-shape"><b>●</b><span>도형</span></button>
        <input type="file" data-editor-image-input accept="image/png,image/jpeg,image/webp,image/gif,image/avif" hidden>
      </aside>

      <aside class="slide-editor-inspector" data-editor-ui>
        <div class="editor-inspector-heading">
          <span>SELECTED OBJECT</span>
          <strong data-editor-selected-label>객체를 선택하세요</strong>
        </div>

        <fieldset data-editor-object-fields disabled>
          <legend>Typography</legend>
          <label>폰트<select data-editor-field="font">${fontOptions}</select></label>
          <label>크기<input data-editor-field="fontSize" type="number" min="8" max="220" placeholder="px"></label>
          <label>색상<input data-editor-field="color" type="color" value="#111827"></label>
          <div class="editor-segmented">
            <button type="button" data-editor-align="left">왼쪽</button>
            <button type="button" data-editor-align="center">가운데</button>
            <button type="button" data-editor-align="right">오른쪽</button>
          </div>
        </fieldset>

        <fieldset data-editor-object-fields disabled>
          <legend>Motion &amp; Position</legend>
          <label>등장 효과<select data-editor-field="animation">${animationOptions}</select></label>
          <div class="editor-position-grid">
            <label>X<input data-editor-field="x" type="number" step="1" value="0"></label>
            <label>Y<input data-editor-field="y" type="number" step="1" value="0"></label>
          </div>
          <button type="button" data-editor-action="preview-animation">애니메이션 미리보기</button>
          <button type="button" class="editor-danger" data-editor-action="delete-object">객체 삭제</button>
        </fieldset>

        <section class="editor-comments">
          <div><span>OBJECT COMMENTS</span><b data-editor-comment-count>0</b></div>
          <div data-editor-comment-list class="editor-comment-list"></div>
          <textarea data-editor-comment-input placeholder="이 객체에 대한 수정 지시를 남기세요"></textarea>
          <button type="button" data-editor-action="add-comment">댓글 추가</button>
        </section>
      </aside>

      <footer class="slide-editor-command" data-editor-ui>
        <div class="editor-command-copy">
          <span>COMMAND SKILLS</span>
          <small>선택한 객체에 자연어로 명령하세요</small>
        </div>
        <input data-editor-command-input type="text" placeholder='예: "오른쪽에서 나타나서 왼쪽에 위치하도록 해줘"'>
        <button type="button" data-editor-action="run-command">적용</button>
        <div class="editor-command-examples">
          <button type="button" data-editor-example="오른쪽에서 나타나게 해줘">오른쪽에서 등장</button>
          <button type="button" data-editor-example="폰트를 명조로 바꿔줘">명조 폰트</button>
          <button type="button" data-editor-example="가운데 정렬해줘">가운데 정렬</button>
        </div>
      </footer>
      <div class="slide-editor-toast" data-editor-toast data-editor-ui role="status"></div>
    `;
  }

  function buildShell() {
    shell = document.createElement("div");
    shell.className = "slide-editor-shell";
    shell.dataset.editorUi = "";
    shell.innerHTML = shellMarkup();
    document.body.append(shell);

    shell.addEventListener("click", handleShellClick);
    shell.querySelector("[data-editor-image-input]").addEventListener("change", handleImageUpload);
    shell.querySelectorAll("[data-editor-field]").forEach((field) => {
      field.addEventListener("change", handleInspectorChange);
    });
    shell.querySelector("[data-editor-command-input]").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        runCommandSkill(event.currentTarget.value);
      }
    });
    window.addEventListener("resize", updateWorkspaceScale);
  }

  function updateWorkspaceScale() {
    if (!editorActive || window.innerWidth <= 900) return;
    const horizontalRoom = window.innerWidth - 88 - 352 - 24;
    const verticalRoom = window.innerHeight - 58 - 78 - 48;
    const scale = Math.max(
      0.35,
      Math.min(horizontalRoom / window.innerWidth, verticalRoom / window.innerHeight, 0.88)
    );
    deck.querySelector(".deck-slides")?.style.setProperty("--editor-deck-scale", scale);
  }

  function workspaceScale() {
    if (window.innerWidth <= 900) return 1;
    return (
      Number.parseFloat(
        getComputedStyle(deck.querySelector(".deck-slides")).getPropertyValue(
          "--editor-deck-scale"
        )
      ) || 1
    );
  }

  async function activateEditor() {
    const draft = await api(`/decks/${deckSlug}`);
    if (!shell) buildShell();
    editorActive = true;
    document.body.classList.add("is-slide-editing");
    updateWorkspaceScale();
    applyState(draft);
    deck.addEventListener("click", selectObjectFromEvent, true);
    deck.addEventListener("dblclick", enableTextEdit, true);
    deck.addEventListener("pointerdown", startDrag, true);
    toast("편집 모드가 열렸습니다.");
  }

  function selectObjectFromEvent(event) {
    if (!editorActive || document.body.classList.contains("slide-editor-preview")) return;
    const target = event.target.closest("[data-editor-id]");
    if (!target || target.closest("[data-editor-ui]")) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    selectObject(target.dataset.editorId);
  }

  function selectObject(id) {
    selectedId = id;
    renderSelection();
    refreshInspector();
  }

  function renderSelection() {
    deck.querySelectorAll(".editor-object-selected").forEach((element) => {
      element.classList.remove("editor-object-selected");
    });
    if (editorActive && selectedId) elementFor(selectedId)?.classList.add("editor-object-selected");
  }

  function enableTextEdit(event) {
    if (!editorActive || document.body.classList.contains("slide-editor-preview")) return;
    const target = event.target.closest("[data-editor-id]");
    if (!target?.matches(TEXT_EDITABLE_SELECTOR)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    selectObject(target.dataset.editorId);
    textEditBaseline = clone(state);
    target.contentEditable = "true";
    target.focus();
    const selection = window.getSelection();
    selection?.selectAllChildren(target);
    selection?.collapseToEnd();
    target.addEventListener(
      "blur",
      () => {
        target.removeAttribute("contenteditable");
        const patch = patchForSelected();
        if (!patch) return;
        patch.text = target.textContent;
        if (textEditBaseline) {
          undoStack.push(textEditBaseline);
          redoStack = [];
          textEditBaseline = null;
        }
        scheduleSave();
      },
      { once: true }
    );
  }

  function selectedAddition() {
    return state.additions.find((addition) => addition.id === selectedId);
  }

  function patchForSelected() {
    if (!selectedId) return null;
    const addition = selectedAddition();
    if (addition) return addition;
    state.objectPatches[selectedId] ||= {
      styles: {},
      animation: "none",
      hidden: false,
    };
    state.objectPatches[selectedId].styles ||= {};
    return state.objectPatches[selectedId];
  }

  function mutate(callback, message) {
    undoStack.push(clone(state));
    if (undoStack.length > 80) undoStack.shift();
    redoStack = [];
    callback();
    applyState(state);
    scheduleSave();
    if (message) toast(message);
  }

  function scheduleSave() {
    state.frozen = false;
    saveQueued = true;
    refreshStatus("변경사항 저장 대기");
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveDraft().catch(handleEditorError), 900);
  }

  async function saveDraft() {
    clearTimeout(saveTimer);
    if (saving) {
      saveQueued = true;
      return activeSave;
    }

    activeSave = (async () => {
      saving = true;
      try {
        do {
          saveQueued = false;
          refreshStatus("저장 중…");
          const saved = await api(`/decks/${deckSlug}`, {
            method: "PUT",
            body: JSON.stringify(state),
          });
          state.revision = saved.revision;
          state.updatedAt = saved.updatedAt;
        } while (saveQueued);
        refreshStatus("자동 저장됨");
      } finally {
        saving = false;
      }
    })();

    return activeSave;
  }

  function refreshStatus(override) {
    if (!shell) return;
    const status = shell.querySelector("[data-editor-status]");
    status.textContent =
      override || `Draft · r${state.revision}${state.frozen ? " · Frozen" : ""}`;
  }

  function refreshInspector() {
    if (!shell) return;
    const element = selectedId ? elementFor(selectedId) : null;
    const patch = selectedId
      ? selectedAddition() || state.objectPatches[selectedId] || { styles: {}, animation: "none" }
      : null;
    shell.querySelector("[data-editor-selected-label]").textContent = element
      ? `${element.tagName.toLowerCase()} · ${selectedId}`
      : "객체를 선택하세요";
    shell.querySelectorAll("[data-editor-object-fields]").forEach((field) => {
      field.disabled = !element;
    });

    if (element && patch) {
      const computed = getComputedStyle(element);
      const fontKey =
        Object.entries(FONT_OPTIONS).find(([, value]) =>
          computed.fontFamily.includes(value.split(",")[0].replaceAll('"', ""))
        )?.[0] || "Pretendard";
      shell.querySelector('[data-editor-field="font"]').value = fontKey;
      shell.querySelector('[data-editor-field="fontSize"]').value =
        Number.parseFloat(patch.styles?.fontSize || computed.fontSize) || "";
      shell.querySelector('[data-editor-field="color"]').value = rgbToHex(
        patch.styles?.color || computed.color
      );
      shell.querySelector('[data-editor-field="animation"]').value =
        patch.animation || "none";
      shell.querySelector('[data-editor-field="x"]').value =
        Number.parseFloat(patch.styles?.translateX) || 0;
      shell.querySelector('[data-editor-field="y"]').value =
        Number.parseFloat(patch.styles?.translateY) || 0;
    }
    renderComments();
  }

  function rgbToHex(color) {
    if (/^#[0-9a-f]{6}$/i.test(color || "")) return color;
    const values = String(color || "").match(/\d+/g)?.slice(0, 3).map(Number);
    return values?.length === 3
      ? `#${values.map((value) => value.toString(16).padStart(2, "0")).join("")}`
      : "#111827";
  }

  function handleInspectorChange(event) {
    if (!selectedId) return;
    const field = event.currentTarget.dataset.editorField;
    mutate(() => {
      const patch = patchForSelected();
      patch.styles ||= {};
      if (field === "font") patch.styles.fontFamily = FONT_OPTIONS[event.currentTarget.value];
      if (field === "fontSize") patch.styles.fontSize = `${event.currentTarget.value}px`;
      if (field === "color") patch.styles.color = event.currentTarget.value;
      if (field === "animation") patch.animation = event.currentTarget.value;
      if (field === "x") patch.styles.translateX = `${event.currentTarget.value}px`;
      if (field === "y") patch.styles.translateY = `${event.currentTarget.value}px`;
    });
  }

  function handleShellClick(event) {
    const example = event.target.closest("[data-editor-example]");
    if (example) {
      const input = shell.querySelector("[data-editor-command-input]");
      input.value = example.dataset.editorExample;
      input.focus();
      return;
    }

    const align = event.target.closest("[data-editor-align]");
    if (align && selectedId) {
      mutate(() => {
        patchForSelected().styles.textAlign = align.dataset.editorAlign;
      }, `${align.textContent} 정렬을 적용했습니다.`);
      return;
    }

    const button = event.target.closest("[data-editor-action]");
    if (!button) return;
    const action = button.dataset.editorAction;

    if (action === "previous" || action === "next") navigate(action === "next" ? 1 : -1);
    if (action === "undo") undo();
    if (action === "redo") redo();
    if (action === "save") saveDraft().then(() => toast("저장했습니다.")).catch(handleEditorError);
    if (action === "preview") togglePreview();
    if (action === "publish") publish().catch(handleEditorError);
    if (action === "close") closeEditor();
    if (action === "add-text") addText();
    if (action === "add-image") shell.querySelector("[data-editor-image-input]").click();
    if (action === "add-shape") addShape();
    if (action === "delete-object") deleteSelected();
    if (action === "preview-animation") previewAnimation();
    if (action === "add-comment") addComment();
    if (action === "run-command") {
      runCommandSkill(shell.querySelector("[data-editor-command-input]").value);
    }
  }

  function currentSlideIndex() {
    const match = location.hash.match(/^#\/?(\d+)$/);
    return Math.max(0, Math.min(slides.length - 1, (Number(match?.[1]) || 1) - 1));
  }

  function navigate(delta) {
    const next = Math.max(0, Math.min(slides.length - 1, currentSlideIndex() + delta));
    location.hash = `/${next + 1}`;
    selectedId = null;
    renderSelection();
    refreshInspector();
  }

  function addText(text = "새 텍스트를 입력하세요") {
    const id = `added-${crypto.randomUUID()}`;
    mutate(() => {
      state.additions.push({
        id,
        type: "text",
        slideIndex: currentSlideIndex(),
        text,
        src: "",
        alt: "",
        styles: {
          left: "50%",
          top: "50%",
          width: "18rem",
          fontSize: "2rem",
          color: "#111827",
          textAlign: "center",
        },
        animation: "none",
        hidden: false,
      });
      selectedId = id;
    }, "텍스트 객체를 추가했습니다.");
  }

  function addShape() {
    const id = `added-${crypto.randomUUID()}`;
    mutate(() => {
      state.additions.push({
        id,
        type: "shape",
        slideIndex: currentSlideIndex(),
        text: "",
        src: "",
        alt: "",
        styles: {
          left: "50%",
          top: "50%",
          width: "8rem",
          height: "8rem",
          borderRadius: "50%",
          backgroundColor: "#1788f5",
          opacity: "0.85",
        },
        animation: "none",
        hidden: false,
      });
      selectedId = id;
    }, "도형을 추가했습니다.");
  }

  async function handleImageUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.size > 8 * 1024 * 1024) {
      toast("이미지는 8MB 이하만 추가할 수 있습니다.", true);
      return;
    }
    const data = await fileToDataUrl(file);
    refreshStatus("이미지 업로드 중…");
    try {
      const uploaded = await api(`/uploads/${deckSlug}`, {
        method: "POST",
        body: JSON.stringify({ name: file.name, type: file.type, data }),
      });
      const id = `added-${crypto.randomUUID()}`;
      mutate(() => {
        state.additions.push({
          id,
          type: "image",
          slideIndex: currentSlideIndex(),
          text: "",
          src: uploaded.previewUrl,
          alt: file.name,
          styles: {
            left: "50%",
            top: "50%",
            width: "24rem",
            borderRadius: "1rem",
          },
          animation: "none",
          hidden: false,
        });
        selectedId = id;
      }, "이미지를 추가했습니다.");
    } catch (error) {
      handleEditorError(error);
    }
  }

  function fileToDataUrl(file) {
    return new Promise((resolvePromise, rejectPromise) => {
      const reader = new FileReader();
      reader.onload = () => resolvePromise(reader.result);
      reader.onerror = rejectPromise;
      reader.readAsDataURL(file);
    });
  }

  function deleteSelected() {
    if (!selectedId) return;
    mutate(() => {
      const additionIndex = state.additions.findIndex((item) => item.id === selectedId);
      if (additionIndex >= 0) state.additions.splice(additionIndex, 1);
      else patchForSelected().hidden = true;
      selectedId = null;
    }, "객체를 숨겼습니다.");
  }

  function undo() {
    const previous = undoStack.pop();
    if (!previous) return;
    redoStack.push(clone(state));
    applyState(previous);
    scheduleSave();
  }

  function redo() {
    const next = redoStack.pop();
    if (!next) return;
    undoStack.push(clone(state));
    applyState(next);
    scheduleSave();
  }

  function togglePreview() {
    document.body.classList.toggle("slide-editor-preview");
    shell
      .querySelector('[data-editor-action="preview"]')
      .classList.toggle("is-active", document.body.classList.contains("slide-editor-preview"));
  }

  function previewAnimation() {
    const element = selectedId ? elementFor(selectedId) : null;
    if (!element) return;
    element.classList.remove("editor-animation-preview");
    void element.offsetWidth;
    element.classList.add("editor-animation-preview");
  }

  function closeEditor() {
    const url = new URL(location.href);
    url.searchParams.delete("edit");
    location.href = url.toString();
  }

  async function publish() {
    if (
      !window.confirm(
        "현재 Draft를 공개본으로 Freeze하고 즉시 배포할까요? 댓글은 공개되지 않습니다."
      )
    ) {
      return;
    }
    await saveDraft();
    refreshStatus("프로덕션 배포 중…");
    const result = await api(`/publish/${deckSlug}`, {
      method: "POST",
      body: "{}",
    });
    state.frozen = true;
    refreshStatus(`Published · r${result.revision}`);
    toast(
      result.warnings?.length
        ? `배포 완료 · ${result.warnings.join(" ")}`
        : "공개본 배포와 Git 동기화를 완료했습니다."
    );
  }

  function renderComments() {
    if (!shell) return;
    const list = shell.querySelector("[data-editor-comment-list]");
    const comments = state.comments.filter((comment) => comment.objectId === selectedId);
    list.replaceChildren();
    for (const comment of comments) {
      const item = document.createElement("article");
      item.className = comment.status === "resolved" ? "is-resolved" : "";
      const body = document.createElement("p");
      body.textContent = comment.body;
      const meta = document.createElement("div");
      const time = document.createElement("time");
      time.textContent = new Date(comment.createdAt).toLocaleString("ko-KR");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = comment.status === "resolved" ? "다시 열기" : "해결";
      button.addEventListener("click", () => {
        mutate(() => {
          comment.status = comment.status === "resolved" ? "open" : "resolved";
        });
      });
      meta.append(time, button);
      item.append(body, meta);
      list.append(item);
    }
    shell.querySelector("[data-editor-comment-count]").textContent = String(comments.length);
    shell.querySelector("[data-editor-comment-input]").disabled = !selectedId;
    shell.querySelector('[data-editor-action="add-comment"]').disabled = !selectedId;
  }

  function addComment(text) {
    const input = shell.querySelector("[data-editor-comment-input]");
    const body = String(text || input.value).trim();
    if (!selectedId || !body) return;
    input.value = "";
    mutate(() => {
      state.comments.push({
        id: `comment-${crypto.randomUUID()}`,
        objectId: selectedId,
        body,
        status: "open",
        createdAt: new Date().toISOString(),
      });
    }, "객체에 댓글을 남겼습니다.");
  }

  const commandSkills = [
    {
      id: "comment",
      match: (command) => command.match(/^(?:댓글|메모)\s*[:：]\s*(.+)$/i),
      run: (match) => addComment(match[1]),
    },
    {
      id: "add-text",
      match: (command) => command.match(/^(?:텍스트|문구)\s*추가\s*[:：]\s*(.+)$/i),
      run: (match) => addText(match[1]),
    },
    {
      id: "add-image",
      match: (command) => (/이미지.*추가|사진.*추가/i.test(command) ? [command] : null),
      run: () => shell.querySelector("[data-editor-image-input]").click(),
    },
    {
      id: "enter-right",
      match: (command) =>
        /오른쪽.*(?:나타|등장)|(?:나타|등장).*오른쪽|from\s+right/i.test(command)
          ? [command]
          : null,
      run: (_, command) =>
        updateSelected((patch) => {
          patch.animation = "slide-in-right";
          if (/왼쪽.*(?:위치|정렬)/.test(command)) {
            patch.styles.textAlign = "left";
            if (selectedAddition()) patch.styles.left = "25%";
            else patch.styles.translateX = "-18vw";
          }
        }, "오른쪽 등장 애니메이션을 적용했습니다."),
    },
    {
      id: "enter-left",
      match: (command) =>
        /왼쪽.*(?:나타|등장)|(?:나타|등장).*왼쪽|from\s+left/i.test(command)
          ? [command]
          : null,
      run: () =>
        updateSelected((patch) => {
          patch.animation = "slide-in-left";
        }, "왼쪽 등장 애니메이션을 적용했습니다."),
    },
    {
      id: "fade",
      match: (command) => (/서서히|페이드|fade/i.test(command) ? [command] : null),
      run: () =>
        updateSelected((patch) => {
          patch.animation = "fade-in";
        }, "페이드 애니메이션을 적용했습니다."),
    },
    {
      id: "zoom",
      match: (command) => (/확대.*나타|줌|zoom/i.test(command) ? [command] : null),
      run: () =>
        updateSelected((patch) => {
          patch.animation = "zoom-in";
        }, "확대 애니메이션을 적용했습니다."),
    },
    {
      id: "font",
      match: (command) =>
        command.match(/(?:폰트|글꼴)(?:를|을)?\s*([가-힣A-Za-z]+)(?:로|으로)?/i),
      run: (match) => {
        const requested = match[1].replace(/(?:로|으로)$/u, "");
        const key =
          Object.keys(FONT_OPTIONS).find((font) =>
            requested.toLowerCase().includes(font.toLowerCase())
          ) ||
          (requested.includes("명조") ? "명조" : requested.includes("고딕") ? "고딕" : null);
        if (!key) return toast(`지원하는 폰트: ${Object.keys(FONT_OPTIONS).join(", ")}`, true);
        updateSelected((patch) => {
          patch.styles.fontFamily = FONT_OPTIONS[key];
        }, `${key} 폰트를 적용했습니다.`);
      },
    },
    {
      id: "align",
      match: (command) => command.match(/(왼쪽|가운데|중앙|오른쪽).*(?:정렬|위치)/),
      run: (match) =>
        updateSelected((patch) => {
          patch.styles.textAlign =
            match[1] === "왼쪽" ? "left" : match[1] === "오른쪽" ? "right" : "center";
        }, `${match[1]} 정렬을 적용했습니다.`),
    },
    {
      id: "font-size",
      match: (command) => command.match(/(?:크기|사이즈).*?(\d{1,3})\s*(?:px|픽셀)?/i),
      run: (match) =>
        updateSelected((patch) => {
          patch.styles.fontSize = `${Math.max(8, Math.min(220, Number(match[1])))}px`;
        }, `글자 크기를 ${match[1]}px로 변경했습니다.`),
    },
    {
      id: "delete",
      match: (command) => (/삭제|지워/.test(command) ? [command] : null),
      run: deleteSelected,
    },
  ];

  function updateSelected(callback, message) {
    if (!selectedId) return toast("먼저 슬라이드에서 객체를 선택하세요.", true);
    mutate(() => {
      const patch = patchForSelected();
      patch.styles ||= {};
      callback(patch);
    }, message);
  }

  function runCommandSkill(rawCommand) {
    const command = String(rawCommand || "").trim();
    if (!command) return;
    for (const skill of commandSkills) {
      const match = skill.match(command);
      if (match) {
        skill.run(match, command);
        shell.querySelector("[data-editor-command-input]").value = "";
        return;
      }
    }
    toast("아직 이해하지 못한 명령입니다. 애니메이션·폰트·정렬·크기·댓글 명령을 사용해보세요.", true);
  }

  function startDrag(event) {
    if (
      !editorActive ||
      document.body.classList.contains("slide-editor-preview") ||
      event.button !== 0
    ) {
      return;
    }
    const element = event.target.closest("[data-editor-id]");
    if (!element || element.dataset.editorId !== selectedId || element.isContentEditable) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const baseline = clone(state);
    const patch = patchForSelected();
    patch.styles ||= {};
    const start = {
      x: event.clientX,
      y: event.clientY,
      translateX: Number.parseFloat(patch.styles.translateX) || 0,
      translateY: Number.parseFloat(patch.styles.translateY) || 0,
      left: Number.parseFloat(patch.styles.left) || 50,
      top: Number.parseFloat(patch.styles.top) || 50,
      addition: Boolean(selectedAddition()),
    };

    function move(moveEvent) {
      const deltaX = moveEvent.clientX - start.x;
      const deltaY = moveEvent.clientY - start.y;
      if (start.addition) {
        const slideRect = slides[currentSlideIndex()].getBoundingClientRect();
        patch.styles.left = `${start.left + (deltaX / slideRect.width) * 100}%`;
        patch.styles.top = `${start.top + (deltaY / slideRect.height) * 100}%`;
      } else {
        const scale = workspaceScale();
        patch.styles.translateX = `${start.translateX + deltaX / scale}px`;
        patch.styles.translateY = `${start.translateY + deltaY / scale}px`;
      }
      applyState(state);
    }

    function end() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      undoStack.push(baseline);
      redoStack = [];
      scheduleSave();
    }

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end, { once: true });
  }

  function toast(message, error = false) {
    if (!shell) return;
    const element = shell.querySelector("[data-editor-toast]");
    element.textContent = message;
    element.classList.toggle("is-error", error);
    element.classList.add("is-visible");
    clearTimeout(element._timer);
    element._timer = setTimeout(() => element.classList.remove("is-visible"), 3200);
  }

  function handleEditorError(error) {
    if (error.message === "AUTH_REQUIRED") {
      showLogin();
      return;
    }
    console.error(error);
    toast(error.message || "편집기 오류가 발생했습니다.", true);
  }

  assignEditorIds();
  (async () => {
    applyState(await loadPublishedState());
    const editRequested = new URL(location.href).searchParams.get("edit") === "1";
    if (!editRequested) return;
    if (editorToken()) await activateEditor();
    else showLogin();
  })().catch(handleEditorError);
})();
