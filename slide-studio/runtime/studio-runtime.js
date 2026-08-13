(() => {
  "use strict";

  const root = document.querySelector("[data-studio-deck]");
  const deck = JSON.parse(document.querySelector("#deck-data")?.textContent || "{}");
  const config = JSON.parse(document.querySelector("#deck-config")?.textContent || "{}");
  if (!root || !Array.isArray(deck.slides)) return;

  const viewport = document.createElement("div");
  viewport.className = "studio-viewport";
  const renderedSlides = [];
  let current = Math.max(0, Math.min(deck.slides.length - 1, Number(location.hash.slice(2)) - 1 || 0));
  let clickStep = 0;

  const assetUrl = (assetId) => config.assetMap?.[assetId] || `${config.assetBase || ""}${encodeURIComponent(assetId)}`;
  const applyStyles = (element, styles = {}) => {
    for (const [key, value] of Object.entries(styles)) {
      if (/^[a-zA-Z][a-zA-Z0-9]*$/.test(key) && !["position", "left", "top", "width", "height", "zIndex", "transform"].includes(key)) {
        element.style[key] = String(value);
      }
    }
  };

  function renderObject(object) {
    const element = document.createElement("div");
    element.className = `slide-object slide-object--${object.type}${object.className ? ` ${object.className}` : ""}`;
    element.dataset.objectId = object.id;
    Object.assign(element.style, {
      left: `${object.x}px`, top: `${object.y}px`, width: `${object.width}px`, height: `${object.height}px`,
      zIndex: String(object.zIndex), transform: `rotate(${object.rotation || 0}deg)`
    });
    applyStyles(element, object.styles);

    if (object.type === "text") {
      const text = document.createElement("div");
      text.className = "slide-object__text";
      text.textContent = object.content || "";
      element.append(text);
    } else if (object.type === "shape") {
      element.setAttribute("aria-hidden", "true");
    } else if (object.type === "image") {
      const image = document.createElement("img");
      image.src = assetUrl(object.assetId);
      image.alt = object.content || "발표자료 이미지";
      element.append(image);
    } else if (object.type === "video") {
      const video = document.createElement("video");
      video.src = assetUrl(object.assetId);
      video.controls = true;
      video.preload = "metadata";
      element.append(video);
    } else if (object.type === "audio") {
      const audio = document.createElement("audio");
      audio.src = assetUrl(object.assetId);
      audio.controls = true;
      audio.preload = "metadata";
      element.append(audio);
    } else if (object.type === "pdf") {
      const frame = document.createElement("iframe");
      frame.src = assetUrl(object.assetId);
      frame.title = object.content || "PDF 자료";
      frame.loading = "lazy";
      element.append(frame);
    } else {
      element.classList.add("slide-object--attachment");
      const link = document.createElement("a");
      link.href = assetUrl(object.assetId);
      link.download = object.content || "attachment";
      link.textContent = object.content || "첨부 파일 다운로드";
      element.append(link);
    }

    if (object.animation?.name) {
      element.classList.add("has-animation");
      element.dataset.animationTrigger = object.animation.trigger;
      element.style.setProperty("--object-animation", object.animation.name);
      element.style.setProperty("--object-duration", `${object.animation.durationMs}ms`);
      element.style.setProperty("--object-delay", `${object.animation.delayMs}ms`);
      element.style.setProperty("--object-easing", object.animation.easing || "ease");
      element.style.setProperty("--object-iterations", String(object.animation.iterationCount || 1));
    }
    return element;
  }

  for (const slide of deck.slides) {
    const section = document.createElement("section");
    section.className = "studio-slide";
    section.setAttribute("aria-label", slide.title || "슬라이드");
    section.style.background = slide.background || "#fff";
    [...slide.objects].sort((a, b) => a.zIndex - b.zIndex).forEach((object) => section.append(renderObject(object)));
    viewport.append(section);
    renderedSlides.push(section);
  }
  root.append(viewport);

  const controls = document.createElement("nav");
  controls.className = "studio-controls";
  controls.setAttribute("aria-label", "발표 제어");
  controls.innerHTML = '<button data-action="overview" title="개요 (O)">▦</button><button data-action="previous" title="이전">←</button><span class="studio-counter"></span><button data-action="next" title="다음">→</button><button data-action="fullscreen" title="전체화면 (F)">⛶</button>';
  document.body.append(controls);

  const overview = document.createElement("div");
  overview.className = "studio-overview";
  deck.slides.forEach((slide, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${index + 1}. ${slide.title}`;
    button.addEventListener("click", () => { overview.classList.remove("is-open"); show(index); });
    overview.append(button);
  });
  document.body.append(overview);

  function scale() {
    const value = Math.min(innerWidth / deck.width, innerHeight / deck.height);
    viewport.style.transform = `scale(${value})`;
  }

  function show(index) {
    current = Math.max(0, Math.min(renderedSlides.length - 1, index));
    clickStep = 0;
    renderedSlides.forEach((slide, slideIndex) => {
      slide.classList.toggle("is-active", slideIndex === current);
      slide.querySelectorAll(".has-animation").forEach((element) => element.classList.remove("is-visible"));
    });
    renderedSlides[current]?.querySelectorAll('[data-animation-trigger]:not([data-animation-trigger="click"])')
      .forEach((element) => requestAnimationFrame(() => element.classList.add("is-visible")));
    controls.querySelector(".studio-counter").textContent = `${current + 1} / ${renderedSlides.length}`;
    history.replaceState(null, "", `#/${current + 1}`);
  }

  function next() {
    const pending = [...renderedSlides[current].querySelectorAll('[data-animation-trigger="click"]:not(.is-visible)')];
    if (pending.length) {
      pending[0].classList.add("is-visible");
      clickStep += 1;
      return;
    }
    show(current + 1);
  }

  function previous() {
    const visible = [...renderedSlides[current].querySelectorAll('[data-animation-trigger="click"].is-visible')];
    if (visible.length) {
      visible.at(-1).classList.remove("is-visible");
      clickStep = Math.max(0, clickStep - 1);
      return;
    }
    show(current - 1);
  }

  controls.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "next") next();
    if (action === "previous") previous();
    if (action === "overview") overview.classList.toggle("is-open");
    if (action === "fullscreen") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
  addEventListener("keydown", (event) => {
    if (["ArrowRight", " ", "PageDown"].includes(event.key)) { event.preventDefault(); next(); }
    if (["ArrowLeft", "PageUp"].includes(event.key)) { event.preventDefault(); previous(); }
    if (event.key.toLowerCase() === "o") overview.classList.toggle("is-open");
    if (event.key.toLowerCase() === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    if (event.key === "Escape") overview.classList.remove("is-open");
  });
  addEventListener("resize", scale);
  scale();
  show(current);
})();
