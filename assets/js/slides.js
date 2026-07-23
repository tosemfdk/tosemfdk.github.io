(() => {
  "use strict";

  const deck = document.querySelector("[data-deck]");
  if (!deck) return;

  const slides = Array.from(deck.querySelectorAll("[data-slide]"));
  if (slides.length === 0) return;

  const progress = deck.querySelector(".deck-progress");
  const progressBar = progress?.querySelector("span");
  const currentLabel = deck.querySelector("[data-current-slide]");
  const totalLabel = deck.querySelector("[data-total-slides]");
  const helpDialog = document.querySelector("[data-help-dialog]");
  const overviewButton = deck.querySelector('[data-action="overview"]');
  const fullscreenButton = deck.querySelector('[data-action="fullscreen"]');

  let currentIndex = 0;
  let visibleFragments = 0;
  let overviewOpen = false;
  let pointerStart = null;

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
  const fragmentsFor = (slide) => Array.from(slide.querySelectorAll("[data-fragment]"));

  function indexFromHash() {
    const match = window.location.hash.match(/^#\/?(\d+)$/);
    return match ? clamp(Number(match[1]) - 1, 0, slides.length - 1) : 0;
  }

  function syncMedia() {
    slides.forEach((slide, index) => {
      slide.querySelectorAll("video, audio").forEach((media) => {
        if (index === currentIndex) {
          const playback = media.play();
          if (playback) playback.catch(() => {});
        } else {
          media.pause();
        }
      });
    });
  }

  function render({ updateHash = true } = {}) {
    slides.forEach((slide, index) => {
      const active = index === currentIndex;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", active ? "false" : "true");
      slide.tabIndex = active ? 0 : -1;

      fragmentsFor(slide).forEach((fragment, fragmentIndex) => {
        const visible = active && fragmentIndex < visibleFragments;
        fragment.classList.toggle("is-visible", visible);
        fragment.setAttribute("aria-hidden", visible ? "false" : "true");
      });
    });

    const current = currentIndex + 1;
    const percent = (current / slides.length) * 100;

    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progress) {
      progress.setAttribute("aria-valuenow", String(current));
      progress.setAttribute("aria-valuemax", String(slides.length));
    }
    if (currentLabel) currentLabel.textContent = String(current);
    if (totalLabel) totalLabel.textContent = String(slides.length);

    deck.dataset.firstSlide = String(currentIndex === 0 && visibleFragments === 0);
    deck.dataset.lastSlide = String(
      currentIndex === slides.length - 1 &&
      visibleFragments === fragmentsFor(slides[currentIndex]).length
    );

    if (updateHash) {
      history.replaceState(null, "", `#/${current}`);
    }

    syncMedia();
  }

  function goTo(index, { revealFragments = false, updateHash = true } = {}) {
    currentIndex = clamp(index, 0, slides.length - 1);
    visibleFragments = revealFragments ? fragmentsFor(slides[currentIndex]).length : 0;
    closeOverview();
    render({ updateHash });
  }

  function next() {
    if (overviewOpen) {
      closeOverview();
      return;
    }

    const fragments = fragmentsFor(slides[currentIndex]);
    if (visibleFragments < fragments.length) {
      visibleFragments += 1;
      render();
      return;
    }

    if (currentIndex < slides.length - 1) goTo(currentIndex + 1);
  }

  function previous() {
    if (overviewOpen) {
      closeOverview();
      return;
    }

    if (visibleFragments > 0) {
      visibleFragments -= 1;
      render();
      return;
    }

    if (currentIndex > 0) goTo(currentIndex - 1, { revealFragments: true });
  }

  function openOverview() {
    overviewOpen = true;
    document.body.classList.add("is-overview");
    overviewButton?.setAttribute("aria-pressed", "true");
    slides.forEach((slide) => {
      slide.setAttribute("aria-hidden", "false");
      slide.tabIndex = 0;
    });
  }

  function closeOverview() {
    if (!overviewOpen) return;
    overviewOpen = false;
    document.body.classList.remove("is-overview");
    overviewButton?.setAttribute("aria-pressed", "false");
    render();
  }

  function toggleOverview() {
    if (overviewOpen) closeOverview();
    else openOverview();
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      // Fullscreen can be blocked by browser or embedding policy.
    }
  }

  function toggleHelp() {
    if (!helpDialog) return;
    if (helpDialog.open) helpDialog.close();
    else helpDialog.showModal();
  }

  deck.addEventListener("click", (event) => {
    const actionButton = event.target.closest("[data-action]");
    if (actionButton) {
      const action = actionButton.dataset.action;
      if (action === "next") next();
      if (action === "previous") previous();
      if (action === "overview") toggleOverview();
      if (action === "fullscreen") toggleFullscreen();
      return;
    }

    if (overviewOpen) {
      const selectedSlide = event.target.closest("[data-slide]");
      if (selectedSlide) goTo(slides.indexOf(selectedSlide));
    }
  });

  deck.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse") return;
    pointerStart = { x: event.clientX, y: event.clientY };
  });

  deck.addEventListener("pointerup", (event) => {
    if (!pointerStart || event.pointerType === "mouse") return;

    const deltaX = event.clientX - pointerStart.x;
    const deltaY = event.clientY - pointerStart.y;
    pointerStart = null;

    if (Math.abs(deltaX) < 45 || Math.abs(deltaX) < Math.abs(deltaY)) return;
    if (deltaX < 0) next();
    else previous();
  });

  window.addEventListener("keydown", (event) => {
    const tagName = document.activeElement?.tagName;
    if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") return;

    if (event.key === "ArrowRight" || event.key === "ArrowDown" || event.key === "PageDown" || event.key === " ") {
      event.preventDefault();
      next();
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp" || event.key === "PageUp") {
      event.preventDefault();
      previous();
    } else if (event.key === "Home") {
      event.preventDefault();
      goTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      goTo(slides.length - 1, { revealFragments: true });
    } else if (event.key.toLowerCase() === "o") {
      event.preventDefault();
      toggleOverview();
    } else if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      toggleFullscreen();
    } else if (event.key === "?") {
      event.preventDefault();
      toggleHelp();
    } else if (event.key === "Escape" && overviewOpen) {
      closeOverview();
    }
  });

  window.addEventListener("hashchange", () => {
    goTo(indexFromHash(), { updateHash: false });
  });

  document.addEventListener("fullscreenchange", () => {
    fullscreenButton?.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
  });

  currentIndex = indexFromHash();
  render({ updateHash: false });
})();
