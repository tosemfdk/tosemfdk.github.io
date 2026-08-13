import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { toPng } from "html-to-image";
import { api, assetContentUrl, AuthenticationError, type ProjectDetail } from "./api";
import { clampObject, cloneDeck, createAssetObject, createSlide, findObject } from "./deck-utils";
import type { AiJob, Asset, Deck, DeckObject, Project, SelectionContext, Slide } from "./types";

type CanvasTool = "select" | "point" | "region";
type SaveState = "saved" | "dirty" | "saving" | "error";

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshProjects = useCallback(async () => {
    try {
      const values = await api.listProjects();
      setProjects(values);
      setAuthenticated(true);
    } catch (caught) {
      if (caught instanceof AuthenticationError) setAuthenticated(false);
      else setError(message(caught));
    }
  }, [projectId]);

  useEffect(() => { refreshProjects(); }, [refreshProjects]);

  if (authenticated === null) return <div className="center-screen"><span className="spinner" /> Slide Studio를 불러오는 중…</div>;
  if (!authenticated) return <Login onSuccess={() => { setAuthenticated(true); refreshProjects(); }} />;
  if (!projectId) return <ProjectHome projects={projects} onOpen={setProjectId} onCreated={(project) => { setProjects((items) => [project, ...items]); setProjectId(project.id); }} />;
  return (
    <Studio
      key={projectId}
      projectId={projectId}
      projects={projects}
      onProjectChange={setProjectId}
      onProjectsChange={setProjects}
      onExit={() => setProjectId(null)}
      globalError={error}
      clearGlobalError={() => setError(null)}
    />
  );
}

function Login({ onSuccess }: { onSuccess: () => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <main className="login-screen">
      <form className="login-card" onSubmit={async (event) => {
        event.preventDefault(); setBusy(true); setError("");
        try { await api.login(token); onSuccess(); } catch (caught) { setError(message(caught)); } finally { setBusy(false); }
      }}>
        <p className="eyebrow">PRIVATE AUTHORING</p>
        <h1>LOE Slide Studio</h1>
        <p>관리자 토큰 또는 Cloudflare Access 인증으로 들어갑니다.</p>
        <label>Administrator token<input autoFocus type="password" value={token} onChange={(event) => setToken(event.target.value)} /></label>
        {error && <p className="form-error">{error}</p>}
        <button className="primary" disabled={busy}>{busy ? "확인 중…" : "편집기 열기"}</button>
      </form>
    </main>
  );
}

function ProjectHome({ projects, onOpen, onCreated }: { projects: Project[]; onOpen: (id: string) => void; onCreated: (project: Project) => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <main className="project-home">
      <header><div><p className="eyebrow">AI PRESENTATION WORKSPACE</p><h1>LOE Slide Studio</h1></div></header>
      <section className="new-project-panel">
        <h2>새 발표자료</h2>
        <form onSubmit={async (event) => {
          event.preventDefault(); if (!name.trim()) return; setBusy(true);
          try { const project = await api.createProject(name); onCreated(project); } finally { setBusy(false); }
        }}>
          <input placeholder="발표자료 제목" value={name} onChange={(event) => setName(event.target.value)} />
          <button className="primary" disabled={busy}>{busy ? "생성 중…" : "새 덱 만들기"}</button>
        </form>
      </section>
      <section><h2>프로젝트</h2><div className="project-grid">
        {projects.map((project) => <button className="project-card" key={project.id} onClick={() => onOpen(project.id)}><span className="project-card__cover">16:9</span><strong>{project.name}</strong><small>{new Date(project.updatedAt).toLocaleString("ko-KR")}</small></button>)}
        {!projects.length && <p className="empty-state">첫 발표자료를 만들어보세요.</p>}
      </div></section>
    </main>
  );
}

interface StudioProps {
  projectId: string;
  projects: Project[];
  onProjectChange: (id: string) => void;
  onProjectsChange: (projects: Project[]) => void;
  onExit: () => void;
  globalError: string | null;
  clearGlobalError: () => void;
}

function Studio(props: StudioProps) {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [deck, setDeck] = useState<Deck | null>(null);
  const [slideId, setSlideId] = useState<string>("");
  const [selectedObjectIds, setSelectedObjectIds] = useState<string[]>([]);
  const [tool, setTool] = useState<CanvasTool>("select");
  const [point, setPoint] = useState<SelectionContext["point"]>();
  const [region, setRegion] = useState<SelectionContext["region"]>();
  const [scale, setScale] = useState(0.5);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [toast, setToast] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [activeJob, setActiveJob] = useState<AiJob | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [previewKey, setPreviewKey] = useState(0);
  const [showVersions, setShowVersions] = useState(false);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const undoStack = useRef<Deck[]>([]);
  const redoStack = useRef<Deck[]>([]);
  const dirtyRef = useRef(false);

  const loadProject = useCallback(async () => {
    const value = await api.getProject(props.projectId);
    setDetail(value); setDeck(value.deck); setSlideId((current) => value.deck.slides.some((slide) => slide.id === current) ? current : value.deck.slides[0].id);
    setSelectedObjectIds([]); setSaveState("saved"); dirtyRef.current = false;
    const reviewable = value.jobs.find((job) => ["queued", "running", "ready"].includes(job.status));
    setActiveJob(reviewable || null);
  }, [props.projectId]);

  useEffect(() => { loadProject().catch((caught) => setLocalError(message(caught))); }, [loadProject]);

  useEffect(() => {
    if (!canvasHostRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setScale(Math.min((entry.contentRect.width - 56) / 1920, (entry.contentRect.height - 56) / 1080, 1));
    });
    observer.observe(canvasHostRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!deck || !dirtyRef.current) return;
    const timer = setTimeout(async () => {
      setSaveState("saving");
      try { await api.saveDeck(props.projectId, deck); dirtyRef.current = false; setSaveState("saved"); }
      catch (caught) { setSaveState("error"); setLocalError(message(caught)); }
    }, 700);
    return () => clearTimeout(timer);
  }, [deck, props.projectId]);

  useEffect(() => {
    if (!activeJob || !["queued", "running"].includes(activeJob.status)) return;
    const source = new EventSource(`/api/ai-jobs/${activeJob.id}/events`, { withCredentials: true });
    source.addEventListener("snapshot", (event) => {
      const value = JSON.parse((event as MessageEvent).data); setActiveJob(value.job); setJobLogs(value.logs || []);
    });
    source.addEventListener("log", (event) => {
      const value = JSON.parse((event as MessageEvent).data); setJobLogs((logs) => [...logs.slice(-99), value.line]);
    });
    source.addEventListener("status", (event) => {
      const value = JSON.parse((event as MessageEvent).data); setActiveJob(value.job);
      if (!["queued", "running"].includes(value.job.status)) source.close();
    });
    return () => source.close();
  }, [activeJob?.id, activeJob?.status]);

  const currentSlide = deck?.slides.find((slide) => slide.id === slideId) || deck?.slides[0];
  const selectedObjects = useMemo(() => {
    if (!deck || !currentSlide) return [];
    return selectedObjectIds.flatMap((id) => {
      const object = findObject(deck, currentSlide.id, id);
      return object ? [object] : [];
    });
  }, [currentSlide, deck, selectedObjectIds]);
  const selectedObject = selectedObjects.length === 1 ? selectedObjects[0] : undefined;
  const context: SelectionContext = useMemo(() => ({
    slideId: currentSlide?.id || "",
    selectedObjectIds: selectedObjects.map((object) => object.id),
    ...(point ? { point } : {}),
    ...(region ? { region } : {})
  }), [currentSlide?.id, point, region, selectedObjects]);

  const selectObject = useCallback((objectId: string, additive = false) => {
    setSelectedObjectIds((current) => {
      if (!additive) return current.length === 1 && current[0] === objectId ? current : [objectId];
      return current.includes(objectId) ? current.filter((id) => id !== objectId) : [...current, objectId];
    });
  }, []);

  const updateDeck = useCallback((producer: (next: Deck) => void, record = true) => {
    setDeck((previous) => {
      if (!previous) return previous;
      if (record) { undoStack.current.push(cloneDeck(previous)); if (undoStack.current.length > 100) undoStack.current.shift(); redoStack.current = []; }
      const next = cloneDeck(previous); producer(next); dirtyRef.current = true; setSaveState("dirty"); return next;
    });
  }, []);

  const undo = useCallback(() => {
    const previous = undoStack.current.pop(); if (!previous || !deck) return;
    redoStack.current.push(cloneDeck(deck)); setDeck(previous); dirtyRef.current = true; setSaveState("dirty");
  }, [deck]);
  const redo = useCallback(() => {
    const next = redoStack.current.pop(); if (!next || !deck) return;
    undoStack.current.push(cloneDeck(deck)); setDeck(next); dirtyRef.current = true; setSaveState("dirty");
  }, [deck]);

  const addObject = useCallback((object: DeckObject) => {
    if (!currentSlide) return;
    updateDeck((next) => next.slides.find((slide) => slide.id === currentSlide.id)!.objects.push(object));
    setSelectedObjectIds([object.id]); setTool("select");
  }, [currentSlide, updateDeck]);

  const uploadFiles = useCallback(async (files: File[]) => {
    if (!files.length) return; setUploading(true);
    try {
      const addedObjectIds: string[] = [];
      for (const file of files) {
        const asset = await api.uploadAsset(props.projectId, file);
        setDetail((value) => value ? { ...value, assets: [asset, ...value.assets] } : value);
        const object = createAssetObject(asset, point);
        addObject(object);
        addedObjectIds.push(object.id);
      }
      setSelectedObjectIds(addedObjectIds);
      setToast(`${files.length}개 자료를 캔버스에 추가했습니다.`);
    } catch (caught) { setLocalError(message(caught)); } finally { setUploading(false); }
  }, [addObject, point, props.projectId]);

  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      if ((event.target as HTMLElement)?.closest("input, textarea, [contenteditable=true]")) return;
      const files = [...(event.clipboardData?.files || [])];
      if (files.length) { event.preventDefault(); uploadFiles(files); }
    };
    addEventListener("paste", paste); return () => removeEventListener("paste", paste);
  }, [uploadFiles]);

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.closest("input, textarea, select, [contenteditable=true]")) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a" && currentSlide) {
        event.preventDefault(); setSelectedObjectIds(currentSlide.objects.map((object) => object.id));
      }
      if (event.key === "Escape") setSelectedObjectIds([]);
      if ((event.key === "Delete" || event.key === "Backspace") && selectedObjectIds.length && currentSlide) {
        event.preventDefault();
        const selected = new Set(selectedObjectIds);
        updateDeck((next) => { const slide = next.slides.find((item) => item.id === currentSlide.id)!; slide.objects = slide.objects.filter((object) => !selected.has(object.id)); });
        setSelectedObjectIds([]);
      }
    };
    addEventListener("keydown", keydown); return () => removeEventListener("keydown", keydown);
  }, [currentSlide, redo, selectedObjectIds, undo, updateDeck]);

  if (!detail || !deck || !currentSlide) return <div className="center-screen"><span className="spinner" /> 프로젝트를 여는 중…</div>;

  const patchSelected = (patch: (object: DeckObject) => void) => {
    if (!selectedObject) return;
    updateDeck((next) => {
      const object = next.slides.find((slide) => slide.id === currentSlide.id)!.objects.find((item) => item.id === selectedObject.id)!;
      patch(object);
    });
  };

  const deleteSelected = () => {
    if (!selectedObjectIds.length) return;
    const selected = new Set(selectedObjectIds);
    updateDeck((next) => { const slide = next.slides.find((item) => item.id === currentSlide.id)!; slide.objects = slide.objects.filter((object) => !selected.has(object.id)); });
    setSelectedObjectIds([]);
  };

  const runJob = async () => {
    if (!prompt.trim() || activeJob && ["queued", "running"].includes(activeJob.status)) return;
    setJobLogs([]); setLocalError(null);
    try {
      let screenshotDataUrl: string | undefined;
      if (canvasRef.current) {
        try { screenshotDataUrl = await toPng(canvasRef.current, { pixelRatio: 0.45, cacheBust: true, backgroundColor: currentSlide.background }); }
        catch { setToast("스크린샷 없이 구조·좌표 컨텍스트로 Codex를 실행합니다."); }
      }
      const job = await api.createJob(props.projectId, { prompt, context, screenshotDataUrl });
      setActiveJob(job); setPrompt("");
    } catch (caught) { setLocalError(message(caught)); }
  };

  const publish = async () => {
    try { const result = await api.publish(props.projectId); setToast(`발행 완료: ${result.publicUrl}`); await loadProject(); window.open(result.publicUrl, "_blank", "noopener"); }
    catch (caught) { setLocalError(message(caught)); }
  };

  return (
    <main className="studio-shell">
      <header className="studio-topbar">
        <button className="brand" onClick={props.onExit}><span>LOE</span> Slide Studio</button>
        <select value={props.projectId} onChange={(event) => props.onProjectChange(event.target.value)}>{props.projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select>
        <div className="topbar-spacer" />
        <span className={`save-state save-state--${saveState}`}>{saveState === "saved" ? "저장됨" : saveState === "saving" ? "저장 중…" : saveState === "dirty" ? "변경됨" : "저장 실패"}</span>
        <button onClick={() => setPreviewKey((value) => value + 1)}>미리보기</button>
        <button onClick={() => { setShowVersions(true); }}>버전</button>
        <a className="button" href={`/api/projects/${props.projectId}/export`}>HTML 내보내기</a>
        <button className="primary" onClick={publish}>링크 발행</button>
      </header>

      <aside className="slides-panel">
        <div className="panel-title"><strong>슬라이드</strong><button title="슬라이드 추가" onClick={() => updateDeck((next) => { const slide = createSlide(`슬라이드 ${next.slides.length + 1}`); next.slides.push(slide); setSlideId(slide.id); })}>＋</button></div>
        <div className="slide-list">{deck.slides.map((slide, index) => <button key={slide.id} className={`slide-thumbnail ${slide.id === currentSlide.id ? "is-active" : ""}`} onClick={() => { setSlideId(slide.id); setSelectedObjectIds([]); }}><span>{index + 1}</span><div style={{ background: slide.background }}><strong>{slide.title}</strong><small>{slide.objects.length} objects</small></div></button>)}</div>
        <button className="danger-text" disabled={deck.slides.length === 1} onClick={() => {
          const index = deck.slides.findIndex((slide) => slide.id === currentSlide.id);
          updateDeck((next) => { next.slides.splice(index, 1); });
          setSlideId(deck.slides[Math.max(0, index - 1)].id); setSelectedObjectIds([]);
        }}>현재 슬라이드 삭제</button>
      </aside>

      <section className="workspace">
        <nav className="canvas-toolbar">
          <div className="tool-group"><button className={tool === "select" ? "is-active" : ""} onClick={() => setTool("select")}>선택</button><button className={tool === "point" ? "is-active" : ""} onClick={() => setTool("point")}>점 좌표</button><button className={tool === "region" ? "is-active" : ""} onClick={() => setTool("region")}>영역 좌표</button></div>
          <div className="tool-group"><button onClick={() => addObject(textObject(point))}>T 텍스트</button><button onClick={() => addObject(shapeObject(point))}>○ 도형</button><button onClick={() => fileInputRef.current?.click()}>{uploading ? "업로드 중…" : "＋ 자료"}</button><input ref={fileInputRef} hidden multiple type="file" onChange={(event) => { uploadFiles([...event.target.files || []]); event.target.value = ""; }} /></div>
          <div className="tool-group"><button onClick={undo} disabled={!undoStack.current.length}>↶</button><button onClick={redo} disabled={!redoStack.current.length}>↷</button><span>{Math.round(scale * 100)}%</span></div>
        </nav>
        <div className="canvas-host" ref={canvasHostRef}>
          <div className="canvas-scaled-wrap" style={{ width: 1920 * scale, height: 1080 * scale }}>
            <div
              ref={canvasRef}
              className={`slide-canvas tool-${tool}`}
              style={{ transform: `scale(${scale})`, background: currentSlide.background }}
              onPointerDown={(event) => handleCanvasPointer(event, tool, canvasRef.current!, setPoint, setRegion, () => { setSelectedObjectIds([]); }, () => setTool("select"))}
            >
              <style>{detail.themeCss}{"\n"}{detail.animationsCss}</style>
              {[...currentSlide.objects].sort((a, b) => a.zIndex - b.zIndex).map((object) => (
                <CanvasObject key={object.id} object={object} projectId={props.projectId} selected={selectedObjectIds.includes(object.id)} showResizeHandle={selectedObjectIds.length === 1 && selectedObjectIds[0] === object.id} scale={scale} tool={tool} onSelect={(additive) => selectObject(object.id, additive)} onChange={(nextObject, baseline) => {
                  if (baseline) undoStack.current.push(baseline);
                  setDeck((previous) => {
                    if (!previous) return previous; const next = cloneDeck(previous); const slide = next.slides.find((item) => item.id === currentSlide.id)!;
                    slide.objects = slide.objects.map((item) => item.id === nextObject.id ? clampObject(nextObject) : item); dirtyRef.current = true; setSaveState("dirty"); return next;
                  });
                }} deckSnapshot={() => cloneDeck(deck)} />
              ))}
              {point && <span className="point-marker" style={{ left: point.x, top: point.y }}><i />{Math.round(point.x)}, {Math.round(point.y)}</span>}
              {region && <span className="region-marker" style={{ left: region.x, top: region.y, width: region.width, height: region.height }}><i>{Math.round(region.width)}×{Math.round(region.height)}</i></span>}
            </div>
          </div>
        </div>
      </section>

      <aside className="inspector-panel">
        <div className="inspector-scroll">
          <section><div className="panel-title"><strong>슬라이드</strong></div><label>제목<input value={currentSlide.title} onChange={(event) => updateDeck((next) => { next.slides.find((slide) => slide.id === currentSlide.id)!.title = event.target.value; })} /></label><label>배경<input type="color" value={colorValue(currentSlide.background)} onChange={(event) => updateDeck((next) => { next.slides.find((slide) => slide.id === currentSlide.id)!.background = event.target.value; })} /></label></section>
          <section><div className="panel-title"><strong>선택 객체</strong><span>{selectedObjects.length > 1 ? `${selectedObjects.length}개` : selectedObject?.type || "없음"}</span></div>{selectedObject ? <ObjectInspector object={selectedObject} onPatch={patchSelected} onDelete={deleteSelected} /> : selectedObjects.length > 1 ? <div className="multi-selection"><p className="hint"><strong>{selectedObjects.length}개 객체가 Codex 편집 대상으로 선택되었습니다.</strong><br />개별 속성 편집은 한 객체만 선택했을 때 사용할 수 있습니다.</p><button className="danger" onClick={deleteSelected}>선택 객체 {selectedObjects.length}개 삭제</button></div> : <p className="hint">객체를 클릭하면 위치·스타일·애니메이션을 직접 조정할 수 있습니다.</p>}</section>
          <section><div className="panel-title"><strong>레이어</strong><span>{currentSlide.objects.length}</span></div><div className="layer-list">{[...currentSlide.objects].sort((a, b) => b.zIndex - a.zIndex).map((object) => <button key={object.id} aria-pressed={selectedObjectIds.includes(object.id)} title="클릭: 단일 선택 · Shift/Ctrl/Cmd+클릭: 추가 선택" className={selectedObjectIds.includes(object.id) ? "is-active" : ""} onClick={(event) => selectObject(object.id, event.shiftKey || event.ctrlKey || event.metaKey)}><span>{selectedObjectIds.includes(object.id) ? "✓ " : ""}{object.type}</span><strong>{object.content || object.id}</strong></button>)}</div><p className="selection-help">Shift/Ctrl/Cmd+클릭으로 여러 객체 선택 · Ctrl/Cmd+A로 전체 선택</p></section>
        </div>
        <section className="ai-panel">
          <div className="panel-title"><strong>Codex 디자인 편집</strong><span className={activeJob ? `job-${activeJob.status}` : ""}>{activeJob?.status || "ready"}</span></div>
          <div className="context-chips"><span>@slide({deck.slides.indexOf(currentSlide) + 1})</span>{selectedObjects.map((object, index) => <button className="object-context-chip" key={object.id} title="Codex 편집 대상에서 제거" onClick={() => setSelectedObjectIds((ids) => ids.filter((id) => id !== object.id))}>@object({index + 1}:{object.type}:{object.id.slice(-8)}) ×</button>)}{point && <button onClick={() => setPoint(undefined)}>@point({Math.round(point.x)},{Math.round(point.y)}) ×</button>}{region && <button onClick={() => setRegion(undefined)}>@region({Math.round(region.x)},{Math.round(region.y)},{Math.round(region.width)},{Math.round(region.height)}) ×</button>}</div>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="예: 선택한 객체들을 지정한 영역에 함께 배치하고, 차례로 등장하는 애니메이션을 만들어줘" onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runJob(); }} />
          <button className="primary ai-run" disabled={!prompt.trim() || Boolean(activeJob && ["queued", "running"].includes(activeJob.status))} onClick={runJob}>Codex 변경안 만들기 <kbd>⌘↵</kbd></button>
          {activeJob && ["queued", "running"].includes(activeJob.status) && <div className="job-progress"><span className="spinner" /><strong>격리 작업공간에서 수정 중…</strong><button onClick={() => api.cancelJob(activeJob.id).then(() => setActiveJob({ ...activeJob, status: "cancelled" }))}>취소</button><details><summary>실행 로그</summary><pre>{jobLogs.join("\n")}</pre></details></div>}
          {activeJob?.status === "ready" && <button className="review-button" onClick={() => setPreviewKey((value) => value + 1)}>변경안 비교·승인하기</button>}
          {activeJob?.status === "failed" && <p className="form-error">{activeJob.error}</p>}
        </section>
      </aside>

      {previewKey > 0 && <PreviewDialog key={previewKey} projectId={props.projectId} job={activeJob?.status === "ready" ? activeJob : null} onClose={() => setPreviewKey(0)} onAccept={async () => { await api.acceptJob(activeJob!.id); setActiveJob(null); setPreviewKey(0); await loadProject(); setToast("Codex 변경안을 Draft에 적용했습니다."); }} onReject={async () => { await api.rejectJob(activeJob!.id); setActiveJob(null); setPreviewKey(0); setToast("변경안을 폐기했습니다."); }} />}
      {showVersions && <VersionDialog versions={detail.versions} onClose={() => setShowVersions(false)} onRestore={async (id) => { await api.restoreVersion(props.projectId, id); setShowVersions(false); await loadProject(); setToast("선택한 버전으로 복원했습니다."); }} />}
      {(localError || props.globalError) && <div className="error-banner"><span>{localError || props.globalError}</span><button onClick={() => { setLocalError(null); props.clearGlobalError(); }}>×</button></div>}
      {toast && <div className="toast" onAnimationEnd={() => setToast(null)}>{toast}</div>}
    </main>
  );
}

function CanvasObject({ object, projectId, selected, showResizeHandle, scale, tool, onSelect, onChange, deckSnapshot }: { object: DeckObject; projectId: string; selected: boolean; showResizeHandle: boolean; scale: number; tool: CanvasTool; onSelect: (additive: boolean) => void; onChange: (object: DeckObject, baseline?: Deck) => void; deckSnapshot: () => Deck }) {
  const elementStyle: CSSProperties = {
    position: "absolute", left: object.x, top: object.y, width: object.width, height: object.height,
    zIndex: object.zIndex, transform: `rotate(${object.rotation}deg)`, ...object.styles as CSSProperties
  };
  const startGesture = (event: ReactPointerEvent, mode: "move" | "resize") => {
    if (tool !== "select" || event.button !== 0) return;
    event.stopPropagation(); event.preventDefault();
    const additive = mode === "move" && (event.shiftKey || event.ctrlKey || event.metaKey);
    onSelect(additive);
    if (additive) return;
    const baseline = deckSnapshot(); const initial = { ...object }; let latest = initial; const startX = event.clientX; const startY = event.clientY;
    const move = (moveEvent: PointerEvent) => {
      const dx = (moveEvent.clientX - startX) / scale; const dy = (moveEvent.clientY - startY) / scale;
      latest = mode === "move" ? { ...initial, x: Math.round(initial.x + dx), y: Math.round(initial.y + dy) } : { ...initial, width: Math.max(24, Math.round(initial.width + dx)), height: Math.max(24, Math.round(initial.height + dy)) };
      onChange(latest);
    };
    const end = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", end); onChange(latest, baseline); };
    addEventListener("pointermove", move); addEventListener("pointerup", end, { once: true });
  };
  return (
    <div className={`slide-object editor-object slide-object--${object.type} ${object.className || ""} ${selected ? "is-selected" : ""}`} style={elementStyle} onPointerDown={(event) => startGesture(event, "move")}>
      {object.type === "text" && <div className="slide-object__text">{object.content}</div>}
      {object.type === "shape" && null}
      {object.type === "image" && <img draggable={false} src={assetContentUrl(projectId, object.assetId!)} alt={object.content || ""} style={{ objectFit: (object.styles.objectFit || "contain") as CSSProperties["objectFit"] }} />}
      {object.type === "video" && <video src={assetContentUrl(projectId, object.assetId!)} controls preload="metadata" />}
      {object.type === "audio" && <audio src={assetContentUrl(projectId, object.assetId!)} controls preload="metadata" />}
      {object.type === "pdf" && <iframe src={assetContentUrl(projectId, object.assetId!)} title={object.content || "PDF"} />}
      {object.type === "attachment" && <a href={assetContentUrl(projectId, object.assetId!)} download={object.content}>{object.content}</a>}
      {showResizeHandle && <button className="resize-handle" aria-label="크기 조절" onPointerDown={(event) => startGesture(event, "resize")} />}
    </div>
  );
}

function ObjectInspector({ object, onPatch, onDelete }: { object: DeckObject; onPatch: (patch: (object: DeckObject) => void) => void; onDelete: () => void }) {
  const numberField = (label: string, key: "x" | "y" | "width" | "height" | "rotation" | "zIndex") => <label>{label}<input type="number" value={Math.round(object[key])} onChange={(event) => onPatch((next) => { next[key] = Number(event.target.value); })} /></label>;
  return <div className="object-inspector">
    {(object.type === "text" || object.type === "attachment") && <label className="full">내용<textarea value={object.content || ""} onChange={(event) => onPatch((next) => { next.content = event.target.value; })} /></label>}
    <div className="field-grid">{numberField("X", "x")}{numberField("Y", "y")}{numberField("W", "width")}{numberField("H", "height")}{numberField("회전", "rotation")}{numberField("레이어", "zIndex")}</div>
    <div className="field-grid"><label>글자색<input type="color" value={colorValue(String(object.styles.color || "#111827"))} onChange={(event) => onPatch((next) => { next.styles.color = event.target.value; })} /></label><label>배경색<input type="color" value={colorValue(String(object.styles.backgroundColor || "#ffffff"))} onChange={(event) => onPatch((next) => { next.styles.backgroundColor = event.target.value; })} /></label><label>글자 크기<input value={String(object.styles.fontSize || "32px")} onChange={(event) => onPatch((next) => { next.styles.fontSize = event.target.value; })} /></label><label>모서리<input value={String(object.styles.borderRadius || "0px")} onChange={(event) => onPatch((next) => { next.styles.borderRadius = event.target.value; })} /></label></div>
    <fieldset><legend>CSS 애니메이션</legend><label>Keyframe 이름<input value={object.animation?.name || ""} placeholder="fade-in" onChange={(event) => onPatch((next) => { const name = event.target.value; if (!name) { delete next.animation; return; } next.animation = next.animation || { name, trigger: "click", durationMs: 700, delayMs: 0, easing: "ease", iterationCount: 1 }; next.animation.name = name; })} /></label>{object.animation && <><label>시작<select value={object.animation.trigger} onChange={(event) => onPatch((next) => { next.animation!.trigger = event.target.value as NonNullable<DeckObject["animation"]>["trigger"]; })}><option value="click">클릭</option><option value="slide-enter">슬라이드 진입</option><option value="with-previous">이전과 함께</option><option value="after-previous">이전 다음</option></select></label><div className="field-grid"><label>시간(ms)<input type="number" value={object.animation.durationMs} onChange={(event) => onPatch((next) => { next.animation!.durationMs = Number(event.target.value); })} /></label><label>지연(ms)<input type="number" value={object.animation.delayMs} onChange={(event) => onPatch((next) => { next.animation!.delayMs = Number(event.target.value); })} /></label></div><label>Easing<input value={object.animation.easing} onChange={(event) => onPatch((next) => { next.animation!.easing = event.target.value; })} /></label></>}</fieldset>
        <button className="danger" onClick={onDelete}>객체 삭제</button>
  </div>;
}

function PreviewDialog({ projectId, job, onClose, onAccept, onReject }: { projectId: string; job: AiJob | null; onClose: () => void; onAccept: () => Promise<void>; onReject: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const previewHint = <small>파란색 → 버튼 또는 Space로 애니메이션 재생</small>;
  return <div className="modal-backdrop"><section className="preview-dialog"><header><div><p className="eyebrow">REVIEW BEFORE APPLY</p><h2>{job ? "Codex 변경안 비교" : "현재 Draft 미리보기"}</h2></div><button onClick={onClose}>×</button></header><div className={`preview-grid ${job ? "" : "single"}`}><div><span>현재 Draft {previewHint}</span><iframe src={`/api/projects/${projectId}/preview?at=${Date.now()}`} title="현재 Draft" /></div>{job && <div><span>Codex 변경안 {previewHint}</span><iframe src={`/api/ai-jobs/${job.id}/preview?at=${Date.now()}`} title="Codex 변경안" /></div>}</div>{job && <footer><pre>{job.summary}</pre><div><button disabled={busy} onClick={async () => { setBusy(true); await onReject(); }}>폐기</button><button className="primary" disabled={busy} onClick={async () => { setBusy(true); await onAccept(); }}>Draft에 적용</button></div></footer>}</section></div>;
}

function VersionDialog({ versions, onClose, onRestore }: { versions: ProjectDetail["versions"]; onClose: () => void; onRestore: (id: string) => Promise<void> }) {
  return <div className="modal-backdrop"><section className="version-dialog"><header><h2>버전 기록</h2><button onClick={onClose}>×</button></header>{versions.length ? versions.map((version) => <article key={version.id}><div><strong>{version.reason}</strong><time>{new Date(version.createdAt).toLocaleString("ko-KR")}</time></div><button onClick={() => onRestore(version.id)}>복원</button></article>) : <p className="empty-state">Codex 변경을 적용하거나 버전을 복원하면 안전 스냅샷이 생성됩니다.</p>}</section></div>;
}

function handleCanvasPointer(event: ReactPointerEvent<HTMLDivElement>, tool: CanvasTool, canvas: HTMLDivElement, setPoint: (point: SelectionContext["point"]) => void, setRegion: (region: SelectionContext["region"]) => void, clearSelection: () => void, finishTool: () => void) {
  if (event.button !== 0 || event.target !== canvas) return;
  const rect = canvas.getBoundingClientRect(); const ratioX = 1920 / rect.width; const ratioY = 1080 / rect.height;
  const at = (source: PointerEvent | ReactPointerEvent) => ({ x: Math.max(0, Math.min(1920, (source.clientX - rect.left) * ratioX)), y: Math.max(0, Math.min(1080, (source.clientY - rect.top) * ratioY)) });
  if (tool === "select") { clearSelection(); return; }
  if (tool === "point") { setPoint(at(event)); finishTool(); return; }
  const start = at(event); setRegion({ ...start, width: 0, height: 0 });
  const move = (moveEvent: PointerEvent) => { const end = at(moveEvent); setRegion({ x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(end.x - start.x), height: Math.abs(end.y - start.y) }); };
  const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); finishTool(); };
  addEventListener("pointermove", move); addEventListener("pointerup", up, { once: true });
}

function textObject(point?: { x: number; y: number }): DeckObject {
  return { id: `object-${crypto.randomUUID()}`, type: "text", x: (point?.x || 960) - 300, y: (point?.y || 540) - 70, width: 600, height: 140, rotation: 0, zIndex: 10, content: "텍스트를 입력하세요", styles: { fontSize: "48px", fontWeight: "700", color: "#111827", textAlign: "center", padding: "12px" } };
}
function shapeObject(point?: { x: number; y: number }): DeckObject {
  return { id: `object-${crypto.randomUUID()}`, type: "shape", x: (point?.x || 960) - 100, y: (point?.y || 540) - 100, width: 200, height: 200, rotation: 0, zIndex: 5, styles: { backgroundColor: "#5b7cfa", borderRadius: "50%", opacity: 0.9 } };
}
function colorValue(value: string): string { return /^#[0-9a-f]{6}$/i.test(value) ? value : "#ffffff"; }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
