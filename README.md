# tosemfdk.com

Jekyll과 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)를 사용하는 개인 블로그입니다.
프로덕션 빌드는 이 Mac에서 Caddy를 통해 `localhost:5555`로 제공되고,
Cloudflare Tunnel이 `https://tosemfdk.com`으로 연결합니다.

> macOS는 TCP `55555` 포트를 시스템 용도로 예약하므로 애플리케이션이 해당
> 포트에 바인딩할 수 없습니다. Cloudflare Tunnel의 `tosemfdk.com` origin은
> `http://localhost:5555`로 설정해야 합니다.

## 로컬 운영

필수 도구:

```shell
brew install ruby@3.4 caddy
```

콘텐츠를 수정한 뒤 다음 명령으로 검증하고 배포합니다.

```shell
./tools/deploy.sh
```

배포는 새 릴리스를 별도 디렉터리에 빌드한 뒤 `.deploy/current` 심볼릭 링크를
원자적으로 교체합니다. 빌드나 내부 링크 검사가 실패하면 현재 사이트는 바뀌지
않으며 최근 릴리스 세 개가 보존됩니다.

최초 한 번 다음 명령으로 launchd 서비스를 설치합니다.

```shell
./tools/install-service.sh
```

서비스 확인과 재시작:

```shell
launchctl print "gui/$(id -u)/com.tosemfdk.web"
launchctl kickstart -k "gui/$(id -u)/com.tosemfdk.web"
curl -I http://localhost:5555
```

로그:

```text
~/Library/Logs/tosemfdk/access.log
~/Library/Logs/tosemfdk/service.log
~/Library/Logs/tosemfdk/service-error.log
```

직전 릴리스로 되돌리려면 다음 명령을 사용합니다.

```shell
./tools/rollback.sh
```

## 개발 및 검사

명령이 Homebrew Ruby 3.4를 사용하도록 먼저 PATH를 설정합니다.

```shell
export PATH="/opt/homebrew/opt/ruby@3.4/bin:/opt/homebrew/lib/ruby/gems/3.4.0/bin:$PATH"
bundle install
bash tools/test.sh
```

개발 서버는 `bash tools/run.sh`로 실행할 수 있습니다. 프로덕션 서비스와 포트가
겹치지 않도록 기본 Jekyll 포트인 `4000`을 사용합니다.

## 발표자료 작성

`_slides/`에 HTML 파일을 추가하면 `/slides/<파일명>/`에 독립적인 전체화면
발표자료가 생성되고 `/slides/` 목록에도 자동으로 표시됩니다.

```html
---
title: "발표 제목"
description: "목록과 검색엔진에 표시할 설명"
date: 2026-07-23
theme: loe
accent: "#8b6cff"
slide_count: 2
published: true
---

<section data-slide aria-label="표지">
  <h1>발표 제목</h1>
</section>

<section data-slide aria-label="두 번째 슬라이드">
  <h2>핵심 메시지</h2>
  <p data-fragment>한 단계씩 나타나는 설명</p>
</section>
```

공통 인터랙션은 `assets/js/slides.js`, 레이아웃은
`assets/css/slides/core.css`, 덱 테마는 `assets/css/slides/<theme>.css`에서
관리합니다. 화살표·스페이스바 이동, 터치 스와이프, URL 해시, 개요(`O`),
전체화면(`F`), 프래그먼트가 기본 제공됩니다.

### 브라우저 Slide Studio

`editor_enabled: true`인 덱은 공개 발표 화면을 유지하면서 소유자 전용 Draft를
브라우저에서 편집할 수 있습니다. 최초 한 번 API 서비스를 설치합니다.

```shell
./tools/install-slide-editor-service.sh
./tools/install-service.sh
```

설치 명령은 256-bit Editor Key를
`~/.config/tosemfdk/slide-editor-token`에 권한 `0600`으로 만들고 클립보드에
복사합니다. 키는 Git이나 정적 사이트에 포함되지 않습니다. 편집기는 다음 주소로
엽니다.

```text
https://tosemfdk.com/slides/active-scene-change-detection/?edit=1
```

편집 모드에서 제공하는 기능:

- 슬라이드 객체 클릭 선택, 드래그 이동, 텍스트 더블클릭 편집
- 텍스트·이미지·도형 추가와 객체 삭제
- 폰트·크기·색상·정렬·위치·등장 애니메이션 변경
- 객체별 댓글 작성, 해결, 다시 열기
- 실행 취소·다시 실행, 자동 저장, 애니메이션/캔버스 미리보기
- 자연어 명령 스킬: `오른쪽에서 나타나서 왼쪽에 위치하도록 해줘`,
  `폰트를 명조로 바꿔줘`, `크기를 48px로`, `댓글: 그래프를 더 크게`

Draft와 객체 댓글은 `.slide-editor/` 아래에만 저장되어 공개 빌드와 Git에서
제외됩니다. Draft 이미지도 인증 쿠키가 있어야 미리 볼 수 있습니다.
`Freeze & Publish`를 누를 때만 댓글을 제거한 공개 JSON을 만들고, Jekyll의
원자적 배포를 실행한 뒤 공개 변경분을 Git에 커밋·푸시합니다.

서비스 상태와 로그:

```shell
launchctl print "gui/$(id -u)/com.tosemfdk.slide-editor"
curl http://localhost:5555/slide-editor-api/health
tail -f ~/Library/Logs/tosemfdk/slide-editor-error.log
```

## Notion 가져오기

`tools/notion_to_jekyll.py`는 Notion 콘텐츠를 `_posts/`로 가져옵니다.

- 기본 모드: `NOTION_IMPORT_MODE=single`
- 직접 자식 일괄 모드: `NOTION_IMPORT_MODE=direct_children`
- 카테고리 강제 지정: `NOTION_IMPORT_CATEGORY_OVERRIDE=...`

단일 페이지 예시:

```shell
NOTION_PAGE_ID=342cbb7d793780f7af67f18a6256482e \
NOTION_IMPORT_CATEGORY_OVERRIDE='UNIST' \
uv run python tools/notion_to_jekyll.py
```

직접 자식 일괄 가져오기 예시:

```shell
NOTION_IMPORT_MODE=direct_children \
NOTION_IMPORT_ROOT_PAGE_ID=24dcbb7d79378091bb83df2ae86685f4 \
NOTION_IMPORT_CATEGORY_OVERRIDE='서울대학교 여름방학 인턴' \
NOTION_DIRECT_CHILD_MAX_GIF_MB=25 \
uv run python tools/notion_to_jekyll.py
```

실제 Notion 토큰은 `.env`에만 저장하고 커밋하지 않습니다.

## 저장소 역할

기존 `tosemfdk/tosemfdk.github.io` 원격은 소스 코드와 이력 백업용으로 유지합니다.
GitHub Pages 배포는 사용하지 않으며 실제 서비스는 이 기기의 Caddy가 담당합니다.
