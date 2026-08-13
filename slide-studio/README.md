# LOE Slide Studio

블로그의 Jekyll 배포와 분리된 1920×1080 HTML 발표자료 제작 웹앱입니다. 파일을 붙여넣거나 업로드하고, 캔버스의 객체·점·영역을 채팅 컨텍스트로 지정해 격리된 Codex CLI 작업으로 디자인과 CSS 애니메이션을 변경합니다.

## 구현 범위

- React + TypeScript 캔버스, 슬라이드/레이어/속성 패널, Undo/Redo
- 이미지 클립보드 붙여넣기와 모든 파일 형식의 스트리밍 업로드
- 이미지·영상·오디오·PDF 직접 표시, 나머지는 첨부 카드로 표시
- `@object`, `@point(x,y)`, `@region(x,y,w,h)` Codex 컨텍스트
- 프로젝트별 Codex 작업공간, 허용 파일 제한, JSON/CSS/자산 검증
- 변경 전후 iframe 비교 후 승인·폐기와 안전 버전 복원
- 링크 공개 URL과 독립 실행 HTML ZIP 내보내기
- SQLite 메타데이터와 Git 외부 콘텐츠 해시 자산 저장소

## 개발

```shell
cd slide-studio
npm install
npm run dev
```

- 스튜디오: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:5560`
- 개발 모드에서 인증 환경변수가 없으면 localhost 접근을 허용합니다.

검증:

```shell
npm test
npm run build
npx playwright install chromium # 최초 한 번
npm run test:e2e
```

## Codex 작업 경계

서버는 요청마다 현재 `deck.json`, `theme.css`, `animations.css`를 별도 작업 디렉터리로 복사하고 다음과 같이 Codex를 실행합니다.

- `--ephemeral --ignore-user-config --sandbox workspace-write`
- 원본 자산 대신 `assets.json`의 메타데이터와 현재 캔버스 PNG 전달
- 수정 가능 파일은 덱 JSON과 두 CSS 파일뿐
- 새 파일 생성이나 `assets.json`, 스크린샷, 실행 계약 변경 시 작업 실패
- JSON 스키마, 자산 UUID, 외부 URL·스크립트성 CSS를 검증한 뒤에만 리뷰 가능

Mac mini에서 서비스를 실행하는 동일한 계정으로 먼저 `codex login`을 완료하고 비대화형 `codex exec`가 동작하는지 확인해야 합니다.
설치기는 현재 셸의 `codex` 절대 경로를 launchd 환경에 기록하므로 설치할 때 Codex가 `PATH`에 있어야 합니다.

## Mac mini 설치

Node와 Codex CLI를 설치하고 저장소 루트에서 실행합니다.

```shell
chmod +x tools/install-slide-studio-service.sh slide-studio/tools/run-production.sh
./tools/install-slide-studio-service.sh
curl http://127.0.0.1:5560/api/health
```

설치기는 다음 작업을 수행합니다.

- `npm ci`와 프로덕션 빌드
- `~/.config/tosemfdk/slide-studio-token`에 256-bit 토큰 생성 (`0600`)
- `~/.local/share/tosemfdk-slide-studio`에 Git 외부 데이터 저장
- 현재 저장소 경로를 사용해 launchd plist를 동적으로 생성
- 로그를 `~/Library/Logs/tosemfdk/slide-studio*.log`에 기록

Cloudflare Tunnel의 `slides.tosemfdk.com` origin은 `http://127.0.0.1:5560`으로 지정할 수 있습니다. Caddy를 중간에 둘 경우 `ops/Caddyfile.snippet`을 기존 Caddy 설정에 합치고 origin을 `http://127.0.0.1:5561`로 지정합니다.

스튜디오 API를 Cloudflare Access로 보호하려면 `/api/*`에 단일 관리자 정책을 설정하고 launchd 설치 전에 다음 값을 지정합니다.

```shell
SLIDE_STUDIO_ADMIN_EMAIL='owner@example.com' \
SLIDE_STUDIO_PUBLIC_URL='https://slides.tosemfdk.com' \
./tools/install-slide-studio-service.sh
```

Access를 사용하지 않으면 설치 시 클립보드에 복사된 관리자 토큰으로 로그인합니다. `/p/{slug}`, `/published/*`, 런타임 파일은 공개 발표를 위해 인증 없이 제공되며 `noindex,nofollow`가 기본입니다.

## 주요 환경변수

| 이름 | 기본값 | 역할 |
| --- | --- | --- |
| `SLIDE_STUDIO_DATA_DIR` | `~/.local/share/tosemfdk-slide-studio` | SQLite, 자산, 작업본, 발행본 |
| `SLIDE_STUDIO_ADMIN_EMAIL` | 없음 | 허용할 Cloudflare Access 이메일 |
| `SLIDE_STUDIO_ADMIN_TOKEN` | 토큰 파일 | 단일 관리자 로그인 토큰 |
| `SLIDE_STUDIO_PUBLIC_URL` | 요청 origin | 발행 URL 기준 주소 |
| `SLIDE_STUDIO_MAX_FILE_BYTES` | 2 GiB | 파일당 업로드 한도 |
| `SLIDE_STUDIO_MAX_PROJECT_BYTES` | 20 GiB | 프로젝트당 논리 용량 한도 |
| `SLIDE_STUDIO_CODEX_TIMEOUT_MS` | 480000 | Codex 작업 제한시간 |

운영 데이터와 실제 인증값은 Git에 커밋하지 않습니다.
