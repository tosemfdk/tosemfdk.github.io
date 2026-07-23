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
