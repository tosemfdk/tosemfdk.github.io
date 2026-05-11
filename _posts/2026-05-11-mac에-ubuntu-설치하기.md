---
layout: post
title: "Mac에 ubuntu 설치하기"
date: 2026-05-11T08:10:00.000Z
math: true
image:
  path: "/assets/img/posts/35dcbb7d-7937-80bc-91b2-e3fbbc857629.webp"
categories:
  - "development tips"
---

### mac에 ubuntu 설치법

지금까지 linux를 mac에 설치하려면 UTM같은 Virtual machine을 사용하거나 docker 를 활용해야 했다. 하지만 VM ware는 cpu나 ram 자원 자체를 할당해줘야 했기 때문에 장기적으로 활용하기는 아쉬었다. 또한 커널이 하드웨어와 완전히 호환되지 않아서 카메라나 라이다 같은 센서를 붙이고 싶은 나로서는 최악의 선택지였다.

docker도 마찬가지로 mac chip이기 때문에 설치할 수 있는 이미지의 수가 적고 native linux와 다르게 장치를 연결해 줄 수 없었다..

이렇게 포기한 와중에 발견한 것이 바로 아래의 방법이다.

### Ashahi Linux

![](/assets/img/posts/35dcbb7d-7937-80bc-91b2-e3fbbc857629.webp)

![](/assets/img/posts/35dcbb7d-7937-8012-9db1-fbb0ee8d1ea6.webp)

mac의 유래인 사과품종을 아사히라고 부르기때문에 해당 커널의 이름을 ashahi lunux라고 한다.

아사히 리눅스는 이름도 마음에 들지만 좋은점은 m1, m2 chip에 맞춰서 하드웨어 장치의 호환성을 reverse engineering해서 (엄청난 노가다,,) 설계된 커널이기 때문에 오디오, usb, display, track pad와 같은 기능이 전부 네이티브로 지원이 된다~~ 그니까 실제 네이티브 우분투와 같이 사용할 수 있다는 말이다.

조금 아쉬운점은 m3,m4 chip부터는 설계가 많이 바뀌어 지원하지는 않다는 점이다. mac mini m4가 지원됐다면 정말 좋았을텐데 아쉽다.

### 설치법

[https://ubuntuasahi.org/](https://ubuntuasahi.org/)

[
Ubuntu Asahi | Ubuntu images for Apple hardware
Ubuntu is a trademark of Canonical Ltd. Asahi Linux is a project by The Asahi Linux Contributors. Linux is a Registered Trademark of Linus Torvalds. Ubuntu Asahi is not affiliated with Canonical Ltd or The Asahi Linux Contributors. All other product names,
ubuntuasahi.org](https://ubuntuasahi.org/)

```bash
curl -sL https://ubuntuasahi.org/install | sh
```

설치는 간단하다 위의 스크립트를 터미널에서 실행하고 비밀번호를 쳐주면 된다.. 나는 이미 세팅 완료했기 때문에 다시 하기는 어렵지만 아래의 유튜브 링크를 보면 설치하기 편할 것 같다.  일단 서비스가 길고 호환이 잘되는 ubuntu 24.04 LTS를 설치하였다.

[https://www.youtube.com/watch?v=8F-eE2zW_HQ](https://www.youtube.com/watch?v=8F-eE2zW_HQ)

### 설치 후

mac의 전원 버튼을 꾹 눌러스 booting 선택창을 load한다. 이후 realsense camera를 연결햇더니 정상적으로 video가 잡히는것을 확인할 수 있다.

![](/assets/img/posts/35dcbb7d-7937-80c2-bc37-d446eb04a6db.webp)

![](/assets/img/posts/35dcbb7d-7937-8005-8512-ed4f2427060b.webp)

![](/assets/img/posts/35dcbb7d-7937-80a8-8ce8-cfae8d333633.webp)

![](/assets/img/posts/35dcbb7d-7937-802d-9dd8-ca41f92b151b.webp)

