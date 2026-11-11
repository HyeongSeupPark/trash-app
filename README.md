# trash-app
쓰레기 분리수거 앱 캡스톤 프로젝트

## 학습 데이터 출처

| 데이터 | 내용 | 라이선스 |
|---|---|---|
| [TrashNet](https://github.com/garythung/trashnet) | 흰 배경 쓰레기 사진 2,527장 | MIT |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (UCI Machine Learning Repository) | 실제 폐기물 사진 3,551장 사용 | CC BY 4.0 |
| [TACO](http://tacodataset.org) (Proença & Simões, 2020) | 길거리 쓰레기 사진에서 물체 755개를 잘라 사용 | CC BY 4.0 |
| [Wikimedia Commons](https://commons.wikimedia.org) | 테이크아웃 플라스틱 컵 사진 56장 | CC BY / CC BY-SA / CC0 (사진별 상이) |

RealWaste의 Plastic 폴더 중 비닐(포장재·봉투) 71장은 vinyl로 다시 분류했고, 스티로폼 등 36장은 제외했다.
학습 스크립트: `model_train_v2.py`
