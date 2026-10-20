# 에피소드별 독립 지연 Monte Carlo 검증

기존 H200 프로젝트의 고정 모델로 **새 폐루프 시뮬레이션**을 실행하는 패키지입니다. 기존 지연 측정 풀을 에피소드·반복별로 독립 추출합니다. 재학습하거나 새 하드웨어 지연을 측정하지 않습니다. 이 패키지는 아직 연구 실험을 실행하지 않았으며, 로컬에서는 합성 테스트와 보관 자료의 읽기 검증만 수행했습니다.

## 실행

압축을 푼 뒤 `Monte_Carlo_Validation` 폴더에서 실행합니다. 아래 경로를 기존 모델·데이터·캐시가 있는 실제 프로젝트 경로로 바꾸세요.

```bash
unzip Monte_Carlo_Validation_20261013.zip
cd Monte_Carlo_Validation
bash run_validation.sh /실제/프로젝트/경로
```

스크립트가 기존 `scripts/activate_jev.sh`를 실행하므로 현재 셸의 Conda 환경과 무관하게 원래 `jev` 환경을 사용합니다. 원본 코드가 지정한 Linux H200, 단일 GPU, 자원 제한과 공유 잠금을 확인합니다. 패키지 외에 기존 원본 프로젝트가 반드시 필요합니다. 의존성 설치나 모델 다운로드는 하지 않습니다.

실행 예산은 **처음 시작하기 전에** 결정합니다.

| 반복 수 | 신규 실행 | 기준 재현 | 합계 |
|---|---:|---:|---:|
| 기본 16 | 2,304 | 240 | 2,544 |
| 권장 64 이상 중 64 | 9,216 | 240 | 9,456 |

각 반복은 5개 방법 × 24개 에피소드와 FV hold 24개입니다. 16회는 예비 예산이며, 순위 확률의 정밀도를 높이려면 64회 이상을 먼저 지정하세요. 확률 0.5 부근에서 Wilson 95% 구간 반폭은 16회 약 0.22, 64회 약 0.12입니다. 더 많은 반복도 새 하드웨어·학습 seed 불확실성을 해결하지 않습니다. 여러 시간에서 수일이 걸릴 수 있으며 전체 기간은 GPU/로딩/저장 속도에 따라 달라집니다.

```bash
# 64회로 처음부터 고정
bash run_validation.sh /실제/프로젝트/경로 --replicates 64

# 백그라운드 실행
nohup bash run_validation.sh /실제/프로젝트/경로 --replicates 64 > monte_carlo.log 2>&1 &
tail -f monte_carlo.log
```

## 실행 전 점검과 재개

```bash
# GPU 없이 설계만 보기
python -B run_suite.py --plan --replicates 16

# 원본·환경만 확인하고 설계를 저장 (모델 추론/시뮬레이션 없음)
bash run_validation.sh /실제/프로젝트/경로 --replicates 64 --check-only

# 위 로그에 나온 결과 경로로 계속하기 (저장된 반복 수를 자동 사용)
bash run_validation.sh /실제/프로젝트/경로 --resume --output /로그에/나온/결과경로
```

재개 시 소스·모델·환경·설계가 달라지면 중단합니다. 완료 사례는 봉인 체크섬을 검증하고 보존하며 중단된 사례만 새 attempt에서 다시 시작합니다. 결과를 보고 반복 수를 바꾸는 것은 같은 실행의 재개로 허용하지 않습니다.

단계를 나누려면 처음 `--phase baseline`으로 240건을 재현한 뒤 같은 출력에 `--resume --phase monte_carlo`를 사용합니다. 기준 재현이 모두 통과해야 신규 실행이 시작됩니다.

## 무엇이 달라지나

- 각 방법의 기존 3회 측정 기록을 합친 normal 30,000행에서 IID 복원추출합니다. fallback은 각 3,000행의 전용 풀을 사용하며 요청의 fallback reason과 가장 가까운 token 수를 맞춥니다. fallback 자료가 없으면 중단합니다.
- 에피소드와 반복마다 다른 결정적 난수 키를 사용합니다. FV zero/hold에는 같은 잠재 추출 계획을 적용합니다. feedback으로 실제 경로가 달라지면 경로·시각 차이를 따로 보고하며 잘못된 지연 풀로 강제 일치시키지 않습니다.
- 원래 interleaved 지연 로더를 유지하되 이를 연속 시간열로 해석하지 않습니다. IID는 시간 상관을 제거하므로 결과는 이 가정과 보관된 풀에 조건부입니다.
- 플랜트 적분 간격은 10 ms입니다. 후보·모델·alpha·gate·관측은 고정합니다. 네트워크 지연, 실제 설비, 추가 seed 학습은 포함하지 않습니다.

## 결과

완료 시 로그의 `Return this ZIP:`에 표시되는 `latency_monte_carlo_..._results.zip`을 전달하면 됩니다. 실패한 경우에도 생성된 ZIP과 로그를 보존하세요. 전체 연구 완료는 `analysis/summary.json`의 `COMPLETE_VERIFIED`로 확인합니다.

- `ranking_probabilities.csv`: 반복별 순위 역전 등의 조건부 비율과 Wilson 구간
- `method_rankings.json`: 각 반복의 5개 방법 순위, FV zero/hold 대체 비교와 동률
- `cost_effect_distribution.csv`: 비용 변화의 반복 간 분포 요약
- `paired_fv_effects.csv`: FV hold–zero 대응 차이
- `episode_metrics.csv`: 기준 및 신규 실행의 모든 에피소드 결과
- `coupling_audit.csv`: FV 대응 실행의 잠재 계획·실제 경로·시각 일치 여부
- 원시 trace·scores·masks·명령·지연 source row·최종장·적분 기록·봉인 자료

결과 ZIP에는 재분석을 위한 측정 풀도 포함하므로 기존 패키지보다 클 수 있습니다. 모델 가중치는 결과 ZIP에 복사하지 않습니다. 자세한 가정과 통계 정의는 [PROTOCOL.md](PROTOCOL.md)에 있습니다. 완전한 실행이 가설 지지나 hold의 안전성을 뜻하지 않습니다.

## 로컬 테스트

합성 fixture만 사용하는 명령입니다. 원래 플랜트나 모델을 불러오지 않습니다.

```bash
python -B -m unittest -v test_validation.py
```
