"""
==============================================================================
 프로그램명 : download
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   NYC Yellow Taxi 2026-05 Parquet 원본 데이터를 data/raw/ 로 다운로드한다.
     1) download_taxi_data() : URL에서 parquet 파일을 스트리밍 다운로드
        (이미 파일이 있으면 재다운로드하지 않는다)

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  스트리밍 다운로드 로직 구현
==============================================================================
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = _PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"


# ---------------------------------------------------------------------------
# 1) download_taxi_data
# ---------------------------------------------------------------------------
def download_taxi_data(
    dest: Path = DEFAULT_DEST, url: str = DATA_URL, chunk_size: int = 1 << 20
) -> Path:
    """url의 parquet 파일을 스트리밍으로 dest에 저장하고 경로를 반환한다.

    dest가 이미 존재하면 다운로드를 건너뛰고 그대로 반환한다 (팀원 각자 반복 실행 시
    70MB 파일을 매번 다시 받지 않도록).
    """
    tag = "[download_taxi_data]"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("%s 이미 존재해 다운로드 생략 : %s", tag, dest)
        return dest

    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
    except requests.RequestException as e:
        logger.error("%s 다운로드 실패 : %s", tag, e)
        raise

    logger.info("%s 다운로드 완료 : %s (%.1f MB)", tag, dest, dest.stat().st_size / 1024**2)
    return dest


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== 1단계 : 데이터 다운로드 ===\n")
    path = download_taxi_data()
    assert path.exists(), "Checkpoint : 다운로드된 파일이 존재해야 한다"

    print(f"\n\nCheckpoint : 데이터 준비 완료 ({path})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
