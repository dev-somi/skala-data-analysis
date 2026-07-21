"""
==============================================================================
 프로그램명 : <프로그램 이름>
 작성자     : 광주 4반 박소미
 작성일     : YYYY-MM-DD
------------------------------------------------------------------------------
 [프로그램 설명]
   <이 파일이 하는 일을 한두 문장으로 요약. 이어서 처리 단계를 번호로 나열>
     1) <1단계 함수명()> : <한 줄 설명>
     2) <2단계 클래스/함수명> : <한 줄 설명>
     3) <3단계 함수명()> : <한 줄 설명>

 [변경 내역]
   v1.0  YYYY-MM-DD  박소미  최초 작성 (<요약>)
==============================================================================
"""

# NOTE: 아래 logging 설정은 "이 파일 스타일"의 필수 요소가 아니다.
#       파일 I/O·예외 처리가 있는 스크립트에서만 쓰는 선택적 패턴이며,
#       단순 로직/유틸 파일이라면 이 블록과 tag/try-except-finally 구조는
#       통째로 빼고 아래 "핵심 스타일"만 가져다 쓰면 된다.
#         - 헤더 블록 (프로그램명/작성자/설명/변경내역)
#         - # --- 섹션 구분선 + 번호
#         - 계약(contract) 중심의 간결한 docstring
#         - WHY만 남기는 인라인 주석
#         - main()의 단계별 print + assert Checkpoint
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) <섹션 제목>
# ---------------------------------------------------------------------------
# 아래 함수는 "파일 I/O + 예외 처리가 있는 경우"의 예시다.
# 로깅/tag/try-except-finally 없이 순수 로직 함수라면 그냥
#   def example_step_one(...) -> ...:
#       """<한 줄 설명>"""
#       ...
# 처럼 섹션 구분선 + docstring만 유지해도 충분하다.
def example_step_one(path: str) -> Optional[list[dict]]:
    """<무엇을 하는지 한 줄> 실패하면 None 을 반환한다.

    <반환값/예외 계약을 간결하게 설명. WHAT이 아니라 계약(contract) 중심으로.>
    """
    tag = "[example_step_one]"
    try:
        raise NotImplementedError
    except FileNotFoundError:
        logger.error("%s 파일을 찾을 수 없습니다 : %s", tag, path)
    finally:
        print(f"{tag} 처리 종료 : {path}")
    return None


# ---------------------------------------------------------------------------
# 2) <섹션 제목>
# ---------------------------------------------------------------------------
# WHY만 남기는 인라인 주석 예시:
# 여기서는 왜 이 방식을 택했는지, 어떤 비직관적 제약이 있는지만 설명한다.


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== 1단계 : <설명> ===\n")
    result = example_step_one("dummy_path")
    assert result is None, "Checkpoint 설명"

    print("\n\nCheckpoint : 모든 항목이 통과했습니다.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
