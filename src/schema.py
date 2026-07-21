"""
==============================================================================
 프로그램명 : schema
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   정제된 trip 레코드의 필드 타입·범위를 Pydantic v2 모델로 검증한다.
     1) TripRecord : tip_amount, fare_amount, trip_distance, payment_type,
                      passenger_count 등의 타입·범위 계약 정의

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (필드 뼈대만, 검증 로직은 팀원 작업 예정)
==============================================================================
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1) TripRecord — 정제된 trip 레코드 스키마
# ---------------------------------------------------------------------------
# TODO: 팀원 작업 예정 — 실제 컬럼명/제약조건에 맞춰 Field 범위 확정
class TripRecord(BaseModel):
    payment_type: int = Field(..., description="결제 수단 코드 (1 = 신용카드)")
    fare_amount: float = Field(..., gt=0, description="운행 요금")
    trip_distance: float = Field(..., gt=0, description="운행 거리")
    tip_amount: float = Field(..., ge=0, description="팁 금액")
    passenger_count: int = Field(..., ge=0, description="승객 수")
