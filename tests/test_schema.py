"""
==============================================================================
 프로그램명 : test_schema
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   src/schema.py 의 TripRecord Pydantic 모델 검증 테스트.
     1) test_valid_trip_record_passes() : 정상 데이터 검증 통과 확인
     2) test_invalid_trip_record_raises() : 범위를 벗어난 데이터의 검증 실패 확인

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (스텁만, 팀원 작업 예정)
   v1.1  2026-07-21  장상민  테스트 로직 구현
==============================================================================
"""

import pytest
from pydantic import ValidationError

from src.schema import TripRecord

VALID_PAYLOAD = {
    "payment_type": 1,
    "fare_amount": 12.5,
    "trip_distance": 3.2,
    "tip_amount": 2.5,
    "passenger_count": 1,
}


def test_valid_trip_record_passes():
    record = TripRecord(**VALID_PAYLOAD)
    assert record.payment_type == 1
    assert record.fare_amount == 12.5
    assert record.trip_distance == 3.2


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("fare_amount", -5.0),   # 요금은 0보다 커야 함 (gt=0)
        ("trip_distance", 0.0),  # 거리는 0보다 커야 함 (gt=0)
        ("tip_amount", -1.0),    # 팁은 음수일 수 없음 (ge=0)
        ("passenger_count", -1), # 승객 수는 음수일 수 없음 (ge=0)
    ],
)
def test_invalid_trip_record_raises(field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    with pytest.raises(ValidationError):
        TripRecord(**payload)
