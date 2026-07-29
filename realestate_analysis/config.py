from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexConfig:
    name: str
    aliases: tuple[str, ...]
    lawd_codes: tuple[str, ...]
    start_ym: str
    legal_dong: str
    jibun: str
    type_by_area: dict[float, str]


TARGET_COMPLEX = ComplexConfig(
    name="동탄시범한빛마을한화꿈에그린",
    aliases=(
        "동탄시범한빛마을한화꿈에그린",
        "한빛마을한화꿈에그린",
        "시범한빛마을한화꿈에그린",
    ),
    # 동탄구 코드가 과거 계약분까지 소급해 반환하는 것을 실제 API 응답으로 확인했다.
    lawd_codes=("41597",),
    start_ym="200703",
    legal_dong="반송동",
    jibun="21",
    type_by_area={84.80: "A", 84.73: "B", 84.79: "C"},
)


def normalized_name(value: str) -> str:
    return "".join(str(value).split()).replace("아파트", "")
