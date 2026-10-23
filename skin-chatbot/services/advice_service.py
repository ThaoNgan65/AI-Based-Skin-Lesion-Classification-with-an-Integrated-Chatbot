# -*- coding: utf-8 -*-
"""
services/advice_service.py

Tư vấn đơn giản dựa trên nhãn bệnh dự đoán từ skin_model.predict().
Đây là lời khuyên chung (rule-based), KHÔNG phải chỉ định điều trị,
luôn đi kèm disclaimer y tế.
"""

DISCLAIMER = (
    "Đây là công cụ hỗ trợ sàng lọc bằng AI, KHÔNG thay thế chẩn đoán y khoa. "
    "Kết quả chỉ mang tính tham khảo và có thể sai. Bạn nên đến cơ sở y tế / "
    "gặp bác sĩ da liễu để được thăm khám trực tiếp, đặc biệt nếu tổn thương "
    "thay đổi kích thước, màu sắc, hoặc gây đau/chảy máu."
)

ADVICE = {
    "mel": (
        "Đây là nhóm có nguy cơ cao nhất trong 7 loại (u hắc tố ác tính). "
        "Bạn NÊN đi khám bác sĩ da liễu càng sớm càng tốt để được sinh thiết "
        "xác nhận. Trong lúc chờ lịch khám, tránh cào/nặn/tác động lên tổn "
        "thương và chụp lại ảnh để theo dõi thay đổi theo thời gian."
    ),
    "bcc": (
        "Ung thư biểu mô tế bào đáy thường phát triển chậm, ít di căn, nhưng "
        "vẫn cần bác sĩ da liễu kiểm tra và điều trị (thường là phẫu thuật cắt "
        "bỏ nhỏ) để tránh tổn thương lan rộng thêm."
    ),
    "akiec": (
        "Đây là tổn thương tiền ung thư do tia UV gây ra. Nên đi khám da liễu "
        "để được theo dõi hoặc điều trị sớm (áp lạnh, thuốc bôi...). Hạn chế "
        "phơi nắng trực tiếp và dùng kem chống nắng phổ rộng hằng ngày."
    ),
    "bkl": (
        "Thường là tổn thương lành tính (ví dụ dày sừng tiết bã). Nếu không "
        "thay đổi kích thước/màu sắc thì ít đáng lo ngại, nhưng nếu bạn không "
        "chắc chắn hoặc tổn thương mới xuất hiện gần đây, vẫn nên khám để loại "
        "trừ các nguyên nhân khác."
    ),
    "df": (
        "U xơ da lành tính, thường không cần điều trị nếu không gây khó chịu. "
        "Chỉ cần theo dõi nếu kích thước tăng bất thường hoặc gây đau khi chạm vào."
    ),
    "nv": (
        "Đây là nốt ruồi thông thường, đa số lành tính. Bạn vẫn nên tự kiểm tra "
        "định kỳ theo quy tắc ABCDE: Asymmetry (bất đối xứng), Border (viền "
        "không đều), Color (màu không đều), Diameter (đường kính > 6mm), "
        "Evolving (thay đổi theo thời gian). Nếu có bất kỳ dấu hiệu nào ở trên, "
        "hãy đi khám sớm."
    ),
    "vasc": (
        "Tổn thương mạch máu lành tính (ví dụ u máu, giãn mạch), thường không "
        "nguy hiểm. Có thể tham khảo bác sĩ da liễu nếu cần về mặt thẩm mỹ hoặc "
        "nếu tổn thương chảy máu/thay đổi bất thường."
    ),
}

HIGH_RISK_CODES = {"mel", "bcc", "akiec"}


def get_advice(class_code: str) -> str:
    return ADVICE.get(class_code, "Chưa có thông tin tư vấn cho nhãn này.")


def get_disclaimer() -> str:
    return DISCLAIMER


def is_high_risk(class_code: str) -> bool:
    return class_code in HIGH_RISK_CODES
