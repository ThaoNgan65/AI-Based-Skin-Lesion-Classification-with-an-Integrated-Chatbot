# -*- coding: utf-8 -*-
"""
services/dialog.py

Điều phối hội thoại:
  - build_diagnosis_reply(): ghép kết quả model + tư vấn thành câu trả lời chat
  - get_text_reply(): trả lời rành sạch (rule-based) khi người dùng gõ text
    thay vì gửi ảnh (chào hỏi, hướng dẫn, hỏi về độ tin cậy...)
"""

from services import advice_service

RISK_LABEL_VI = {
    "high": "nguy cơ cao",
    "medium": "cần chú ý",
    "low": "nguy cơ thấp",
}


def build_diagnosis_reply(results):
    """
    results: list từ skin_model.predict(), đã sắp xếp giảm dần theo prob.
    Trả về dict {"summary": str, "top1": {...}, "advice": str, "disclaimer": str}
    để frontend hiển thị.
    """
    top1 = results[0]
    confidence_pct = round(top1["prob"] * 100, 1)
    risk_vi = RISK_LABEL_VI.get(top1["risk"], "")

    if confidence_pct < 50:
        confidence_note = (
            f" Độ tin cậy của dự đoán này không cao ({confidence_pct}%), "
            "bạn nên chụp thêm ảnh rõ nét hơn (đủ sáng, lấy nét vào tổn thương) "
            "hoặc hỏi trực tiếp bác sĩ để có kết quả chính xác hơn."
        )
    else:
        confidence_note = ""

    summary = (
        f"Dựa trên hình ảnh bạn gửi, khả năng cao nhất là '{top1['name_vi']}' "
        f"với độ tin cậy khoảng {confidence_pct}% ({risk_vi})." + confidence_note
    )

    advice_text = advice_service.get_advice(top1["code"])

    return {
        "summary": summary,
        "top1": top1,
        "top_results": results,
        "advice": advice_text,
        "disclaimer": advice_service.get_disclaimer(),
        "high_risk": advice_service.is_high_risk(top1["code"]),
    }


# ------------------------------------------------------------------
# Rule-based fallback cho tin nhắn text (khi chưa gửi ảnh)
# ------------------------------------------------------------------
_GREETINGS = ("chao", "hi", "hello", "xin chao", "alo")
_HELP_KEYWORDS = ("giup", "huong dan", "lam sao", "cach dung", "help")
_THANKS_KEYWORDS = ("cam on", "thanks", "thank you")


def _normalize(text: str) -> str:
    return text.strip().lower()


def get_text_reply(user_text: str) -> str:
    text = _normalize(user_text)

    if any(k in text for k in _GREETINGS):
        return (
            "Xin chào! Tôi là chatbot hỗ trợ sàng lọc bệnh về da bằng AI. "
            "Bạn hãy gửi một bức ảnh chụp rõ tổn thương trên da (kèm tuổi, giới "
            "tính, vị trí nếu có), tôi sẽ đưa ra dự đoán và lời khuyên ban đầu."
        )

    if any(k in text for k in _HELP_KEYWORDS):
        return (
            "Cách dùng: 1) Chụp ảnh rõ nét, đủ sáng vào vùng da cần kiểm tra. "
            "2) Nhấn 'Chọn ảnh' và tùy chọn điền thêm tuổi/giới tính/vị trí trên "
            "cơ thể để tăng độ chính xác. 3) Nhấn 'Chẩn đoán' để xem kết quả và "
            "lời khuyên. Lưu ý đây chỉ là công cụ tham khảo, không thay thế bác sĩ."
        )

    if any(k in text for k in _THANKS_KEYWORDS):
        return "Không có gì! Chúc bạn nhiều sức khỏe. Nếu cần kiểm tra thêm tổn thương khác, cứ gửi ảnh nhé."

    return (
        "Tôi chủ yếu hỗ trợ chẩn đoán sơ bộ qua HÌNH ẢNH. Bạn hãy gửi ảnh chụp "
        "tổn thương trên da để tôi phân tích, hoặc gõ 'hướng dẫn' để xem cách sử dụng."
    )
