# -*- coding: utf-8 -*-
"""
server.py

Chatbot chan doan benh da co ban:
  - Giu khung Flask tuong tu ban goc (cb106-AIWatsonLab), nhung route chinh
    la /diagnose: nhan anh (+ metadata tuy chon) -> services.skin_model.predict()
    -> services.dialog.build_diagnosis_reply() -> tra ve JSON cho UI chat.
  - /ask_text: fallback tra loi text ranh sanh (chao hoi, huong dan...) khi
    nguoi dung go chu thay vi gui anh.
"""

import os
from flask import Flask, render_template, request, jsonify

from services import skin_model, dialog

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_TMP_DIR = os.path.join(BASE_DIR, "tmp_uploads")
os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH_MB = 10
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024


@app.route("/")
def index():
    return render_template(
        "upload.html",
        localizations=skin_model.get_localizations(),
    )


@app.route("/ask_text", methods=["POST"])
def ask_text():
    """Tra loi cac tin nhan text don gian (khong kem anh)."""
    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify(error="Vui long nhap noi dung."), 400
    reply = dialog.get_text_reply(user_text)
    return jsonify(reply_text=reply)


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Nhan anh (form field 'image') + metadata tuy chon (age, sex, localization).
    Tra ve JSON gom top-3 du doan, tom tat, loi khuyen, disclaimer.
    """
    f = request.files.get("image")
    if f is None or f.filename == "":
        return jsonify(error="Vui long chon mot buc anh de chan doan."), 400

    extn = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if extn not in ALLOWED_EXTENSIONS:
        return jsonify(error="Dinh dang anh khong duoc ho tro. Vui long dung jpg/png/webp."), 400

    age = request.form.get("age", "")
    sex = request.form.get("sex", "unknown")
    localization = request.form.get("localization", "unknown")

    safe_name = f"upload_{os.getpid()}_{abs(hash(f.filename))}.{extn}"
    save_path = os.path.join(UPLOAD_TMP_DIR, safe_name)
    f.save(save_path)

    try:
        results = skin_model.predict(
            save_path,
            age=age,
            sex=sex,
            localization=localization,
        )
        reply = dialog.build_diagnosis_reply(results)
        return jsonify(**reply)
    except Exception as excp:
        print("Loi khi chan doan:", excp)
        return jsonify(error="Co loi xay ra khi xu ly anh. Vui long thu lai."), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
