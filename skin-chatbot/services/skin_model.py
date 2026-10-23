"""
services/skin_model.py

Load model chẩn đoán bệnh da (best_model.pth) và cung cấp hàm predict().

Kiến trúc y hệt notebook gốc (skin-diseases.ipynb) - PHẢI giữ nguyên số
chiều/thứ tự layer thì state_dict mới load đúng:

    ImageEncoder (EfficientNetV2-S, pretrained backbone -> embed 512)
    MetadataEncoder (MLP 19 -> 64 -> 128)
    MultimodalClassifier (fusion 640 -> 256 -> 7 lớp)

7 nhãn bệnh (dx) theo đúng thứ tự alphabet của sklearn.LabelEncoder trên
tập HAM10000: akiec, bcc, bkl, df, mel, nv, vasc.

Metadata đầu vào (19 chiều), đúng thứ tự như lúc train:
    [age_norm, sex_male, sex_female, sex_unknown] + 15 cột loc_<vị_trí>
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thứ tự 7 nhãn đúng như LabelEncoder().fit_transform(df["dx"]) (alphabet)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Index của các lớp ác tính cần threshold riêng (đồng bộ với notebook
# skin-diseases.ipynb, cell "Evaluation")
MEL_IDX = CLASSES.index("mel")   # = 4
BCC_IDX = CLASSES.index("bcc")   # = 1
NV_IDX = CLASSES.index("nv")     # = 5

# Threshold đã chốt ở notebook (cell "Inference — Hàm predict chính thức
# để deploy"): mel=0.30, bcc=0.20. Hạ ngưỡng để giảm bỏ sót các lớp ác tính.
MEL_THRESHOLD = 0.30
BCC_THRESHOLD = 0.20

# Thứ tự 15 vị trí trên cơ thể, đúng như pd.get_dummies(df["localization"])
LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown",
    "upper extremity",
]

# Nhãn hiển thị cho người dùng (tiếng Việt, không dùng thuật ngữ chuyên môn quá nặng)
CLASS_INFO = {
    "akiec": {
        "name_vi": "Dày sừng quang hóa / ung thư biểu mô tại chỗ (AKIEC)",
        "risk": "high",
    },
    "bcc": {
        "name_vi": "Ung thư biểu mô tế bào đáy (BCC)",
        "risk": "high",
    },
    "bkl": {
        "name_vi": "Tổn thương dạng sừng lành tính (BKL)",
        "risk": "medium",
    },
    "df": {
        "name_vi": "U xơ da lành tính (Dermatofibroma)",
        "risk": "low",
    },
    "mel": {
        "name_vi": "U hắc tố ác tính - Melanoma",
        "risk": "high",
    },
    "nv": {
        "name_vi": "Nốt ruồi thường (Melanocytic nevi)",
        "risk": "low",
    },
    "vasc": {
        "name_vi": "Tổn thương mạch máu lành tính",
        "risk": "low",
    },
}

_AGE_MIN, _AGE_MAX = 0.0, 100.0  # xấp xỉ khoảng tuổi dùng khi normalize lúc train


# ------------------------------------------------------------------
# Kiến trúc model (copy từ notebook, không được đổi số chiều)
# ------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=19, embed_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=7, metadata_dim=19, image_embed_dim=512,
                 meta_embed_dim=128, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=image_embed_dim)
        self.metadata_encoder = MetadataEncoder(input_dim=metadata_dim, embed_dim=meta_embed_dim)
        concat_dim = image_embed_dim + meta_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, image, metadata):
        img_feat = self.image_encoder(image)
        meta_feat = self.metadata_encoder(metadata)
        fused = self.fusion(torch.cat([img_feat, meta_feat], dim=1))
        return self.classifier(fused)


# ------------------------------------------------------------------
# Load model 1 lần duy nhất khi module được import
# ------------------------------------------------------------------
_model = MultimodalClassifier()
_checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
_state_dict = _checkpoint["model_state_dict"] if isinstance(_checkpoint, dict) and "model_state_dict" in _checkpoint else _checkpoint
_model.load_state_dict(_state_dict)
_model.to(DEVICE)
_model.eval()

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_mel_bcc_threshold(all_probs, mel_threshold=MEL_THRESHOLD, bcc_threshold=BCC_THRESHOLD):
    """
    Hàm áp dụng threshold cuối cùng cho mel/bcc — y hệt notebook
    skin-diseases.ipynb (dùng chung cho evaluation VÀ deploy).

    all_probs: np.ndarray shape (N, 7) - xác suất softmax của N mẫu.
    Trả về preds: np.ndarray shape (N,) - index lớp dự đoán sau khi áp threshold.
    """
    preds = np.argmax(all_probs, axis=1).copy()
    preds[all_probs[:, MEL_IDX] >= mel_threshold] = MEL_IDX
    # Chỉ ghi đè bcc cho mẫu CHƯA bị gán mel (tránh chồng lấn)
    mask = (preds != MEL_IDX) & (all_probs[:, BCC_IDX] >= bcc_threshold)
    preds[mask] = BCC_IDX
    return preds


def get_localizations():
    """Danh sách vị trí trên cơ thể để hiển thị lên dropdown UI."""
    return LOCALIZATIONS


def _build_metadata_tensor(age, sex, localization):
    age = float(age) if age not in (None, "") else 40.0
    age_norm = (min(max(age, _AGE_MIN), _AGE_MAX) - _AGE_MIN) / (_AGE_MAX - _AGE_MIN)

    sex = (sex or "unknown").lower()
    sex_male = 1.0 if sex == "male" else 0.0
    sex_female = 1.0 if sex == "female" else 0.0
    sex_unknown = 1.0 if sex not in ("male", "female") else 0.0

    loc = localization if localization in LOCALIZATIONS else "unknown"
    loc_vec = [1.0 if c == loc else 0.0 for c in LOCALIZATIONS]

    vec = [age_norm, sex_male, sex_female, sex_unknown] + loc_vec
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


def predict(image_path, age=40, sex="unknown", localization="unknown", top_k=3,
            mel_threshold=MEL_THRESHOLD, bcc_threshold=BCC_THRESHOLD):
    """
    Chạy ảnh + metadata qua model, áp dụng threshold mel/bcc đã chốt trong
    notebook (mel=0.30, bcc=0.20) để giảm bỏ sót các lớp ác tính, rồi trả về
    danh sách dự đoán:
        [{"code": "mel", "name_vi": "...", "risk": "high", "prob": 0.83,
          "confidence": 0.83, "is_thresholded": True}, ...]
    Phần tử đầu tiên (top1) là nhãn dự đoán CUỐI CÙNG sau khi áp threshold
    (đồng bộ với hàm predict() trong notebook skin-diseases.ipynb).
    Các phần tử sau là top_k-1 lớp còn lại có xác suất cao nhất, để tham khảo.
    """
    image = Image.open(image_path).convert("RGB")
    img_tensor = _transform(image).unsqueeze(0).to(DEVICE)
    meta_tensor = _build_metadata_tensor(age, sex, localization).to(DEVICE)

    with torch.no_grad():
        logits = _model(img_tensor, meta_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape (1, 7)

    # Nhãn dự đoán sau khi áp threshold mel/bcc (giống notebook 1:1)
    thresholded_idx = int(apply_mel_bcc_threshold(
        probs, mel_threshold=mel_threshold, bcc_threshold=bcc_threshold
    )[0])
    # "Độ tin cậy": xác suất của lớp ĐÃ ĐƯỢC DỰ ĐOÁN (không nhất thiết là
    # xác suất cao nhất, vì threshold có thể đã ghi đè argmax gốc)
    confidence = float(probs[0, thresholded_idx])

    argmax_idx = int(np.argmax(probs[0]))
    threshold_applied = thresholded_idx != argmax_idx

    # Xếp hạng các lớp theo xác suất giảm dần, để hiện "top_k" tham khảo
    order = np.argsort(-probs[0])
    ordered_indices = [thresholded_idx] + [i for i in order if i != thresholded_idx]
    ordered_indices = ordered_indices[:max(top_k, 1)]

    results = []
    for rank, idx in enumerate(ordered_indices):
        code = CLASSES[idx]
        info = CLASS_INFO[code]
        prob = float(probs[0, idx])
        results.append({
            "code": code,
            "name_vi": info["name_vi"],
            "risk": info["risk"],
            "prob": round(prob, 4),
            "confidence": round(prob, 4),
            "is_thresholded": rank == 0 and threshold_applied,
        })
    return results
