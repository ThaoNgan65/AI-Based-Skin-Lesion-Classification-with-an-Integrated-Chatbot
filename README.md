# Soi Da — Chatbot sang loc benh ve da bang AI

Chatbot web don gian: nguoi dung gui anh chup ton thuong tren da, model
multimodal (EfficientNetV2-S + metadata) du doan 1 trong 7 nhom benh cua
tap HAM10000, sau do chatbot dua ra tom tat + loi khuyen ban dau (khong
thay the chan doan y khoa).

## Kien truc

```
[Nguoi dung]
   |  (anh + tuoi/gioi tinh/vi tri - tuy chon)
   v
Flask server.py --route /diagnose-->
   services/skin_model.py   : load best_model.pth, tien xu ly anh + metadata,
                               forward qua model, ap dung threshold rieng cho
                               mel/bcc (xem muc "Threshold mel/bcc" ben duoi),
                               tra ve top-3 nhan + xac suat + do tin cay
   services/advice_service.py: map nhan benh -> loi khuyen + disclaimer
   services/dialog.py        : ghep ket qua model + loi khuyen thanh cau tra loi
                                (va xu ly rieng tin nhan text ranh sanh qua /ask_text)
   |
   v
templates/upload.html + static/  : giao dien chat, xem truoc anh trong
                                    "scan ring", hien thi ket qua + canh bao
```

## 7 nhan benh (theo tap HAM10000)

| Ma    | Ten                                          | Nhom nguy co |
|-------|-----------------------------------------------|--------------|
| akiec | Day sung quang hoa / ung thu bieu mo tai cho  | cao          |
| bcc   | Ung thu bieu mo te bao day                    | cao          |
| bkl   | Ton thuong dang sung lanh tinh                | trung binh   |
| df    | U xo da lanh tinh (Dermatofibroma)            | thap         |
| mel   | U hac to ac tinh (Melanoma)                   | cao          |
| nv    | Not ruoi thuong (Melanocytic nevi)             | thap         |
| vasc  | Ton thuong mach mau lanh tinh                  | thap         |

## Threshold rieng cho mel/bcc (quan trong)

De giam bo sot 2 nhom benh ac tinh de nham lan nhat (melanoma va basal cell
carcinoma), `services/skin_model.py` **khong dung argmax/softmax thuan** de
chon nhan du doan cuoi cung. Thay vao do, ham `apply_mel_bcc_threshold()`
(dong bo 1:1 voi cell "Inference" trong notebook `skin-diseases.ipynb`) ap
dung quy tac:

1. Neu `prob(mel) >= MEL_THRESHOLD (0.30)` → gan nhan **mel**, bat ke lop nao
   co xac suat cao nhat theo argmax.
2. Neu chua bi gan mel va `prob(bcc) >= BCC_THRESHOLD (0.20)` → gan nhan
   **bcc**.
3. Neu khong roi vao 2 truong hop tren → dung argmax binh thuong.

Ket qua tra ve tu `skin_model.predict()`:
- Phan tu dau tien (`top1`) la nhan **sau khi da ap threshold** (co the
  khac voi lop co xac suat cao nhat), kem co `is_thresholded=True` neu
  threshold da ghi de argmax goc.
- Cac phan tu con lai la top-k lop khac co xac suat cao, chi de tham khao.

Hai hang so `MEL_THRESHOLD`, `BCC_THRESHOLD` khai bao dau file
`skin_model.py` — chinh o do neu can tune lai nguong (vd sau khi co them
du lieu / danh gia lai tren tap test moi).

## Cai dat

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`best_model.pth` da co san trong thu muc goc du an (cung cap boi ban, lay
tu notebook `skin-diseases-newmodel.ipynb` - kien truc EfficientNetV2-S + MLP
metadata giong het ban dau, chi cai tien qua trinh training - vd Focal Loss /
weighted sampler de xu ly class imbalance tot hon. val_acc ~86.0%, val_bacc
(balanced accuracy) ~88.2% - can bang giua cac lop tot hon dang ke so ban cu,
dac biet cac lop it du lieu nhu akiec, df, vasc (xem confusion matrix ban gui).

## Chay thu

```bash
python server.py
```

Mo trinh duyet tai `http://localhost:8080`:
1. Nhan "Chon anh" de tai len anh chup vung da can kiem tra.
2. (Tuy chon) Nhan "Them thong tin" de dien tuoi / gioi tinh / vi tri tren
   co the — giup model du doan chinh xac hon vi day la model multimodal.
3. Nhan "Gui" de chan doan. Chatbot tra ve top-3 kha nang (nhan dau tien da
   duoc ap threshold rieng cho mel/bcc — xem muc ben tren), loi khuyen ban
   dau, va luon kem canh bao "khong thay the bac si".
4. Co the go text (vd "huong dan", "xin chao") de duoc tra loi ranh sanh
   khong can gui anh.

## Luu y quan trong

- Day la **cong cu ho tro sang loc**, khong phai cong cu chan doan y khoa.
  Ket qua co the sai; nguoi dung luon duoc nhac di kham bac si da lieu,
  dac biet voi nhom nguy co cao (mel, bcc, akiec).
- Model yeu cau dong thoi ca ANH va METADATA (tuoi/gioi tinh/vi tri) vi day
  la kien truc multimodal luc train; neu nguoi dung khong dien, he thong se
  dung gia tri "khong ro/unknown" da duoc train san (khong loi, nhung do
  chinh xac co the thap hon so voi khi co day du thong tin).
- Muon nang cap loi tu van tu nhien hon: co the goi them LLM API (Claude/OpenAI)
  trong `services/dialog.py` de dien dat lai `advice_service.get_advice()`
  bang van phong gan gui hon, nhung nen giu nguyen phan disclaimer va khong
  de LLM tu them chi dinh dieu tri cu the.

## Thay doi gan day

- **Threshold rieng cho mel/bcc**: `services/skin_model.py` da duoc cap nhat
  de dong bo voi ban nang cap cua notebook (`skin-diseases.ipynb`) — them
  hang so `MEL_IDX`/`BCC_IDX`/`NV_IDX`, ham `apply_mel_bcc_threshold()`, va
  sua ham `predict()` de ap dung threshold mel=0.30 / bcc=0.20 thay vi chi
  dung argmax/top-k theo softmax thuan. Xem chi tiet o muc "Threshold rieng
  cho mel/bcc" ben tren.
- **Bo nut "Bo anh"**: nut nay trong giao dien (`templates/upload.html`,
  gan `scan-preview`) khong hoat dong dung nhu mong doi nen da duoc go bo
  khoi giao dien (`templates/upload.html`) va phan JS lien quan
  (`static/script.js`, bien `clearImageBtn` va event listener tuong ung).
  Neu muon lam lai tinh nang nay trong tuong lai, can kiem tra lai dong bo
  giua `selectedFile` va gia tri cua `<input type="file" id="imageInput">`.
