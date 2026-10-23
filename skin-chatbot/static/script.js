// static/script.js
// Xu ly UI chat: chon anh (preview trong "scan ring"), gui form len /diagnose,
// hoac gui text len /ask_text, roi render bong bong chat tuong ung.

const chatLog = document.getElementById("chatLog");
const composer = document.getElementById("composer");
const imageInput = document.getElementById("imageInput");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const scanPreview = document.getElementById("scanPreview");
const previewImg = document.getElementById("previewImg");
const scanLine = document.getElementById("scanLine");
const toggleDetailsBtn = document.getElementById("toggleDetails");
const detailsRow = document.getElementById("detailsRow");
const ageInput = document.getElementById("ageInput");
const sexInput = document.getElementById("sexInput");
const locInput = document.getElementById("locInput");

let selectedFile = null;

const RISK_LABEL = { high: "Nguy cơ cao", medium: "Cần chú ý", low: "Nguy cơ thấp" };

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addBotBubble(html) {
  const msg = document.createElement("div");
  msg.className = "msg msg-bot";
  msg.innerHTML = `<div class="bubble">${html}</div>`;
  chatLog.appendChild(msg);
  scrollToBottom();
}

function addUserBubble({ text, imageDataUrl }) {
  const msg = document.createElement("div");
  msg.className = "msg msg-user";
  const imgHtml = imageDataUrl
    ? `<img class="result-photo" src="${imageDataUrl}" alt="Ảnh đã gửi">`
    : "";
  const textHtml = text ? `<div>${escapeHtml(text)}</div>` : "";
  msg.innerHTML = `<div class="bubble">${imgHtml}${textHtml}</div>`;
  chatLog.appendChild(msg);
  scrollToBottom();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderDiagnosisReply(data) {
  const rows = data.top_results.map(r => {
    const pct = (r.prob * 100).toFixed(1);
    return `<div class="result-row risk-${r.risk}">
              <span class="name">${escapeHtml(r.name_vi)}</span>
              <span class="prob">${pct}%</span>
            </div>`;
  }).join("");

  const highRiskFlag = data.high_risk
    ? `<div class="disclaimer-inline">🔺 Nhóm nguy cơ cao — nên ưu tiên đi khám bác sĩ da liễu sớm.</div>`
    : "";

  const html = `
    <div>${escapeHtml(data.summary)}</div>
    <div class="result-card">${rows}</div>
    <div class="advice-block">
      <span class="label">Gợi ý ban đầu</span>
      ${escapeHtml(data.advice)}
    </div>
    ${highRiskFlag}
    <div class="disclaimer-inline">${escapeHtml(data.disclaimer)}</div>
  `;
  addBotBubble(html);
}

// ---- Image selection / preview ----
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    scanPreview.hidden = false;
  };
  reader.readAsDataURL(file);
});

toggleDetailsBtn.addEventListener("click", () => {
  detailsRow.hidden = !detailsRow.hidden;
  toggleDetailsBtn.textContent = detailsRow.hidden ? "Thêm thông tin ▾" : "Thu gọn ▴";
});

// ---- Submit (send) ----
composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();

  if (!selectedFile && !text) return;

  sendBtn.disabled = true;
  scanLine.hidden = false;

  try {
    if (selectedFile) {
      const imageDataUrl = previewImg.src;
      addUserBubble({ text, imageDataUrl });

      const formData = new FormData();
      formData.append("image", selectedFile);
      if (ageInput.value) formData.append("age", ageInput.value);
      formData.append("sex", sexInput.value);
      formData.append("localization", locInput.value);

      const res = await fetch("/diagnose", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        addBotBubble(`⚠ ${escapeHtml(data.error || "Có lỗi xảy ra, vui lòng thử lại.")}`);
      } else {
        renderDiagnosisReply(data);
      }

      selectedFile = null;
      imageInput.value = "";
      scanPreview.hidden = true;
    } else {
      addUserBubble({ text });
      const res = await fetch("/ask_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      addBotBubble(escapeHtml(data.reply_text || data.error || ""));
    }

    textInput.value = "";
  } catch (err) {
    console.error(err);
    addBotBubble("⚠ Không thể kết nối tới máy chủ. Vui lòng thử lại.");
  } finally {
    sendBtn.disabled = false;
    scanLine.hidden = true;
  }
});
