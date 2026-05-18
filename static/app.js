const fileInput = document.getElementById("fileInput");
const extractBtn = document.getElementById("extractBtn");
const fileName = document.getElementById("fileName");
const output = document.getElementById("output");
const copyBtn = document.getElementById("copyBtn");
const progressWrap = document.getElementById("progressWrap");
const progressLabel = document.getElementById("progressLabel");
const progressPercent = document.getElementById("progressPercent");
const progressBar = document.getElementById("progressBar");

const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;

let selectedFile = null;
let extractedText = "";

function formatFileSize(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setProgress(percent, label) {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  progressWrap.hidden = false;
  progressBar.style.width = `${safePercent}%`;
  progressPercent.textContent = `${safePercent}%`;
  progressLabel.textContent = label;
}

function resetProgress() {
  progressWrap.hidden = true;
  progressBar.style.width = "0%";
  progressPercent.textContent = "0%";
  progressLabel.textContent = "Preparing upload...";
}

function uploadForExtraction(formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/extract");

    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress((event.loaded / event.total) * 100);
      }
    });

    request.addEventListener("load", () => {
      const responseText = request.responseText || "";
      let data = {};
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch {
        reject(new Error(responseText || "Server returned a non-JSON response."));
        return;
      }

      if (request.status < 200 || request.status >= 300) {
        reject(new Error(data.detail || "Extraction failed."));
        return;
      }

      resolve(data);
    });

    request.addEventListener("error", () => {
      reject(new Error("Network error while uploading file."));
    });

    request.addEventListener("abort", () => {
      reject(new Error("Upload was cancelled."));
    });

    request.send(formData);
  });
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0] || null;
  selectedFile = file;
  fileName.textContent = file ? `${file.name} (${formatFileSize(file.size)})` : "No file selected";
  extractBtn.disabled = !file;
  extractedText = "";
  resetProgress();
});

extractBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  if (selectedFile.size > MAX_UPLOAD_BYTES) {
    output.textContent = "Error: File is too large. Please try a file smaller than 300 MB.";
    return;
  }

  extractBtn.disabled = true;
  copyBtn.disabled = true;
  output.textContent = "Extracting text…";
  output.classList.add("output--loading");
  setProgress(0, "Preparing upload...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const data = await uploadForExtraction(formData, (percent) => {
      setProgress(percent, `Uploading ${selectedFile.name}...`);
      if (percent >= 100) {
        progressLabel.textContent = "Upload complete. Extracting pages...";
      }
    });

    extractedText = data.text || "";
    output.textContent = extractedText || "(No text found)";
    if (data.total_pages) {
      setProgress(100, `Extracted ${data.pages_extracted || data.total_pages} of ${data.total_pages} pages.`);
    } else {
      setProgress(100, "Extraction complete.");
    }
    copyBtn.disabled = false;
  } catch (error) {
    extractedText = "";
    output.textContent = `Error: ${error.message}`;
    setProgress(0, "Extraction failed.");
  } finally {
    output.classList.remove("output--loading");
    extractBtn.disabled = false;
  }
});

copyBtn.addEventListener("click", async () => {
  const text = output.textContent || "";
  if (!text || text.startsWith("Error:")) return;
  await navigator.clipboard.writeText(text);
  copyBtn.textContent = "Copied!";
  setTimeout(() => {
    copyBtn.textContent = "Copy to clipboard";
  }, 1000);
});
