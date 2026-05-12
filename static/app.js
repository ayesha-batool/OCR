const fileInput = document.getElementById("fileInput");
const extractBtn = document.getElementById("extractBtn");
const fileName = document.getElementById("fileName");
const output = document.getElementById("output");
const copyBtn = document.getElementById("copyBtn");

let selectedFile = null;
let extractedText = "";

fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0] || null;
  selectedFile = file;
  fileName.textContent = file ? file.name : "No file selected";
  extractBtn.disabled = !file;
  extractedText = "";
});

extractBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  extractBtn.disabled = true;
  copyBtn.disabled = true;
  output.textContent = "Extracting text…";
  output.classList.add("output--loading");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/extract", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Extraction failed.");
    }
    extractedText = data.text || "";
    output.textContent = extractedText || "(No text found)";
    copyBtn.disabled = false;
  } catch (error) {
    extractedText = "";
    output.textContent = `Error: ${error.message}`;
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
