// common.js - shared helpers used by every example
export async function initGPU(canvas) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No suitable GPU adapter found.");
  }
  const device = await adapter.requestDevice();

  // Helpful diagnostics: you will thank yourself later.
  device.lost.then((info) => {
    console.error(`Device lost (${info.reason}): ${info.message}`);
  });
  device.addEventListener("uncapturederror", (event) => {
    console.error("WebGPU error:", event.error.message);
  });

  let context = null;
  let format = null;
  if (canvas) {
    context = canvas.getContext("webgpu");
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
  }
  return { adapter, device, context, format };
}

// Prints to the console AND to the <pre id="log"> element on the page.
export function log(...args) {
  console.log(...args);
  const el = document.getElementById("log");
  if (el) el.textContent += args.join(" ") + "\n";
}
