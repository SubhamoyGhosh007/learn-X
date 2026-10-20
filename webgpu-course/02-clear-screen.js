import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context } = await initGPU(canvas);

function render() {
  // 1. Record commands
  const encoder = device.createCommandEncoder({ label: "clear-encoder" });

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(), // where to draw
      clearValue: { r: 0.05, g: 0.15, b: 0.35, a: 1.0 },
      loadOp: "clear",   // start by filling with clearValue
      storeOp: "store",  // keep the result when the pass ends
    }],
  });
  pass.end();            // no draw calls: we only wanted the clear

  // 2. Finish recording and submit to the GPU
  device.queue.submit([encoder.finish()]);
}

render();
