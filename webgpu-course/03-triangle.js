import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

const shader = device.createShaderModule({
  label: "triangle shader",
  code: /* wgsl */ `
    @vertex
    fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
      var positions = array<vec2f, 3>(
        vec2f( 0.0,  0.6),   // top
        vec2f(-0.6, -0.6),   // bottom-left
        vec2f( 0.6, -0.6)    // bottom-right
      );
      return vec4f(positions[i], 0.0, 1.0);
    }

    @fragment
    fn fs() -> @location(0) vec4f {
      return vec4f(1.0, 0.45, 0.15, 1.0);   // orange (r, g, b, a)
    }
  `,
});

const pipeline = device.createRenderPipeline({
  label: "triangle pipeline",
  layout: "auto",
  vertex:   { module: shader, entryPoint: "vs" },
  fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
  primitive: { topology: "triangle-list" },
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
    loadOp: "clear",
    storeOp: "store",
  }],
});
pass.setPipeline(pipeline);
pass.draw(3);            // 3 vertices -> vertex shader runs 3 times
pass.end();
device.queue.submit([encoder.finish()]);
