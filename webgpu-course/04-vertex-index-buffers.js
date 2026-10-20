import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

// x, y, r, g, b  (interleaved: 5 floats = 20 bytes per vertex)
const vertices = new Float32Array([
  -0.5, -0.5,   1, 0, 0,   // 0 bottom-left  red
   0.5, -0.5,   0, 1, 0,   // 1 bottom-right green
   0.5,  0.5,   0, 0, 1,   // 2 top-right    blue
  -0.5,  0.5,   1, 1, 0,   // 3 top-left     yellow
]);
// Two triangles that share vertices (counter-clockwise winding)
const indices = new Uint16Array([0, 1, 2,   0, 2, 3]);

const vertexBuffer = device.createBuffer({
  label: "vertex buffer",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const indexBuffer = device.createBuffer({
  label: "index buffer",
  size: indices.byteLength,            // 12 bytes: multiple of 4, good
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(indexBuffer, 0, indices);

const shader = device.createShaderModule({
  code: /* wgsl */ `
    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs(@location(0) position: vec2f,
          @location(1) color: vec3f) -> VSOut {
      var output: VSOut;
      output.pos = vec4f(position, 0.0, 1.0);
      output.color = color;
      return output;
    }

    @fragment
    fn fs(input: VSOut) -> @location(0) vec4f {
      return vec4f(input.color, 1.0);   // color is interpolated per pixel
    }
  `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shader,
    entryPoint: "vs",
    buffers: [{
      arrayStride: 5 * 4,                       // bytes between vertices
      attributes: [
        { shaderLocation: 0, offset: 0,     format: "float32x2" }, // position
        { shaderLocation: 1, offset: 2 * 4, format: "float32x3" }, // color
      ],
    }],
  },
  fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
  primitive: { topology: "triangle-list" },
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
    loadOp: "clear", storeOp: "store",
  }],
});
pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setIndexBuffer(indexBuffer, "uint16");
pass.drawIndexed(6);
pass.end();
device.queue.submit([encoder.finish()]);
