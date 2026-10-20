import { initGPU } from "./common.js";
import { mat4 } from "./math.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

// 8 corners: x, y, z, r, g, b
const vertices = new Float32Array([
  -1, -1,  1,  0.2, 0.2, 1.0,   // 0
   1, -1,  1,  1.0, 0.2, 0.2,   // 1
   1,  1,  1,  0.2, 1.0, 0.2,   // 2
  -1,  1,  1,  1.0, 1.0, 0.2,   // 3
  -1, -1, -1,  1.0, 0.2, 1.0,   // 4
   1, -1, -1,  0.2, 1.0, 1.0,   // 5
   1,  1, -1,  1.0, 1.0, 1.0,   // 6
  -1,  1, -1,  0.5, 0.5, 0.5,   // 7
]);
// 12 triangles, every one counter-clockwise when seen from OUTSIDE the cube
const indices = new Uint16Array([
  0, 1, 2,  0, 2, 3,    // front  (+z)
  5, 4, 7,  5, 7, 6,    // back   (-z)
  1, 5, 6,  1, 6, 2,    // right  (+x)
  4, 0, 3,  4, 3, 7,    // left   (-x)
  3, 2, 6,  3, 6, 7,    // top    (+y)
  4, 5, 1,  4, 1, 0,    // bottom (-y)
]);

const vertexBuffer = device.createBuffer({
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);
const indexBuffer = device.createBuffer({
  size: indices.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(indexBuffer, 0, indices);

// One mat4x4f = 16 floats = 64 bytes
const uniformBuffer = device.createBuffer({
  size: 64,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// The depth buffer decides which surface is in front at each pixel
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const shader = device.createShaderModule({
  code: /* wgsl */ `
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs(@location(0) position: vec3f,
          @location(1) color: vec3f) -> VSOut {
      var output: VSOut;
      output.pos = u.mvp * vec4f(position, 1.0);
      output.color = color;
      return output;
    }

    @fragment
    fn fs(input: VSOut) -> @location(0) vec4f {
      return vec4f(input.color, 1.0);
    }
  `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shader, entryPoint: "vs",
    buffers: [{
      arrayStride: 6 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0,      format: "float32x3" },
        { shaderLocation: 1, offset: 3 * 4,  format: "float32x3" },
      ],
    }],
  },
  fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
  primitive: { topology: "triangle-list", cullMode: "back" }, // skip back faces
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",          // nearer fragments win
  },
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
});

const projection = mat4.perspective(
  Math.PI / 4, canvas.width / canvas.height, 0.1, 100
);
const view = mat4.translation(0, 0, -5);       // move the world away from camera

function frame(nowMs) {
  const t = nowMs / 1000;
  const model = mat4.multiply(mat4.rotationX(t * 0.7), mat4.rotationY(t));
  // MVP = Projection * View * Model   (rightmost is applied first)
  const mvp = mat4.multiply(projection, mat4.multiply(view, model));
  device.queue.writeBuffer(uniformBuffer, 0, mvp);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.06, g: 0.06, b: 0.1, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setIndexBuffer(indexBuffer, "uint16");
  pass.drawIndexed(indices.length);
  pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
