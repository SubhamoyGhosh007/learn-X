import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

// ---- 1. Make an 8x8 checkerboard on the CPU (RGBA, 1 byte per channel)
const size = 8;
const pixels = new Uint8Array(size * size * 4);
for (let y = 0; y < size; y++) {
  for (let x = 0; x < size; x++) {
    const i = (y * size + x) * 4;
    const on = (x + y) % 2 === 0;
    pixels[i + 0] = on ? 240 : 30;
    pixels[i + 1] = on ? 200 : 60;
    pixels[i + 2] = on ? 40 : 160;
    pixels[i + 3] = 255;
  }
}

// ---- 2. Create the GPU texture and upload the pixels
const texture = device.createTexture({
  label: "checkerboard",
  size: [size, size],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
device.queue.writeTexture(
  { texture },
  pixels,
  { bytesPerRow: size * 4 },
  [size, size]
);

// ---- 3. A sampler says HOW to read the texture
const sampler = device.createSampler({
  magFilter: "nearest",       // crisp pixels when magnified
  minFilter: "nearest",
  addressModeU: "repeat",     // uv > 1 wraps around (tiling)
  addressModeV: "repeat",
});

// x, y, u, v   (uv goes 0..4, so the texture repeats 4 times)
const vertices = new Float32Array([
  -0.8, -0.8,  0, 4,
   0.8, -0.8,  4, 4,
   0.8,  0.8,  4, 0,
  -0.8,  0.8,  0, 0,
]);
const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
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

const shader = device.createShaderModule({
  code: /* wgsl */ `
    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex
    fn vs(@location(0) position: vec2f,
          @location(1) uv: vec2f) -> VSOut {
      var output: VSOut;
      output.pos = vec4f(position, 0.0, 1.0);
      output.uv = uv;
      return output;
    }

    @group(0) @binding(0) var samp: sampler;
    @group(0) @binding(1) var tex: texture_2d<f32>;

    @fragment
    fn fs(input: VSOut) -> @location(0) vec4f {
      return textureSample(tex, samp, input.uv);
    }
  `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shader, entryPoint: "vs",
    buffers: [{
      arrayStride: 16,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
      ],
    }],
  },
  fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: sampler },
    { binding: 1, resource: texture.createView() },
  ],
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
    loadOp: "clear", storeOp: "store",
  }],
});
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.setVertexBuffer(0, vertexBuffer);
pass.setIndexBuffer(indexBuffer, "uint16");
pass.drawIndexed(6);
pass.end();
device.queue.submit([encoder.finish()]);
