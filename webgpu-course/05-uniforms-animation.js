import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

const vertices = new Float32Array([
  -0.5, -0.5,  1, 0, 0,
   0.5, -0.5,  0, 1, 0,
   0.5,  0.5,  0, 0, 1,
  -0.5,  0.5,  1, 1, 0,
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

// Uniform buffer: { time: f32, aspect: f32 }  (8 bytes; we allocate 16)
const uniformData = new Float32Array(4);
const uniformBuffer = device.createBuffer({
  label: "uniforms",
  size: uniformData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const shader = device.createShaderModule({
  code: /* wgsl */ `
    struct Uniforms {
      time: f32,
      aspect: f32,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs(@location(0) position: vec2f,
          @location(1) color: vec3f) -> VSOut {
      let c = cos(u.time);
      let s = sin(u.time);
      let rotated = vec2f(position.x * c - position.y * s,
                          position.x * s + position.y * c);
      var output: VSOut;
      // divide x by aspect so the square stays square on a wide canvas
      output.pos = vec4f(rotated.x / u.aspect, rotated.y, 0.0, 1.0);
      output.color = color;
      return output;
    }

    @fragment
    fn fs(input: VSOut) -> @location(0) vec4f {
      let pulse = 0.75 + 0.25 * sin(u.time * 3.0);
      return vec4f(input.color * pulse, 1.0);
    }
  `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shader, entryPoint: "vs",
    buffers: [{
      arrayStride: 20,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x3" },
      ],
    }],
  },
  fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
});

// The bind group connects our actual buffer to @group(0) @binding(0)
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
});

function frame(nowMs) {
  uniformData[0] = nowMs / 1000;                   // time in seconds
  uniformData[1] = canvas.width / canvas.height;   // aspect ratio
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),  // fresh view EVERY frame
      clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
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

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
