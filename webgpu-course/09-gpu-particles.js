import { initGPU } from "./common.js";

const canvas = document.getElementById("gpu-canvas");
const { device, context, format } = await initGPU(canvas);

const N = 20000;                       // number of particles

// Each particle: pos (vec2f) + vel (vec2f) = 4 floats = 16 bytes
const init = new Float32Array(N * 4);
for (let i = 0; i < N; i++) {
  init[i * 4 + 0] = Math.random() * 2 - 1;        // pos.x
  init[i * 4 + 1] = Math.random() * 2 - 1;        // pos.y
  init[i * 4 + 2] = (Math.random() - 0.5) * 0.6;  // vel.x
  init[i * 4 + 3] = (Math.random() - 0.5) * 0.6;  // vel.y
}
const particleBuffer = device.createBuffer({
  label: "particles",
  size: init.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(particleBuffer, 0, init);

const paramsBuffer = device.createBuffer({
  label: "params",
  size: 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// ---------- compute shader: move particles, bounce off the walls
const computeModule = device.createShaderModule({
  code: /* wgsl */ `
    struct Particle { pos: vec2f, vel: vec2f };
    struct Params { dt: f32 };

    @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
    @group(0) @binding(1) var<uniform> params: Params;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      let i = gid.x;
      if (i >= arrayLength(&particles)) { return; }

      var p = particles[i];
      p.pos = p.pos + p.vel * params.dt;

      if (abs(p.pos.x) > 1.0) {
        p.vel.x = -p.vel.x;
        p.pos.x = clamp(p.pos.x, -1.0, 1.0);
      }
      if (abs(p.pos.y) > 1.0) {
        p.vel.y = -p.vel.y;
        p.pos.y = clamp(p.pos.y, -1.0, 1.0);
      }
      particles[i] = p;
    }
  `,
});

// ---------- render shader: draw each particle as a tiny instanced quad
const renderModule = device.createShaderModule({
  code: /* wgsl */ `
    struct Particle { pos: vec2f, vel: vec2f };
    @group(0) @binding(0) var<storage, read> particles: array<Particle>;

    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f,
    };

    @vertex
    fn vs(@builtin(vertex_index) vi: u32,
          @builtin(instance_index) ii: u32) -> VSOut {
      var corners = array<vec2f, 6>(
        vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1),
        vec2f(-1, -1), vec2f(1, 1),  vec2f(-1, 1)
      );
      let p = particles[ii];
      var output: VSOut;
      output.pos = vec4f(p.pos + corners[vi] * 0.006, 0.0, 1.0);
      output.color = vec3f(0.5 + 0.5 * p.pos.x, 0.5 + 0.5 * p.pos.y, 1.0);
      return output;
    }

    @fragment
    fn fs(input: VSOut) -> @location(0) vec4f {
      return vec4f(input.color, 1.0);
    }
  `,
});

const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: computeModule, entryPoint: "main" },
});
const renderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex:   { module: renderModule, entryPoint: "vs" },
  fragment: { module: renderModule, entryPoint: "fs", targets: [{ format }] },
});

// Same particle buffer, bound into two different pipelines
const computeBG = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: particleBuffer } },
    { binding: 1, resource: { buffer: paramsBuffer } },
  ],
});
const renderBG = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: particleBuffer } }],
});

let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);   // clamp huge gaps
  last = now;
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([dt]));

  const encoder = device.createCommandEncoder();

  const cpass = encoder.beginComputePass();          // 1) simulate
  cpass.setPipeline(computePipeline);
  cpass.setBindGroup(0, computeBG);
  cpass.dispatchWorkgroups(Math.ceil(N / 64));
  cpass.end();

  const rpass = encoder.beginRenderPass({            // 2) draw
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0.03, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
  });
  rpass.setPipeline(renderPipeline);
  rpass.setBindGroup(0, renderBG);
  rpass.draw(6, N);                                  // 6 vertices x N instances
  rpass.end();

  device.queue.submit([encoder.finish()]);           // ONE submit per frame
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
