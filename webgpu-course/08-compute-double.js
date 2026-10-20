import { initGPU, log } from "./common.js";

const { device } = await initGPU();

const N = 1_000_000;
const input = new Float32Array(N);
for (let i = 0; i < N; i++) input[i] = i;
const byteSize = input.byteLength;

// Buffer the shader reads AND writes
const storageBuffer = device.createBuffer({
  label: "data",
  size: byteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(storageBuffer, 0, input);

// Storage buffers cannot be mapped to the CPU, so we copy into a
// "staging" buffer that CAN be mapped for reading.
const readBuffer = device.createBuffer({
  label: "readback",
  size: byteSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const shader = device.createShaderModule({
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      if (i >= arrayLength(&data)) { return; }   // guard the last workgroup
      data[i] = data[i] * 2.0;
    }
  `,
});

const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: shader, entryPoint: "main" },
});
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: storageBuffer } }],
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(N / 64));      // 15,625 workgroups x 64 threads
pass.end();
encoder.copyBufferToBuffer(storageBuffer, 0, readBuffer, 0, byteSize);
device.queue.submit([encoder.finish()]);

// Wait for the GPU, then read the result
await readBuffer.mapAsync(GPUMapMode.READ);
const result = new Float32Array(readBuffer.getMappedRange().slice(0));
readBuffer.unmap();

log("first 5 :", Array.from(result.slice(0, 5)).join(", "));   // 0, 2, 4, 6, 8
log("last 3  :", Array.from(result.slice(-3)).join(", "));
log("expected:", (N - 1) * 2, "for the last element");
