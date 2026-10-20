import { initGPU, log } from "./common.js";

const { adapter, device } = await initGPU();   // no canvas needed

const info = adapter.info ?? {};
log("Vendor:      ", info.vendor ?? "n/a");
log("Architecture:", info.architecture ?? "n/a");
log("Description: ", info.description ?? "n/a");
log("Features:    ", [...adapter.features].join(", ") || "(none)");

log("maxTextureDimension2D:", device.limits.maxTextureDimension2D);
log("maxBufferSize:", device.limits.maxBufferSize);
log("maxStorageBufferBindingSize:", device.limits.maxStorageBufferBindingSize);
log("maxComputeInvocationsPerWorkgroup:",
    device.limits.maxComputeInvocationsPerWorkgroup);
log("maxComputeWorkgroupSizeX:", device.limits.maxComputeWorkgroupSizeX);
