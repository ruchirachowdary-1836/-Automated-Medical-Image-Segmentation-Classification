import "@testing-library/jest-dom/vitest";

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (RO as unknown as typeof ResizeObserver);
HTMLCanvasElement.prototype.getContext = (() => ({ putImageData() {} })) as never;
if (!globalThis.ImageData) {
  (globalThis as unknown as { ImageData: unknown }).ImageData = class {
    constructor(public data: Uint8ClampedArray, public width: number, public height: number) {}
  };
}
