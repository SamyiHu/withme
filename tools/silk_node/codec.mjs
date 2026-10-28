// SILK 编解码兜底（silk-wasm，纯 WASM 零编译）：
//   node codec.mjs encode <in.pcm> <out.silk> [rate]   24kHz PCM -> SILK(0x02#!SILK_V3)
//   node codec.mjs decode <in.silk> <out.pcm> [rate]   SILK -> PCM16/mono
// 依赖：tools/silk_node 下 npm install（src/silk_audio.py 首次调用会自动装）。
import { encode, decode } from "silk-wasm";
import { readFileSync, writeFileSync } from "node:fs";

const [, , mode, inPath, outPath, rateArg] = process.argv;
const rate = Number(rateArg || 24000);
if (!mode || !inPath || !outPath) {
  console.error("usage: node codec.mjs encode|decode <in> <out> [rate]");
  process.exit(2);
}
const input = readFileSync(inPath);
if (mode === "encode") {
  const { data } = await encode(input, rate);
  writeFileSync(outPath, Buffer.from(data));
} else if (mode === "decode") {
  const { data } = await decode(input, rate);
  writeFileSync(outPath, Buffer.from(data));
} else {
  console.error("mode 必须是 encode | decode");
  process.exit(2);
}
