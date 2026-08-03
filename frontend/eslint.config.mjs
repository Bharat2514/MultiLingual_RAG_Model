// eslint-config-next 16 exports a flat config array directly. FlatCompat is no
// longer needed and in fact throws a circular-JSON error when used with it.
import next from "eslint-config-next";

const config = [
  ...next,
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
