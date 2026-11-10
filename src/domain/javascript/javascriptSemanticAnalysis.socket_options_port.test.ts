import { expect, it } from "vitest";

import { analyzeJavaScriptSemantics } from "./javascriptSemanticAnalysis.js";

/**
 * `net.connect` and `tls.connect` judge the same port value whether it arrives positionally
 * or inside an options object, and both spellings are read by one shared value rule.
 *
 * Measured on Node 24.22.1:
 *
 * | call | outcome |
 * | --- | --- |
 * | `net.connect({port: -1})`, `{port: 70000}`, `{port: "x"}`, `{port: null}` | THREW `ERR_SOCKET_BAD_PORT` |
 * | `net.connect({port: 8080})`, `{port: "8080"}`, `{port: 0}` | socket created |
 * | `tls.connect({path: "/tmp/x.sock", port: -1})` | socket created — the path neutralises the port |
 */
const acquisitionsFor = (body: string) =>
  analyzeJavaScriptSemantics(
    `import net from "node:net";\nimport tls from "node:tls";\nfunction f() { ${body}; }`,
  ).resourceOperations.filter(({ kind }) => kind === "acquire");

it.each([
  [`net.connect({ port: -1 })`],
  [`net.connect({ port: 70000 })`],
  [`net.connect({ port: "x" })`],
  [`net.connect({ port: null })`],
  [`tls.connect({ port: -1 })`],
] as const)("emits no acquisition for a rejected options port: %s", (call) => {
  expect(acquisitionsFor(call)).toEqual([]);
});

it.each([
  [`net.connect({ port: 8080 })`],
  // A numeric string is coerced with `Number`, so it is a usable port.
  [`net.connect({ port: "8080" })`],
  [`net.connect({ port: 0 })`],
  [`tls.connect({ port: 443 })`],
  // A non-empty path becomes the Unix socket, so the port is never read.
  [`tls.connect({ path: "/tmp/x.sock", port: -1 })`],
] as const)(
  "keeps the acquisition for an accepted options port: %s",
  (call) => {
    expect(acquisitionsFor(call)).toHaveLength(1);
  },
);

it.each([
  // A string in the positional slot is a Unix path, so only an empty one fails.
  [`net.connect("-1")`],
  [`net.connect("/tmp/x.sock")`],
  [`tls.connect("/tmp/x.sock")`],
  [`tls.connect("/tmp/x.sock", {})`],
  // `tls` reads an options object in slot 1 after an empty path.
  [`tls.connect("", { port: 80 })`],
  [`tls.connect("", { path: "/tmp/x" })`],
] as const)("keeps the acquisition for a path spelling: %s", (call) => {
  expect(acquisitionsFor(call)).toHaveLength(1);
});

it.each([
  // An empty path leaves nothing to connect to, and the options object does not supply one.
  [`net.connect("")`],
  [`tls.connect("", {})`],
] as const)(
  "emits no acquisition for an empty path with no target: %s",
  (call) => {
    expect(acquisitionsFor(call)).toEqual([]);
  },
);

it.each([
  // `tls` merges slot 1 over slot 0, so the effective port can come from either.
  [`tls.connect({ port: -1 }, { port: 443 })`],
  [`tls.connect({ port: -1 }, { path: "/tmp/x" })`],
  [`net.connect({ path: "/tmp/sock", port: -1 })`],
  [`net.connect({ get side() { this.port = 80; }, port: -1 })`],
] as const)(
  "keeps the acquisition for a merged or accessor-written port: %s",
  (call) => {
    expect(acquisitionsFor(call)).toHaveLength(1);
  },
);

it.each([
  [`tls.connect({ port: 443 }, { port: -1 })`],
  [`tls.connect({ port: -1 }, {})`],
  [`net.connect({ path: "", port: -1 })`],
  [`net.connect({ path: null, port: -1 })`],
  [`tls.connect("", { path: true, port: 80 })`],
  // A setter or a plain method never runs during Node's copy, so the literal still decides.
  [`net.connect({ set port(_) {}, port: -1 })`],
  [`net.connect({ set side(_) {}, port: -1 })`],
  [`net.connect({ method() {}, port: -1 })`],
] as const)("emits no acquisition for a decided bad port: %s", (call) => {
  expect(acquisitionsFor(call)).toEqual([]);
});

it.each([
  // Node reads `path` only when it is truthy, so a falsy non-string falls back to the port.
  [`net.connect({ path: false, port: 80 })`],
  [`net.connect({ path: 0, port: 80 })`],
  [`net.connect({ path: 0n, port: 80 })`],
] as const)("keeps the acquisition for a falsy path: %s", (call) => {
  expect(acquisitionsFor(call)).toHaveLength(1);
});

it("emits no acquisition for a truthy non-string path", () => {
  expect(acquisitionsFor(`net.connect({ path: true, port: 80 })`)).toEqual([]);
});

it.each([
  // An omitted `path` leaves the target one in place, so the Unix socket still wins.
  [`tls.connect({ path: "/tmp/x" }, { port: -1 })`],
  // A getter on the merge target runs again and can rewrite what the source writes.
  [`tls.connect({ get side() { this.port = 80; }, port: 443 }, { port: -1 })`],
  // A class in slot 2 carries enumerable static fields that the merge copies.
  [`tls.connect({ port: -1 }, null, class { static port = 443 })`],
] as const)("keeps the acquisition for a merged override: %s", (call) => {
  expect(acquisitionsFor(call)).toHaveLength(1);
});

it.each([
  // An explicit nullish or empty override clears the target's path.
  [`tls.connect({ path: "/tmp/x" }, { path: null, port: -1 })`],
  [`tls.connect({ path: "/tmp/x" }, { path: "", port: -1 })`],
  [`tls.connect({ path: "/tmp/x" }, { path: undefined, port: -1 })`],
  // A class in slot 1 is not an options object, so the target still decides.
  [`tls.connect({ port: -1 }, class { static port = 443 })`],
] as const)(
  "emits no acquisition for a cleared path or a non-options class: %s",
  (call) => {
    expect(acquisitionsFor(call)).toEqual([]);
  },
);

it("keeps the acquisition when a shadowed undefined is not the global value", () => {
  // A binding named `undefined` is not the global, so assigning it cannot clear
  // the target's Unix socket path and the port still has to decide.
  expect(
    acquisitionsFor(
      `(function (undefined) { return tls.connect({ path: "/tmp/x" }, { path: undefined, port: -1 }); })("x")`,
    ),
  ).toHaveLength(1);
});

it.each([
  // `net` ignores a `socket` member and still validates the port.
  [`net.connect({ socket: new net.Socket(), port: -1 })`],
  // A falsy `socket` is not a stream, so TLS falls through to the port.
  [`tls.connect({ socket: false, port: -1 })`],
  [`tls.connect({ socket: null, port: -1 })`],
] as const)(
  "emits no acquisition when the socket is not a wrapped stream: %s",
  (call) => {
    expect(acquisitionsFor(call)).toEqual([]);
  },
);

it("keeps the acquisition for a wrapped stream", () => {
  // TLS wraps the supplied stream instead of connecting, so the port is never read.
  expect(
    acquisitionsFor(`tls.connect({ socket: new net.Socket(), port: -1 })`),
  ).toHaveLength(1);
});

it.each([
  // A `typeof` result is a nonempty string, and Node reads a positional string as a path.
  [`net.connect(typeof missing)`],
  [`tls.connect(typeof missing)`],
  [`net.connect(typeof 1)`],
  // A setter inherited through `__proto__` runs during the merge.
  [
    `tls.connect({ __proto__: { set port(_) { this.path = "/tmp/x"; } } }, { path: null, port: -1 })`,
  ],
] as const)(
  "keeps the acquisition for a positional string or inherited setter: %s",
  (call) => {
    expect(acquisitionsFor(call)).toHaveLength(1);
  },
);

it.each([
  // The same value inside `port` is judged as a port, not as a path.
  [`net.connect({ port: typeof missing })`],
  [`tls.connect({ __proto__: {} }, { path: null, port: -1 })`],
] as const)("emits no acquisition when the port decides: %s", (call) => {
  expect(acquisitionsFor(call)).toEqual([]);
});
