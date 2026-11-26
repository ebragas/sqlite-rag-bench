import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { ConfigError } from "../models/errors.js";
import { runContractSmokeSuite, type SpawnedServerOptions } from "../usecases/contract-smoke.js";

function parsePositiveInt(raw: string | undefined, flagName: string): number | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new ConfigError(`Flag ${flagName} must be a positive integer, got "${raw}".`);
  }
  return value;
}

export async function runSmokeCli(argv: readonly string[]): Promise<void> {
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    options: {
      url: { type: "string", default: "http://127.0.0.1:8080" },
      spawn: { type: "string" },
      question: { type: "string" },
      "startup-timeout-ms": { type: "string" },
      "shutdown-timeout-ms": { type: "string" },
      "request-timeout-ms": { type: "string" }
    }
  });

  const startupTimeoutMs = parsePositiveInt(values["startup-timeout-ms"], "--startup-timeout-ms");
  const shutdownTimeoutMs = parsePositiveInt(
    values["shutdown-timeout-ms"],
    "--shutdown-timeout-ms"
  );
  const requestTimeoutMs = parsePositiveInt(values["request-timeout-ms"], "--request-timeout-ms");

  let spawnOptions: SpawnedServerOptions | undefined;
  if (values.spawn !== undefined) {
    spawnOptions = {
      command: values.spawn,
      ...(startupTimeoutMs !== undefined ? { startupTimeoutMs } : {}),
      ...(shutdownTimeoutMs !== undefined ? { shutdownTimeoutMs } : {})
    };
  }

  const report = await runContractSmokeSuite({
    baseUrl: values.url,
    ...(values.question !== undefined ? { sampleQuestion: values.question } : {}),
    ...(requestTimeoutMs !== undefined ? { requestTimeoutMs } : {}),
    ...(spawnOptions !== undefined ? { spawn: spawnOptions } : {})
  });

  const lines = [
    `Contract smoke PASS (${report.baseUrl})`,
    ...report.checks.map((check) => `  [PASS] ${check.name}: ${check.detail}`)
  ];
  process.stdout.write(`${lines.join("\n")}\n`);
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  runSmokeCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`smoke failed: ${message}\n`);
    process.exitCode = 1;
  });
}
