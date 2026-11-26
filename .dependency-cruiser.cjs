/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: "no-circular",
      severity: "error",
      comment: "Prevent circular dependencies across all modules.",
      from: {},
      to: { circular: true }
    },
    {
      name: "utils-layer-boundary",
      severity: "error",
      comment: "src/utils must stay pure and cannot import higher layers.",
      from: { path: "^src/utils/" },
      to: { path: "^src/(models|adapters|usecases|services)/" }
    },
    {
      name: "models-layer-boundary",
      severity: "error",
      comment: "src/models cannot import adapters, usecases, or services.",
      from: { path: "^src/models/" },
      to: { path: "^src/(adapters|usecases|services)/" }
    },
    {
      name: "adapters-layer-boundary",
      severity: "error",
      comment: "src/adapters cannot import usecases or services.",
      from: { path: "^src/adapters/" },
      to: { path: "^src/(usecases|services)/" }
    },
    {
      name: "usecases-layer-boundary",
      severity: "error",
      comment: "src/usecases cannot import runnable services.",
      from: { path: "^src/usecases/" },
      to: { path: "^src/services/" }
    },
    {
      name: "services-independent",
      severity: "error",
      comment: "Services must not import sibling services; place shared logic in src/usecases.",
      from: { path: "^src/services/([^/]+)" },
      to: { path: "^src/services/", pathNot: "^src/services/$1" }
    }
  ],
  options: {
    tsPreCompilationDeps: true,
    tsConfig: { fileName: "tsconfig.json" },
    doNotFollow: { path: "node_modules" }
  }
};
