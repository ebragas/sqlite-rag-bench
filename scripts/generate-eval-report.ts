import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  aggregateReliabilityTrials,
  type QuestionStratum,
  type ReliabilityItemRecord,
  type ReliabilityReport
} from "./compute-reliability.js";

interface RawItem {
  readonly questionId: string;
  readonly stratum: QuestionStratum;
  readonly question: string;
  readonly answer: string;
  readonly returnedDocumentIds: readonly string[];
  readonly expectedDocumentIds: readonly string[];
  readonly verdict: {
    readonly answerScore: number;
    readonly hasFabricatedOrContradictingClaim: boolean;
    readonly reasoning: string;
  };
  readonly effectiveAnswerScore: number;
  readonly citationScore: {
    readonly precision: number;
    readonly recall: number;
    readonly f1: number;
    readonly invalidCount: number;
  };
  readonly compositeScore: number;
  readonly abstained: boolean;
  readonly latencyMs: number;
}

interface RawManifest {
  readonly gitSha: string;
  readonly profileName: string;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly items: readonly RawItem[];
}

interface OnyxQuestionMeta {
  readonly question_id: string;
  readonly question_type: string;
  readonly source_types?: readonly string[];
}

interface CategoryStats {
  readonly category: string;
  readonly count: number;
  readonly composite: number;
  readonly answer: number;
  readonly citation: number;
  readonly fabrications: number;
  readonly falseAbstentions: number;
  readonly invalidCitations: number;
}

interface PairedRow {
  readonly questionId: string;
  readonly category: string;
  readonly stratum: QuestionStratum;
  readonly question: string;
  readonly prodComposite: number;
  readonly l0Composite: number;
  readonly delta: number;
  readonly prodCitationF1: number;
  readonly retDocs: number;
  readonly expDocs: number;
  readonly prodLatencyMs: number;
  readonly l0Contradiction: boolean;
  readonly reasoning: string;
}

interface OnyxBenchmarkSummary {
  readonly generatedAt: string;
  readonly benchmarkName: string;
  readonly benchmarkLicense: string;
  readonly benchmarkUrl: string;
  readonly corpusNote: string;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly totalQuestions: number;
  readonly overall: {
    readonly composite: number;
    readonly answer: number;
    readonly citation: number;
    readonly abstentionRecall: number;
    readonly falseAbstentionRate: number;
    readonly invalidCitationsTotal: number;
    readonly fabricationsTotal: number;
    readonly latencyP50Ms: number;
    readonly latencyP95Ms: number;
  };
  readonly byCategory: readonly CategoryStats[];
  readonly byStratum: readonly CategoryStats[];
  readonly paired100: {
    readonly count: number;
    readonly l0Composite: number;
    readonly prodComposite: number;
    readonly deltaComposite: number;
    readonly l0FalseAbstentionRate: number;
    readonly prodFalseAbstentionRate: number;
    readonly byCategory: readonly {
      readonly category: string;
      readonly count: number;
      readonly l0Composite: number;
      readonly prodComposite: number;
      readonly delta: number;
    }[];
    readonly items: readonly PairedRow[];
  };
  readonly hardReliabilityK5: ReliabilityReport | null;
  readonly spendUsd: {
    readonly responder: number;
    readonly judge: number;
    readonly total: number;
  };
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function sec(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx] ?? 0;
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function round4(v: number): number {
  return Number(v.toFixed(4));
}

function summarizeGroup(
  label: string,
  items: readonly RawItem[]
): CategoryStats {
  return {
    category: label,
    count: items.length,
    composite: round4(mean(items.map((i) => i.compositeScore))),
    answer: round4(mean(items.map((i) => i.effectiveAnswerScore))),
    citation: round4(mean(items.map((i) => i.citationScore.f1))),
    fabrications: items.filter((i) => i.verdict.hasFabricatedOrContradictingClaim).length,
    falseAbstentions: items.filter((i) => i.stratum !== "unanswerable" && i.abstained).length,
    invalidCitations: items.reduce((acc, i) => acc + i.citationScore.invalidCount, 0)
  };
}

function loadManifestsInDir(dirPath: string): RawManifest[] {
  if (!existsSync(dirPath)) {
    return [];
  }
  return readdirSync(dirPath)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => JSON.parse(readFileSync(join(dirPath, f), "utf8")) as RawManifest);
}

function buildSummaryFromRuns(repoRoot: string): OnyxBenchmarkSummary {
  const runsRoot = resolve(repoRoot, "eval/runs-onyx");
  const questionsPath = resolve(repoRoot, "onyx-questions.jsonl");
  const typeById = new Map<string, string>();
  if (existsSync(questionsPath)) {
    for (const line of readFileSync(questionsPath, "utf8").trim().split("\n")) {
      if (!line.trim()) continue;
      const q = JSON.parse(line) as OnyxQuestionMeta;
      typeById.set(q.question_id, q.question_type);
    }
  }

  const allProdItems: RawItem[] = [];
  let responderModel = "gemini-3.8-flash";
  let judgeModel = "gemini-4-neon";
  let gitSha = "main";

  for (let b = 1; b <= 5; b += 1) {
    for (const m of loadManifestsInDir(join(runsRoot, `prod-b${String(b)}`))) {
      responderModel = m.responderModel;
      judgeModel = m.judgeModel;
      gitSha = m.gitSha;
      allProdItems.push(...m.items);
    }
  }

  const prodById = new Map(allProdItems.map((i) => [i.questionId, i]));

  const unanswerable = allProdItems.filter((i) => i.stratum === "unanswerable");
  const answerable = allProdItems.filter((i) => i.stratum !== "unanswerable");
  const abstentionRecall =
    unanswerable.length === 0
      ? 1
      : unanswerable.filter((i) => i.abstained && i.returnedDocumentIds.length === 0).length /
        unanswerable.length;
  const falseAbstentionRate =
    answerable.length === 0
      ? 0
      : answerable.filter((i) => i.abstained).length / answerable.length;

  const byCatMap = new Map<string, RawItem[]>();
  for (const item of allProdItems) {
    const cat = typeById.get(item.questionId) ?? "other";
    const arr = byCatMap.get(cat) ?? [];
    arr.push(item);
    byCatMap.set(cat, arr);
  }
  const byCategory = [...byCatMap.entries()]
    .map(([cat, items]) => summarizeGroup(cat, items))
    .sort((a, b) => b.count - a.count || a.category.localeCompare(b.category));

  const strataOrder: readonly QuestionStratum[] = [
    "unanswerable",
    "single_doc",
    "two_doc",
    "multi_doc"
  ];
  const byStratum = strataOrder.map((st) =>
    summarizeGroup(
      st,
      allProdItems.filter((i) => i.stratum === st)
    )
  );

  // Paired 100 L0 vs prod
  const l0Manifests = loadManifestsInDir(join(runsRoot, "l0-100"));
  const l0Items = l0Manifests.flatMap((m) => m.items);
  const pairedRows: PairedRow[] = [];
  for (const l0 of l0Items) {
    const prod = prodById.get(l0.questionId);
    if (prod === undefined) continue;
    pairedRows.push({
      questionId: l0.questionId,
      category: typeById.get(l0.questionId) ?? "other",
      stratum: prod.stratum,
      question: prod.question,
      prodComposite: round4(prod.compositeScore),
      l0Composite: round4(l0.compositeScore),
      delta: round4(prod.compositeScore - l0.compositeScore),
      prodCitationF1: round4(prod.citationScore.f1),
      retDocs: prod.returnedDocumentIds.length,
      expDocs: prod.expectedDocumentIds.length,
      prodLatencyMs: Math.round(prod.latencyMs),
      l0Contradiction: l0.verdict.hasFabricatedOrContradictingClaim,
      reasoning: prod.verdict.reasoning.slice(0, 190)
    });
  }

  const pairedByCatMap = new Map<string, PairedRow[]>();
  for (const row of pairedRows) {
    const arr = pairedByCatMap.get(row.category) ?? [];
    arr.push(row);
    pairedByCatMap.set(row.category, arr);
  }
  const pairedByCat = [...pairedByCatMap.entries()]
    .map(([category, rows]) => {
      const l0C = round4(mean(rows.map((r) => r.l0Composite)));
      const pC = round4(mean(rows.map((r) => r.prodComposite)));
      return { category, count: rows.length, l0Composite: l0C, prodComposite: pC, delta: round4(pC - l0C) };
    })
    .sort((a, b) => b.delta - a.delta);

  const l0Answerable = l0Items.filter((i) => i.stratum !== "unanswerable");
  const prodPairedAnswerable = l0Items
    .map((l0) => prodById.get(l0.questionId))
    .filter((i): i is RawItem => i !== undefined && i.stratum !== "unanswerable");

  // Spend ledger
  let spendResp = 0;
  let spendJudge = 0;
  const ledgerPath = resolve(repoRoot, "eval/spend-ledger-onyx.jsonl");
  if (existsSync(ledgerPath)) {
    for (const line of readFileSync(ledgerPath, "utf8").trim().split("\n")) {
      if (!line.trim()) continue;
      const row = JSON.parse(line) as { readonly bucket: string; readonly usd: number };
      if (row.bucket === "responder") spendResp += row.usd;
      else if (row.bucket === "judge") spendJudge += row.usd;
    }
  }

  // Hard k=5 consistency trials
  let hardReliabilityK5: ReliabilityReport | null = null;
  const hardRoot = join(runsRoot, "hard-k5");
  if (existsSync(hardRoot)) {
    const trialDirs = readdirSync(hardRoot)
      .filter((d) => /^t\d+$/.test(d))
      .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
    const trials: ReliabilityItemRecord[][] = [];
    for (const dir of trialDirs) {
      const items = loadManifestsInDir(join(hardRoot, dir)).flatMap((m) =>
        m.items.map(
          (it): ReliabilityItemRecord => ({
            questionId: it.questionId,
            split: "val",
            stratum: it.stratum,
            question: it.question,
            compositeScore: it.compositeScore,
            effectiveAnswerScore: it.effectiveAnswerScore,
            citationF1: it.citationScore.f1,
            invalidCitations: it.citationScore.invalidCount,
            fabricatedOrContradicted: it.verdict.hasFabricatedOrContradictingClaim,
            abstained: it.abstained,
            latencyMs: it.latencyMs
          })
        )
      );
      if (items.length > 0) trials.push(items);
    }
    if (trials.length > 0) {
      hardReliabilityK5 = aggregateReliabilityTrials(trials, {
        gitSha,
        profileName: "prod",
        responderModel,
        judgeModel,
        spendUsd: { responder: 0, judge: 0, total: 0 }
      });
    }
  }

  return {
    generatedAt: new Date().toISOString(),
    benchmarkName: "Onyx EnterpriseRAG-Bench (500 Public Questions)",
    benchmarkLicense: "MIT",
    benchmarkUrl: "https://github.com/onyx-dot-app/EnterpriseRAG-Bench",
    corpusNote:
      "Evaluated on all 500 official MIT-licensed Onyx EnterpriseRAG-Bench questions over a 21,612-document multi-source slice (all 9 enterprise sources, 38,774 chunks, 100% expected-document coverage).",
    responderModel,
    judgeModel,
    totalQuestions: allProdItems.length,
    overall: {
      composite: round4(mean(allProdItems.map((i) => i.compositeScore))),
      answer: round4(mean(allProdItems.map((i) => i.effectiveAnswerScore))),
      citation: round4(mean(allProdItems.map((i) => i.citationScore.f1))),
      abstentionRecall: round4(abstentionRecall),
      falseAbstentionRate: round4(falseAbstentionRate),
      invalidCitationsTotal: allProdItems.reduce((acc, i) => acc + i.citationScore.invalidCount, 0),
      fabricationsTotal: allProdItems.filter((i) => i.verdict.hasFabricatedOrContradictingClaim).length,
      latencyP50Ms: Math.round(percentile(allProdItems.map((i) => i.latencyMs), 50)),
      latencyP95Ms: Math.round(percentile(allProdItems.map((i) => i.latencyMs), 95))
    },
    byCategory,
    byStratum,
    paired100: {
      count: pairedRows.length,
      l0Composite: round4(mean(pairedRows.map((r) => r.l0Composite))),
      prodComposite: round4(mean(pairedRows.map((r) => r.prodComposite))),
      deltaComposite: round4(
        mean(pairedRows.map((r) => r.prodComposite)) - mean(pairedRows.map((r) => r.l0Composite))
      ),
      l0FalseAbstentionRate:
        l0Answerable.length === 0
          ? 0
          : round4(l0Answerable.filter((i) => i.abstained).length / l0Answerable.length),
      prodFalseAbstentionRate:
        prodPairedAnswerable.length === 0
          ? 0
          : round4(
              prodPairedAnswerable.filter((i) => i.abstained).length / prodPairedAnswerable.length
            ),
      byCategory: pairedByCat,
      items: pairedRows
    },
    hardReliabilityK5,
    spendUsd: {
      responder: Number(spendResp.toFixed(2)),
      judge: Number(spendJudge.toFixed(2)),
      total: Number((spendResp + spendJudge).toFixed(2))
    }
  };
}

function renderCategoryBarSvg(categories: readonly CategoryStats[]): string {
  const width = 820;
  const rowH = 30;
  const padLeft = 185;
  const padRight = 72;
  const padTop = 16;
  const height = padTop + categories.length * rowH + 18;
  const barArea = width - padLeft - padRight;

  const rows = categories
    .map((c, idx) => {
      const y = padTop + idx * rowH;
      const w = Math.max(2, c.composite * barArea);
      const color = c.composite >= 0.9 ? "#059669" : c.composite >= 0.8 ? "#2563eb" : "#d97706";
      return `<text x="${padLeft - 10}" y="${y + 17}" text-anchor="end" font-size="12" font-weight="600" fill="#1e293b">${escapeHtml(c.category)} (${String(c.count)})</text>
      <rect x="${padLeft}" y="${y + 4}" width="${barArea}" height="18" rx="4" fill="#f1f5f9" />
      <rect x="${padLeft}" y="${y + 4}" width="${w.toFixed(1)}" height="18" rx="4" fill="${color}" />
      <text x="${(padLeft + w + 8).toFixed(1)}" y="${y + 17}" font-size="11.5" font-weight="700" fill="#0f172a">${pct(c.composite)}</text>`;
    })
    .join("\n");

  return `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="Composite score across all 10 Onyx EnterpriseRAG-Bench categories">${rows}</svg>`;
}

function renderPairedSvg(
  cats: readonly { readonly category: string; readonly l0Composite: number; readonly prodComposite: number }[]
): string {
  const width = 820;
  const height = 270;
  const padLeft = 48;
  const padRight = 18;
  const padTop = 26;
  const padBottom = 68;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const groupW = plotW / Math.max(1, cats.length);
  const barW = Math.min(24, groupW * 0.34);

  const yAt = (v: number): number => padTop + (1 - v) * plotH;
  const grid = [0, 0.25, 0.5, 0.75, 1.0]
    .map((v) => {
      const y = yAt(v).toFixed(1);
      return `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="#e2e8f0" stroke-width="1" />
      <text x="${padLeft - 8}" y="${(yAt(v) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#64748b">${Math.round(v * 100)}%</text>`;
    })
    .join("\n");

  const bars = cats
    .map((c, idx) => {
      const cx = padLeft + idx * groupW + groupW / 2;
      const x1 = cx - barW - 2;
      const x2 = cx + 2;
      const y1 = yAt(c.l0Composite);
      const y2 = yAt(c.prodComposite);
      const h1 = Math.max(2, padTop + plotH - y1);
      const h2 = Math.max(2, padTop + plotH - y2);
      const shortLabel = c.category.replace("intra_document_reasoning", "intra_doc").replace("conflicting_info", "conflict");
      return `<rect x="${x1.toFixed(1)}" y="${y1.toFixed(1)}" width="${barW.toFixed(1)}" height="${h1.toFixed(1)}" rx="3" fill="#f59e0b" />
      <rect x="${x2.toFixed(1)}" y="${y2.toFixed(1)}" width="${barW.toFixed(1)}" height="${h2.toFixed(1)}" rx="3" fill="#059669" />
      <text x="${cx.toFixed(1)}" y="${height - 46}" text-anchor="end" font-size="10.5" fill="#334155" transform="rotate(-28 ${cx.toFixed(1)} ${height - 46})">${escapeHtml(shortLabel)}</text>`;
    })
    .join("\n");

  return `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="Paired L0 versus prod Composite Score across 10 Onyx categories">${grid}${bars}</svg>`;
}

function renderReliabilitySvg(report: ReliabilityReport): string {
  const kMax = report.trialsPerQuestion;
  const width = 820;
  const height = 270;
  const padLeft = 52;
  const padRight = 28;
  const padTop = 24;
  const padBottom = 42;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const xAt = (idx: number): number => padLeft + (kMax <= 1 ? 0 : (idx / (kMax - 1)) * plotW);
  const yAt = (v: number): number => padTop + (1 - Math.max(0, Math.min(1, v))) * plotH;

  const grid = [0, 0.25, 0.5, 0.75, 1.0]
    .map((v) => {
      const y = yAt(v).toFixed(1);
      return `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="#e2e8f0" stroke-width="1" />
      <text x="${padLeft - 8}" y="${(yAt(v) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#64748b">${Math.round(v * 100)}%</text>`;
    })
    .join("\n");

  const xLabels = Array.from({ length: kMax }, (_, idx) =>
    `<text x="${xAt(idx).toFixed(1)}" y="${height - 12}" text-anchor="middle" font-size="11.5" fill="#334155">k=${String(idx + 1)}</text>`
  ).join("\n");

  const lineOf = (values: readonly number[], color: string, dashed = false): string => {
    const pts = values.map((v, idx) => `${xAt(idx).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
    const dashAttr = dashed ? ' stroke-dasharray="5 4"' : "";
    const dots = values
      .map((v, idx) => {
        const cx = xAt(idx).toFixed(1);
        const cy = yAt(v).toFixed(1);
        return `<circle cx="${cx}" cy="${cy}" r="${dashed ? 3.5 : 4.5}" fill="${color}" stroke="#ffffff" stroke-width="1.5" />
        <text x="${cx}" y="${Math.max(14, yAt(v) - 8).toFixed(1)}" text-anchor="middle" font-size="10.5" font-weight="700" fill="${color}">${Math.round(v * 100)}%</text>`;
      })
      .join("\n");
    return `<polyline fill="none" stroke="${color}" stroke-width="${dashed ? 2 : 2.75}"${dashAttr} points="${pts}" />\n${dots}`;
  };

  return `<svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="pass@k versus pass^k curves on hard Onyx slice">
    ${grid}${xLabels}
    ${lineOf(report.overall.strict.passAtK, "#059669", false)}
    ${lineOf(report.overall.strict.passPowK, "#7c3aed", false)}
    ${lineOf(report.overall.safety.passPowK, "#0284c7", true)}
    ${lineOf(report.overall.perfect.passPowK, "#d97706", true)}
  </svg>`;
}

function renderHtml(summary: OnyxBenchmarkSummary): string {
  const catRows = summary.byCategory
    .map(
      (c) => `<tr>
      <td><strong>${escapeHtml(c.category)}</strong></td>
      <td class="num">${String(c.count)}</td>
      <td class="num"><strong>${pct(c.composite)}</strong></td>
      <td class="num">${pct(c.answer)}</td>
      <td class="num">${pct(c.citation)}</td>
      <td class="num">${String(c.fabrications)}</td>
      <td class="num">${String(c.falseAbstentions)}</td>
      <td class="num">${String(c.invalidCitations)}</td>
    </tr>`
    )
    .join("\n");

  const pairedItemRows = summary.paired100.items
    .map((item) => {
      const deltaClass = item.delta > 0.05 ? "delta-pos" : item.delta < -0.05 ? "delta-neg" : "delta-flat";
      const sign = item.delta > 0 ? "+" : "";
      const pill =
        item.prodComposite >= 0.999 ? "pill-green" : item.prodComposite >= 0.8 ? "pill-blue" : "pill-amber";
      return `<tr data-cat="${escapeHtml(item.category)}" data-delta="${item.delta.toFixed(3)}" data-score="${item.prodComposite.toFixed(3)}">
        <td class="mono">${escapeHtml(item.questionId)}</td>
        <td><span class="tag">${escapeHtml(item.category)}</span></td>
        <td>
          <div class="q-text">${escapeHtml(item.question)}</div>
          <div class="q-meta">${escapeHtml(item.reasoning)}</div>
        </td>
        <td class="num"><span class="pill ${pill}">${pct(item.prodComposite)}</span></td>
        <td class="num">${pct(item.l0Composite)}</td>
        <td class="num ${deltaClass}">${sign}${(item.delta * 100).toFixed(1)}%</td>
        <td class="num">${pct(item.prodCitationF1)} (${String(item.retDocs)}/${String(item.expDocs)})</td>
        <td class="num">${sec(item.prodLatencyMs)}</td>
      </tr>`;
    })
    .join("\n");

  const rel = summary.hardReliabilityK5;
  const relSection =
    rel !== null
      ? `<section>
      <h2>3. Hard-Slice Consistency Benchmark (<code>pass@1..${String(rel.trialsPerQuestion)}</code> vs. <code>pass^1..${String(rel.trialsPerQuestion)}</code>)</h2>
      <p class="sec-lead">
        Evaluated <strong>${String(rel.trialsPerQuestion)} fresh trials</strong> across <strong>${String(rel.overall.questionCount)} hard Onyx questions</strong>
        (5 each from <code>conflicting_info</code>, <code>completeness</code>, <code>project_related</code>, and <code>info_not_found</code>, ${String(rel.totalEvaluations)} total runs at <code>temperature: 0</code>).
      </p>
      <div class="grid-3" style="margin-bottom:16px">
        <div class="card card-accent">
          <div class="card-kicker">Mean Composite Across ${String(rel.trialsPerQuestion)} Trials</div>
          <div class="metric-big">${pct(rel.trialMeanComposite)} &plusmn; ${(rel.trialStdComposite * 100).toFixed(2)}%</div>
          <p class="muted">Trial range: ${pct(Math.min(...rel.trialCompositeScores))} to ${pct(Math.max(...rel.trialCompositeScores))}</p>
        </div>
        <div class="card">
          <div class="card-kicker">Strict Pass (&ge; 80%, 0 Fab, 0 Inv)</div>
          <div class="metric-big">pass@${String(rel.trialsPerQuestion)} ${pct(rel.overall.strict.passAtK[rel.trialsPerQuestion - 1] ?? 0)} / pass^${String(rel.trialsPerQuestion)} ${pct(rel.overall.strict.passPowK[rel.trialsPerQuestion - 1] ?? 0)}</div>
          <p class="muted">pass@1: ${pct(rel.overall.strict.passAtK[0] ?? 0)} &middot; pass^3: ${pct(rel.overall.strict.passPowK[Math.min(2, rel.trialsPerQuestion - 1)] ?? 0)}</p>
        </div>
        <div class="card">
          <div class="card-kicker">Safety Across ${String(rel.totalEvaluations)} Hard Runs</div>
          <div class="metric-big">pass^${String(rel.trialsPerQuestion)} ${pct(rel.overall.safety.passPowK[rel.trialsPerQuestion - 1] ?? 0)}</div>
          <p class="muted">Fabrications: ${String(rel.totalFabrications)} &middot; Invalid Citations: ${String(rel.totalInvalidCitations)} &middot; False Abstentions: ${String(rel.totalFalseAbstentions)}</p>
        </div>
      </div>
      <div class="chart-box">
        <div class="legend">
          <span><span class="swatch" style="background:#059669"></span>Strict pass@k (capability)</span>
          <span><span class="swatch" style="background:#7c3aed"></span>Strict pass^k (consistency)</span>
          <span><span class="swatch" style="background:#0284c7"></span>Safety pass^k (dashed)</span>
          <span><span class="swatch" style="background:#d97706"></span>Perfect 100% pass^k (dashed)</span>
        </div>
        ${renderReliabilitySvg(rel)}
      </div>
    </section>`
      : "";

  const categoryButtons = [...new Set(summary.paired100.items.map((i) => i.category))]
    .sort()
    .map((cat) => `<button class="filter-btn" data-filter="${escapeHtml(cat)}">${escapeHtml(cat)}</button>`)
    .join("");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>sqlite-rag-bench: EnterpriseRAG-Bench Evaluation Report</title>
  <style>
    :root { --bg:#f8fafc; --surface:#ffffff; --ink:#0f172a; --muted:#475569; --border:#e2e8f0; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif; background:var(--bg); color:var(--ink); line-height:1.55; }
    .wrap { max-width:1120px; margin:0 auto; padding:36px 24px 72px; }
    header { background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%); color:#f8fafc; border-radius:14px; padding:32px 36px; margin-bottom:28px; }
    .eyebrow { text-transform:uppercase; letter-spacing:0.08em; font-size:12px; font-weight:700; color:#38bdf8; margin-bottom:8px; }
    h1 { margin:0 0 10px; font-size:28px; font-weight:800; }
    header p { margin:0; color:#cbd5e1; max-width:860px; font-size:15px; }
    header a { color:#7dd3fc; }
    .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-bottom:28px; }
    .kpi { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }
    .kpi-primary { border-color:#a7f3d0; background:#ecfdf5; }
    .kpi-label { font-size:12px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }
    .kpi-val { font-size:26px; font-weight:800; margin:4px 0 2px; }
    .kpi-primary .kpi-val { color:#047857; }
    .kpi-sub { font-size:12px; color:var(--muted); }
    section { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:26px 28px; margin-bottom:24px; }
    h2 { margin:0 0 6px; font-size:20px; font-weight:750; }
    .sec-lead { margin:0 0 18px; color:var(--muted); font-size:14px; }
    .chart-box { border:1px solid var(--border); border-radius:10px; padding:14px 12px 8px; background:#fcfdff; margin-bottom:18px; }
    .chart-svg { width:100%; height:auto; display:block; }
    .legend { display:flex; flex-wrap:wrap; gap:18px; font-size:12.5px; color:var(--muted); margin-bottom:10px; padding:0 8px; }
    .swatch { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
    th { font-size:11.5px; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); background:#f8fafc; white-space:nowrap; }
    .num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:12px; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }
    .pill-green { background:#dcfce7; color:#166534; }
    .pill-blue { background:#dbeafe; color:#1e40af; }
    .pill-amber { background:#fef3c7; color:#92400e; }
    .tag { display:inline-block; background:#f1f5f9; color:#334155; padding:2px 7px; border-radius:6px; font-size:11.5px; font-weight:600; white-space:nowrap; }
    .delta-pos { color:#059669; font-weight:700; }
    .delta-neg { color:#dc2626; font-weight:700; }
    .delta-flat { color:#64748b; }
    .q-text { font-weight:600; margin-bottom:3px; }
    .q-meta { font-size:12px; color:var(--muted); }
    .filter-bar { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
    .filter-btn { border:1px solid var(--border); background:var(--surface); color:var(--ink); padding:6px 11px; border-radius:8px; font-size:12px; font-weight:600; cursor:pointer; }
    .filter-btn.active { background:#0f172a; color:#ffffff; border-color:#0f172a; }
    .grid-3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
    .card { border:1px solid var(--border); border-radius:10px; padding:18px 20px; background:#fcfdff; }
    .card-accent { border-color:#a7f3d0; background:#f0fdf4; }
    .card-kicker { font-size:11.5px; text-transform:uppercase; letter-spacing:0.05em; font-weight:700; color:var(--muted); margin-bottom:6px; }
    .metric-big { font-size:21px; font-weight:800; margin-bottom:8px; }
    .muted { color:var(--muted); font-size:13.5px; }
    footer { text-align:center; color:var(--muted); font-size:12.5px; margin-top:28px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">sqlite-rag-bench &middot; MIT Open Benchmark Report</div>
      <h1>Full 500-Question Onyx EnterpriseRAG-Bench Evaluation</h1>
      <p>
        Single-file SQLite (FTS5 + <code>Float32Array</code> vectors + deterministic Link Graph) benchmarked across all
        <strong>${String(summary.totalQuestions)} public MIT-licensed questions</strong> from
        <a href="${summary.benchmarkUrl}">onyx-dot-app/EnterpriseRAG-Bench</a>, judged by <code>${escapeHtml(summary.judgeModel)}</code>
        with strict Onyx contradiction zeroing and live Langfuse v4 OpenTelemetry tracing.
      </p>
    </header>

    <div class="kpi-grid">
      <div class="kpi kpi-primary">
        <div class="kpi-label">500-Question Composite</div>
        <div class="kpi-val">${pct(summary.overall.composite)}</div>
        <div class="kpi-sub">Answer ${pct(summary.overall.answer)} &middot; Citation ${pct(summary.overall.citation)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Gain vs. L0 Baseline (100q)</div>
        <div class="kpi-val">+${(summary.paired100.deltaComposite * 100).toFixed(1)}%</div>
        <div class="kpi-sub">${pct(summary.paired100.prodComposite)} prod vs. ${pct(summary.paired100.l0Composite)} L0</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Abstention Safety</div>
        <div class="kpi-val">${pct(summary.overall.abstentionRecall)}</div>
        <div class="kpi-sub">${pct(summary.overall.falseAbstentionRate)} false abstention rate</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Invalid Citations</div>
        <div class="kpi-val">${String(summary.overall.invalidCitationsTotal)}</div>
        <div class="kpi-sub">Across all ${String(summary.totalQuestions)} questions</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Latency (p50 / p95)</div>
        <div class="kpi-val">${sec(summary.overall.latencyP95Ms)}</div>
        <div class="kpi-sub">p95 (p50: ${sec(summary.overall.latencyP50Ms)})</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Total Benchmark Spend</div>
        <div class="kpi-val">$${summary.spendUsd.total.toFixed(2)}</div>
        <div class="kpi-sub">Resp $${summary.spendUsd.responder.toFixed(2)} &middot; Judge $${summary.spendUsd.judge.toFixed(2)}</div>
      </div>
    </div>

    <section>
      <h2>1. Results Across All 10 Official Onyx Categories (${String(summary.totalQuestions)} Questions)</h2>
      <p class="sec-lead">${escapeHtml(summary.corpusNote)}</p>
      <div class="chart-box">${renderCategoryBarSvg(summary.byCategory)}</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Onyx Category</th>
              <th class="num">Questions</th>
              <th class="num">Composite</th>
              <th class="num">Answer</th>
              <th class="num">Citation F1</th>
              <th class="num">Fabrications</th>
              <th class="num">False Abs.</th>
              <th class="num">Invalid Cit.</th>
            </tr>
          </thead>
          <tbody>${catRows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>2. Paired Baseline Ablation: Single-Pass FTS5 (<code>l0-fts</code>) vs. <code>prod</code> (${String(summary.paired100.count)} Stratified Questions)</h2>
      <p class="sec-lead">
        10 questions per official Onyx category evaluated on both <code>l0-fts</code> (single-pass keyword search) and <code>prod</code>
        (hybrid FTS5 + <code>gemini-embedding-2</code> + Link Graph + Turn 0 pre-search + <code>temperature: 0</code>).
      </p>
      <div class="chart-box">
        <div class="legend">
          <span><span class="swatch" style="background:#f59e0b"></span>L0 Single-Pass FTS5 (${pct(summary.paired100.l0Composite)}, ${pct(summary.paired100.l0FalseAbstentionRate)} false abstention)</span>
          <span><span class="swatch" style="background:#059669"></span>Promoted prod (${pct(summary.paired100.prodComposite)}, ${pct(summary.paired100.prodFalseAbstentionRate)} false abstention)</span>
        </div>
        ${renderPairedSvg(summary.paired100.byCategory)}
      </div>

      <div class="filter-bar" id="paired-filters">
        <button class="filter-btn active" data-filter="all">All (${String(summary.paired100.count)})</button>
        <button class="filter-btn" data-filter="wins">Biggest Wins (&ge; +25%)</button>
        <button class="filter-btn" data-filter="misses">Misses (&lt; 80%)</button>
        ${categoryButtons}
      </div>
      <div class="table-wrap">
        <table id="paired-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Category</th>
              <th>Public Onyx Question &amp; Judge Reasoning</th>
              <th class="num">prod</th>
              <th class="num">L0</th>
              <th class="num">Delta</th>
              <th class="num">Citation F1</th>
              <th class="num">Latency</th>
            </tr>
          </thead>
          <tbody>${pairedItemRows}</tbody>
        </table>
      </div>
    </section>

    ${relSection}

    <footer>
      sqlite-rag-bench &middot; MIT License &middot; Generated by <code>scripts/generate-eval-report.ts</code> from public MIT-licensed Onyx EnterpriseRAG-Bench runs.
    </footer>
  </div>

  <script>
    (function () {
      var bar = document.getElementById("paired-filters");
      var rows = Array.prototype.slice.call(document.querySelectorAll("#paired-table tbody tr"));
      if (!bar) return;
      bar.addEventListener("click", function (ev) {
        var btn = ev.target.closest("button[data-filter]");
        if (!btn) return;
        var mode = btn.getAttribute("data-filter");
        bar.querySelectorAll(".filter-btn").forEach(function (b) { b.classList.toggle("active", b === btn); });
        rows.forEach(function (row) {
          var cat = row.getAttribute("data-cat");
          var delta = parseFloat(row.getAttribute("data-delta") || "0");
          var score = parseFloat(row.getAttribute("data-score") || "0");
          var show = mode === "all" || mode === cat || (mode === "wins" && delta >= 0.25) || (mode === "misses" && score < 0.8);
          row.style.display = show ? "" : "none";
        });
      });
    })();
  </script>
</body>
</html>
`;
}

function main(): void {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = resolve(scriptDir, "..");
  const summaryPath = resolve(repoRoot, "eval/onyx/benchmark-summary.json");
  const htmlPath = resolve(repoRoot, "docs/index.html");

  let summary: OnyxBenchmarkSummary;
  if (existsSync(resolve(repoRoot, "eval/runs-onyx/prod-b1"))) {
    summary = buildSummaryFromRuns(repoRoot);
    mkdirSync(dirname(summaryPath), { recursive: true });
    writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    console.log(`[generate-eval-report] Wrote ${summaryPath}`);
  } else if (existsSync(summaryPath)) {
    summary = JSON.parse(readFileSync(summaryPath, "utf8")) as OnyxBenchmarkSummary;
  } else {
    throw new Error("No Onyx benchmark runs or eval/onyx/benchmark-summary.json found.");
  }

  const html = renderHtml(summary);
  mkdirSync(dirname(htmlPath), { recursive: true });
  writeFileSync(htmlPath, html, "utf8");
  console.log(`[generate-eval-report] Wrote ${htmlPath} (${Buffer.byteLength(html, "utf8")} bytes)`);
}

main();
