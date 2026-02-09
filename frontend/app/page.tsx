"use client";

import clsx from "clsx";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { ComponentType, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type GraphNode = {
  id: string;
  node_type: "transaction" | "entity";
  label: string;
  risk_score?: number;
  risk_color?: string;
  isFraud?: number;
  amount?: number;
  timestamp?: number;
  x?: number;
  y?: number;
  entity_type?: string;
  degree?: number;
};

type GraphLink = {
  source: string | GraphNode;
  target: string | GraphNode;
  relation: string;
};

type GraphResponse = {
  nodes: GraphNode[];
  links: GraphLink[];
  meta?: {
    node_count?: number;
    link_count?: number;
    hops?: number;
  };
};

type TransactionDetail = {
  TransactionID: number;
  isFraud: number;
  split: string;
  TransactionDT: number;
  TransactionAmt: number;
  pred_score: number;
  uid_clean: string;
  ProductCD: string;
  DeviceInfo: string;
  id_30: string;
  id_31: string;
};

type ExplainResponse = {
  backend: string;
  final_probability: number;
  base_probability?: number;
  waterfall: Array<{
    feature: string;
    delta_logit: number;
    prob_before: number;
    prob_after: number;
  }>;
};

type TimelineResponse = {
  points: Array<{
    bucket: number;
    transaction_count: number;
    fraud_count: number;
    fraud_rate: number;
    avg_amount: number;
    avg_pred_score: number;
  }>;
};

type PatternItem = {
  node_id?: string;
  entity_type?: string;
  entity_value?: string;
  degree?: number;
  risk_mean?: number;
  tx_a?: number;
  tx_b?: number;
  shared_relation_count?: number;
  score?: number;
};

type EmbeddingPoint = {
  TransactionID: number;
  x: number;
  y: number;
  risk_score: number;
  isFraud: number;
};

type DatasetName = "main" | "demo";

type SnapshotBundle = {
  dataset: DatasetName;
  meta: {
    relation_columns: string[];
    transaction_count: number;
  };
  overview: GraphResponse;
  patterns: {
    stars: { patterns: PatternItem[] };
    rings: { patterns: PatternItem[] };
  };
  embedding: { points: EmbeddingPoint[] };
  neighborhoods: Record<string, GraphResponse>;
  transactions: Record<string, TransactionDetail>;
  explanations: Record<string, ExplainResponse>;
  timelines_by_uid: Record<string, TimelineResponse>;
  search_index: {
    transaction_ids: string[];
    entities: Array<{ node_id: string; entity_type: string; entity_value: string }>;
  };
};

type ForceGraph2DProps = {
  graphData: GraphResponse;
  width: number;
  height: number;
  cooldownTicks?: number;
  d3VelocityDecay?: number;
  d3AlphaDecay?: number;
  linkColor?: (link: GraphLink) => string;
  linkWidth?: number;
  linkCurvature?: number;
  linkDirectionalParticles?: number;
  linkDirectionalParticleWidth?: number;
  linkDirectionalParticleSpeed?: () => number;
  onNodeClick?: (node: GraphNode) => void;
  nodeCanvasObject?: (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => void;
};

const ForceGraph2D = dynamic(
  async () =>
    (await import("react-force-graph-2d")).default as unknown as ComponentType<ForceGraph2DProps>,
  { ssr: false },
) as ComponentType<ForceGraph2DProps>;

const API_BASE = process.env.NEXT_PUBLIC_FRAUD_API_BASE_URL ?? "http://127.0.0.1:8000";
const DATA_SOURCE = (process.env.NEXT_PUBLIC_DATA_SOURCE ?? "api") as "api" | "static";
const IS_STATIC = DATA_SOURCE === "static";
const RELATION_COLORS: Record<string, string> = {
  card1: "rgba(167, 74, 62, 0.48)",
  addr1: "rgba(37, 130, 164, 0.45)",
  P_emaildomain: "rgba(35, 138, 107, 0.45)",
  R_emaildomain: "rgba(93, 126, 194, 0.45)",
  DeviceInfo: "rgba(148, 84, 176, 0.45)",
  id_30: "rgba(235, 142, 31, 0.45)",
  id_31: "rgba(214, 78, 116, 0.45)",
};

function withDataset(path: string, dataset: DatasetName): string {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}dataset=${dataset}`;
}

async function fetchApi<T>(path: string, dataset: DatasetName): Promise<T> {
  const response = await fetch(`${API_BASE}${withDataset(path, dataset)}`, {
    method: "GET",
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status}): ${body}`);
  }
  return (await response.json()) as T;
}

async function fetchSnapshot(dataset: DatasetName): Promise<SnapshotBundle> {
  const response = await fetch(`/snapshots/${dataset}.json`, {
    method: "GET",
    cache: "force-cache",
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Snapshot load failed (${response.status}): ${body}`);
  }
  return (await response.json()) as SnapshotBundle;
}

function linkKey(link: GraphLink): string {
  const source = typeof link.source === "string" ? link.source : link.source.id;
  const target = typeof link.target === "string" ? link.target : link.target.id;
  return `${source}|${target}|${link.relation}`;
}

function mergeGraph(base: GraphResponse, incoming: GraphResponse): GraphResponse {
  const nodes = new Map<string, GraphNode>();
  for (const n of base.nodes) nodes.set(n.id, n);
  for (const n of incoming.nodes) nodes.set(n.id, n);

  const links = new Map<string, GraphLink>();
  for (const l of base.links) links.set(linkKey(l), l);
  for (const l of incoming.links) links.set(linkKey(l), l);

  return {
    nodes: [...nodes.values()],
    links: [...links.values()],
    meta: incoming.meta ?? base.meta,
  };
}

function riskColor(score: number | undefined): string {
  const s = Math.max(0, Math.min(1, score ?? 0));
  if (s >= 0.75) return "#d1493f";
  if (s >= 0.45) return "#f2a93b";
  return "#1c9c7c";
}

export default function Page() {
  const [mounted, setMounted] = useState(false);
  const [graph, setGraph] = useState<GraphResponse>({ nodes: [], links: [] });
  const [loadingGraph, setLoadingGraph] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedTxId, setSelectedTxId] = useState<number | null>(null);
  const [detail, setDetail] = useState<TransactionDetail | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [stars, setStars] = useState<PatternItem[]>([]);
  const [rings, setRings] = useState<PatternItem[]>([]);
  const [embeddingPoints, setEmbeddingPoints] = useState<EmbeddingPoint[]>([]);
  const [searchInput, setSearchInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [expandMode, setExpandMode] = useState<"focus" | "merge">("focus");
  const [dataset, setDataset] = useState<DatasetName>("demo");
  const snapshotRef = useRef<Partial<Record<DatasetName, SnapshotBundle>>>({});

  const ensureSnapshot = useCallback(
    async (ds: DatasetName): Promise<SnapshotBundle | null> => {
      if (!IS_STATIC) return null;
      const cached = snapshotRef.current[ds];
      if (cached) return cached;
      const loaded = await fetchSnapshot(ds);
      snapshotRef.current[ds] = loaded;
      return loaded;
    },
    [],
  );

  const refreshOverview = useCallback(async () => {
    try {
      setLoadingGraph(true);
      setError(null);
      if (IS_STATIC) {
        const snap = await ensureSnapshot(dataset);
        if (!snap) throw new Error("Static snapshot not available.");
        setGraph(snap.overview);
        setStars(snap.patterns.stars.patterns ?? []);
        setRings(snap.patterns.rings.patterns ?? []);
        setEmbeddingPoints(snap.embedding.points ?? []);
        return;
      }
      const [overview, starData, ringData, embedData] = await Promise.all([
        fetchApi<GraphResponse>("/api/v1/graph/overview?max_transactions=80&hops=2&max_nodes=900&max_edges=3800", dataset),
        fetchApi<{ patterns: PatternItem[] }>("/api/v1/graph/patterns/stars?top_k=10&min_degree=6", dataset),
        fetchApi<{ patterns: PatternItem[] }>("/api/v1/graph/patterns/rings?top_k=10&max_seed_transactions=1500", dataset),
        fetchApi<{ points: EmbeddingPoint[] }>("/api/v1/embedding-space?sample_size=1200", dataset),
      ]);
      setGraph(overview);
      setStars(starData.patterns);
      setRings(ringData.patterns);
      setEmbeddingPoints(embedData.points);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setLoadingGraph(false);
    }
  }, [dataset, ensureSnapshot]);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedTxId(null);
    setDetail(null);
    setExplain(null);
    setTimeline(null);
  }, [dataset]);

  useEffect(() => {
    void refreshOverview();
  }, [refreshOverview]);

  useEffect(() => {
    if (!selectedTxId) {
      setDetail(null);
      setExplain(null);
      setTimeline(null);
      return;
    }
    const loadTransactionPanels = async () => {
      try {
        if (IS_STATIC) {
          const snap = await ensureSnapshot(dataset);
          if (!snap) throw new Error("Static snapshot not available.");
          const txDetail = snap.transactions[String(selectedTxId)];
          if (!txDetail) {
            setError(`Transaction ${selectedTxId} not present in static snapshot.`);
            return;
          }
          setDetail(txDetail);
          setExplain(snap.explanations[String(selectedTxId)] ?? null);
          setTimeline(snap.timelines_by_uid[txDetail.uid_clean] ?? { points: [] });
          return;
        }
        const txDetail = await fetchApi<TransactionDetail>(`/api/v1/transactions/${selectedTxId}`, dataset);
        setDetail(txDetail);
        const [txExplain, txTimeline] = await Promise.all([
          fetchApi<ExplainResponse>(`/api/v1/transactions/${selectedTxId}/explain?top_k=10`, dataset),
          fetchApi<TimelineResponse>(
            `/api/v1/timeline/entity?entity_type=uid_clean&entity_value=${encodeURIComponent(txDetail.uid_clean)}&bucket=day`,
            dataset,
          ),
        ]);
        setExplain(txExplain);
        setTimeline(txTimeline);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
      }
    };
    void loadTransactionPanels();
  }, [selectedTxId, dataset, ensureSnapshot]);

  const handleNodeClick = useCallback(
    async (node: GraphNode) => {
      setSelectedNodeId(node.id);
      if (node.id.startsWith("tx:")) {
        setSelectedTxId(Number.parseInt(node.id.replace("tx:", ""), 10));
      } else {
        setSelectedTxId(null);
      }
      try {
        if (IS_STATIC) {
          const snap = await ensureSnapshot(dataset);
          if (!snap) throw new Error("Static snapshot not available.");
          const expanded = snap.neighborhoods[node.id];
          if (!expanded) {
            setError("No precomputed neighborhood for this node in static mode.");
            return;
          }
          setGraph((prev) => (expandMode === "merge" ? mergeGraph(prev, expanded) : expanded));
          return;
        }
        const expanded = await fetchApi<GraphResponse>(
          `/api/v1/graph/neighborhood?node_id=${encodeURIComponent(node.id)}&hops=2&max_nodes=850&max_edges=4500&max_entity_degree=700&max_entity_fanout=120&include_missing_entities=false`,
          dataset,
        );
        setGraph((prev) => (expandMode === "merge" ? mergeGraph(prev, expanded) : expanded));
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
      }
    },
    [dataset, expandMode, ensureSnapshot],
  );

  const handleSearch = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = searchInput.trim();
      if (!trimmed) return;
      try {
        setError(null);
        if (IS_STATIC) {
          const snap = await ensureSnapshot(dataset);
          if (!snap) throw new Error("Static snapshot not available.");
          if (/^\d+$/.test(trimmed)) {
            const matched = snap.search_index.transaction_ids.find((id) => id.startsWith(trimmed));
            if (!matched) {
              setError("Transaction not found in static snapshot.");
              return;
            }
            const nodeId = `tx:${matched}`;
            const txGraph = snap.neighborhoods[nodeId];
            if (!txGraph) {
              setError("No precomputed transaction neighborhood in static snapshot.");
              return;
            }
            setGraph(txGraph);
            setSelectedNodeId(nodeId);
            setSelectedTxId(Number.parseInt(matched, 10));
            return;
          }
          const q = trimmed.toLowerCase();
          const entity = snap.search_index.entities.find(
            (e) =>
              e.entity_value.toLowerCase().includes(q) ||
              e.entity_type.toLowerCase().includes(q),
          );
          if (!entity) {
            setError("No matching entity in static snapshot.");
            return;
          }
          const resultGraph = snap.neighborhoods[entity.node_id];
          if (!resultGraph) {
            setError("No precomputed neighborhood for matched entity.");
            return;
          }
          setGraph(resultGraph);
          setSelectedNodeId(entity.node_id);
          setSelectedTxId(null);
          return;
        }

        if (/^\d+$/.test(trimmed)) {
          const txId = Number.parseInt(trimmed, 10);
          const txGraph = await fetchApi<GraphResponse>(
            `/api/v1/graph/transaction/${txId}?hops=2&max_nodes=850&max_edges=4500&max_entity_degree=700&include_missing_entities=false`,
            dataset,
          );
          setGraph(txGraph);
          setSelectedNodeId(`tx:${txId}`);
          setSelectedTxId(txId);
          return;
        }
        const search = await fetchApi<{ entities: Array<{ entity_type: string; entity_value: string }> }>(
          `/api/v1/search?q=${encodeURIComponent(trimmed)}&limit=1`,
          dataset,
        );
        const first = search.entities[0];
        if (first) {
          const nodeId = `ent:${first.entity_type}:${first.entity_value}`;
          const resultGraph = await fetchApi<GraphResponse>(
            `/api/v1/graph/neighborhood?node_id=${encodeURIComponent(nodeId)}&hops=2&max_nodes=850&max_edges=4500&max_entity_degree=700&max_entity_fanout=120&include_missing_entities=false`,
            dataset,
          );
          setGraph(resultGraph);
          setSelectedNodeId(nodeId);
          setSelectedTxId(null);
        } else {
          setError("No matching transactions or entities found.");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
      }
    },
    [dataset, searchInput, ensureSnapshot],
  );

  const graphStats = useMemo(
    () => ({
      nodes: graph.meta?.node_count ?? graph.nodes.length,
      links: graph.meta?.link_count ?? graph.links.length,
      risky: graph.nodes.filter((n) => (n.risk_score ?? 0) >= 0.65).length,
    }),
    [graph],
  );

  return (
    <div className="forensic-shell px-4 py-4 md:px-8 md:py-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="mx-auto flex w-full max-w-[1500px] flex-col gap-4"
      >
        <header className="glass-panel rounded-2xl p-4 md:p-5">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="mono text-xs tracking-[0.18em] text-[var(--accent-link)]">FRAUDSCOPE // INVESTIGATION GRID</p>
              <h1 className="mt-1 text-2xl font-semibold md:text-3xl">Graph-Centric Fraud Ring Explorer</h1>
            </div>
            <div className="flex items-center gap-2 rounded-md border border-[var(--line-soft)] bg-white/95 px-2 py-1">
              <button
                type="button"
                onClick={() => setDataset("demo")}
                className={clsx(
                  "rounded px-2 py-1 text-xs font-semibold",
                  dataset === "demo" ? "bg-[var(--bg-deep)] text-white" : "text-[var(--text-muted)]",
                )}
              >
                Demo Dataset
              </button>
              <button
                type="button"
                onClick={() => setDataset("main")}
                className={clsx(
                  "rounded px-2 py-1 text-xs font-semibold",
                  dataset === "main" ? "bg-[var(--bg-deep)] text-white" : "text-[var(--text-muted)]",
                )}
              >
                Main Dataset
              </button>
            </div>
            <form onSubmit={handleSearch} className="flex w-full max-w-xl flex-col gap-2 sm:flex-row">
              <input
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="TransactionID or entity value (email, device, uid...)"
                className="w-full rounded-xl border border-[var(--line-soft)] bg-white px-4 py-2.5 text-sm outline-none transition focus:border-[var(--accent-link)]"
              />
              <button
                type="submit"
                className="rounded-xl bg-[var(--bg-deep)] px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90"
              >
                Investigate
              </button>
              <button
                type="button"
                onClick={() => void refreshOverview()}
                className="rounded-xl border border-[var(--line-soft)] bg-white px-4 py-2.5 text-sm font-semibold text-[var(--text-strong)] transition hover:bg-slate-50"
              >
                Reset
              </button>
            </form>
          </div>
          {error ? <p className="mt-3 text-sm text-[var(--accent-risk)]">{error}</p> : null}
        </header>

        <section className="grid gap-4 lg:grid-cols-[1.65fr_0.95fr]">
          <article className="glass-panel relative min-h-[580px] overflow-hidden rounded-2xl p-2 md:p-3">
            <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex justify-between px-4 pt-3 text-xs text-[var(--text-muted)]">
              <span className="mono">Click nodes to expand 2-hop neighborhood</span>
              <span className="mono">
                {dataset.toUpperCase()} • {DATA_SOURCE.toUpperCase()} • {graphStats.nodes} nodes • {graphStats.links} links • {graphStats.risky} high-risk
              </span>
            </div>
            <div className="absolute inset-x-0 top-9 z-10 flex flex-wrap items-center justify-between gap-2 px-4 text-[11px]">
              <div className="flex items-center gap-2 rounded-md border border-[var(--line-soft)] bg-white/90 px-2 py-1">
                <button
                  type="button"
                  onClick={() => setExpandMode("focus")}
                  className={clsx(
                    "rounded px-2 py-1 font-semibold",
                    expandMode === "focus" ? "bg-[var(--bg-deep)] text-white" : "text-[var(--text-muted)]",
                  )}
                >
                  Focus mode
                </button>
                <button
                  type="button"
                  onClick={() => setExpandMode("merge")}
                  className={clsx(
                    "rounded px-2 py-1 font-semibold",
                    expandMode === "merge" ? "bg-[var(--bg-deep)] text-white" : "text-[var(--text-muted)]",
                  )}
                >
                  Merge mode
                </button>
              </div>
              <div className="flex flex-wrap items-center gap-2 rounded-md border border-[var(--line-soft)] bg-white/90 px-2 py-1">
                {Object.entries(RELATION_COLORS).map(([key, color]) => (
                  <span key={key} className="mono flex items-center gap-1 text-[10px] text-[var(--text-muted)]">
                    <i className="block h-1.5 w-3 rounded" style={{ background: color }} />
                    {key}
                  </span>
                ))}
              </div>
            </div>
            {loadingGraph ? (
              <div className="flex h-[560px] items-center justify-center text-sm text-[var(--text-muted)]">
                Building investigation graph...
              </div>
            ) : (
              <ForceGraph2D
                graphData={graph}
                width={980}
                height={560}
                cooldownTicks={60}
                d3VelocityDecay={0.35}
                d3AlphaDecay={0.06}
                linkColor={(link: GraphLink) =>
                  RELATION_COLORS[link.relation] ?? "rgba(17,37,45,0.20)"
                }
                linkWidth={1.1}
                linkCurvature={0.05}
                linkDirectionalParticles={1}
                linkDirectionalParticleWidth={1.2}
                linkDirectionalParticleSpeed={() => 0.003}
                onNodeClick={handleNodeClick}
                nodeCanvasObject={(node: GraphNode, ctx: CanvasRenderingContext2D, scale: number) => {
                  const label = node.label;
                  const radius = node.node_type === "transaction" ? 6.5 : 7.8;
                  ctx.save();
                  ctx.globalAlpha = selectedNodeId === node.id ? 1 : 0.92;
                  ctx.beginPath();
                  if (node.node_type === "transaction") {
                    ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, Math.PI * 2, false);
                    ctx.fillStyle = node.risk_color ?? riskColor(node.risk_score);
                    ctx.fill();
                  } else {
                    ctx.fillStyle = selectedNodeId === node.id ? "#0f2c35" : "#1d5f74";
                    ctx.fillRect((node.x ?? 0) - radius, (node.y ?? 0) - radius, radius * 2, radius * 2);
                  }
                  if (selectedNodeId === node.id || scale < 1.2) {
                    const fontSize = 12 / scale;
                    ctx.font = `${fontSize}px var(--font-code-mono)`;
                    ctx.fillStyle = "#11252d";
                    ctx.fillText(label.slice(0, 30), (node.x ?? 0) + radius + 3, (node.y ?? 0) + radius);
                  }
                  ctx.restore();
                }}
              />
            )}
          </article>

          <motion.aside
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15, duration: 0.35 }}
            className="glass-panel flex min-h-[580px] flex-col gap-3 rounded-2xl p-4"
          >
            <h2 className="text-lg font-semibold">Why Is This Risky?</h2>
            {detail ? (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div className="metric-tile rounded-xl p-2.5">
                    <p className="mono text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Transaction</p>
                    <p className="mono text-sm">{detail.TransactionID}</p>
                  </div>
                  <div className="metric-tile rounded-xl p-2.5">
                    <p className="mono text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Risk Score</p>
                    <p className={clsx("mono text-sm", detail.pred_score > 0.7 && "text-[var(--accent-risk)]")}>
                      {(detail.pred_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="metric-tile rounded-xl p-2.5">
                    <p className="mono text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Amount</p>
                    <p className="mono text-sm">${detail.TransactionAmt.toFixed(2)}</p>
                  </div>
                  <div className="metric-tile rounded-xl p-2.5">
                    <p className="mono text-[10px] uppercase tracking-wider text-[var(--text-muted)]">UID</p>
                    <p className="mono truncate text-sm">{detail.uid_clean}</p>
                  </div>
                </div>

                <div className="mt-1 rounded-xl border border-[var(--line-soft)] bg-white/80 p-2">
                  <p className="mono mb-2 text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                    Explainability Waterfall
                  </p>
                  <div className="h-44 w-full">
                    {mounted ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={(explain?.waterfall ?? []).map((w) => ({
                            feature: w.feature.slice(0, 14),
                            delta: Number(w.delta_logit.toFixed(3)),
                          }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e4ded2" />
                          <XAxis dataKey="feature" tick={{ fontSize: 10 }} interval={0} angle={-18} height={44} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Bar dataKey="delta" fill="#1d5f74" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : null}
                  </div>
                  <p className="mono mt-1 text-xs text-[var(--text-muted)]">
                    Final probability: {((explain?.final_probability ?? detail.pred_score) * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="rounded-xl border border-[var(--line-soft)] bg-white/80 p-2">
                  <p className="mono mb-2 text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                    Velocity Timeline (UID)
                  </p>
                  <div className="h-32 w-full">
                    {mounted ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={timeline?.points ?? []}>
                          <defs>
                            <linearGradient id="txnArea" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#1c9c7c" stopOpacity={0.75} />
                              <stop offset="95%" stopColor="#1c9c7c" stopOpacity={0.07} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e4ded2" />
                          <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Area type="monotone" dataKey="transaction_count" stroke="#1c9c7c" fill="url(#txnArea)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : null}
                  </div>
                </div>
              </>
            ) : (
              <p className="text-sm text-[var(--text-muted)]">
                Select a transaction node to view model explanation and timeline context.
              </p>
            )}
          </motion.aside>
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_1fr_1fr]">
          <article className="glass-panel rounded-2xl p-3">
            <h3 className="mb-2 text-sm font-semibold">Embedding Space Snapshot</h3>
            <div className="h-56 w-full">
              {mounted ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid stroke="#e4ded2" />
                    <XAxis type="number" dataKey="x" tick={{ fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" tick={{ fontSize: 10 }} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={embeddingPoints} fill="#1d5f74" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : null}
            </div>
          </article>

          <article className="glass-panel rounded-2xl p-3">
            <h3 className="mb-2 text-sm font-semibold">Star Patterns (High Fan-Out)</h3>
            <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
              {stars.map((item) => (
                <div key={`${item.node_id ?? "star"}-${item.entity_value ?? ""}`} className="metric-tile rounded-xl p-2">
                  <p className="mono text-xs text-[var(--text-strong)]">
                    {item.entity_type}: {item.entity_value}
                  </p>
                  <p className="mono mt-1 text-[11px] text-[var(--text-muted)]">
                    degree {item.degree} • risk {(100 * (item.risk_mean ?? 0)).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </article>

          <article className="glass-panel rounded-2xl p-3">
            <h3 className="mb-2 text-sm font-semibold">Ring Candidates</h3>
            <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
              {rings.map((item) => (
                <div key={`${item.tx_a ?? ""}-${item.tx_b ?? ""}`} className="metric-tile rounded-xl p-2">
                  <p className="mono text-xs">
                    TX {item.tx_a} ↔ TX {item.tx_b}
                  </p>
                  <p className="mono mt-1 text-[11px] text-[var(--text-muted)]">
                    shared relations: {item.shared_relation_count} • score {(item.score ?? 0).toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          </article>
        </section>
      </motion.div>
    </div>
  );
}
