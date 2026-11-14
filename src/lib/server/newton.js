import { ATAI_API_KEY, ATAI_API_ENDPOINT } from '$env/static/private';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { OmegaSession } from '$lib/inference-client.js';

// The nine drilling sensor channels, in the order the library was built with.
export const DATA_COLUMNS = ['BPOS', 'DBTM', 'FLWI', 'HDTH', 'HKLD', 'ROP', 'RPM', 'SPPA', 'WOB'];

export const DEFAULT_CONFIG = {
	windowSize: 128,
	stepSize: 128,
	nNeighbors: 3
};

// ──────────────────────────────────────────────────────────────────────
// Global per-channel robust scaler (data/scaler.json), loaded lazily.
// Scaling every window with these fixed stats, rather than normalizing each
// window on its own, preserves the cross-window amplitude signal. See scripts/build-scaler.js.
// Scaled values are clipped to ±SCALE_CLIP so sensor-glitch sentinels in the
// raw well data (e.g. SPPA spikes to ~1e31) can't dominate the embedding. The
// library was built with the same clip, so live and reference stay comparable.
// ──────────────────────────────────────────────────────────────────────

const SCALE_CLIP = 10;

let SCALER = null;
let SCALER_ERROR = null;
function ensureScaler() {
	if (SCALER || SCALER_ERROR) return;
	const path = resolve('data/scaler.json');
	if (!existsSync(path)) {
		SCALER_ERROR = new Error(
			'Missing data/scaler.json — run `node scripts/build-scaler.js` first.'
		);
		return;
	}
	SCALER = JSON.parse(readFileSync(path, 'utf-8'));
}

function applyScaler(channelFirstWindow, columns) {
	ensureScaler();
	if (SCALER_ERROR) throw SCALER_ERROR;
	const out = new Array(columns.length);
	for (let c = 0; c < columns.length; c++) {
		const col = columns[c];
		const m = SCALER.mean[col] ?? 0;
		const s = SCALER.std[col] ?? 1;
		const src = channelFirstWindow[c];
		const dst = new Array(src.length);
		for (let i = 0; i < src.length; i++) {
			const z = (src[i] - m) / s;
			dst[i] = z > SCALE_CLIP ? SCALE_CLIP : z < -SCALE_CLIP ? -SCALE_CLIP : z;
		}
		out[c] = dst;
	}
	return out;
}

// ──────────────────────────────────────────────────────────────────────
// KNN library (loaded once from data/knn-library.json)
// ──────────────────────────────────────────────────────────────────────

let LIBRARY = null;
let LIBRARY_ERROR = null;
function ensureLibrary() {
	if (LIBRARY || LIBRARY_ERROR) return;
	const path = resolve('data/knn-library.json');
	if (!existsSync(path)) {
		LIBRARY_ERROR = new Error(
			'Missing data/knn-library.json — run `node scripts/build-knn-library.js` first.'
		);
		return;
	}
	const raw = JSON.parse(readFileSync(path, 'utf-8'));
	raw.embeddings = raw.embeddings.map((e) => ({ label: e.label, vec: Float32Array.from(e.vec) }));
	LIBRARY = raw;
}

export function getLibraryConfig() {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	return LIBRARY.config;
}

// ──────────────────────────────────────────────────────────────────────
// Inference API: Omega 1.5 embeddings from one long-lived session
// ──────────────────────────────────────────────────────────────────────

// One session per server process, kept on globalThis so Vite's hot reload
// reuses it instead of opening another. The session is cancelled on Ctrl-C
// (see vite.config.js) and on SIGINT/SIGTERM when running a production build.
function getOmegaSession() {
	if (!globalThis.__drillingOmegaSession) {
		const session = new OmegaSession({
			endpoint: ATAI_API_ENDPOINT,
			apiKey: ATAI_API_KEY,
			windowSize: DEFAULT_CONFIG.windowSize,
			maxInFlight: 6
		});
		const shutdown = () => session.close().finally(() => process.exit(0));
		process.once('SIGINT', shutdown);
		process.once('SIGTERM', shutdown);
		globalThis.__drillingOmegaSession = session;
	}
	return globalThis.__drillingOmegaSession;
}

// Embed a [channels x windowSize] window in one query. Returns the per-channel
// 768-d vectors concatenated, in channel order, for KNN.
export function embedWindow(channelFirstWindow) {
	return getOmegaSession().embed(channelFirstWindow);
}

// ──────────────────────────────────────────────────────────────────────
// Local KNN classifier
// ──────────────────────────────────────────────────────────────────────

function euclideanSq(a, b) {
	let s = 0;
	for (let i = 0; i < a.length; i++) {
		const d = a[i] - b[i];
		s += d * d;
	}
	return s;
}

function classifyEmbedding(embedding, k = DEFAULT_CONFIG.nNeighbors) {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	const lib = LIBRARY.embeddings;
	const dists = new Array(lib.length);
	for (let i = 0; i < lib.length; i++) {
		dists[i] = { d: euclideanSq(lib[i].vec, embedding), label: lib[i].label };
	}
	dists.sort((a, b) => a.d - b.d);
	const top = dists.slice(0, k);
	const votes = {};
	for (const t of top) votes[t.label] = (votes[t.label] || 0) + 1;
	let winner = null;
	let max = -1;
	for (const [label, n] of Object.entries(votes)) {
		if (n > max) {
			max = n;
			winner = label;
		}
	}
	return {
		label: winner,
		votes,
		neighbors: top.map((t) => ({ label: t.label, dist: Math.sqrt(t.d) }))
	};
}

function extractWindow(rows, columns) {
	return columns.map((col) =>
		rows.map((row) => {
			const v = parseFloat(row[col]);
			return isNaN(v) ? 0 : v;
		})
	);
}

// Embed one window of sensor rows with Omega and classify it with KNN.
// Returns { label, votes, neighbors }.
export async function classifyWindow(rows, { k = DEFAULT_CONFIG.nNeighbors } = {}) {
	const win = extractWindow(rows, DATA_COLUMNS);
	const scaled = applyScaler(win, DATA_COLUMNS);
	const embedding = await embedWindow(scaled);
	return classifyEmbedding(embedding, k);
}
