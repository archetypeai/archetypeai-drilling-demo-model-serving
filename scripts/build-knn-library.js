#!/usr/bin/env node
// Build the n-shot KNN library from the held-out reference wells
// (src/lib/reference-wells.js) — NOT from the wells the demo classifies, so the
// split is leakage-free.
//
// For each reference well it slides a window across the rows; a window whose
// ACTC codes are unanimously drilling (1/2) or not-drilling (3/4/8/9) becomes a
// labeled reference, mixed/other windows are skipped. The unanimous windows are
// pooled per class across the reference wells and evenly subsampled to CAP each
// (a balanced library that doesn't take forever to embed), then each window is
// embedded with Omega 1.5 through an Inference API session, all nine channels
// in one query.
//
// Output: data/knn-library.json
//   { config: { windowSize, stepSize, model, referenceWells, builtAt },
//     columns: [...],
//     embeddings: [{ label: 'DRILLING'|'NOT_DRILLING', vec: number[] }, ...] }
//
// Usage: node scripts/build-knn-library.js [--window=128] [--step=128] [--cap=18]

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { OmegaSession, OMEGA_MODEL } from '../src/lib/inference-client.js';
import { REFERENCE_WELLS } from '../src/lib/reference-wells.js';
import { DATA_COLUMNS, applyScaler, collectWindows, subsample } from './windows.js';


function loadEnv() {
	const env = {};
	for (const line of readFileSync('.env', 'utf-8').split('\n')) {
		const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
		if (m) env[m[1]] = m[2].trim();
	}
	return env;
}

function parseArgs() {
	const args = { window: 128, step: 128, cap: 18 };
	for (const a of process.argv.slice(2)) {
		const m = a.match(/^--(\w+)=(\d+)$/);
		if (m) args[m[1]] = parseInt(m[2]);
	}
	return args;
}

let session = null;

async function main() {
	const env = loadEnv();
	const args = parseArgs();
	console.log(`Endpoint: ${env.ATAI_API_ENDPOINT}`);
	console.log(`Window: ${args.window}, Step: ${args.step}, Cap/class: ${args.cap}`);
	console.log(`Reference wells: ${REFERENCE_WELLS.join(', ')}`);

	// Pool unanimous windows per class across the reference wells.
	const pool = { DRILLING: [], NOT_DRILLING: [] };
	for (const well of REFERENCE_WELLS) {
		const found = collectWindows(well, args.window, args.step);
		pool.DRILLING.push(...found.DRILLING);
		pool.NOT_DRILLING.push(...found.NOT_DRILLING);
		console.log(
			`  ${well}: ${found.DRILLING.length} drilling / ${found.NOT_DRILLING.length} not-drilling windows`
		);
	}

	const selected = {
		DRILLING: subsample(pool.DRILLING, args.cap),
		NOT_DRILLING: subsample(pool.NOT_DRILLING, args.cap)
	};
	console.log(
		`Selected: ${selected.DRILLING.length} drilling + ${selected.NOT_DRILLING.length} not-drilling`
	);

	session = new OmegaSession({
		endpoint: env.ATAI_API_ENDPOINT,
		apiKey: env.ATAI_API_KEY,
		windowSize: args.window
	});
	const t0 = Date.now();
	const jobs = [];
	for (const label of ['DRILLING', 'NOT_DRILLING']) {
		for (const win of selected[label]) {
			jobs.push(session.embed(applyScaler(win, DATA_COLUMNS)).then((vec) => ({ label, vec: Array.from(vec) })));
		}
	}
	const embeddings = await Promise.all(jobs);
	await session.close();

	const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
	const lib = {
		config: {
			windowSize: args.window,
			stepSize: args.step,
			model: OMEGA_MODEL,
			referenceWells: REFERENCE_WELLS,
			builtAt: new Date().toISOString()
		},
		columns: DATA_COLUMNS,
		embeddings
	};

	const outPath = resolve('data/knn-library.json');
	mkdirSync(dirname(outPath), { recursive: true });
	writeFileSync(outPath, JSON.stringify(lib));
	const sizeMb = (readFileSync(outPath).length / 1024 / 1024).toFixed(2);
	console.log(`\nWrote ${outPath} (${sizeMb} MB, ${embeddings.length} windows) in ${elapsed}s`);
}

main().catch(async (err) => {
	console.error('FATAL:', err);
	await session?.close();
	process.exit(1);
});
