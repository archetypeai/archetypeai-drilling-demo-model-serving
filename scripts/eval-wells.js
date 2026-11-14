#!/usr/bin/env node
// Score the KNN library on every playable well (the reference wells are left
// out). For each well it takes up to CAP unanimous-ACTC windows per class,
// evenly spaced through the well, embeds them with Omega 1.5 through an
// Inference API session, and classifies each against data/knn-library.json the
// same way /api/classify does. DRILLING is the positive class.
//
// Usage: node scripts/eval-wells.js [--cap=15] [--k=3]

import { readFileSync, readdirSync } from 'fs';
import { resolve } from 'path';
import { OmegaSession } from '../src/lib/inference-client.js';
import { REFERENCE_WELLS } from '../src/lib/reference-wells.js';
import { DATA_COLUMNS, WELLS_DIR, applyScaler, collectWindows, subsample } from './windows.js';

function loadEnv() {
	const env = {};
	for (const line of readFileSync('.env', 'utf-8').split('\n')) {
		const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
		if (m) env[m[1]] = m[2].trim();
	}
	return env;
}

function parseArgs() {
	const args = { cap: 15, k: 3 };
	for (const a of process.argv.slice(2)) {
		const m = a.match(/^--(\w+)=(\d+)$/);
		if (m) args[m[1]] = parseInt(m[2]);
	}
	return args;
}

// Same naming as the well selector, with the operator added for the two F-5s.
function wellName(file, all) {
	const name = file.replace('.csv', '').replace(/\$47\$/g, '');
	const short = name.match(/F-[\w]+(?: [\w]+)?$/)?.[0] ?? name;
	const operator = name.replace(/-15_.*$/, '').replace(/^Norway-/, '');
	const shared = all.filter((f) => f !== file && f.match(/F-[\w]+(?: [\w]+)?\.csv$/)?.[0] === `${short}.csv`);
	return shared.length ? `${short} (${operator})` : short;
}

function classify(library, vec, k) {
	const scored = library.embeddings.map((e) => {
		let d = 0;
		for (let i = 0; i < vec.length; i++) d += (e.vec[i] - vec[i]) ** 2;
		return { d, label: e.label };
	});
	scored.sort((a, b) => a.d - b.d);
	const votes = {};
	for (const { label } of scored.slice(0, k)) votes[label] = (votes[label] || 0) + 1;
	return Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
}

let session = null;

async function main() {
	const env = loadEnv();
	const args = parseArgs();
	const library = JSON.parse(readFileSync(resolve('data/knn-library.json'), 'utf-8'));
	const { windowSize } = library.config;
	console.log(`Endpoint: ${env.ATAI_API_ENDPOINT}  library: ${library.config.model}, window ${windowSize}, k=${args.k}`);

	session = new OmegaSession({ endpoint: env.ATAI_API_ENDPOINT, apiKey: env.ATAI_API_KEY, windowSize });
	const all = readdirSync(WELLS_DIR).filter((f) => f.endsWith('.csv'));
	const wells = all.filter((f) => !REFERENCE_WELLS.includes(f));
	const rows = [];
	const t0 = Date.now();

	for (const file of wells) {
		const found = collectWindows(file, windowSize, windowSize);
		const picked = ['DRILLING', 'NOT_DRILLING'].flatMap((label) =>
			subsample(found[label], args.cap).map((win) => ({ label, win }))
		);
		const predicted = await Promise.all(
			picked.map(({ win }) => session.embed(applyScaler(win, DATA_COLUMNS)).then((v) => classify(library, v, args.k)))
		);
		let tp = 0, fp = 0, fn = 0, correct = 0;
		picked.forEach(({ label }, i) => {
			const p = predicted[i];
			if (p === label) correct++;
			if (p === 'DRILLING' && label === 'DRILLING') tp++;
			if (p === 'DRILLING' && label !== 'DRILLING') fp++;
			if (p !== 'DRILLING' && label === 'DRILLING') fn++;
		});
		const f1 = tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
		rows.push({
			well: wellName(file, all),
			acc: correct / picked.length,
			f1,
			n: picked.length,
			notDrilling: found.NOT_DRILLING.length
		});
		console.log(`  ${rows.at(-1).well}: ${correct}/${picked.length}`);
	}
	await session.close();

	rows.sort((a, b) => b.acc - a.acc);
	console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s\n`);
	console.log('| Well | Accuracy | F1 | Eval windows | Not-drilling windows in well |');
	console.log('|---|---:|---:|---:|---:|');
	for (const r of rows) {
		console.log(`| ${r.well} | ${(r.acc * 100).toFixed(0)}% | ${(r.f1 * 100).toFixed(0)}% | ${r.n} | ${r.notDrilling} |`);
	}
}

main().catch(async (err) => {
	console.error('FATAL:', err);
	await session?.close();
	process.exit(1);
});
