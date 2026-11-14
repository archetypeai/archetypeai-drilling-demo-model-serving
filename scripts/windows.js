// Window helpers shared by build-knn-library.js and eval-wells.js.

import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { ACTC_DRILLING, ACTC_NOT_DRILLING } from '../src/lib/reference-wells.js';

export const DATA_COLUMNS = ['BPOS', 'DBTM', 'FLWI', 'HDTH', 'HKLD', 'ROP', 'RPM', 'SPPA', 'WOB'];
export const WELLS_DIR = 'static/data/wells';
const DRILL = new Set(ACTC_DRILLING);
const NOT = new Set(ACTC_NOT_DRILLING);

const SCALER_PATH = resolve('data/scaler.json');
const SCALER = existsSync(SCALER_PATH) ? JSON.parse(readFileSync(SCALER_PATH, 'utf-8')) : null;
if (!SCALER) {
	console.error('Missing data/scaler.json — run `node scripts/build-scaler.js` first.');
	process.exit(1);
}
const SCALE_CLIP = 10; // must match src/lib/server/newton.js
export function applyScaler(channelFirstWindow, columns) {
	const out = new Array(columns.length);
	for (let c = 0; c < columns.length; c++) {
		const m = SCALER.mean[columns[c]] ?? 0;
		const s = SCALER.std[columns[c]] ?? 1;
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

function readCsv(filePath) {
	const text = readFileSync(filePath, 'utf-8');
	const lines = text.split(/\r?\n/).filter((l) => l.trim());
	const headers = lines[0].split(',').map((h) => h.trim());
	return { headers, rows: lines.slice(1).map((l) => l.split(',')) };
}

// Collect unanimous-label windows (channel-first 2D) from one well.
export function collectWindows(well, windowSize, stepSize) {
	const { headers, rows } = readCsv(resolve(WELLS_DIR, well));
	const colIdx = DATA_COLUMNS.map((c) => headers.indexOf(c));
	const actcIdx = headers.indexOf('ACTC');
	if (colIdx.some((i) => i < 0) || actcIdx < 0) throw new Error(`${well}: missing column`);
	const found = { DRILLING: [], NOT_DRILLING: [] };
	for (let start = 0; start + windowSize <= rows.length; start += stepSize) {
		let allD = true;
		let allN = true;
		for (let r = 0; r < windowSize; r++) {
			const code = rows[start + r][actcIdx]?.trim();
			if (DRILL.has(code)) allN = false;
			else if (NOT.has(code)) allD = false;
			else {
				allD = false;
				allN = false;
			}
		}
		const label = allD ? 'DRILLING' : allN ? 'NOT_DRILLING' : null;
		if (!label) continue;
		const win = colIdx.map((ci) => {
			const ch = new Array(windowSize);
			for (let r = 0; r < windowSize; r++) {
				const v = parseFloat(rows[start + r][ci]);
				ch[r] = isNaN(v) ? 0 : v;
			}
			return ch;
		});
		found[label].push(win);
	}
	return found;
}

// Evenly subsample `cap` windows across the pool (temporal diversity, deterministic).
export function subsample(arr, cap) {
	if (arr.length <= cap) return arr;
	const step = arr.length / cap;
	const out = [];
	for (let i = 0; i < cap; i++) out.push(arr[Math.floor(i * step)]);
	return out;
}
