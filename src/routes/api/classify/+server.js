import { json } from '@sveltejs/kit';
import { classifyWindow, DEFAULT_CONFIG } from '$lib/server/newton.js';

// Per-window classification: Omega 1.5 embeddings (Inference API) + local KNN.
//   Request:  { rows: [{ BPOS: '...', ... }, ...], k? }   length == windowSize
//   Response: { label: 'DRILLING'|'NOT_DRILLING', votes, neighbors: [...] }
export async function POST({ request }) {
	try {
		const { rows, k } = await request.json();
		if (!Array.isArray(rows) || rows.length === 0) {
			return json({ error: 'Missing or empty rows' }, { status: 400 });
		}
		const result = await classifyWindow(rows, { k: k || DEFAULT_CONFIG.nNeighbors });
		return json(result);
	} catch (err) {
		return json({ error: err.message }, { status: 500 });
	}
}
