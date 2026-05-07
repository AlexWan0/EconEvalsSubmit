// static/js/ui/components/D5Summary.js
import { el } from '../dom.js';

function fmtScore(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '';
  // Show to 3 decimals (e.g., 0.871)
  return x.toFixed(3);
}

/**
 * @param {{ items: Array<[string, number, string[]]> }} props
 */
export function D5Summary({ items }) {
  const wrap = el('section', { class: 'px-1 pb-4 pt-2' });

  const title = el('h3', { class: 'px-1 text-xs font-semibold text-gray-700 mb-2' }, 'Comparison clusters');
  wrap.appendChild(title);

  if (!Array.isArray(items) || items.length === 0) {
    wrap.appendChild(el('div', { class: 'text-xs text-gray-500' }, 'No comparison data.'));
    return wrap;
  }

  const list = el('div', { class: 'space-y-2' });

  items.forEach((row, idx) => {
    // row is [label: string, score: number, prompts: string[]]
    const label = row?.[0] ?? '';
    const score = row?.[1];
    const prompts = Array.isArray(row?.[2]) ? row[2] : [];

    const card = el('div', { class: 'border border-gray-200 rounded-lg bg-white' });

    const header = el('div', { class: 'px-2 py-1.5 flex items-center justify-between bg-gray-50 rounded-t-lg' });
    header.appendChild(el('div', { class: 'text-[11px] font-medium text-gray-800' }, label));
    header.appendChild(el('div', { class: 'text-[11px] text-gray-600 tabular-nums' }, fmtScore(score)));
    card.appendChild(header);

    if (prompts.length) {
      const details = el('details', { class: 'px-2 py-1.5' });
      const sum = el('summary', { class: 'cursor-pointer text-[11px] text-gray-700 hover:underline' }, `Prompts`);
      details.appendChild(sum);

      const body = el('div', { class: 'mt-1 max-h-56 overflow-auto pr-1' });
      const ul = el('ul', { class: 'list-none px-2 space-y-1' });

      // Render all prompts (they may be long; container scrolls)
      for (const p of prompts) {
        ul.appendChild(el('li', { class: 'text-[11px] leading-snug text-gray-800 whitespace-pre-wrap border-b-2 pb-2' }, String(p)));
      }
      body.appendChild(ul);
      details.appendChild(body);
      card.appendChild(details);
    }

    list.appendChild(card);
  });

  wrap.appendChild(list);
  return wrap;
}
