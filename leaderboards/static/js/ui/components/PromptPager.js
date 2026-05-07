import { el } from '../dom.js';

/**
 * Collapsible prompt block that shows the last N lines when collapsed.
 * Pure/presentational: relies on props.expanded and calls onToggle() to request a re-render.
 */
function buildCollapsiblePromptBlockByLines({ text, expanded, onToggle, maxLines = 5 }) {
  const wrap = el('div', { class: 'relative border border-gray-200 rounded bg-gray-50 overflow-hidden cursor-pointer', attrs: { tabIndex: 0 } });

  // Clip region
  const clip = el('div', { class: 'relative overflow-hidden' });
  wrap.appendChild(clip);

  const content = el('div', { class: 'p-3 whitespace-pre-wrap leading-5 text-gray-900' }, text || '');
  clip.appendChild(content);

  const topFade = el('div', { class: 'pointer-events-none absolute top-0 left-0 right-0 h-7 bg-gradient-to-b from-gray-200 to-transparent opacity-100' });
  clip.appendChild(topFade);

  function applyLayout() {
    const cs = getComputedStyle(content);
    let lineHeight = parseFloat(cs.lineHeight);
    if (Number.isNaN(lineHeight)) {
      lineHeight = parseFloat(cs.fontSize) * 1.25;
    }
    const padTop = parseFloat(cs.paddingTop) || 0;
    const padBot = parseFloat(cs.paddingBottom) || 0;

    // Minimum stable height so the pager doesn’t jump
    const minWrapHeight = 5 * lineHeight + padTop + padBot;
    wrap.style.minHeight = `${minWrapHeight}px`;

    const targetContentHeight = maxLines * lineHeight;
    const fullH = content.scrollHeight;
    const needsCollapse = fullH > Math.ceil(targetContentHeight + 1);

    if (!needsCollapse || expanded) {
      clip.style.height = '';
      content.style.transform = '';
      topFade.style.display = 'none';
      wrap.setAttribute('aria-expanded', 'true');
      return;
    }

    // Collapsed: show only the last N lines
    clip.style.height = `${targetContentHeight + padTop + padBot}px`;
    const overflow = Math.max(0, fullH - targetContentHeight);
    content.style.transform = `translateY(-${overflow}px)`;
    topFade.style.display = '';
    wrap.setAttribute('aria-expanded', 'false');
  }

  // Toggle on click / keyboard
  const doToggle = () => {
    onToggle?.(); // controller flips state + re-renders promptly
  };
  wrap.addEventListener('click', (e) => {
    const sel = window.getSelection?.();
    if (sel && String(sel).length > 0) return; // don’t toggle while selecting text
    doToggle();
  });
  wrap.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      doToggle();
    }
  });

  const ro = new ResizeObserver(() => applyLayout());
  ro.observe(clip);
  ro.observe(wrap);

  queueMicrotask(applyLayout);
  return wrap;
}

/**
 * PromptPager
 * Props:
 *  - panelKey: 'left' | 'right'
 *  - lb: leaderboard data (full, with .prompts)
 *  - state: full app state (read-only)
 *  - onAdvance(delta: number)
 *  - responsesNodeFactory: () => HTMLElement
 *  - onToggleExpand: () => void    <-- NEW: triggers immediate re-render
 */
export function PromptPager({ panelKey, lb, state, onAdvance, responsesNodeFactory, onToggleExpand }) {
  const promptCount = Array.isArray(lb?.prompts) ? lb.prompts.length : 0;
  if (!promptCount) return el('div'); // nothing to render

  const idx = Math.min(
    Math.max((state.panels[panelKey].promptIndex || 0), 0),
    promptCount - 1
  );

  const wrap = el('div', { class: 'mt-2 overflow-hidden pb-3' });

  // Header
  const hdr = el('div', { class: 'px-3 py-1 m-0.5 rounded-lg border border-gray-100 text-xs flex items-center justify-between' });
  const title = el('div', { class: 'text-gray-700 font-medium' }, `Prompt ${idx + 1} of ${promptCount}`);
  const ctrls = el('div', { class: 'flex items-center gap-2' });

  const btnBase = 'px-2 py-1 rounded border text-xs disabled:opacity-50 disabled:cursor-not-allowed';
  const prevBtn = el('button', { class: `${btnBase} border-gray-300 hover:bg-gray-50` }, '◀ Prev');
  prevBtn.disabled = idx <= 0;
  prevBtn.addEventListener('click', () => onAdvance?.(-1));

  const nextBtn = el('button', { class: `${btnBase} border-gray-300 hover:bg-gray-50` }, 'Next ▶');
  nextBtn.disabled = idx >= promptCount - 1;
  nextBtn.addEventListener('click', () => onAdvance?.(+1));

  const hint = el('span', { class: 'text-gray-400 text-[11px] ml-2' }, 'Shortcut: left / right arrow keys');

  ctrls.appendChild(hint);
  ctrls.appendChild(prevBtn);
  ctrls.appendChild(nextBtn);

  hdr.appendChild(title);
  hdr.appendChild(ctrls);
  wrap.appendChild(hdr);

  // Body
  const body = el('div', { class: 'p-0.5 text-xs space-y-3' });
  wrap.appendChild(body);

  // Collapsible prompt
  const isExpanded = !!(state.ui?.promptExpanded?.[panelKey]);
  const promptText = String(lb.prompts[idx] ?? '').replace(/\\n/g, '\n');
  const promptBlock = buildCollapsiblePromptBlockByLines({
    text: promptText,
    expanded: isExpanded,
    onToggle: () => onToggleExpand?.(),
    maxLines: 5
  });
  body.appendChild(promptBlock);

  // Responses (selected models)
  const respNode = responsesNodeFactory?.();
  if (respNode) body.appendChild(respNode);

  return wrap;
}
