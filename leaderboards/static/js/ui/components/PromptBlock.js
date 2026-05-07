import { el } from '../dom.js';
import { PROMPT_MAX_LINES, PROMPT_MIN_LINES } from '../../utils/constants.js';

export function PromptBlock({ text, isExpanded, onToggle }) {
  const wrap = el('div', { class: 'relative border border-gray-200 rounded bg-gray-50 overflow-hidden cursor-pointer', attrs: { tabIndex: 0 } });

  const clip = el('div', { class: 'relative overflow-hidden' });
  const content = el('div', { class: 'p-3 whitespace-pre-wrap leading-5 text-gray-900' }, text);
  clip.appendChild(content);

  const topFade = el('div', { class: 'pointer-events-none absolute top-0 left-0 right-0 h-7 bg-gradient-to-b from-gray-200 to-transparent opacity-100' });
  clip.appendChild(topFade);

  wrap.appendChild(clip);

  function applyLayout() {
    const cs = getComputedStyle(content);
    let lineHeight = parseFloat(cs.lineHeight);
    if (Number.isNaN(lineHeight)) lineHeight = parseFloat(cs.fontSize) * 1.25;

    const targetContentHeight = PROMPT_MAX_LINES * lineHeight;
    const fullH = content.scrollHeight;

    const padTop = parseFloat(cs.paddingTop) || 0;
    const padBot = parseFloat(cs.paddingBottom) || 0;
    const minWrapHeight = PROMPT_MIN_LINES * lineHeight + padTop + padBot;
    wrap.style.minHeight = `${minWrapHeight}px`;

    if (isExpanded || fullH <= Math.ceil(targetContentHeight + 1)) {
      clip.style.height = '';
      content.style.transform = '';
      topFade.style.display = 'none';
      wrap.setAttribute('aria-expanded', 'true');
      return;
    }

    clip.style.height = `${targetContentHeight + padTop + padBot}px`;
    const overflow = Math.max(0, fullH - targetContentHeight);
    content.style.transform = `translateY(-${overflow}px)`;
    topFade.style.display = '';
    wrap.setAttribute('aria-expanded', 'false');
  }

  const ro = new ResizeObserver(() => applyLayout());
  ro.observe(clip);
  ro.observe(wrap);
  queueMicrotask(applyLayout);

  const toggle = () => { onToggle?.(); };

  wrap.addEventListener('click', (e) => {
    const sel = window.getSelection?.();
    if (sel && String(sel).length > 0) return;
    toggle();
  });
  wrap.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });

  return wrap;
}
