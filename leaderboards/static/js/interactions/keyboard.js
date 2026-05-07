import { panelSetPromptIndex, setLastActivePanel } from '../state/index.js';
import { getPromptCount } from '../state/selectors.js';

function isTypingTarget(el) {
  if (!el) return false;
  const tag = (el.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  if (el.isContentEditable) return true;
  return false;
}

export function installKeyboard({ getState, rerenderPanel }) {
  function onKeydown(e) {
    if (e.defaultPrevented) return;
    if (isTypingTarget(e.target)) return;

    const state = getState();
    const which = state.ui.lastActivePanel || 'left';
    const p = state.panels[which];
    const data = p?.data;
    const count = getPromptCount(data);

    if (!count) return;

    if (e.key === 'ArrowLeft') {
      const idx = Math.max(0, (p.promptIndex || 0) - 1);
      panelSetPromptIndex(which, idx);
      rerenderPanel(which);
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      const idx = Math.min(count - 1, (p.promptIndex || 0) + 1);
      panelSetPromptIndex(which, idx);
      rerenderPanel(which);
      e.preventDefault();
    }
  }

  document.addEventListener('keydown', onKeydown);
  return () => document.removeEventListener('keydown', onKeydown);
}
