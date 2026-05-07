import { el } from '../dom.js';
import { setExpansion } from '../../state/index.js';

/**
 * Renders the list of selected-model responses (for current prompt).
 * @param {{ panelKey: 'left'|'right', items: Array<{modelName:string, bgClass:string, sections:Array<{key:string,label:string,text:string}>}>, expansion: Map<string, Map<string, boolean>> }} props
 */
export function Responses({ panelKey, items, expansion }) {
  if (!items?.length) {
    return el('div', { class: 'text-gray-500' }, 'No responses for selected models on this prompt.');
  }

  const frag = document.createDocumentFragment();

  items.forEach(({ modelName, bgClass, sections }) => {
    const modelOpen = expansion.get(modelName)?.get('__MODEL__');
    const outer = el('details', { class: 'rounded border border-gray-200', attrs: { open: modelOpen !== undefined ? modelOpen : true } });
    outer.addEventListener('toggle', () => setExpansion(panelKey, modelName, '__MODEL__', outer.open));

    const sum = el('summary', { class: `px-2 py-1.5 cursor-pointer ${bgClass} rounded text-gray-900` }, modelName);
    outer.appendChild(sum);

    const innerWrap = el('div', { class: 'px-1 py-2 space-y-2' });

    sections.forEach(({ key, label, text }) => {
      const secOpen = expansion.get(modelName)?.get(key);
      const sec = el('details', { class: 'border border-gray-100 rounded', attrs: { open: secOpen !== undefined ? secOpen : (key === 'judge_scores') } });
      const ssum = el('summary', { class: 'px-2 py-1 bg-gray-50 cursor-pointer rounded' }, label);
      const content = el('div', { class: 'px-2 py-2 whitespace-pre-wrap text-gray-800' }, text);
      sec.append(ssum, content);
      sec.addEventListener('toggle', () => setExpansion(panelKey, modelName, key, sec.open));
      innerWrap.appendChild(sec);
    });

    outer.appendChild(innerWrap);
    frag.appendChild(outer);
  });

  return frag;
}
