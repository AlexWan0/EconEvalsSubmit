// newline / text utilities

export function sanitizeMultiline(s) {
  return typeof s === 'string' ? s.replace(/\\n/g, '\n') : (s == null ? '' : String(s));
}

export function decodeNewlines(s) {
  return typeof s === 'string' ? s.replace(/\\n/g, '\n') : s;
}
