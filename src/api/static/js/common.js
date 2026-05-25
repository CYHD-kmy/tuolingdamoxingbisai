/* ═══════════════════════════════════════════════════════
   智投未来 — 共享 JavaScript 工具
   ═══════════════════════════════════════════════════════ */

// ── Navigation ────────────────────────────
function highlightNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === path);
  });
}

// ── API helper ────────────────────────────
async function apiFetch(path, date) {
  const sep = path.includes('?') ? '&' : '?';
  const url = date ? path + sep + 'date=' + date : path;
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ── Formatting ────────────────────────────
function fmtMoney(val) {
  if (val == null) return '-';
  if (val >= 1e8) return '¥' + (val / 1e8).toFixed(1) + '亿';
  if (val >= 1e4) return '¥' + (val / 1e4).toFixed(0) + '万';
  return '¥' + Number(val).toFixed(0);
}

function fmtPct(val) {
  if (val == null) return '-';
  return (val > 1 ? Number(val).toFixed(1) : (Number(val) * 100).toFixed(0)) + '%';
}

function fmtNum(v) { return v != null ? Number(v).toFixed(2) : '-'; }

// ── DOM helpers ───────────────────────────
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

// ── Status indicator ──────────────────────
async function updateStatusDot() {
  try {
    const d = await apiFetch('/api/status');
    const dot = $('#statusDot');
    const txt = $('#statusText');
    if (dot && txt) {
      if (d.ready) {
        dot.classList.add('online');
        txt.textContent = '数据就绪';
      } else {
        dot.classList.remove('online');
        txt.textContent = '暂无数据';
      }
    }
  } catch(e) { /* server not running */ }
}

// ── Date selector ─────────────────────────
async function populateDateSelect(selectEl) {
  try {
    const d = await apiFetch('/api/history');
    selectEl.innerHTML = '<option value="">最新</option>';
    for (const t of (d.traces || [])) {
      const opt = document.createElement('option');
      opt.value = t.date;
      opt.textContent = t.date + ' (' + t.size_kb + 'KB)';
      selectEl.appendChild(opt);
    }
  } catch(e) { console.error(e); }
}

// ── Markdown to HTML (简易渲染器) ──────────
function renderMarkdown(md) {
  if (!md) return '<p>暂无内容</p>';

  let html = md;

  // Escape HTML entities in non-code text
  // (skip this for simplicity - report content is trusted)

  // Code blocks (fenced) - must be before inline code
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return '<pre><code>' + code.trim() + '</code></pre>';
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr>');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Tables: detect |---|---| pattern, then process surrounding lines
  html = html.replace(/(\|[^\n]+\|\n\|[-:\s|]+\|\n((?:\|[^\n]+\|\n?)*))/g, (match) => {
    const lines = match.trim().split('\n');
    if (lines.length < 2) return match;
    let table = '<table>';
    // Header
    table += '<thead><tr>' + lines[0].split('|').filter(c => c.trim()).map(c =>
      '<th>' + c.trim() + '</th>').join('') + '</tr></thead>';
    // Body (skip separator line)
    table += '<tbody>';
    for (let i = 2; i < lines.length; i++) {
      table += '<tr>' + lines[i].split('|').filter(c => c.trim()).map(c =>
        '<td>' + c.trim() + '</td>').join('') + '</tr>';
    }
    table += '</tbody></table>';
    return table;
  });

  // Unordered lists: consecutive "- item" lines
  html = html.replace(/((?:^\- .+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(line =>
      '<li>' + line.replace(/^\- /, '') + '</li>').join('');
    return '<ul>' + items + '</ul>';
  });

  // Ordered lists: consecutive "1. item" lines
  html = html.replace(/((?:^\d+\. .+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(line =>
      '<li>' + line.replace(/^\d+\. /, '') + '</li>').join('');
    return '<ol>' + items + '</ol>';
  });

  // Paragraphs: double newlines become paragraph breaks
  // Split by double newlines and wrap non-tag blocks in <p>
  const blocks = html.split(/\n\n+/);
  html = blocks.map(block => {
    block = block.trim();
    if (!block) return '';
    // Skip if already wrapped in block-level tag
    if (/^<(h[1-4]|table|ul|ol|pre|hr|div|li|thead|tbody|tr|th|td)/.test(block)) {
      return block;
    }
    // Skip single <br>-like
    return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
  }).join('\n');

  return '<div class="report-content">' + html + '</div>';
}

// ── Init ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  highlightNav();
  updateStatusDot();
  // Refresh status dot every 60s
  setInterval(updateStatusDot, 60000);
});

// ── URL params ─────────────────────────────
function getUrlDate() {
  const params = new URLSearchParams(window.location.search);
  return params.get('date') || '';
}
