/* 中英对照阅读器：markdown 渲染 + 章节锚点匹配 + 双向滚动同步 */
(function () {
  'use strict';

  /* ================= 工具 ================= */
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* ================= Markdown 渲染 =================
     支持：标题/段落/表格/列表/引用/粗体/斜体/行内代码/链接/分隔线
     数学：$$...$$ 与 $...$ 原样保留，KaTeX auto-render 就绪后渲染，
     CDN 不可用时保持 LaTeX 原文显示 */

  var MATH_STORE = [];
  function protectMath(text) {
    MATH_STORE = [];
    var t = text.replace(/\$\$([\s\S]+?)\$\$/g, function (m) {
      MATH_STORE.push(m); return '@@M' + (MATH_STORE.length - 1) + '@@';
    });
    t = t.replace(/\$([^$\n]+?)\$/g, function (m) {
      MATH_STORE.push(m); return '@@M' + (MATH_STORE.length - 1) + '@@';
    });
    return t;
  }
  function restoreMath(html) {
    return html.replace(/@@M(\d+)@@/g, function (m, i) {
      return esc(MATH_STORE[+i]);
    });
  }
  function inline(s) {
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }
  function fmt(text) { return restoreMath(inline(esc(protectMath(text)))); }

  function splitRow(line) {
    var t = line.trim();
    if (t.charAt(0) === '|') t = t.slice(1);
    if (t.charAt(t.length - 1) === '|' && t.charAt(t.length - 2) !== '\\') t = t.slice(0, -1);
    var cells = t.split('|');
    for (var i = 0; i < cells.length; i++) cells[i] = cells[i].trim();
    return cells;
  }
  function isSepRow(line) {
    var t = line.trim();
    if (t.indexOf('-') < 0) return false;
    return /^[\s|:-]+$/.test(t);
  }

  function renderTable(lines) {
    var rows = [];
    var aligns = null;
    for (var i = 0; i < lines.length; i++) {
      if (i === 1 && isSepRow(lines[i])) { aligns = splitRow(lines[i]); continue; }
      rows.push(splitRow(lines[i]));
    }
    if (!rows.length) return '';
    var head = rows[0];
    var html = '<div class="table-wrap"><table>';
    html += '<thead><tr>';
    for (var h = 0; h < head.length; h++) {
      html += '<th>' + fmt(head[h]) + '</th>';
    }
    html += '</tr></thead><tbody>';
    for (var r = 1; r < rows.length; r++) {
      html += '<tr>';
      for (var c = 0; c < head.length; c++) {
        html += '<td>' + (rows[r][c] !== undefined ? fmt(rows[r][c]) : '') + '</td>';
      }
      html += '</tr>';
    }
    html += '</tbody></table></div>';
    return html;
  }

  function renderList(lines, ordered) {
    var html = ordered ? '<ol>' : '<ul>';
    var inSub = false;
    for (var i = 0; i < lines.length; i++) {
      var m = lines[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (!m) continue;
      var depth = Math.floor(m[1].replace(/\t/g, '  ').length / 2);
      if (depth > 0) {
        if (!inSub) { html += '<ul>'; inSub = true; }
        html += '<li>' + fmt(m[3]) + '</li>';
      } else {
        if (inSub) { html += '</ul>'; inSub = false; }
        html += '<li>' + fmt(m[3]) + '</li>';
      }
    }
    if (inSub) html += '</ul>';
    html += ordered ? '</ol>' : '</ul>';
    return html;
  }

  function renderMarkdown(md) {
    var lines = md.replace(/\r\n?/g, '\n').split('\n');
    var out = [], i = 0;
    while (i < lines.length) {
      var line = lines[i];
      if (/^\s*$/.test(line)) { i++; continue; }
      if (/^(---+|\*\*\*+|___+)\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        var lv = Math.min(h[1].length, 4);
        var txt = h[2].trim();
        var key = zhSecKey(txt);
        out.push('<h' + lv + ' class="anchorable"' +
          (key ? ' data-sec="' + esc(key) + '"' : '') + '>' + fmt(txt) + '</h' + lv + '>');
        i++; continue;
      }
      if (line.trim().charAt(0) === '|') {
        var tbl = [];
        while (i < lines.length && lines[i].trim().charAt(0) === '|') { tbl.push(lines[i]); i++; }
        out.push(renderTable(tbl)); continue;
      }
      if (/^\s*>\s?/.test(line)) {
        var q = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          q.push(lines[i].replace(/^\s*>\s?/, '')); i++;
        }
        out.push('<blockquote>' + q.map(fmt).join('<br>') + '</blockquote>'); continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {
        var ul = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { ul.push(lines[i]); i++; }
        out.push(renderList(ul, false)); continue;
      }
      if (/^\s*\d+\.\s+/.test(line)) {
        var ol = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { ol.push(lines[i]); i++; }
        out.push(renderList(ol, true)); continue;
      }
      if (/^\s*\$\$/.test(line)) {
        var single = line.trim();
        if (single.length > 4 && single.indexOf('$$', 2) > 0 && /\$\$\s*$/.test(single)) {
          out.push('<div class="math-display">' + esc(single) + '</div>'); i++; continue;
        }
        var blk = [line];
        i++;
        while (i < lines.length) {
          blk.push(lines[i]);
          if (/\$\$\s*$/.test(lines[i])) { i++; break; }
          i++;
        }
        out.push('<div class="math-display">' + esc(blk.join('\n')) + '</div>'); continue;
      }
      var para = [line]; i++;
      while (i < lines.length && !/^\s*$/.test(lines[i]) &&
        !/^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\s*\$\$|\s*>|\s*\|)/.test(lines[i])) {
        para.push(lines[i]); i++;
      }
      out.push('<p>' + para.map(fmt).join(' ') + '</p>');
    }
    return out.join('\n');
  }

  /* ================= 章节号识别 ================= */
  function normalizeSec(t) {
    return String(t).trim().replace(/\s+/g, '').replace(/[,;:.]$/, '');
  }
  function zhSecKey(text) {
    var m = text.match(/^\s*(\d+(?:\.\d+)*)[\s、．.:：)）]/);
    if (m) return m[1];
    var a = text.match(/^\s*附\s*录?\s*([A-Za-z])(?:[.．]\s*(\d+))?/);
    if (a) return a[1].toUpperCase() + (a[2] ? '.' + a[2] : '');
    if (/^\s*摘要/.test(text)) return 'abs';
    return null;
  }
  function enSecKey(el) {
    var tag = el.querySelector('.ltx_tag');
    if (tag) {
      var t = normalizeSec(tag.textContent);
      if (t) {
        t = t.replace(/^appendix/i, '');   // "AppendixA" -> "A"
        if (t) return t;
      }
    }
    var txt = (el.textContent || '').trim().toLowerCase();
    if (/^abstract\b/.test(txt)) return 'abs';
    if (/^(references|bibliography)\b/.test(txt)) return null;
    if (/^appendix\b/.test(txt)) {
      var am = txt.match(/^appendix\s*([a-z])\b/);
      if (am) return am[1].toUpperCase();
      return null;
    }
    var num = txt.match(/^(\d+(?:\.\d+)*)\s/);
    if (num) return num[1];
    return null;
  }

  /* ================= 初始化 ================= */
  var enBody = document.getElementById('enBody');
  var zhSrcEl = document.getElementById('zhSrc');
  var zhRoot = document.getElementById('mdRoot');
  var zhBody = document.getElementById('zhBody');
  var syncToggle = document.getElementById('syncToggle');

  if (zhSrcEl && zhRoot) {
    zhRoot.innerHTML = renderMarkdown(zhSrcEl.textContent || '');
  }

  function fixImgs(root) {
    if (!root) return;
    var imgs = root.querySelectorAll('img');
    for (var i = 0; i < imgs.length; i++) {
      imgs[i].addEventListener('error', function () {
        this.outerHTML = '<span class="img-miss">[图片需联网加载]</span>';
      });
    }
  }
  fixImgs(enBody);
  fixImgs(zhRoot);

  /* ================= 英文侧超宽表格自适应 ================= */
  /* LaTeXML 对超宽表用内联 transform + 固定尺寸包裹层做缩放，其缩放系数按
     LaTeXML 自带样式的紧凑内边距估算；本站的表格边框/内边距会让实际尺寸
     远超预留盒，缩放后的表格会溢出并与下方正文重叠。此处先解除内联缩放，
     再按窗格实际宽度重新适配——zoom 参与布局计算，不会像 transform 那样
     把内容压到正文上。 */
  function fitEnTables() {
    if (!enBody) return;
    var inners = enBody.querySelectorAll('.ltx_transformed_inner, .ltx_transformed_outer');
    for (var k = 0; k < inners.length; k++) {
      inners[k].style.transform = '';
      inners[k].style.width = '';
      inners[k].style.height = '';
      inners[k].style.verticalAlign = '';
    }
    var figs = enBody.querySelectorAll('figure.ltx_table');
    for (var i = 0; i < figs.length; i++) {
      var fig = figs[i];
      var tbs = fig.querySelectorAll('table');
      for (var j = 0; j < tbs.length; j++) tbs[j].style.zoom = '';
      /* 清除 zoom 后按溢出量整体缩放（含同卡多子表的情况）。
         卡片左右 padding 各 12px：可用内容宽 = clientWidth - 26（含 2px 余量），
         溢出内容自然宽 = scrollWidth - 24；缩放后复核一次，消除舍入残差 */
      var cw = fig.clientWidth, sw = fig.scrollWidth;
      if (sw > cw + 4 && sw > 0) {
        /* 窄屏设缩放下限：保证字号仍可读，剩余宽度由卡片横向滚动兜底 */
        var zfloor = (window.innerWidth < 760) ? 0.45 : 0.3;
        var avail = cw - 26, nat = Math.max(sw - 24, avail);
        var s = Math.max(zfloor, Math.round(avail / nat * 1000) / 1000);
        for (var a = 0; a < tbs.length; a++) tbs[a].style.zoom = String(s);
        var sw2 = fig.scrollWidth, avail2 = fig.clientWidth - 26;
        if (sw2 - 24 > avail2 + 2) {
          var s2 = Math.max(zfloor, Math.round(s * avail2 / (sw2 - 24) * 1000) / 1000);
          for (var b = 0; b < tbs.length; b++) tbs[b].style.zoom = String(s2);
        }
      }
    }
  }
  fitEnTables();
  window.addEventListener('load', fitEnTables);

  if (syncToggle) syncToggle.checked = true;

  /* ================= 锚点与滚动同步 ================= */
  var pairs = [];   // {from: 英文侧锚点位置, to: 中文侧锚点位置}，均递增
  var recomputeTimer = null;

  function relTop(el, container) {
    var r = el.getBoundingClientRect(), c = container.getBoundingClientRect();
    return r.top - c.top + container.scrollTop;
  }

  function collectZhAnchors() {
    var heads = zhRoot ? zhRoot.querySelectorAll('[data-sec]') : [];
    var list = [];
    for (var i = 0; i < heads.length; i++) {
      list.push({ key: heads[i].getAttribute('data-sec'), el: heads[i] });
    }
    return list;
  }

  function collectEnAnchors() {
    if (!enBody) return [];
    var heads = enBody.querySelectorAll('h2.ltx_title, h3.ltx_title, h4.ltx_title, h6.ltx_title, h2, h3');
    var list = [];
    for (var i = 0; i < heads.length; i++) {
      var key = enSecKey(heads[i]);
      if (key) list.push({ key: key, el: heads[i] });
    }
    return list;
  }

  function scrollable(el) {
    return el && (el.scrollHeight - el.clientHeight) > 50;
  }

  function recomputeAnchors() {
    pairs = [];
    if (!enBody || !zhRoot || !zhBody) return;
    // 窄屏堆叠布局下 pane 不独立滚动，此时不做同步
    if (!scrollable(enBody) || !scrollable(zhBody)) return;
    var en = collectEnAnchors(), zh = collectZhAnchors();
    var zhByKey = {};
    for (var i = 0; i < zh.length; i++) {
      if (!zhByKey[zh[i].key]) zhByKey[zh[i].key] = zh[i].el;
    }
    var usedZh = {};
    var lastFrom = 0, lastTo = 0;
    for (var j = 0; j < en.length; j++) {
      var k = en[j].key;
      var zhEl = zhByKey[k];
      if (!zhEl || usedZh[k]) continue;
      var from = relTop(en[j].el, enBody);
      var to = relTop(zhEl, zhBody);
      if (from < lastFrom || to < lastTo) continue;
      usedZh[k] = true;
      pairs.push({ from: from, to: to });
      lastFrom = from; lastTo = to;
    }
    if (!pairs.length || pairs[0].from > 400) pairs.unshift({ from: 0, to: 0 });
    var maxFrom = enBody.scrollHeight - enBody.clientHeight;
    var maxTo = zhBody.scrollHeight - zhBody.clientHeight;
    if (maxFrom > 0 && maxTo > 0) pairs.push({ from: maxFrom, to: maxTo });
    // 锚点对过少时同步意义不大，仅保留首尾
  }

  function scheduleRecompute() {
    if (recomputeTimer) clearTimeout(recomputeTimer);
    recomputeTimer = setTimeout(recomputeAnchors, 180);
  }

  /* rev=false: en→zh（from 为源）；rev=true: zh→en（to 为源） */
  function applySync(srcEl, dstEl, rev) {
    if (!pairs.length) return;
    var st = srcEl.scrollTop;
    var a = null, b = null;
    for (var i = 0; i < pairs.length; i++) {
      var pos = rev ? pairs[i].to : pairs[i].from;
      if (pos <= st + 2) { a = pairs[i]; b = pairs[i + 1] || null; }
      else break;
    }
    if (!a) { a = pairs[0]; b = pairs[1] || null; }
    var srcMax = srcEl.scrollHeight - srcEl.clientHeight;
    var dstMax = dstEl.scrollHeight - dstEl.clientHeight;
    var target;
    if (!b) {
      var ratio = srcMax > 0 ? st / srcMax : 0;
      target = ratio * dstMax;
    } else {
      var aSrc = rev ? a.to : a.from;
      var bSrc = rev ? b.to : b.from;
      var aDst = rev ? a.from : a.to;
      var bDst = rev ? b.from : b.to;
      var span = bSrc - aSrc;
      var prog = span > 0 ? Math.min(1, Math.max(0, (st - aSrc) / span)) : 0;
      target = aDst + prog * (bDst - aDst);
    }
    dstEl.scrollTop = target;
  }

  var lock = 0;
  function guardAndRun(fn, delay) {
    var now = Date.now();
    if (now - lock < 120) return;
    lock = now;
    if (syncToggle && !syncToggle.checked) return;
    if (delay) { setTimeout(fn, 60); } else { fn(); }
  }
  function bindSync(pane, fn) {
    pane.addEventListener('scroll', function () { guardAndRun(fn, false); }, { passive: true });
    // 补充：直接由用户输入事件触发（个别环境下程序化/惯性滚动的 scroll 事件可能缺失）
    ['wheel', 'touchmove', 'keydown', 'pointerdown'].forEach(function (type) {
      pane.addEventListener(type, function () { guardAndRun(fn, true); }, { passive: true });
    });
  }

  if (enBody && zhBody) {
    bindSync(enBody, function () { applySync(enBody, zhBody, false); });
    bindSync(zhBody, function () { applySync(zhBody, enBody, true); });

    /* 点击标题：另一侧跳到对应章节 */
    var enHeads = enBody.querySelectorAll('h2.ltx_title, h3.ltx_title, h4.ltx_title, h6.ltx_title, h2, h3');
    for (var i = 0; i < enHeads.length; i++) {
      (function (h) {
        h.addEventListener('click', function () {
          var key = enSecKey(h);
          if (!key || !zhRoot) return;
          var m = zhRoot.querySelector('[data-sec="' + key + '"]');
          if (m) zhBody.scrollTop = relTop(m, zhBody) - 8;
        });
      })(enHeads[i]);
    }
    var zhHeads = zhRoot ? zhRoot.querySelectorAll('[data-sec]') : [];
    for (var z = 0; z < zhHeads.length; z++) {
      (function (h) {
        h.addEventListener('click', function () {
          var key = h.getAttribute('data-sec');
          var en = collectEnAnchors();
          for (var k = 0; k < en.length; k++) {
            if (en[k].key === key) {
              enBody.scrollTop = relTop(en[k].el, enBody) - 8;
              break;
            }
          }
        });
      })(zhHeads[z]);
    }

    recomputeAnchors();
    scheduleRecompute();
    setTimeout(function () { fitEnTables(); recomputeAnchors(); }, 1500);
    if (enBody.addEventListener) {
      enBody.addEventListener('load', function () { scheduleRecompute(); }, true);
    }
    window.addEventListener('resize', function () {
      fitEnTables();
      scheduleRecompute();
    });
  }

  /* 调试接口 */
  window.__reader = {
    recompute: recomputeAnchors,
    getPairs: function () { return pairs; },
    apply: applySync
  };

  /* KaTeX 就绪后渲染公式（CDN 失败则保持 LaTeX 原文可读） */
  if (zhRoot) {
    var katexTries = 0;
    var katexTimer = setInterval(function () {
      katexTries++;
      if (window.renderMathInElement) {
        try {
          window.renderMathInElement(zhRoot, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false }
            ],
            throwOnError: false
          });
        } catch (e) { /* ignore */ }
        scheduleRecompute();
        clearInterval(katexTimer);
      } else if (katexTries > 150) {
        clearInterval(katexTimer);
      }
    }, 200);
    window.addEventListener('load', function () { scheduleRecompute(); });
  }
  /* ================= 图片灯箱：点击放大 · 滚轮缩放 · 拖动平移 ================= */
  (function setupLightbox() {
    if (!document.body) return;
    var box = document.createElement('div');
    box.className = 'lightbox';
    box.innerHTML =
      '<div class="lb-close" title="关闭 (Esc)">&times;</div>' +
      '<div class="lb-zoom"></div>' +
      '<img class="lb-img" alt="">' +
      '<div class="lb-cap"></div>' +
      '<div class="lb-hint">滚轮缩放 · 拖动平移 · 点击图片或按 Esc 关闭</div>';
    document.body.appendChild(box);
    var img = box.querySelector('.lb-img');
    var cap = box.querySelector('.lb-cap');
    var zoomEl = box.querySelector('.lb-zoom');
    var scale = 1, tx = 0, ty = 0;
    var dragging = false, moved = false, sx = 0, sy = 0;

    function apply() {
      img.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
      zoomEl.textContent = Math.round(scale * 100) + '%';
    }
    function reset() { scale = 1; tx = 0; ty = 0; apply(); }
    function open(src, caption) {
      reset();
      img.src = src;
      if (caption) { cap.textContent = caption; cap.style.display = ''; }
      else { cap.textContent = ''; cap.style.display = 'none'; }
      box.classList.add('open');
    }
    function close() {
      box.classList.remove('open');
      img.src = '';
    }

    /* 关闭：× 按钮、点击背景、点击图片（非拖动）、Esc */
    box.querySelector('.lb-close').addEventListener('click', close);
    box.addEventListener('click', function (e) {
      if (e.target === img && moved) return;
      close();
    });
    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && box.classList.contains('open')) close();
    });

    /* 滚轮缩放 */
    box.addEventListener('wheel', function (e) {
      e.preventDefault();
      var f = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      var next = Math.min(8, Math.max(0.25, scale * f));
      /* 以鼠标位置为缩放中心 */
      var rect = img.getBoundingClientRect();
      var cx = (e.clientX - rect.left - rect.width / 2) / scale;
      var cy = (e.clientY - rect.top - rect.height / 2) / scale;
      tx -= cx * (next - scale);
      ty -= cy * (next - scale);
      scale = next;
      apply();
    }, { passive: false });

    /* 拖动平移 */
    img.addEventListener('mousedown', function (e) {
      dragging = true; moved = false;
      sx = e.clientX - tx; sy = e.clientY - ty;
      e.preventDefault();
    });
    window.addEventListener('mousemove', function (e) {
      if (!dragging) return;
      tx = e.clientX - sx; ty = e.clientY - sy;
      if (Math.abs(tx) + Math.abs(ty) > 4) moved = true;
      apply();
    });
    window.addEventListener('mouseup', function () { dragging = false; });

    /* 触屏：单指拖动 */
    img.addEventListener('touchstart', function (e) {
      if (e.touches.length === 1) {
        dragging = true; moved = false;
        sx = e.touches[0].clientX - tx; sy = e.touches[0].clientY - ty;
      }
    }, { passive: true });
    img.addEventListener('touchmove', function (e) {
      if (dragging && e.touches.length === 1) {
        tx = e.touches[0].clientX - sx; ty = e.touches[0].clientY - sy;
        moved = true; apply();
      }
    }, { passive: true });
    img.addEventListener('touchend', function () { dragging = false; });

    /* 事件委托：点击文档内任意图片打开灯箱 */
    document.addEventListener('click', function (e) {
      var t = e.target;
      if (!t || t.tagName !== 'IMG') return;
      if (t.classList.contains('lb-img') || t.closest('.lightbox')) return;
      var src = t.getAttribute('src');
      if (!src) return;
      var caption = '';
      var fig = t.closest('figure, .ltx_figure, .ltx_float');
      if (fig) {
        var capEl = fig.querySelector('.ltx_caption, figcaption');
        if (capEl) caption = capEl.textContent.trim();
      }
      e.preventDefault();
      open(src, caption);
    });
  })();
})();
