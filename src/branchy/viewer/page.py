HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>branchy tree</title>
<style>
  :root { --line: #d0d7de; --amber: #b54708; --green: #1a7f37; --gray: #57606a; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0;
         color: #1f2328; background: #fff; }
  header { display: flex; gap: 16px; align-items: center; padding: 8px 14px;
           border-bottom: 1px solid var(--line); background: #f6f8fa;
           position: sticky; top: 0; }
  header h1 { font-size: 15px; margin: 0; }
  header .meta { color: var(--gray); font-size: 12px; }
  header label { font-size: 12px; display: flex; align-items: center; gap: 4px; }
  header button { font-size: 12px; padding: 2px 10px; cursor: pointer; }
  main { display: flex; height: calc(100vh - 41px); }
  #tree-pane { width: 46%; overflow: auto; padding: 10px 8px;
               border-right: 1px solid var(--line); }
  #detail-pane { flex: 1; overflow: auto; padding: 12px 16px;
                 font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
  ul.nodes { list-style: none; margin: 0; padding-left: 18px;
             border-left: 1px solid var(--line); }
  ul.nodes.root { padding-left: 0; border-left: none; }
  li.node { margin: 3px 0; }
  .box { display: inline-block; max-width: 100%; padding: 3px 8px;
         border: 1px solid var(--line); border-radius: 6px; cursor: pointer;
         background: #fff; vertical-align: top; }
  .box:hover { border-color: #8c959f; }
  .box.selected { border-color: #0969da; box-shadow: 0 0 0 1px #0969da; }
  .box .caret { display: inline-block; width: 12px; color: var(--gray);
                user-select: none; }
  .badge { display: inline-block; font-size: 10px; padding: 0 5px;
           border-radius: 8px; margin-right: 4px; border: 1px solid var(--line); }
  .badge.role-u { background: #ddf4ff; }
  .badge.role-a { background: #dafbe1; }
  .badge.tentative { background: #fff8c5; color: var(--amber);
                     border-color: #eac54f; }
  .badge.fork { background: #fbefff; }
  .badge.active { background: #dafbe1; color: var(--green);
                  border-color: #4ac26b; }
  .badge.root { background: #eaeef2; }
  .snippet { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
             max-width: 480px; display: inline-block; vertical-align: bottom; }
  li.node.collapsed > ul.nodes { display: none; }
  #detail-pane h3 { font-family: inherit; font-size: 13px; margin: 14px 0 4px;
                    color: var(--gray); text-transform: uppercase;
                    letter-spacing: .05em; }
  #detail-pane table { border-collapse: collapse; font-size: 12px; }
  #detail-pane td, #detail-pane th { border: 1px solid var(--line);
                                     padding: 3px 8px; text-align: left;
                                     vertical-align: top; }
  #detail-pane pre { background: #f6f8fa; border: 1px solid var(--line);
                     border-radius: 6px; padding: 8px; overflow-x: auto;
                     white-space: pre-wrap; margin: 4px 0; }
  .pair-u { border-left: 3px solid #54aeff; padding-left: 8px; }
  .pair-a { border-left: 3px solid #4ac26b; padding-left: 8px; }
</style>
</head>
<body data-refresh="__REFRESH_SECONDS__">
<header>
  <h1>branchy tree</h1>
  <span class="meta" id="meta"></span>
  <label id="autorefresh-label"><input type="checkbox" id="autorefresh" checked>
    auto-refresh</label>
  <button id="refresh-btn">Refresh</button>
  <button id="expand-all">Expand all</button>
  <button id="collapse-all">Collapse all</button>
</header>
<main>
  <div id="tree-pane"></div>
  <div id="detail-pane"><p>Select a node.</p></div>
</main>
<script>
(function () {
  "use strict";
  var inline = document.getElementById("snapshot");
  var staticMode = !!inline;
  var state = { collapsed: new Set(), selected: null, version: null,
                snap: null, etag: null };
  var refreshMs = parseInt(document.body.getAttribute("data-refresh") || "0", 10) * 1000;

  if (staticMode) {
    document.getElementById("autorefresh-label").style.display = "none";
    document.getElementById("refresh-btn").style.display = "none";
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function fetchSnapshot(cb) {
    if (staticMode) {
      cb(JSON.parse(inline.textContent), null);
      return;
    }
    var opts = state.etag ? { headers: { "If-None-Match": state.etag } } : {};
    fetch("api/tree", opts).then(function (r) {
      if (r.status === 304) { cb(null, state.etag); return null; }
      state.etag = r.headers.get("ETag");
      return r.json();
    }).then(function (snap) { if (snap) cb(snap, state.etag); });
  }

  function allTrees(snap) {
    if (!snap) return [];
    if (snap.trees) return snap.trees;
    return [snap];  // single-tree shape tolerated
  }

  function nodeById(snap, id) {
    var trees = allTrees(snap);
    for (var t = 0; t < trees.length; t++)
      for (var i = 0; i < trees[t].nodes.length; i++)
        if (trees[t].nodes[i].message_id === id) return trees[t].nodes[i];
    return null;
  }

  function treeOf(snap, node) {
    var trees = allTrees(snap);
    for (var t = 0; t < trees.length; t++)
      for (var i = 0; i < trees[t].nodes.length; i++)
        if (trees[t].nodes[i].message_id === node.message_id) return trees[t];
    return null;
  }

  function pathToRoot(snap, node) {
    var tree = treeOf(snap, node);
    var byId = {};
    if (tree) tree.nodes.forEach(function (n) { byId[n.message_id] = n; });
    var path = [], cur = node;
    while (cur) {
      path.unshift(cur);
      cur = cur.parent_message_id ? byId[cur.parent_message_id] : null;
    }
    return path;
  }

  function render() {
    var snap = state.snap;
    var trees = allTrees(snap);
    var totalNodes = 0, totalVersion = 0;
    trees.forEach(function (t) {
      totalNodes += t.nodes.length;
      totalVersion += t.version || 0;
    });
    document.getElementById("meta").textContent =
      "version " + totalVersion + " · " + totalNodes + " nodes" +
      (trees.length > 1 ? " · " + trees.length + " trees" : "");
    var roots = [];
    trees.forEach(function (tree) {
      var byParent = {};
      tree.nodes.forEach(function (n) {
        var p = n.parent_message_id || "";
        (byParent[p] = byParent[p] || []).push(n);
      });
      (byParent[""] || []).forEach(function (r) {
        roots.push({ tree: tree, byParent: byParent, node: r });
      });
    });

    function li(tree, byParent, n) {
      var kids = byParent[n.message_id] || [];
      var key = (tree.tree_id || "default") + ":" + n.message_id;
      var collapsed = state.collapsed.has(key);
      var badges = '<span class="badge role-' +
        (n.node_type === "user" ? "u\">U" : "a\">A") + "</span>";
      if (n.message_id === tree.active_root_id)
        badges += '<span class="badge active">active</span>';
      if (!n.parent_message_id)
        badges += '<span class="badge root">root</span>';
      if (n.tentative)
        badges += '<span class="badge tentative">tentative</span>';
      if (kids.length > 1)
        badges += '<span class="badge fork">fork</span>';
      var caret = kids.length
        ? '<span class="caret">' + (collapsed ? "▸" : "▾") + "</span>"
        : '<span class="caret"></span>';
      var text = n.content.length > 80 ? n.content.slice(0, 80) + "…" : n.content;
      var html = '<li class="node' + (collapsed ? " collapsed" : "") +
        '" data-key="' + key + '" data-id="' + n.message_id + '"><div class="box' +
        (state.selected === n.message_id ? " selected" : "") + '" data-key="' +
        key + '" data-id="' +
        n.message_id + '">' + caret + badges +
        '<span class="snippet">' + esc(text) + "</span></div>";
      if (kids.length) {
        html += '<ul class="nodes">';
        kids.forEach(function (k) { html += li(tree, byParent, k); });
        html += "</ul>";
      }
      return html + "</li>";
    }

    var html = '<ul class="nodes root"><li class="node"><div class="box">' +
      '<span class="badge root">All conversations</span></div><ul class="nodes">';
    var grouped = {};
    var order = [];
    roots.forEach(function (r) {
      var tid = r.tree.tree_id || "default";
      if (!grouped[tid]) { grouped[tid] = []; order.push(tid); }
      grouped[tid].push(r);
    });
    order.forEach(function (tid) {
      if (order.length > 1)
        html += '<li class="node"><div class="box"><span class="badge root">' +
          "tree: " + esc(tid) + " · " + grouped[tid][0].tree.nodes.length +
          " nodes</span></div>";
      grouped[tid].forEach(function (r) {
        html += li(r.tree, r.byParent, r.node);
      });
      if (order.length > 1) html += "</li>";
    });
    html += "</ul></li></ul>";
    document.getElementById("tree-pane").innerHTML = html;

    document.querySelectorAll("#tree-pane .box").forEach(function (box) {
      box.addEventListener("click", function (e) {
        e.stopPropagation();
        var id = box.getAttribute("data-id");
        var key = box.getAttribute("data-key");
        if (!id) return;
        if (e.target.classList.contains("caret")) {
          if (state.collapsed.has(key)) state.collapsed.delete(key);
          else state.collapsed.add(key);
          render();
          return;
        }
        state.selected = id;
        render();
        renderDetail();
      });
    });
    if (state.selected) renderDetail();
  }

  function kvTable(obj) {
    var rows = "";
    Object.keys(obj).forEach(function (k) {
      var v = obj[k];
      if (v === null || v === undefined) v = "";
      if (typeof v === "object") v = JSON.stringify(v);
      rows += "<tr><td>" + esc(k) + "</td><td>" + esc(v) + "</td></tr>";
    });
    return "<table>" + rows + "</table>";
  }

  function renderDetail() {
    var snap = state.snap, id = state.selected;
    var node = id && nodeById(snap, id);
    var pane = document.getElementById("detail-pane");
    if (!node) { pane.innerHTML = "<p>Select a node.</p>"; return; }

    var html = "";
    // Prompt/response pairing
    if (node.node_type === "assistant") {
      var pu = nodeById(snap, node.parent_message_id);
      if (pu) html += "<h3>Prompt</h3><pre class='pair-u'>" +
        esc(pu.content) + "</pre>";
      html += "<h3>Assistant</h3><pre class='pair-a'>" +
        esc(node.content) + "</pre>";
    } else {
      html += "<h3>User</h3><pre class='pair-u'>" + esc(node.content) + "</pre>";
      var kids = node.child_message_ids || [];
      if (kids.length) {
        var child = nodeById(snap, kids[0]);
        if (child) html += "<h3>Response</h3><pre class='pair-a'>" +
          esc(child.content) + "</pre>";
      }
    }

    html += "<h3>Metadata</h3>" + kvTable({
      message_id: node.message_id, node_type: node.node_type,
      parent_message_id: node.parent_message_id,
      child_message_ids: node.child_message_ids,
      created_seq: node.created_seq, created_at: node.created_at,
      tentative: node.tentative,
    });

    if (node.node_type === "user" && node.routing) {
      var r = node.routing;
      html += "<h3>Routing</h3>" + kvTable({
        action: r.action, target_message_id: r.target_message_id,
        margin: r.margin, dominated_root_ids: r.dominated_root_ids,
        reason: r.reason,
      });
      if (r.candidates && r.candidates.length) {
        html += "<h3>Candidates</h3><table><tr><th>node</th><th>kind</th>" +
          "<th>bm25</th><th>recency</th><th>s1</th><th>ev</th></tr>";
        r.candidates.forEach(function (c) {
          function f(v) { return v === null || v === undefined ? "—" :
            (typeof v === "number" ? v.toFixed(3) : v); }
          html += "<tr><td>" + esc(String(c.node_id).slice(0, 8)) + "</td><td>" +
            c.kind + "</td><td>" + f(c.bm25) +
            "</td><td>" + f(c.recency) + "</td><td>" + f(c.s1) + "</td><td>" +
            c.evidence + "</td></tr>";
        });
        html += "</table>";
      }
    }

    var path = pathToRoot(snap, node);
    var rootId = path.length ? path[0].message_id : null;
    var sticker = null;
    (snap.stickers || []).forEach(function (s) {
      if (s.root_id === rootId) sticker = s;
    });
    if (sticker)
      html += "<h3>Tower sticker</h3><pre>" + esc(sticker.text) +
        "</pre><h3>Sticker meta</h3>" + kvTable({
          root_id: sticker.root_id, last_used_seq: sticker.last_used_seq,
        });

    html += "<h3>Path to root</h3><ol>";
    path.forEach(function (n) {
      html += "<li>[" + (n.node_type === "user" ? "U" : "A") + "] " +
        esc(n.content) + "</li>";
    });
    html += "</ol>";

    // conversation from here: full subtree under the clicked node
    var tree = treeOf(snap, node);
    if (tree) {
      var byId = {};
      tree.nodes.forEach(function (n) { byId[n.message_id] = n; });
      var stats = { nodes: 0, leaves: 0, depth: 0 };
      function sub(n, depth) {
        stats.nodes++;
        if (depth > stats.depth) stats.depth = depth;
        var kids = (n.child_message_ids || [])
          .map(function (c) { return byId[c]; }).filter(Boolean);
        if (!kids.length) stats.leaves++;
        var s = "<li>[" + (n.node_type === "user" ? "U" : "A") + "] " +
          esc(n.content) + (n.tentative ? " <em>(tentative)</em>" : "");
        if (kids.length) {
          s += "<ul>";
          kids.forEach(function (k) { s += sub(k, depth + 1); });
          s += "</ul>";
        }
        return s + "</li>";
      }
      html += "<h3>Conversation from here</h3><p>" +
        "<span id='subtree-count'></span></p><ul class='nodes'>" +
        sub(node, 1) + "</ul>";
      html = html.replace(
        "<span id='subtree-count'></span>",
        stats.nodes + " nodes, " + stats.leaves + " leaves, depth " +
        stats.depth
      );
    }

    if (node.node_type === "user") {
      var history = "(none - this message started a new thread)";
      if (node.parent_message_id) {
        var pp = pathToRoot(snap, nodeById(snap, node.parent_message_id));
        history = pp.map(function (n) {
          return (n.node_type === "user" ? "User: " : "Assistant: ") + n.content;
        }).join("\n");
      }
      html += "<h3>History sent to LLM</h3><pre>" + esc(history) + "</pre>";
    }

    pane.innerHTML = html;
  }

  function poll() {
    fetchSnapshot(function (snap) {
      if (!snap) return;  // 304: no change
      var changed = snap.version !== state.version;
      state.snap = snap;
      if (changed || state.version === null) {
        state.version = snap.version;
        render();
      }
    });
  }

  document.getElementById("refresh-btn").addEventListener("click", poll);
  document.getElementById("expand-all").addEventListener("click", function () {
    state.collapsed.clear(); render();
  });
  document.getElementById("collapse-all").addEventListener("click", function () {
    allTrees(state.snap).forEach(function (tree) {
      tree.nodes.forEach(function (n) {
        if ((n.child_message_ids || []).length)
          state.collapsed.add((tree.tree_id || "default") + ":" + n.message_id);
      });
    });
    render();
  });

  poll();
  if (window.location.hash.length > 1)
    state.selected = decodeURIComponent(window.location.hash.slice(1));
  if (!staticMode && refreshMs > 0) {
    setInterval(function () {
      if (document.getElementById("autorefresh").checked) poll();
    }, refreshMs);
  }
})();
</script>
</body>
</html>
"""
