// src/octto/ui/bundle.ts

/**
 * Returns the bundled HTML for the octto UI.
 * Uses nof1 design system - IBM Plex Mono, terminal aesthetic.
 */
export function getHtmlBundle(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Octto</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --background: #ffffff;
      --surface: #ffffff;
      --surface-elevated: #f8f9fa;
      --surface-hover: #f1f3f4;
      --foreground: #000000;
      --foreground-muted: #333333;
      --foreground-subtle: #666666;
      --border: #000000;
      --border-subtle: #cccccc;
      --accent-success: #00aa00;
      --accent-error: #ff0000;
    }

    *, *:before, *:after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html, body {
      height: 100%;
      background: var(--background);
      color: var(--foreground);
      font-family: 'IBM Plex Mono', monospace;
      font-size: 14px;
      line-height: 1.5;
      letter-spacing: -0.02em;
    }

    body {
      position: relative;
    }

    body::before {
      content: "";
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.6' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.15'/%3E%3C/svg%3E");
      background-size: 180px 180px;
      pointer-events: none;
      z-index: 1;
    }

    #root {
      position: relative;
      z-index: 2;
      max-width: 640px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      min-height: 100vh;
    }

    h1, h2, h3 {
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .header {
      text-align: center;
      padding: 3rem 0;
    }

    .header h1 {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
    }

    .header p {
      color: var(--foreground-subtle);
      font-size: 0.875rem;
    }

    .spinner {
      width: 24px;
      height: 24px;
      border: 2px solid var(--border-subtle);
      border-top-color: var(--foreground);
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 1.5rem auto 0;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 1.5rem;
      margin-bottom: 1rem;
    }

    .card-answered {
      background: var(--surface-elevated);
      border-color: var(--border-subtle);
      opacity: 0.7;
      padding: 1rem;
      cursor: pointer;
      transition: opacity 0.15s;
    }

    .card-answered:hover {
      opacity: 0.85;
    }

    .card-answered.expanded {
      opacity: 1;
      cursor: default;
    }

    .card-answered .check {
      color: var(--accent-success);
      margin-right: 0.5rem;
    }

    .card-answered-header {
      display: flex;
      align-items: center;
      cursor: pointer;
    }

    .card-answered-header .toggle {
      margin-left: auto;
      color: var(--foreground-subtle);
      font-size: 0.75rem;
    }

    .card-answered-body {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border-subtle);
    }

    .readonly-answer {
      background: var(--surface-hover);
      padding: 0.75rem;
      margin-top: 0.5rem;
      font-size: 0.875rem;
    }

    .readonly-answer-label {
      font-size: 0.6875rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--foreground-subtle);
      margin-bottom: 0.25rem;
    }

    .readonly-option {
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border-subtle);
      margin-bottom: 0.25rem;
      opacity: 0.6;
    }

    .readonly-option.selected {
      opacity: 1;
      border-color: var(--accent-success);
      background: rgba(0, 170, 0, 0.05);
    }

    .readonly-option .check-mark {
      color: var(--accent-success);
      margin-right: 0.5rem;
    }

    .question-text {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
      line-height: 1.4;
    }

    .context {
      color: var(--foreground-muted);
      font-size: 0.875rem;
      margin-bottom: 1rem;
      padding-left: 1rem;
      border-left: 2px solid var(--border-subtle);
    }

    .options {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .option {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      padding: 0.75rem;
      border: 1px solid var(--border-subtle);
      cursor: pointer;
      transition: none;
    }

    .option:hover {
      background: var(--surface-hover);
      border-color: var(--border);
    }

    .option.recommended {
      border-color: var(--border);
      background: var(--surface-elevated);
    }

    .option input {
      margin-top: 0.125rem;
      accent-color: var(--foreground);
    }

    .option-content {
      flex: 1;
    }

    .option-label {
      font-weight: 500;
    }

    .option-desc {
      font-size: 0.8125rem;
      color: var(--foreground-subtle);
      margin-top: 0.25rem;
    }

    .option-tag {
      font-size: 0.6875rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--foreground-muted);
      margin-left: 0.5rem;
    }

    .btn {
      display: inline-block;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--foreground);
      font-family: 'IBM Plex Mono', monospace;
      font-weight: 500;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.5rem 1rem;
      cursor: pointer;
      transition: none;
    }

    .btn:hover {
      background: var(--surface-hover);
    }

    .btn:active {
      background: var(--foreground);
      color: var(--background);
    }

    .btn-primary {
      background: var(--foreground);
      color: var(--background);
    }

    .btn-primary:hover {
      opacity: 0.9;
    }

    .btn-success {
      border-color: var(--accent-success);
      color: var(--accent-success);
    }

    .btn-success:hover {
      background: var(--accent-success);
      color: var(--background);
    }

    .btn-danger {
      border-color: var(--accent-error);
      color: var(--accent-error);
    }

    .btn-danger:hover {
      background: var(--accent-error);
      color: var(--background);
    }

    .btn-group {
      display: flex;
      gap: 0.5rem;
      margin-top: 1.25rem;
    }

    .input, .textarea {
      width: 100%;
      padding: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--foreground);
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.875rem;
    }

    .input:focus, .textarea:focus {
      outline: none;
      border-color: var(--foreground);
    }

    .textarea {
      resize: vertical;
      min-height: 100px;
    }

    .slider-container {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .slider-container input[type="range"] {
      flex: 1;
      height: 2px;
      background: var(--border-subtle);
      appearance: none;
      -webkit-appearance: none;
    }

    .slider-container input[type="range"]::-webkit-slider-thumb {
      appearance: none;
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      background: var(--foreground);
      cursor: pointer;
    }

    .slider-value {
      font-weight: 600;
      min-width: 3rem;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }

    .slider-labels {
      color: var(--foreground-subtle);
      font-size: 0.75rem;
    }

    .thumbs-container {
      display: flex;
      gap: 1rem;
    }

    .thumb-btn {
      font-size: 2rem;
      padding: 1rem 1.5rem;
      border: 1px solid var(--border-subtle);
      background: var(--surface);
      cursor: pointer;
    }

    .thumb-btn:hover {
      border-color: var(--border);
      background: var(--surface-hover);
    }

    .queue-indicator {
      text-align: center;
      color: var(--foreground-subtle);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 1rem;
    }

    .branch-subtitle {
      font-size: 0.75rem;
      color: var(--foreground-subtle);
      margin-top: 0.25rem;
      margin-bottom: 0.75rem;
    }

    .thinking {
      text-align: center;
      padding: 2rem;
      margin-top: 2rem;
      margin-bottom: 2rem;
      border: 1px dashed var(--border-subtle);
    }

    .thinking-text {
      color: var(--foreground-subtle);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 1rem;
    }

    .thinking .spinner {
      margin: 0 auto;
    }

    .review-content {
      background: var(--surface-elevated);
      border: 1px solid var(--border-subtle);
      padding: 1rem;
      margin-bottom: 1rem;
      font-size: 0.875rem;
      line-height: 1.6;
      max-height: 400px;
      overflow-y: auto;
    }

    .review-content h1, .review-content h2, .review-content h3,
    .review-content h4, .review-content h5, .review-content h6 {
      font-weight: 600;
      margin: 1rem 0 0.5rem 0;
    }

    .review-content h1 { font-size: 1.25rem; }
    .review-content h2 { font-size: 1.125rem; }
    .review-content h3 { font-size: 1rem; }

    .review-content p {
      margin: 0.5rem 0;
    }

    .review-content ul, .review-content ol {
      margin: 0.5rem 0;
      padding-left: 1.5rem;
    }

    .review-content li {
      margin: 0.25rem 0;
    }

    .review-content code {
      background: var(--surface-hover);
      padding: 0.125rem 0.25rem;
      font-size: 0.8125rem;
    }

    .review-content pre {
      background: var(--surface-hover);
      padding: 0.75rem;
      overflow-x: auto;
      margin: 0.5rem 0;
    }

    .review-content pre code {
      background: none;
      padding: 0;
    }

    .review-content blockquote {
      border-left: 2px solid var(--border);
      padding-left: 1rem;
      margin: 0.5rem 0;
      color: var(--foreground-muted);
    }

    .feedback-input {
      margin-top: 1rem;
    }

    .feedback-input label {
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--foreground-subtle);
      margin-bottom: 0.5rem;
    }

    .plan-section {
      margin-bottom: 1.5rem;
    }

    .plan-section-title {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      padding-bottom: 0.25rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    .session-ended {
      text-align: center;
      padding: 4rem 0;
    }

    .session-ended h1 {
      margin-bottom: 0.5rem;
    }

    .session-ended p {
      color: var(--foreground-subtle);
    }

    /* Show Options */
    .options-with-pros {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .option-card {
      border: 1px solid var(--border-subtle);
      padding: 1rem;
    }

    .option-card.recommended {
      border-color: var(--border);
      background: var(--surface-elevated);
    }

    .option-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }

    .pros, .cons {
      font-size: 0.8125rem;
      margin-top: 0.5rem;
    }

    .pros { color: var(--accent-success); }
    .cons { color: var(--accent-error); }

    .pros ul, .cons ul {
      margin: 0.25rem 0 0 1rem;
    }

    /* Show Diff */
    .diff-filepath {
      font-size: 0.75rem;
      color: var(--foreground-subtle);
      margin-bottom: 0.5rem;
      font-family: monospace;
    }

    .diff-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }

    .diff-side {
      border: 1px solid var(--border-subtle);
      overflow: auto;
      max-height: 300px;
    }

    .diff-label {
      font-size: 0.6875rem;
      text-transform: uppercase;
      padding: 0.25rem 0.5rem;
      background: var(--surface-elevated);
      border-bottom: 1px solid var(--border-subtle);
    }

    .diff-before .diff-label { color: var(--accent-error); }
    .diff-after .diff-label { color: var(--accent-success); }

    .diff-side pre {
      margin: 0;
      padding: 0.5rem;
      font-size: 0.75rem;
      white-space: pre-wrap;
    }

    /* Rank */
    .rank-list {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .rank-item {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border-subtle);
      background: var(--surface);
      cursor: grab;
    }

    .rank-item:active, .rank-item.dragging {
      cursor: grabbing;
      opacity: 0.5;
    }

    .rank-handle {
      color: var(--foreground-subtle);
    }

    .rank-num {
      font-weight: 600;
      min-width: 1.5rem;
    }

    /* Rate */
    .rate-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .rate-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .rate-stars {
      display: flex;
      gap: 0.25rem;
    }

    .rate-star {
      width: 2rem;
      height: 2rem;
      border: 1px solid var(--border-subtle);
      background: var(--surface);
      cursor: pointer;
      font-size: 0.75rem;
    }

    .rate-star.selected {
      background: var(--foreground);
      color: var(--background);
      border-color: var(--foreground);
    }

    /* Code Input */
    .code-input {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.8125rem;
    }

    .code-input-label {
      font-size: 0.6875rem;
      text-transform: uppercase;
      color: var(--foreground-subtle);
      margin-bottom: 0.25rem;
    }

    /* File Upload */
    .file-upload {
      margin-bottom: 1rem;
    }

    .file-upload input[type="file"] {
      width: 100%;
      padding: 0.5rem;
      border: 1px dashed var(--border-subtle);
    }

    .image-preview {
      display: flex;
      flex-wrap: wrap;
      margin-top: 0.5rem;
    }

    /* Emoji React */
    .emoji-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .emoji-btn {
      font-size: 2rem;
      padding: 0.75rem;
      border: 1px solid var(--border-subtle);
      background: var(--surface);
      cursor: pointer;
    }

    .emoji-btn:hover {
      background: var(--surface-hover);
      border-color: var(--border);
    }

    /* Keyboard focus styles */
    .thumb-btn:focus,
    .emoji-btn:focus,
    .rate-star:focus,
    .btn:focus {
      outline: 2px solid var(--foreground);
      outline-offset: 2px;
    }
  </style>
</head>
<body>
  <div id="root">
    <div class="header">
      <h1>Octto</h1>
      <p>Connecting to session...</p>
      <div class="spinner"></div>
    </div>
  </div>

  <script>
    const wsUrl = 'ws://' + window.location.host + '/ws';
    let ws = null;
    let questions = [];
    let expandedAnswers = new Set();

    function connect() {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'connected' }));
        render();
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'question') {
          questions.push(msg);
          render();
        } else if (msg.type === 'cancel') {
          questions = questions.filter(q => q.id !== msg.id);
          render();
        } else if (msg.type === 'end') {
          document.getElementById('root').innerHTML =
            '<div class="session-ended"><h1>Session Ended</h1><p>You can close this window.</p></div>';
        }
      };

      ws.onclose = () => {
        setTimeout(connect, 2000);
      };
    }

    function render() {
      const root = document.getElementById('root');

      if (questions.length === 0) {
        root.innerHTML = '<div class="header"><h1>Octto</h1><p>Waiting for questions...</p></div>';
        return;
      }

      const pending = questions.filter(q => !q.answered);
      const answered = questions.filter(q => q.answered);

      let html = '';

      // Show remaining count at top
      if (pending.length > 1) {
        html += '<div class="queue-indicator" style="margin-top: 0; margin-bottom: 1rem;">' + (pending.length - 1) + ' more question(s) remaining</div>';
      }

      // Show current question
      if (pending.length > 0) {
        const q = pending[0];
        html += renderQuestion(q);
      } else if (answered.length > 0) {
        // All answered, waiting for more questions
        html += '<div class="thinking">';
        html += '<div class="thinking-text">Thinking...</div>';
        html += '<div class="spinner"></div>';
        html += '</div>';
      }

      // Show answered questions at bottom (collapsed or expanded)
      for (const q of answered) {
        const isExpanded = expandedAnswers.has(q.id);
        // Extract branch name from context
        let branchName = '';
        const ctx = q.config.context || '';
        const branchMatch = ctx.match(/^\\[([^\\]]+)\\]/);
        if (branchMatch) branchName = branchMatch[1];

        html += '<div class="card card-answered' + (isExpanded ? ' expanded' : '') + '" data-qid="' + q.id + '">';
        html += '<div class="card-answered-header" onclick="toggleAnswered(\\'' + q.id + '\\')">';
        html += '<span class="check">[OK]</span>';
        html += '<div style="flex: 1;">';
        html += '<span>' + escapeHtml(q.config.question) + '</span>';
        if (branchName) html += '<div class="branch-subtitle" style="margin-bottom: 0; margin-top: 0.125rem;">' + escapeHtml(branchName) + '</div>';
        html += '</div>';
        html += '<span class="toggle">' + (isExpanded ? '\\u25B2 collapse' : '\\u25BC view') + '</span>';
        html += '</div>';
        if (isExpanded) {
          html += '<div class="card-answered-body">';
          html += renderAnsweredQuestion(q);
          html += '</div>';
        }
        html += '</div>';
      }

      root.innerHTML = html;
      attachListeners();
    }

    function renderQuestion(q) {
      const config = q.config;
      let html = '<div class="card">';

      // Extract branch from context if present: "[Branch Scope] rest of context"
      let branchName = '';
      let remainingContext = config.context || '';
      const branchMatch = remainingContext.match(/^\\[([^\\]]+)\\]\\s*/);
      if (branchMatch) {
        branchName = branchMatch[1];
        remainingContext = remainingContext.substring(branchMatch[0].length);
      }

      html += '<div class="question-text">' + escapeHtml(config.question) + '</div>';
      if (branchName) {
        html += '<div class="branch-subtitle">' + escapeHtml(branchName) + '</div>';
      }
      if (remainingContext) {
        html += '<div class="context">' + escapeHtml(remainingContext) + '</div>';
      }

      switch (q.questionType) {
        case 'pick_one':
          html += renderPickOne(q);
          break;
        case 'pick_many':
          html += renderPickMany(q);
          break;
        case 'confirm':
          html += renderConfirm(q);
          break;
        case 'ask_text':
          html += renderAskText(q);
          break;
        case 'thumbs':
          html += renderThumbs(q);
          break;
        case 'slider':
          html += renderSlider(q);
          break;
        case 'review_section':
          html += renderReviewSection(q);
          break;
        case 'show_plan':
          html += renderShowPlan(q);
          break;
        case 'show_options':
          html += renderShowOptions(q);
          break;
        case 'show_diff':
          html += renderShowDiff(q);
          break;
        case 'rank':
          html += renderRank(q);
          break;
        case 'rate':
          html += renderRate(q);
          break;
        case 'ask_code':
          html += renderAskCode(q);
          break;
        case 'ask_image':
          html += renderAskImage(q);
          break;
        case 'ask_file':
          html += renderAskFile(q);
          break;
        case 'emoji_react':
          html += renderEmojiReact(q);
          break;
        default:
          html += '<p>Question type "' + q.questionType + '" not yet implemented.</p>';
          html += '<div class="btn-group"><button onclick="submitAnswer(\\'' + q.id + '\\', {})" class="btn">Skip</button></div>';
      }

      html += '</div>';
      return html;
    }

    function renderPickOne(q) {
      const options = q.config.options || [];
      let html = '<div class="options">';
      for (const opt of options) {
        const isRecommended = q.config.recommended === opt.id;
        html += '<label class="option' + (isRecommended ? ' recommended' : '') + '">';
        html += '<input type="radio" name="pick_' + q.id + '" value="' + opt.id + '">';
        html += '<div class="option-content">';
        html += '<div class="option-label">' + escapeHtml(opt.label);
        if (isRecommended) html += '<span class="option-tag">(recommended)</span>';
        html += '</div>';
        if (opt.description) html += '<div class="option-desc">' + escapeHtml(opt.description) + '</div>';
        html += '</div></label>';
      }
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitPickOne(\\'' + q.id + '\\')" class="btn btn-primary">Submit</button></div>';
      return html;
    }

    function renderPickMany(q) {
      const options = q.config.options || [];
      let html = '<div class="options">';
      for (const opt of options) {
        html += '<label class="option">';
        html += '<input type="checkbox" name="pick_' + q.id + '" value="' + opt.id + '">';
        html += '<div class="option-content">';
        html += '<div class="option-label">' + escapeHtml(opt.label) + '</div>';
        if (opt.description) html += '<div class="option-desc">' + escapeHtml(opt.description) + '</div>';
        html += '</div></label>';
      }
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitPickMany(\\'' + q.id + '\\')" class="btn btn-primary">Submit</button></div>';
      return html;
    }

    function renderConfirm(q) {
      const yesLabel = q.config.yesLabel || 'Yes';
      const noLabel = q.config.noLabel || 'No';
      let html = '<div class="btn-group">';
      html += '<button onclick="submitAnswer(\\'' + q.id + '\\', {choice: \\'yes\\'})" class="btn btn-success">' + escapeHtml(yesLabel) + '</button>';
      html += '<button onclick="submitAnswer(\\'' + q.id + '\\', {choice: \\'no\\'})" class="btn btn-danger">' + escapeHtml(noLabel) + '</button>';
      if (q.config.allowCancel) {
        html += '<button onclick="submitAnswer(\\'' + q.id + '\\', {choice: \\'cancel\\'})" class="btn">Cancel</button>';
      }
      html += '</div>';
      return html;
    }

    function renderAskText(q) {
      const multiline = q.config.multiline;
      let html = '';
      if (multiline) {
        html += '<textarea id="text_' + q.id + '" class="textarea" rows="4" placeholder="' + escapeHtml(q.config.placeholder || '') + '"></textarea>';
      } else {
        html += '<input type="text" id="text_' + q.id + '" class="input" placeholder="' + escapeHtml(q.config.placeholder || '') + '">';
      }
      html += '<div class="btn-group"><button onclick="submitText(\\'' + q.id + '\\')" class="btn btn-primary">Submit</button></div>';
      return html;
    }

    function renderThumbs(q) {
      let html = '<div class="thumbs-container">';
      html += '<button onclick="submitAnswer(\\'' + q.id + '\\', {choice: \\'up\\'})" class="thumb-btn">\\uD83D\\uDC4D</button>';
      html += '<button onclick="submitAnswer(\\'' + q.id + '\\', {choice: \\'down\\'})" class="thumb-btn">\\uD83D\\uDC4E</button>';
      html += '</div>';
      return html;
    }

    function renderSlider(q) {
      const min = q.config.min;
      const max = q.config.max;
      const step = q.config.step || 1;
      const defaultVal = q.config.defaultValue || Math.floor((min + max) / 2);
      const labels = q.config.labels || {};
      const minLabel = labels.min || String(min);
      const maxLabel = labels.max || String(max);
      let html = '<div class="slider-container">';
      html += '<span class="slider-labels">' + escapeHtml(minLabel) + '</span>';
      html += '<input type="range" id="slider_' + q.id + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + defaultVal + '">';
      html += '<span class="slider-labels">' + escapeHtml(maxLabel) + '</span>';
      html += '<span id="slider_val_' + q.id + '" class="slider-value">' + defaultVal + '</span>';
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitSlider(\\'' + q.id + '\\')" class="btn btn-primary">Submit</button></div>';
      return html;
    }


    function renderReviewSection(q) {
      let html = '';
      // Render markdown content
      const markdownHtml = typeof marked !== 'undefined' ? marked.parse(q.config.content || '') : escapeHtml(q.config.content || '');
      html += '<div class="review-content">' + markdownHtml + '</div>';
      html += '<div class="feedback-input">';
      html += '<label for="feedback_' + q.id + '">Feedback (optional)</label>';
      html += '<textarea id="feedback_' + q.id + '" class="textarea" rows="3" placeholder="Any suggestions or changes..."></textarea>';
      html += '</div>';
      html += '<div class="btn-group">';
      html += '<button onclick="submitReview(\\'' + q.id + '\\', \\'approve\\')" class="btn btn-success">Approve</button>';
      html += '<button onclick="submitReview(\\'' + q.id + '\\', \\'revise\\')" class="btn btn-danger">Needs Revision</button>';
      html += '</div>';
      return html;
    }

    function renderShowPlan(q) {
      let html = '';

      // Render sections if provided
      if (q.config.sections && q.config.sections.length > 0) {
        for (const section of q.config.sections) {
          html += '<div class="plan-section">';
          html += '<h3 class="plan-section-title">' + escapeHtml(section.title) + '</h3>';
          const sectionHtml = typeof marked !== 'undefined' ? marked.parse(section.content || '') : escapeHtml(section.content || '');
          html += '<div class="review-content">' + sectionHtml + '</div>';
          html += '</div>';
        }
      } else if (q.config.markdown) {
        // Fallback to raw markdown
        const markdownHtml = typeof marked !== 'undefined' ? marked.parse(q.config.markdown) : escapeHtml(q.config.markdown);
        html += '<div class="review-content">' + markdownHtml + '</div>';
      }

      html += '<div class="feedback-input">';
      html += '<label for="feedback_' + q.id + '">Feedback (optional)</label>';
      html += '<textarea id="feedback_' + q.id + '" class="textarea" rows="3" placeholder="Any suggestions or changes..."></textarea>';
      html += '</div>';
      html += '<div class="btn-group">';
      html += '<button onclick="submitReview(\\'' + q.id + '\\', \\'approve\\')" class="btn btn-success">Approve Plan</button>';
      html += '<button onclick="submitReview(\\'' + q.id + '\\', \\'revise\\')" class="btn btn-danger">Needs Changes</button>';
      html += '</div>';
      return html;
    }

    function renderShowOptions(q) {
      const options = q.config.options || [];
      let html = '<div class="options-with-pros">';
      for (const opt of options) {
        const isRecommended = q.config.recommended === opt.id;
        html += '<div class="option-card' + (isRecommended ? ' recommended' : '') + '" data-id="' + opt.id + '">';
        html += '<div class="option-header">';
        html += '<input type="radio" name="opt_' + q.id + '" value="' + opt.id + '">';
        html += '<span class="option-label">' + escapeHtml(opt.label);
        if (isRecommended) html += ' <span class="option-tag">(recommended)</span>';
        html += '</span></div>';
        if (opt.description) html += '<div class="option-desc">' + escapeHtml(opt.description) + '</div>';
        if (opt.pros && opt.pros.length > 0) {
          html += '<div class="pros"><strong>Pros:</strong><ul>';
          for (const pro of opt.pros) html += '<li>' + escapeHtml(pro) + '</li>';
          html += '</ul></div>';
        }
        if (opt.cons && opt.cons.length > 0) {
          html += '<div class="cons"><strong>Cons:</strong><ul>';
          for (const con of opt.cons) html += '<li>' + escapeHtml(con) + '</li>';
          html += '</ul></div>';
        }
        html += '</div>';
      }
      html += '</div>';
      if (q.config.allowFeedback) {
        html += '<div class="feedback-input"><label>Feedback (optional)</label>';
        html += '<textarea id="feedback_' + q.id + '" class="textarea" rows="2"></textarea></div>';
      }
      html += '<div class="btn-group"><button onclick="submitShowOptions(\\'' + q.id + '\\')" class="btn btn-primary">Select</button></div>';
      return html;
    }

    function renderShowDiff(q) {
      let html = '';
      if (q.config.filePath) {
        html += '<div class="diff-filepath">' + escapeHtml(q.config.filePath) + '</div>';
      }
      html += '<div class="diff-container">';
      html += '<div class="diff-side diff-before"><div class="diff-label">Before</div><pre><code>' + escapeHtml(q.config.before || '') + '</code></pre></div>';
      html += '<div class="diff-side diff-after"><div class="diff-label">After</div><pre><code>' + escapeHtml(q.config.after || '') + '</code></pre></div>';
      html += '</div>';
      html += '<div class="feedback-input"><label>Comments (optional)</label>';
      html += '<textarea id="feedback_' + q.id + '" class="textarea" rows="2"></textarea></div>';
      html += '<div class="btn-group">';
      html += '<button onclick="submitDiff(\\'' + q.id + '\\', \\'approve\\')" class="btn btn-success">Approve</button>';
      html += '<button onclick="submitDiff(\\'' + q.id + '\\', \\'reject\\')" class="btn btn-danger">Reject</button>';
      html += '<button onclick="submitDiff(\\'' + q.id + '\\', \\'edit\\')" class="btn">Edit</button>';
      html += '</div>';
      return html;
    }

    function renderRank(q) {
      const options = q.config.options || [];
      let html = '<div class="rank-list" id="rank_' + q.id + '">';
      for (let i = 0; i < options.length; i++) {
        const opt = options[i];
        html += '<div class="rank-item" data-id="' + opt.id + '" draggable="true">';
        html += '<span class="rank-handle">\\u2630</span>';
        html += '<span class="rank-num">' + (i + 1) + '</span>';
        html += '<span class="rank-label">' + escapeHtml(opt.label) + '</span>';
        html += '</div>';
      }
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitRank(\\'' + q.id + '\\')" class="btn btn-primary">Submit Ranking</button></div>';
      return html;
    }

    function renderRate(q) {
      const options = q.config.options || [];
      const min = q.config.min || 1;
      const max = q.config.max || 5;
      const labels = q.config.labels || {};
      let html = '<div class="rate-list">';
      for (const opt of options) {
        html += '<div class="rate-item">';
        html += '<div class="rate-label">' + escapeHtml(opt.label) + '</div>';
        html += '<div class="rate-stars" id="rate_' + q.id + '_' + opt.id + '">';
        for (let i = min; i <= max; i++) {
          html += '<button class="rate-star" data-value="' + i + '" onclick="setRating(\\'' + q.id + '\\', \\'' + opt.id + '\\', ' + i + ')">' + i + '</button>';
        }

        html += '</div>';


        if (labels.min || labels.max) {
          html += '<div class="slider-labels">' + escapeHtml(labels.min || String(min)) + ' / ' + escapeHtml(labels.max || String(max)) + '</div>';
        }
        html += '</div>';
      }
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitRate(\\'' + q.id + '\\')" class="btn btn-primary">Submit Ratings</button></div>';
      return html;
    }

    function renderAskCode(q) {
      let html = '';
      const lang = q.config.language || 'plaintext';
      html += '<div class="code-input-label">Language: ' + escapeHtml(lang) + '</div>';
      html += '<textarea id="code_' + q.id + '" class="textarea code-input" rows="10" placeholder="' + escapeHtml(q.config.placeholder || 'Enter code here...') + '"></textarea>';
      html += '<div class="btn-group"><button onclick="submitCode(\\'' + q.id + '\\')" class="btn btn-primary">Submit Code</button></div>';
      return html;
    }

    function renderAskImage(q) {
      let html = '';
      const multiple = q.config.multiple ? 'multiple' : '';
      const accept = q.config.accept ? q.config.accept.join(',') : 'image/*';
      html += '<div class="file-upload">';
      html += '<input type="file" id="image_' + q.id + '" accept="' + accept + '" ' + multiple + ' onchange="previewImages(\\'' + q.id + '\\')">';
      html += '<div id="preview_' + q.id + '" class="image-preview"></div>';
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitImages(\\'' + q.id + '\\')" class="btn btn-primary">Upload</button></div>';
      return html;
    }


    function renderAskFile(q) {
      let html = '';
      const multiple = q.config.multiple ? 'multiple' : '';
      const accept = q.config.accept ? q.config.accept.join(',') : '';
      html += '<div class="file-upload">';
      html += '<input type="file" id="file_' + q.id + '" ' + (accept ? 'accept="' + accept + '"' : '') + ' ' + multiple + '>';
      html += '<div id="filelist_' + q.id + '" class="file-list"></div>';
      html += '</div>';
      html += '<div class="btn-group"><button onclick="submitFiles(\\'' + q.id + '\\')" class="btn btn-primary">Upload</button></div>';
      return html;
    }

    function renderEmojiReact(q) {
      let html = '';
      const emojis = q.config.emojis || ['\\uD83D\\uDC4D', '\\uD83D\\uDC4E', '\\u2764\\uFE0F', '\\uD83C\\uDF89', '\\uD83D\\uDE15', '\\uD83D\\uDE80'];
      html += '<div class="emoji-grid">';
      for (const emoji of emojis) {
        html += '<button class="emoji-btn" onclick="submitAnswer(\\'' + q.id + '\\', {emoji: \\'' + emoji + '\\'})">' + emoji + '</button>';
      }
      html += '</div>';
      return html;
    }

    function attachListeners() {
      document.querySelectorAll('input[type="range"]').forEach(slider => {
        const id = slider.id.replace('slider_', 'slider_val_');
        slider.oninput = () => {
          document.getElementById(id).textContent = slider.value;
        };
      });
    }

    function submitAnswer(questionId, answer) {
      const q = questions.find(q => q.id === questionId);
      if (q) {
        q.answered = true;
        q.answer = answer;  // Store answer for read-only view
        ws.send(JSON.stringify({ type: 'response', id: questionId, answer }));
        render();
      }
    }

    function showError(questionId, message) {
      const existingError = document.getElementById('error_' + questionId);
      if (existingError) existingError.remove();

      const card = document.querySelector('[data-qid="' + questionId + '"]') || document.querySelector('.card:not(.card-answered)');
      if (card) {
        const errorDiv = document.createElement('div');
        errorDiv.id = 'error_' + questionId;
        errorDiv.style.cssText = 'color: var(--accent-error); font-size: 0.875rem; margin-top: 0.5rem;';
        errorDiv.textContent = message;
        const btnGroup = card.querySelector('.btn-group');
        if (btnGroup) btnGroup.before(errorDiv);
      }
    }

    function submitPickOne(questionId) {
      const selected = document.querySelector('input[name="pick_' + questionId + '"]:checked');
      if (!selected) {
        showError(questionId, 'Please select an option');
        return;
      }
      submitAnswer(questionId, { selected: selected.value });
    }

    function submitPickMany(questionId) {
      const selected = Array.from(document.querySelectorAll('input[name="pick_' + questionId + '"]:checked')).map(el => el.value);
      submitAnswer(questionId, { selected });
    }

    function submitText(questionId) {
      const input = document.getElementById('text_' + questionId);
      if (input) {
        submitAnswer(questionId, { text: input.value });
      }
    }

    function submitSlider(questionId) {
      const slider = document.getElementById('slider_' + questionId);
      if (slider) {
        submitAnswer(questionId, { value: parseFloat(slider.value) });
      }
    }

    function submitReview(questionId, decision) {
      const feedbackEl = document.getElementById('feedback_' + questionId);
      const feedback = feedbackEl ? feedbackEl.value : '';
      submitAnswer(questionId, { decision, feedback: feedback || undefined });
    }

    function submitShowOptions(questionId) {
      const selected = document.querySelector('input[name="opt_' + questionId + '"]:checked');
      if (!selected) {
        showError(questionId, 'Please select an option');
        return;
      }
      const feedbackEl = document.getElementById('feedback_' + questionId);
      const feedback = feedbackEl ? feedbackEl.value : '';
      submitAnswer(questionId, { selected: selected.value, feedback: feedback || undefined });
    }

    function submitDiff(questionId, decision) {
      const feedbackEl = document.getElementById('feedback_' + questionId);
      const feedback = feedbackEl ? feedbackEl.value : '';
      submitAnswer(questionId, { decision, feedback: feedback || undefined });
    }

    function submitRank(questionId) {
      const container = document.getElementById('rank_' + questionId);
      const items = container.querySelectorAll('.rank-item');
      const ranking = Array.from(items).map((item, idx) => ({
        id: item.dataset.id,
        rank: idx + 1
      }));
      submitAnswer(questionId, { ranking });
    }

    function submitRate(questionId) {
      const q = questions.find(q => q.id === questionId);
      if (!q) return;
      const ratings = {};
      for (const opt of (q.config.options || [])) {
        const container = document.getElementById('rate_' + questionId + '_' + opt.id);
        const selected = container.querySelector('.rate-star.selected');
        if (selected) {
          ratings[opt.id] = parseInt(selected.dataset.value);
        }
      }
      submitAnswer(questionId, { ratings });
    }

    function setRating(questionId, optId, value) {
      const container = document.getElementById('rate_' + questionId + '_' + optId);
      container.querySelectorAll('.rate-star').forEach(btn => {
        btn.classList.toggle('selected', parseInt(btn.dataset.value) <= value);
      });
    }

    function submitCode(questionId) {
      const textarea = document.getElementById('code_' + questionId);
      if (textarea) {
        submitAnswer(questionId, { code: textarea.value });
      }
    }

    function submitImages(questionId) {
      const input = document.getElementById('image_' + questionId);
      if (input && input.files.length > 0) {
        // Convert to base64 for transport
        const promises = Array.from(input.files).map(file => {
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, type: file.type, data: reader.result });
            reader.readAsDataURL(file);
          });
        });
        Promise.all(promises).then(images => {
          submitAnswer(questionId, { images });
        });
      }
    }

    function isAllowedFileType(file, allowed) {
      if (!allowed || allowed.length === 0) return true;
      const fileType = file.type || '';
      const fileName = file.name || '';
      return allowed.some(entry => {
        if (!entry) return false;
        if (entry.endsWith('/*')) {
          const prefix = entry.slice(0, -1);
          return fileType.startsWith(prefix);
        }
        if (entry.startsWith('.')) {
          return fileName.toLowerCase().endsWith(entry.toLowerCase());
        }
        return fileType === entry || fileName.toLowerCase().endsWith(entry.toLowerCase());
      });
    }

    function previewImages(questionId) {
      const input = document.getElementById('image_' + questionId);
      const preview = document.getElementById('preview_' + questionId);
      preview.innerHTML = '';
      const q = questions.find(q => q.id === questionId);
      const allowed = q && q.config.accept ? q.config.accept : null;
      for (const file of input.files) {
        if (allowed && allowed.length > 0 && !isAllowedFileType(file, allowed)) {
          const warning = document.createElement('div');
          warning.textContent = 'Warning: ' + file.name + ' does not match allowed types.';
          warning.style.cssText = 'color: var(--accent-error); font-size: 0.75rem; margin: 0.25rem 0;';
          preview.appendChild(warning);
        }
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.style.maxWidth = '100px';
        img.style.maxHeight = '100px';
        img.style.margin = '4px';
        preview.appendChild(img);
      }
    }

    function submitFiles(questionId) {
      const input = document.getElementById('file_' + questionId);
      if (input && input.files.length > 0) {
        const promises = Array.from(input.files).map(file => {
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, type: file.type, size: file.size, data: reader.result });
            reader.readAsDataURL(file);
          });
        });
        Promise.all(promises).then(files => {
          submitAnswer(questionId, { files });
        });
      }
    }

    // Drag and drop for ranking
    document.addEventListener('dragstart', (e) => {
      if (e.target.classList.contains('rank-item')) {
        e.dataTransfer.setData('text/plain', e.target.dataset.id);
        e.target.classList.add('dragging');
      }
    });
    document.addEventListener('dragend', (e) => {
      if (e.target.classList.contains('rank-item')) {
        e.target.classList.remove('dragging');
      }
    });
    document.addEventListener('dragover', (e) => {
      e.preventDefault();
      const dragging = document.querySelector('.rank-item.dragging');
      const rankList = e.target.closest('.rank-list');
      if (dragging && rankList) {
        const siblings = [...rankList.querySelectorAll('.rank-item:not(.dragging)')];
        const nextSibling = siblings.find(sibling => {
          const rect = sibling.getBoundingClientRect();
          return e.clientY < rect.top + rect.height / 2;
        });
        rankList.insertBefore(dragging, nextSibling);
        // Update numbers
        rankList.querySelectorAll('.rank-item').forEach((item, idx) => {
          item.querySelector('.rank-num').textContent = idx + 1;
        });
      }
    });

    function toggleAnswered(questionId) {
      if (expandedAnswers.has(questionId)) {
        expandedAnswers.delete(questionId);
      } else {
        expandedAnswers.add(questionId);
      }
      render();
    }

    function renderAnsweredQuestion(q) {
      const config = q.config;
      const answer = q.answer || {};
      let html = '';

      switch (q.questionType) {
        case 'pick_one':
          html += renderAnsweredPickOne(q, answer);
          break;
        case 'pick_many':
          html += renderAnsweredPickMany(q, answer);
          break;
        case 'confirm':
          html += renderAnsweredConfirm(q, answer);
          break;
        case 'ask_text':
          html += renderAnsweredText(q, answer);
          break;
        case 'thumbs':
          html += renderAnsweredThumbs(q, answer);
          break;
        case 'slider':
          html += renderAnsweredSlider(q, answer);
          break;
        case 'review_section':
        case 'show_plan':
          html += renderAnsweredReview(q, answer);
          break;
        case 'show_options':
          html += renderAnsweredShowOptions(q, answer);
          break;
        case 'show_diff':
          html += renderAnsweredDiff(q, answer);
          break;
        case 'rank':
          html += renderAnsweredRank(q, answer);
          break;
        case 'rate':
          html += renderAnsweredRate(q, answer);
          break;
        case 'ask_code':
          html += renderAnsweredCode(q, answer);
          break;
        case 'ask_image':
        case 'ask_file':
          html += renderAnsweredFile(q, answer);
          break;
        case 'emoji_react':
          html += renderAnsweredEmoji(q, answer);
          break;
        default:
          html += '<div class="readonly-answer"><pre>' + escapeHtml(JSON.stringify(answer, null, 2)) + '</pre></div>';
      }

      return html;
    }

    function renderAnsweredPickOne(q, answer) {
      const options = q.config.options || [];
      let html = '<div class="options">';
      for (const opt of options) {
        const isSelected = answer.selected === opt.id;
        html += '<div class="readonly-option' + (isSelected ? ' selected' : '') + '">';
        if (isSelected) html += '<span class="check-mark">\\u2713</span>';
        html += '<span>' + escapeHtml(opt.label) + '</span>';
        html += '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderAnsweredPickMany(q, answer) {
      const options = q.config.options || [];
      const selected = answer.selected || [];
      let html = '<div class="options">';
      for (const opt of options) {
        const isSelected = selected.includes(opt.id);
        html += '<div class="readonly-option' + (isSelected ? ' selected' : '') + '">';
        if (isSelected) html += '<span class="check-mark">\\u2713</span>';
        html += '<span>' + escapeHtml(opt.label) + '</span>';
        html += '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderAnsweredConfirm(q, answer) {
      const choice = answer.choice;
      const labels = { yes: q.config.yesLabel || 'Yes', no: q.config.noLabel || 'No', cancel: 'Cancel' };
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Answer</div>';
      html += '<strong>' + escapeHtml(labels[choice] || choice) + '</strong>';
      html += '</div>';
      return html;
    }

    function renderAnsweredText(q, answer) {
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Response</div>';
      html += '<div>' + escapeHtml(answer.text || '') + '</div>';
      html += '</div>';
      return html;
    }

    function renderAnsweredThumbs(q, answer) {
      const emoji = answer.choice === 'up' ? '\\uD83D\\uDC4D' : '\\uD83D\\uDC4E';
      let html = '<div class="readonly-answer">';
      html += '<span style="font-size: 2rem;">' + emoji + '</span>';
      html += '</div>';
      return html;
    }

    function renderAnsweredSlider(q, answer) {
      const labels = q.config.labels || {};
      const minLabel = labels.min || String(q.config.min);
      const maxLabel = labels.max || String(q.config.max);
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Value</div>';
      html += '<strong style="font-size: 1.25rem;">' + answer.value + '</strong>';
      html += ' <span style="color: var(--foreground-subtle);">(range: ' + escapeHtml(minLabel) + ' - ' + escapeHtml(maxLabel) + ')</span>';
      html += '</div>';
      return html;
    }

    function renderAnsweredReview(q, answer) {
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Decision</div>';
      html += '<strong>' + (answer.decision === 'approve' ? '\\u2713 Approved' : '\\u2717 Needs Revision') + '</strong>';
      if (answer.feedback) {
        html += '<div style="margin-top: 0.5rem;"><em>Feedback:</em> ' + escapeHtml(answer.feedback) + '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderAnsweredShowOptions(q, answer) {
      const options = q.config.options || [];
      let html = '<div class="options">';
      for (const opt of options) {
        const isSelected = answer.selected === opt.id;
        html += '<div class="readonly-option' + (isSelected ? ' selected' : '') + '">';
        if (isSelected) html += '<span class="check-mark">\\u2713</span>';
        html += '<span>' + escapeHtml(opt.label) + '</span>';
        html += '</div>';
      }
      html += '</div>';
      if (answer.feedback) {
        html += '<div class="readonly-answer"><div class="readonly-answer-label">Feedback</div>' + escapeHtml(answer.feedback) + '</div>';
      }
      return html;
    }

    function renderAnsweredDiff(q, answer) {
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Decision</div>';
      const decisions = { approve: '\\u2713 Approved', reject: '\\u2717 Rejected', edit: '\\u270E Edit Requested' };
      html += '<strong>' + (decisions[answer.decision] || answer.decision) + '</strong>';
      if (answer.feedback) {
        html += '<div style="margin-top: 0.5rem;"><em>Comments:</em> ' + escapeHtml(answer.feedback) + '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderAnsweredRank(q, answer) {
      const ranking = answer.ranking || [];
      let html = '<div class="readonly-answer-label">Final Ranking</div>';
      html += '<div class="options">';
      for (const item of ranking) {
        const opt = (q.config.options || []).find(o => o.id === item.id);
        html += '<div class="readonly-option selected">';
        html += '<strong>' + item.rank + '.</strong> ' + escapeHtml(opt ? opt.label : item.id);
        html += '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderAnsweredRate(q, answer) {
      const ratings = answer.ratings || {};
      const labels = q.config.labels || {};
      const minLabel = labels.min || String(q.config.min || 1);
      const maxLabel = labels.max || String(q.config.max || 5);
      let html = '<div class="readonly-answer-label">Ratings</div>';
      html += '<div class="options">';
      for (const opt of (q.config.options || [])) {
        const rating = ratings[opt.id];
        html += '<div class="readonly-option' + (rating ? ' selected' : '') + '">';
        html += '<span>' + escapeHtml(opt.label) + '</span>';
        html += ' <strong style="margin-left: auto;">' + (rating || '-') + '</strong>';
        html += '</div>';
      }
      html += '</div>';
      if (labels.min || labels.max) {
        html += '<div class="readonly-answer" style="margin-top: 0.5rem;">';
        html += '<div class="readonly-answer-label">Scale</div>';
        html += '<div>' + escapeHtml(minLabel) + ' → ' + escapeHtml(maxLabel) + '</div>';
        html += '</div>';
      }
      return html;
    }

    function renderAnsweredCode(q, answer) {
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Code (' + escapeHtml(q.config.language || 'plaintext') + ')</div>';
      html += '<pre style="margin: 0; white-space: pre-wrap;"><code>' + escapeHtml(answer.code || '') + '</code></pre>';
      html += '</div>';
      return html;
    }

    function renderAnsweredFile(q, answer) {
      const files = answer.images || answer.files || [];
      let html = '<div class="readonly-answer">';
      html += '<div class="readonly-answer-label">Uploaded ' + files.length + ' file(s)</div>';
      html += '<ul style="margin: 0.5rem 0 0 1rem;">';
      for (const f of files) {
        html += '<li>' + escapeHtml(f.name) + '</li>';
      }
      html += '</ul>';
      html += '</div>';
      return html;
    }

    function renderAnsweredEmoji(q, answer) {
      let html = '<div class="readonly-answer">';
      html += '<span style="font-size: 2rem;">' + (answer.emoji || '') + '</span>';
      html += '</div>';
      return html;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    connect();
  </script>
</body>
</html>`;
}
