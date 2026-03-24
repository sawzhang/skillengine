/* ============================================================
   SkillKit - Web UI Frontend
   ============================================================ */

(function () {
  "use strict";

  // --- DOM Elements ---
  const chatMessages = document.getElementById("chat-messages");
  const chatArea = document.getElementById("chat-area");
  const inputForm = document.getElementById("input-form");
  const inputField = document.getElementById("input-field");
  const btnSend = document.getElementById("btn-send");
  const btnClear = document.getElementById("btn-clear");
  const btnTheme = document.getElementById("btn-theme");
  const headerModel = document.getElementById("header-model");
  const headerSkills = document.getElementById("header-skills");

  // --- State ---
  let isStreaming = false;
  let currentAssistantEl = null;
  let currentContentBuffer = "";

  // --- Theme ---
  function initTheme() {
    const saved = localStorage.getItem("ase-theme");
    if (saved) {
      document.documentElement.setAttribute("data-theme", saved);
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("ase-theme", next);
  }

  // --- Auto-resize textarea ---
  function autoResize() {
    inputField.style.height = "auto";
    inputField.style.height = Math.min(inputField.scrollHeight, 150) + "px";
  }

  // --- Scroll to bottom ---
  function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // --- Markdown Rendering (basic) ---
  function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Code blocks: ```lang\n...\n```
    html = html.replace(
      /```(\w*)\n([\s\S]*?)```/g,
      function (_, lang, code) {
        return '<pre><code class="lang-' + lang + '">' + code + "</code></pre>";
      }
    );

    // Inline code: `...`
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

    // Bold: **...**
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic: *...*
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

    // Unordered lists: lines starting with - or *
    html = html.replace(/^(?:[-*])\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists: lines starting with 1. 2. etc
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
    // Wrap consecutive <li> not already in <ul> into <ol>
    html = html.replace(
      /(?<!<\/ul>)((?:<li>.*<\/li>\n?)+)/g,
      function (match) {
        if (match.indexOf("<ul>") !== -1) return match;
        return "<ol>" + match + "</ol>";
      }
    );

    // Blockquote: > ...
    html = html.replace(/^&gt;\s+(.+)$/gm, "<blockquote>$1</blockquote>");

    // Horizontal rule: ---
    html = html.replace(/^---$/gm, "<hr>");

    // Headers: # ## ### etc
    html = html.replace(/^######\s+(.+)$/gm, "<h6>$1</h6>");
    html = html.replace(/^#####\s+(.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^####\s+(.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^###\s+(.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^##\s+(.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^#\s+(.+)$/gm, "<h1>$1</h1>");

    // Links: [text](url)
    html = html.replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );

    // Paragraphs: double newline
    html = html.replace(/\n\n/g, "</p><p>");

    // Single newline to <br> (but not inside pre/code)
    html = html.replace(
      /(?<!\n)\n(?!\n)/g,
      "<br>"
    );

    // Wrap in paragraph
    if (!html.startsWith("<")) {
      html = "<p>" + html + "</p>";
    }

    return html;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  // --- Message Rendering ---
  function clearWelcome() {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();
  }

  function addUserMessage(text) {
    clearWelcome();
    const el = document.createElement("div");
    el.className = "message message-user";
    el.innerHTML =
      '<div class="message-label">You</div>' +
      '<div class="message-bubble"><div class="message-content">' +
      renderMarkdown(text) +
      "</div></div>";
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function createAssistantMessage() {
    clearWelcome();
    const el = document.createElement("div");
    el.className = "message message-assistant";
    el.innerHTML =
      '<div class="message-label">Assistant</div>' +
      '<div class="message-bubble"><div class="message-content"></div></div>';
    chatMessages.appendChild(el);
    currentAssistantEl = el;
    currentContentBuffer = "";
    scrollToBottom();
    return el;
  }

  function appendToAssistant(text) {
    if (!currentAssistantEl) createAssistantMessage();
    currentContentBuffer += text;
    const contentEl = currentAssistantEl.querySelector(".message-content");
    contentEl.innerHTML = renderMarkdown(currentContentBuffer);
    scrollToBottom();
  }

  function finalizeAssistant() {
    if (currentAssistantEl) {
      // Remove streaming cursor if present
      const cursor = currentAssistantEl.querySelector(".streaming-cursor");
      if (cursor) cursor.remove();
    }
    currentAssistantEl = null;
    currentContentBuffer = "";
  }

  function addStreamingCursor() {
    if (!currentAssistantEl) return;
    const contentEl = currentAssistantEl.querySelector(".message-content");
    // Remove existing cursor first
    const existing = contentEl.querySelector(".streaming-cursor");
    if (existing) existing.remove();
    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor";
    contentEl.appendChild(cursor);
  }

  function addThinkingMessage(text) {
    clearWelcome();
    const el = document.createElement("div");
    el.className = "message message-thinking";
    el.innerHTML =
      '<div class="message-label">Thinking</div>' +
      '<div class="message-bubble"><div class="message-content">' +
      renderMarkdown(text) +
      "</div></div>";
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function addToolCallMessage(toolName, args, result) {
    clearWelcome();
    const el = document.createElement("div");
    el.className = "message message-tool";

    let inner =
      '<div class="message-label">Tool Call</div>' +
      '<div class="message-bubble">' +
      '<div class="tool-header"><span class="tool-icon">&#9881;</span> ' +
      escapeHtml(toolName) +
      "</div>";

    if (args) {
      inner += '<div class="tool-args">' + escapeHtml(args) + "</div>";
    }
    if (result) {
      inner += '<div class="tool-result">' + escapeHtml(result) + "</div>";
    }

    inner += "</div>";
    el.innerHTML = inner;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function addErrorMessage(text) {
    clearWelcome();
    const el = document.createElement("div");
    el.className = "message message-error";
    el.innerHTML =
      '<div class="message-label">Error</div>' +
      '<div class="message-bubble"><div class="message-content">' +
      escapeHtml(text) +
      "</div></div>";
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  // --- Clear conversation ---
  function clearConversation() {
    chatMessages.innerHTML =
      '<div class="welcome-message">' +
      "<h2>Welcome</h2>" +
      "<p>Type a message below to start a conversation with the agent.</p>" +
      "</div>";
    currentAssistantEl = null;
    currentContentBuffer = "";
  }

  // --- SSE Chat Streaming ---
  async function sendMessage(message) {
    if (isStreaming || !message.trim()) return;

    isStreaming = true;
    btnSend.disabled = true;

    addUserMessage(message);
    createAssistantMessage();
    addStreamingCursor();

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message }),
      });

      if (!response.ok) {
        const err = await response.json().catch(function () {
          return { error: "Request failed" };
        });
        finalizeAssistant();
        addErrorMessage(err.error || "Request failed with status " + response.status);
        isStreaming = false;
        btnSend.disabled = false;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let toolArgsBuffer = {};

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line.startsWith("data: ")) continue;

          const data = line.slice(6);
          if (data === "[DONE]") {
            finalizeAssistant();
            continue;
          }

          try {
            const event = JSON.parse(data);
            handleStreamEvent(event, toolArgsBuffer);
          } catch (e) {
            // Skip malformed JSON
          }
        }
      }
    } catch (err) {
      finalizeAssistant();
      addErrorMessage("Connection error: " + err.message);
    }

    isStreaming = false;
    btnSend.disabled = false;
    inputField.focus();
  }

  function handleStreamEvent(event, toolArgsBuffer) {
    switch (event.type) {
      case "text_delta":
      case "content_delta":
        if (event.content) {
          appendToAssistant(event.content);
          addStreamingCursor();
        }
        break;

      case "thinking":
      case "thinking_delta":
        if (event.content) {
          addThinkingMessage(event.content);
        }
        break;

      case "tool_call_start":
        if (event.tool_name) {
          toolArgsBuffer[event.tool_call_id || "default"] = {
            name: event.tool_name,
            args: "",
          };
        }
        break;

      case "tool_call_delta":
      case "args_delta":
        if (event.tool_call_id && toolArgsBuffer[event.tool_call_id]) {
          toolArgsBuffer[event.tool_call_id].args += event.args_delta || event.content || "";
        }
        break;

      case "tool_call_end":
      case "tool_result":
        var id = event.tool_call_id || "default";
        var tool = toolArgsBuffer[id];
        if (tool) {
          addToolCallMessage(tool.name, tool.args, event.content || "");
          delete toolArgsBuffer[id];
        } else if (event.tool_name) {
          addToolCallMessage(event.tool_name, "", event.content || "");
        }
        break;

      case "error":
        addErrorMessage(event.error || event.content || "Unknown error");
        break;

      case "finish":
      case "done":
        finalizeAssistant();
        break;

      default:
        // For any unrecognized event with content, append to assistant
        if (event.content && !event.tool_name) {
          appendToAssistant(event.content);
          addStreamingCursor();
        }
        break;
    }
  }

  // --- Load Config ---
  async function loadConfig() {
    try {
      const resp = await fetch("/api/config");
      if (resp.ok) {
        const config = await resp.json();
        if (config.model) {
          headerModel.textContent = config.model;
        }
      }
    } catch (e) {
      // Config endpoint may not be available
    }
  }

  async function loadSkills() {
    try {
      const resp = await fetch("/api/skills");
      if (resp.ok) {
        const data = await resp.json();
        if (data.skills && data.skills.length > 0) {
          headerSkills.textContent = data.skills.length + " skill" + (data.skills.length !== 1 ? "s" : "");
        }
      }
    } catch (e) {
      // Skills endpoint may not be available
    }
  }

  // --- Event Handlers ---
  inputForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const message = inputField.value.trim();
    if (!message) return;

    // Handle /clear command
    if (message === "/clear") {
      clearConversation();
      inputField.value = "";
      autoResize();
      return;
    }

    inputField.value = "";
    autoResize();
    sendMessage(message);
  });

  inputField.addEventListener("input", autoResize);

  inputField.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      inputForm.dispatchEvent(new Event("submit"));
    }
  });

  btnClear.addEventListener("click", function () {
    clearConversation();
  });

  btnTheme.addEventListener("click", toggleTheme);

  // --- Init ---
  initTheme();
  loadConfig();
  loadSkills();
  inputField.focus();
})();
