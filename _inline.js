const SERVER = "";
const AUTH_STORAGE_KEY = "ensemble_access_token";
let currentSession = null;
let currentProfile = null;

function normalizeAppPath() {
  let p = window.location.pathname || "/";
  if (p.length > 1 && p.endsWith("/")) p = p.slice(0, -1);
  return p === "" ? "/" : p;
}

/** Client-side JWT shape/exp check (same secret not verified; server rejects bad tokens on API calls). */
function tokenLooksValid(tok) {
  if (!tok || !String(tok).trim()) return false;
  try {
    const parts = String(tok).trim().split(".");
    if (parts.length !== 3) return false;
    let b64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    const payload = JSON.parse(atob(b64));
    if (payload.exp != null && payload.exp * 1000 < Date.now() - 5000) return false;
    return true;
  } catch (_e) {
    return false;
  }
}

/** Keep URL aligned with auth: `/` when logged in, `/login` when logged out. */
function syncUrlWithAuthState() {
  const ok = tokenLooksValid(localStorage.getItem(AUTH_STORAGE_KEY));
  const path = normalizeAppPath();
  if (ok && path === "/login") {
    window.location.replace("/");
    return;
  }
  if (!ok && path === "/") {
    window.location.replace("/login");
  }
}

(function enforceAuthRoute() {
  syncUrlWithAuthState();
})();

function wipeAllLocalStorage() {
  try {
    localStorage.clear();
  } catch (_e) {}
}

/** Dev escape hatch: wipe client storage and reload at login (same as fresh install for this app). */
function resetBenAppCache() {
  try {
    localStorage.clear();
    sessionStorage.clear();
  } catch (err) {
    console.error("[resetBenAppCache]", err);
    if (err && err.stack) console.error(err.stack);
  }
  window.location.href = "/login";
}

const SEND_BTN_HTML_IDLE = '<i class="fa-solid fa-paper-plane" aria-hidden="true"></i>';
const SEND_BTN_HTML_BUSY = '<i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i>';

function setComposerSending(busy) {
  const row = document.getElementById("composerRow");
  const inp = document.getElementById("userInput");
  const btn = document.getElementById("sendBtn");
  const tools = document.querySelector(".composer-tools");
  if (!inp || !btn) return;
  if (busy) {
    if (row) row.classList.add("composer-loading");
    if (tools) tools.setAttribute("aria-disabled", "true");
    inp.disabled = true;
    btn.disabled = true;
    btn.innerHTML = SEND_BTN_HTML_BUSY;
    btn.setAttribute("aria-busy", "true");
    btn.setAttribute("aria-label", "Sending…");
  } else {
    if (row) row.classList.remove("composer-loading");
    if (tools) tools.removeAttribute("aria-disabled");
    inp.disabled = false;
    btn.disabled = false;
    btn.innerHTML = SEND_BTN_HTML_IDLE;
    btn.removeAttribute("aria-busy");
    btn.setAttribute("aria-label", "Send message");
  }
}

function jsonHeadersWithAuth(extra) {
  const h = {"Content-Type": "application/json", ...(extra || {})};
  const tok = localStorage.getItem(AUTH_STORAGE_KEY);
  if (tok && tok.trim()) h.Authorization = "Bearer " + tok.trim();
  return h;
}

function refreshAuthBadge() {
  const el = document.getElementById("authStatusPill");
  if (!el) return;
  const tok = localStorage.getItem(AUTH_STORAGE_KEY);
  el.textContent = tokenLooksValid(tok) ? "Signed in" : "";
}

function openAuthModal() {
  document.getElementById("authModal").classList.add("visible");
  refreshAuthBadge();
  syncBodyScrollLock();
}

function hideAuthModal() {
  document.getElementById("authModal").classList.remove("visible");
  syncBodyScrollLock();
}

function closeLimitTierModal() {
  const el = document.getElementById("limit-tier-modal");
  if (el) el.classList.remove("visible");
  syncBodyScrollLock();
}

async function openBillingUpgrade() {
  try {
    const r = await fetch(SERVER + "/api/billing/checkout-url");
    const j = await r.json();
    if (j.url) window.location.href = j.url;
    else window.location.href = SERVER + "/upgrade";
  } catch (_e) {
    window.location.href = SERVER + "/upgrade";
  }
}

const AUTH_LOGIN_BTN_IDLE = "Login";

function setAuthLoginBusy(busy) {
  const btn = document.getElementById("authLoginBtn");
  if (!btn) return;
  btn.disabled = !!busy;
  btn.textContent = busy ? "Signing in…" : AUTH_LOGIN_BTN_IDLE;
}

function resetBenLocalState() {
  try {
    localStorage.removeItem(AUTH_STORAGE_KEY);
  } catch (e) {
    console.warn("resetBenLocalState:", e);
  }
  refreshAuthBadge();
  window.location.href = "/login";
}

async function submitAuthLogin() {
  try {
    setAuthLoginBusy(true);
    const email = document.getElementById("authEmail").value.trim();
    const password = document.getElementById("authPassword").value;
    if (!email || !password) {
      if (typeof updateStatus === "function") {
        updateStatus("Email and password required.", true);
      }
      return;
    }
    let r;
    try {
      r = await fetch(SERVER + "/auth/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ email, password }),
      });
    } catch (fetchErr) {
      console.error("submitAuthLogin fetch:", fetchErr);
      if (typeof updateStatus === "function") {
        updateStatus("Login error: " + fetchErr.message, true);
      }
      return;
    }
    let data = {};
    try {
      const text = await r.text();
      if (text) data = JSON.parse(text);
    } catch (parseErr) {
      console.warn("submitAuthLogin parse:", parseErr);
      data = {};
    }
    if (!r.ok) {
      if (typeof updateStatus === "function") {
        updateStatus(typeof data.detail === "string" ? data.detail : "Please login", true);
      }
      return;
    }
    if (data.access_token) {
      localStorage.setItem(AUTH_STORAGE_KEY, data.access_token);
      window.location.href = "/";
      return;
    }
  } catch (err) {
    console.error("submitAuthLogin:", err);
    if (typeof updateStatus === "function") {
      updateStatus("Login error: " + err.message, true);
    }
  } finally {
    setAuthLoginBusy(false);
  }
}

window.submitAuthLogin = submitAuthLogin;
window.loginUser = submitAuthLogin;
window.resetBenLocalState = resetBenLocalState;

async function submitAuthRegister() {
  const email = document.getElementById("authEmail").value.trim();
  const password = document.getElementById("authPassword").value;
  if (!email || password.length < 8) {
    updateStatus("Email and password (8+ chars) required.", true);
    return;
  }
  try {
    const r = await fetch(SERVER + "/auth/register", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ email, password }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      updateStatus(data.detail ? String(data.detail) : "Registration failed", true);
      return;
    }
    if (data.access_token) {
      localStorage.setItem(AUTH_STORAGE_KEY, data.access_token);
      window.location.href = "/";
      return;
    }
  } catch (err) {
    updateStatus("Register error: " + err.message, true);
  }
}

function authLogout() {
  wipeAllLocalStorage();
  window.location.href = "/login";
}

const META = {
  gpt:{label:"GPT", color:"#22c55e", emoji:"⚙️"},
  gemini:{label:"Gemini", color:"#a855f7", emoji:"💻"},
  claude:{label:"Claude", color:"#3b82f6", emoji:"🧠"},
  'gpt-fast':{label:"GPT Mini", color:"#10b981", emoji:"⚡"},
  'gemini-fast':{label:"Gemini Flash", color:"#8b5cf6", emoji:"✨"}
};

let webSearchActive = false;
let isSending = false;
let moneySavedTotal = 0;

/** Stored order: GPT / Gemini / Claude keys; synced with profile.active_tools via PATCH */
let workspaceTools = ["gpt", "gemini", "claude"];

function canonicalWorkspaceTools(ids){
  const order = ["gpt", "gemini", "claude"];
  const next = [];
  order.forEach(k => { if (ids.includes(k)) next.push(k); });
  return next.length ? next : ["gpt"];
}

function hideBenToolPicker(){
  const m = document.getElementById("benToolPickerModal");
  if(m) m.classList.remove("visible");
  syncBodyScrollLock();
}

function showBenToolPicker(){
  syncBenToolToolbar();
  document.getElementById("pickGpt").checked = workspaceTools.includes("gpt");
  document.getElementById("pickGemini").checked = workspaceTools.includes("gemini");
  document.getElementById("pickClaude").checked = workspaceTools.includes("claude");
  document.getElementById("benToolPickerModal").classList.add("visible");
  syncBodyScrollLock();
}

function applyBenToolPicker(){
  const next = canonicalWorkspaceTools([
    ...(document.getElementById("pickGpt").checked ? ["gpt"] : []),
    ...(document.getElementById("pickGemini").checked ? ["gemini"] : []),
    ...(document.getElementById("pickClaude").checked ? ["claude"] : []),
  ]);
  workspaceTools = next;
  syncBenToolToolbar();
  persistWorkspaceTools();
  hideBenToolPicker();
}

function syncBenToolToolbar(){
  const keys = canonicalWorkspaceTools(workspaceTools);
  workspaceTools = keys;
  document.querySelectorAll(".ben-tool-pill[data-tool]").forEach(btn => {
    const k = btn.getAttribute("data-tool");
    btn.classList.toggle("on", keys.includes(k));
  });
}

async function persistWorkspaceTools(){
  if(!currentSession) return;
  try{
    const r = await fetch(SERVER + `/session/${currentSession}/active-tools`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ active_tools: workspaceTools }),
    });
    const d = await r.json();
    if(d.success && d.profile){ currentProfile = d.profile; }
  }catch(_e){}
}

async function toggleWorkspaceTool(mid){
  const set = new Set(workspaceTools);
  if(set.has(mid)){
    if(set.size <= 1){
      updateStatus("Keep at least one analyst enabled.", true);
      return;
    }
    set.delete(mid);
  }else{
    set.add(mid);
  }
  workspaceTools = canonicalWorkspaceTools(Array.from(set));
  syncBenToolToolbar();
  persistWorkspaceTools();
}

function hydrateWorkspaceToolsFromProfile(){
  if(currentProfile && Array.isArray(currentProfile.active_tools)){
    workspaceTools = canonicalWorkspaceTools(currentProfile.active_tools);
  }else{
    workspaceTools = canonicalWorkspaceTools(["gpt", "gemini", "claude"]);
  }
  syncBenToolToolbar();
}

function isMutedLaneText(text){
  const t = String(text || "").toLowerCase();
  return (
    t.includes("analyst is turned off") ||
    t.includes("token saver mode: model skipped") ||
    t.includes("[maintenance]")
  );
}

function applyRound1ToolsActive(group, toolsMap){
  if(!group || !toolsMap || typeof toolsMap !== "object") return;
  ["gpt", "gemini", "claude"].forEach(mid => {
    const card = group.querySelector(`.stream-model-card[data-model="${mid}"]`);
    if(!card) return;
    const active = !!toolsMap[mid];
    card.classList.toggle("lane-off", !active);
    const status = card.querySelector(".sm-status");
    if(status){
      status.classList.remove("offline");
      if(!active){
        status.textContent = "Off";
        status.style.color = "#828282";
      }else{
        status.textContent = "Online";
        status.style.color = "";
      }
    }
    if(active && card.querySelector(".sm-body")?.textContent === "Thinking…"){
      card.classList.add("thinking");
    }
    if(!active){
      card.classList.remove("thinking");
    }
  });
}

function addMoneySaved(amount){
  const n = Number(amount || 0);
  if(!Number.isFinite(n) || n <= 0) return;
  moneySavedTotal += n;
  const el = document.getElementById("moneySavedIndicator");
  if(el) el.textContent = `Money Saved: $${moneySavedTotal.toFixed(6)}`;
}

function resetPipelineBar(){
  const track = document.getElementById("pipelineTrack");
  if(!track) return;
  track.classList.add("hidden");
  const fill = document.getElementById("pipelineBarFill");
  if(fill) fill.style.width = "0%";
  document.querySelectorAll(".pipeline-label").forEach(el => {
    el.classList.remove("active","done");
    const s = el.getAttribute("data-pl");
    if(s === "0") el.textContent = "Researching";
    else if(s === "1") el.textContent = "Analyzing consensus";
    else if(s === "2") el.textContent = "Finalizing answer";
  });
}

/** @param step {number} 0..2 @param label {string|undefined} */
function setPipelineBar(step, label){
  const track = document.getElementById("pipelineTrack");
  if(!track) return;
  track.classList.remove("hidden");
  const fill = document.getElementById("pipelineBarFill");
  if(fill){
    const pct = step <= 0 ? 14 : step >= 2 ? 100 : 52;
    fill.style.width = pct + "%";
  }
  document.querySelectorAll(".pipeline-label").forEach(el => {
    const si = parseInt(el.getAttribute("data-pl"), 10);
    el.classList.remove("active","done");
    if(si < step) el.classList.add("done");
    if(si === step){
      el.classList.add("active");
      if(label && String(label).trim()) el.textContent = String(label).trim();
    }
  });
}

function isModelFailedText(text){
  const t = String(text || "").toLowerCase();
  if(!t.trim()) return true;
  return (
    t.includes(" error:") ||
    t.includes("timed out") ||
    t.includes("authentication") ||
    t.includes("unauthorized") ||
    t.includes("invalid api key") ||
    t.includes("not_found_error") ||
    t.includes("404") ||
    t.includes("[skipped") ||
    t.includes("stream truncated")
  );
}

/** Build concurrent Round 1 cards + optional BEN block inside a message-group */
function mountLiveStreamShell(group){
  group.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "ensemble-r1-grid";

  ["gpt","gemini","claude"].forEach(mid => {
    const meta = META[mid];
    const card = document.createElement("div");
    card.className = "stream-model-card thinking";
    card.dataset.model = mid;
    card.innerHTML = `
      <div class="sm-head">
        <span class="sm-ico" style="background:${meta.color}22">${meta.emoji}</span>
        <span style="color:${meta.color}">${meta.label}</span>
        <span class="sm-status">Online</span>
      </div>
      <div class="sm-body">Thinking…</div>`;
    grid.appendChild(card);
  });
  group.appendChild(grid);

  const benchEl = document.createElement("div");
  benchEl.id = "liveBenchmarkCard";
  benchEl.className = "benchmark-card hidden message assistant";
  group.appendChild(benchEl);

  const benWrap = document.createElement("div");
  benWrap.id = "liveBenSection";
  benWrap.className = "ben-stream-section hidden";
  benWrap.innerHTML = `
    <div class="ben-stream-kicker" id="liveBenKicker"><span class="live-dot" aria-hidden="true"></span><span id="liveBenKickerText">BEN — Supreme Judge</span></div>
    <div class="message assistant ben-stream-body" id="liveBenBody"></div>`;
  group.appendChild(benWrap);

  group._liveBuf = { gpt:"", gemini:"", claude:"" };
}

function showTokenSaverBanner(group, message){
  let el = group.querySelector(".token-saver-badge");
  if(!el){
    el = document.createElement("div");
    el.className = "token-saver-badge";
    group.prepend(el);
  }
  el.textContent = `🍃 ${message || "Token Saver Active: Using high-speed eco-models"}`;
}

function flushLiveRound1Card(model, markdownSource, wrapEl){
  const card = wrapEl.querySelector(`.stream-model-card[data-model="${model}"]`);
  if(!card) return;
  const body = card.querySelector(".sm-body");
  const status = card.querySelector(".sm-status");
  if(!body) return;
  card.classList.remove("thinking");
  let txt = markdownSource || "";
  const muted = isMutedLaneText(txt);
  const failed = !muted && isModelFailedText(txt);
  if(status){
    if(muted){
      status.textContent = "Idle";
      status.style.color = "#8c8c8c";
      status.classList.remove("offline");
    }else{
      status.textContent = failed ? "Offline" : "Online";
      status.classList.toggle("offline", failed);
      if(!failed) status.style.color = "";
    }
  }
  card.classList.toggle("failed", failed);
  try{
    body.innerHTML = marked.parse(txt);
  } catch(_e){
    body.textContent = txt;
  }
}

function revealLiveBenSection(wrapEl, opts){
  const sec = wrapEl.querySelector("#liveBenSection");
  const ktxt = wrapEl.querySelector("#liveBenKickerText");
  if(!sec) return;
  sec.classList.remove("hidden");
  if(opts && opts.kicker && ktxt) ktxt.textContent = opts.kicker;
}

async function onNewChatClick(){
  await createNewSession();
  closeNavDrawer();
}

function syncBodyScrollLock(){
  const nav = document.body.classList.contains("nav-drawer-open");
  const pm = document.getElementById("profileModal");
  const tpm = document.getElementById("benToolPickerModal");
  const am = document.getElementById("authModal");
  const prof = pm && pm.classList.contains("visible");
  const pick = tpm && tpm.classList.contains("visible");
  const auth = am && am.classList.contains("visible");
  document.body.style.overflow = nav || prof || pick || auth ? "hidden" : "";
}

function openNavDrawer(){
  document.body.classList.add("nav-drawer-open");
  syncBodyScrollLock();
}
function closeNavDrawer(){
  document.body.classList.remove("nav-drawer-open");
  syncBodyScrollLock();
}

// ========================
// SESSION MANAGEMENT
// ========================

async function createNewSession(){
  const title = prompt("Chat name:", "New Conversation");
  if(!title) return;

  try {
    const response = await fetch(SERVER + "/session/new", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title})
    });

    const data = await response.json();
    if(data.success){
      currentSession = data.session_id;
      loadSessions();
      loadChatHistory();
      updateStatus("Session created");
    }
  } catch(err) {
    updateStatus("Error creating session: " + err.message, true);
  }
}

function showProfileModal(show){
  const modal = document.getElementById("profileModal");
  modal.classList.toggle("visible", show);
  syncBodyScrollLock();
}

function hideProfileModal(){
  showProfileModal(false);
}

function updateProfileGreeting(profile){
  const greeting = document.getElementById("profileGreeting");
  if(profile && profile.user_name){
    greeting.textContent = `Hello, ${profile.user_name}`;
  } else {
    greeting.textContent = "";
  }
}

function openProfileSettings(){
  const profile = currentProfile || {};
  document.getElementById("profileName").value = profile.user_name || "";
  document.getElementById("profileRole").value = profile.user_role || "developer";
  document.getElementById("profileProjects").value = profile.projects || "";
  showProfileModal(true);
}

async function submitProfileSetup(){
  const name = document.getElementById("profileName").value.trim();
  const role = document.getElementById("profileRole").value;
  const projects = document.getElementById("profileProjects").value.trim();

  if(!name){
    updateStatus("Please enter your name to continue.", true);
    return;
  }

  try {
    let response;
    let data;

    if(currentSession){
      response = await fetch(SERVER + `/session/${currentSession}/profile`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          user_name: name,
          user_role: role,
          projects: projects
        })
      });
      data = await response.json();
    } else {
      response = await fetch(SERVER + "/session/new", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          title: `${name}'s Workspace`,
          user_name: name,
          user_role: role,
          projects: projects
        })
      });
      data = await response.json();
      if(data.success){
        currentSession = data.session_id;
      }
    }

    if(data.success){
      currentProfile = data.profile || {user_name: name, user_role: role, projects: projects};
      hydrateWorkspaceToolsFromProfile();
      updateProfileGreeting(currentProfile);
      showProfileModal(false);
      loadSessions();
      loadChatHistory();
      updateStatus("Profile saved. Welcome, " + name + "!");
    } else {
      updateStatus("Error saving profile: " + (data.error || "Unknown error"), true);
    }
  } catch(err) {
    updateStatus("Error saving profile: " + err.message, true);
  }
}

async function loadSessions(){
  try {
    const response = await fetch(SERVER + "/sessions");
    const data = await response.json();

    if(data.success){
      const list = document.getElementById("sessionsList");
      list.innerHTML = data.sessions.map(s => `
        <div class="session-item ${s.session_id === currentSession ? 'active' : ''}" 
             onclick="switchSession('${s.session_id}')">
          <div class="session-item-title">${escapeHtml(s.title)}</div>
          <div class="session-item-time">${new Date(s.updated_at).toLocaleDateString()}</div>
        </div>
      `).join("");

      if(!currentSession && data.sessions.length > 0){
        currentSession = data.sessions[0].session_id;
      }

      const hasSession = Boolean(currentSession);
      const authed = tokenLooksValid(localStorage.getItem(AUTH_STORAGE_KEY));
      showWelcomeScreen(!hasSession && !authed);
      showProfileModal(!hasSession);

      if(hasSession){
        loadChatHistory();
      }
    }
  } catch(err) {
    console.error("Error loading sessions:", err);
  }
}

async function switchSession(sessionId){
  currentSession = sessionId;
  closeNavDrawer();
  showWelcomeScreen(false);
  loadSessions();
  loadChatHistory();
}

async function loadChatHistory(){
  if(!currentSession) return;

  try {
    const response = await fetch(SERVER + `/session/${currentSession}`);
    const data = await response.json();

    if(data.success){
      currentProfile = data.profile || null;
      hydrateWorkspaceToolsFromProfile();
      updateProfileGreeting(currentProfile);
      setDocumentBadge(Boolean(currentProfile && currentProfile.has_uploaded_document));
      if (data.trial_count !== undefined) {
        const remaining = Math.max(0, 3 - data.trial_count);
        document.getElementById('trial-indicator').textContent = `Free questions remaining: ${remaining}/3`;
      }
      if(!currentProfile){
        showProfileModal(true);
      }
      document.getElementById("sessionSubtitle").textContent =
        `Created: ${new Date(data.created_at).toLocaleString()}`;

      const chat = document.getElementById("chatArea");
      chat.innerHTML = "";

      let pendingRound1 = {gpt: "", gemini: "", claude: ""};
      let pendingRound2 = {gpt: "", gemini: "", claude: ""};
      let lastUserMessage = null;

      data.messages.forEach(msg => {
        if(msg.model === "ensemble" && msg.role === "user"){
          if (msg.content !== lastUserMessage) appendUserMessage(msg.content);
          lastUserMessage = msg.content;
        } else if(msg.model === "ensemble-round1"){
          pendingRound1[msg.role] = msg.content;
        } else if(msg.model === "ensemble-round2"){
          pendingRound2[msg.role] = msg.content;
        } else if(msg.model === "ensemble-final" && msg.role === "assistant"){
          appendAssistantMessage(msg.content, pendingRound1, pendingRound2, null, "technical", "");
          pendingRound1 = {gpt: "", gemini: "", claude: ""};
          pendingRound2 = {gpt: "", gemini: "", claude: ""};
          lastUserMessage = null;
        } else if(msg.role === "user"){
          if (msg.content !== lastUserMessage) appendUserMessage(msg.content);
          lastUserMessage = msg.content;
        } else if(msg.role === "assistant"){
          appendAssistantMessage(msg.content, {}, {}, null, "technical", "");
        }
      });

      chat.scrollTop = chat.scrollHeight;
    }
  } catch(err) {
    console.error("Error loading chat history:", err);
  }
}

// ========================
// UI HELPERS
// ========================

function escapeHtml(text){
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function updateStatus(msg, isError = false){
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = isError ? "status error" : "status";
  if(msg) setTimeout(() => {el.textContent = ""}, 3000);
}

function showWelcomeScreen(show){
  const main = document.getElementById("mainContent");
  const welcome = document.getElementById("welcomeScreen");
  const status = document.getElementById("status");
  main.classList.toggle("welcome-active", show);
  welcome.classList.toggle("visible", show);
  status.style.display = show ? "none" : "block";
  if(show) setDocumentBadge(false);
}

async function useExample(question){
  if(!currentSession){
    try {
      const response = await fetch(SERVER + "/session/new", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({title: "Quick Start"})
      });
      const data = await response.json();
      
      if(data.success){
        currentSession = data.session_id;
        loadSessions();
        loadChatHistory();
      } else {
        if (data.trial_exceeded) {
          document.getElementById('upgrade-modal').classList.add('visible');
        }
        throw new Error(data.error || "Failed to create session");
      }
    } catch(err) {
      updateStatus("Error creating quick start session: " + err.message, true);
      return;
    }
  }

  document.getElementById("userInput").value = question;
  sendMessage();
}

function appendUserMessage(content){
  const chat = document.getElementById("chatArea");
  const group = document.createElement("div");
  group.className = "message-group";
  
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.textContent = content;
  
  group.appendChild(msg);
  chat.appendChild(group);
  chat.scrollTop = chat.scrollHeight;
}

function appendAssistantMessage(content, round1, round2, consensusData = null, category = "technical", learningStatsMsg = ""){
  const chat = document.getElementById("chatArea");
  const group = document.createElement("div");
  group.className = "message-group";
  
  // Learning Engine Banner
  if(learningStatsMsg) {
    const banner = document.createElement("div");
    banner.className = "learning-banner";
    banner.innerHTML = `<i class="fa-solid fa-brain" aria-hidden="true"></i><span>${escapeHtml(learningStatsMsg)}</span>`;
    group.appendChild(banner);
  }

  // Consensus Map Panel
  if(consensusData && consensusData.length > 0) {
    const panel = document.createElement("div");
    panel.className = "consensus-panel";
    
    const header = document.createElement("div");
    header.className = "consensus-header";
    header.innerHTML = `<i class="fa-solid fa-scale-balanced" aria-hidden="true"></i><span>Semantic consensus map</span>`;
    panel.appendChild(header);

    consensusData.forEach(item => {
      let badgeClass = "unverified";
      let status = (item.status || "UNVERIFIED").toUpperCase();
      if(status.includes("HIGH")) badgeClass = "high";
      else if(status.includes("MEDIUM")) badgeClass = "medium";
      else if(status.includes("CONTRADIC")) badgeClass = "contradiction";
      
      const row = document.createElement("div");
      row.className = "consensus-item";
      row.innerHTML = `
        <div class="consensus-claim">${escapeHtml(item.claim)}</div>
        <div class="badge ${badgeClass}">${escapeHtml(item.status)}</div>
      `;
      panel.appendChild(row);
    });
    group.appendChild(panel);
  }

  // Main synthesis response
  const msg = document.createElement("div");
  msg.className = "message assistant";
  
  // Process BEN's structured response
  let processedContent = content;
  
  // Style TL;DR
  processedContent = processedContent.replace(/## TL;DR\n([\s\S]*?)(?=##|$)/, (match, p1) => {
    return `<div class="tldr-section">TL;DR: ${p1.trim()}</div>`;
  });

  // Handle Trust Map colors
  processedContent = processedContent.replace(/✅/g, '<span class="text-success">✅</span>');
  processedContent = processedContent.replace(/⚠️/g, '<span class="text-warning">⚠️</span>');
  processedContent = processedContent.replace(/❌/g, '<span class="text-danger">❌</span>');

  // Extract Next Action for button
  let nextActionText = "";
  processedContent = processedContent.replace(/## Next Action\n([\s\S]*?)(?=##|$)/, (match, p1) => {
    nextActionText = p1.trim();
    return ""; // Remove from main text flow
  });

  // Render markdown
  msg.innerHTML = marked.parse(processedContent);
  
  // Add 'Run' buttons to code blocks
  msg.querySelectorAll('pre code.language-python').forEach(block => {
    const pre = block.parentElement;
    pre.style.position = 'relative';
    const runBtn = document.createElement('button');
    runBtn.className = 'code-run-btn';
    runBtn.innerHTML = '<i class="fa-solid fa-play" aria-hidden="true"></i><span style="margin-left:6px">Run</span>';
    runBtn.setAttribute('aria-label', 'Run Python code');
    runBtn.onclick = () => executeCode(block.textContent, pre);
    pre.appendChild(runBtn);
  });

  // Append Next Action button if found
  if(nextActionText) {
    const btnContainer = document.createElement("div");
    btnContainer.className = "next-action-container";
    const btn = document.createElement("button");
    btn.className = "next-action-btn";
    btn.textContent = nextActionText;
    btn.onclick = () => {
        document.getElementById('userInput').value = nextActionText;
        sendMessage();
    };
    btnContainer.appendChild(btn);
    msg.appendChild(btnContainer);
  }

  // Smart toolbar (same for every synthesized reply)
  const smartToolbar = document.createElement("div");
  smartToolbar.className = "smart-toolbar";
  
  const smartTools = [
    { title: "Generate code", icon: "fa-bolt", action: () => runSmartTool('generate_code', content, smartToolbar) },
    { title: "Research examples", icon: "fa-magnifying-glass", action: () => runSmartTool('research_examples', content, smartToolbar) },
    { title: "Export PDF", icon: "fa-file-pdf", action: () => runSmartTool('export_pdf', content, smartToolbar) },
    { title: "Review Auto-Fix", icon: "fa-wand-magic-sparkles", action: () => window.open('/review-auto-fix', '_blank', 'noopener,noreferrer') }
  ];

  smartTools.forEach(t => {
    const sBtn = document.createElement("button");
    sBtn.className = "smart-btn";
    sBtn.type = "button";
    sBtn.title = t.title;
    sBtn.setAttribute("aria-label", t.title);
    sBtn.innerHTML = `<i class="fa-solid ${t.icon}" aria-hidden="true"></i>`;
    sBtn.onclick = t.action;
    smartToolbar.appendChild(sBtn);
  });
  
  msg.appendChild(smartToolbar);

  group.appendChild(msg);
  
  // Collapsible details section
  const collapsible = document.createElement("div");
  collapsible.className = "collapsible-section";
  
  const header = document.createElement("div");
  header.className = "collapsible-header";
  header.onclick = () => toggleCollapsible(header, content_div);
  header.innerHTML = `
    <span>See how each model thought</span>
    <i class="fa-solid fa-chevron-down toggle-icon" aria-hidden="true"></i>
  `;
  collapsible.appendChild(header);
  
  const content_div = document.createElement("div");
  content_div.className = "collapsible-content";
  
  const roundsContainer = document.createElement("div");
  roundsContainer.className = "rounds-container";
  
  // Round 1
  if(round1 && Object.keys(round1).length > 0){
    const r1Section = document.createElement("div");
    r1Section.className = "round-section";
    r1Section.innerHTML = `<div class="round-label">Round 1 - Initial Analysis</div>`;
    
    const grid = document.createElement("div");
    grid.className = "round-grid";
    grid.innerHTML = [
      cardHtml("gpt", round1.gpt || "", category),
      cardHtml("gemini", round1.gemini || "", category),
      cardHtml("claude", round1.claude || "", category)
    ].join("");
    
    r1Section.appendChild(grid);
    roundsContainer.appendChild(r1Section);
  }
  
  // Round 2
  if(round2 && Object.keys(round2).length > 0){
    const r2Section = document.createElement("div");
    r2Section.className = "round-section";
    r2Section.innerHTML = `<div class="round-label">Round 2 - Cross Critique</div>`;
    
    const grid = document.createElement("div");
    grid.className = "round-grid";
    grid.innerHTML = [
      cardHtml("gpt", round2.gpt || "", category),
      cardHtml("gemini", round2.gemini || "", category),
      cardHtml("claude", round2.claude || "", category)
    ].join("");
    
    r2Section.appendChild(grid);
    roundsContainer.appendChild(r2Section);
  }
  
  // Round 3 (BEN — Supreme Judge)
  if(true){ // In the new system, we treat the main assistant message as Round 3 results
    const r3Section = document.createElement("div");
    r3Section.className = "round-section";
    r3Section.innerHTML = `<div class="round-label">BEN — Supreme Judge (Final Verdict)</div>`;
    
    const r3Msg = document.createElement("div");
    r3Msg.className = "card-body";
    r3Msg.style.maxHeight = "none";
    r3Msg.style.fontSize = "13px";
    r3Msg.innerHTML = "BEN successfully synthesized all inputs into the final response shown above.";
    
    r3Section.appendChild(r3Msg);
    roundsContainer.appendChild(r3Section);
  }
  
  content_div.appendChild(roundsContainer);
  collapsible.appendChild(content_div);
  group.appendChild(collapsible);
  
  chat.appendChild(group);
  chat.scrollTop = chat.scrollHeight;
}

function toggleCollapsible(header, content){
  header.classList.toggle("open");
  header.querySelector(".toggle-icon").classList.toggle("open");
  content.classList.toggle("open");
}

async function submitFeedback(model, category, value, btn) {
  if(!currentSession) return;
  try {
    const response = await fetch(SERVER + "/ensemble/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: currentSession,
        model: model,
        feedback_value: value,
        category: category
      })
    });
    const data = await response.json();
    if(data.success) {
      const parent = btn.parentElement;
      parent.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active-up', 'active-down'));
      if(value === 1) btn.classList.add('active-up');
      else btn.classList.add('active-down');
    }
  } catch(err) {
    console.error("Feedback error:", err);
  }
}

function cardHtml(model, text, category){
  const m = META[model];
  const safeText = escapeHtml(String(text || ""));
  const failed = isModelFailedText(text);
  const statusHtml = failed
    ? '<span class="card-status offline"><i class="fa-solid fa-xmark" aria-hidden="true"></i> Offline</span>'
    : '<span class="card-status online">Online</span>';
  const catParam = category ? `'${category}'` : "'technical'";
  return `
    <div class="card">
      <div class="card-top">
        <div class="icon" style="background:${m.color}22">${m.emoji}</div>
        <div>
          <div class="card-name" style="color:${m.color}">${m.label}${statusHtml}</div>
        </div>
        <div class="feedback-actions">
          <button type="button" class="feedback-btn" title="Good response" aria-label="Good response" onclick="submitFeedback('${model}', ${catParam}, 1, this)"><i class="fa-regular fa-thumbs-up" aria-hidden="true"></i></button>
          <button type="button" class="feedback-btn" title="Poor response" aria-label="Poor response" onclick="submitFeedback('${model}', ${catParam}, -1, this)"><i class="fa-regular fa-thumbs-down" aria-hidden="true"></i></button>
        </div>
      </div>
      <div class="card-body">${safeText}</div>
    </div>
  `;
}

// ========================
// SEND MESSAGE (UNIFIED)
// ========================

async function sendMessage(){
  if (isSending) return;
  if(!currentSession) {
    updateStatus("Create or select a session first", true);
    return;
  }

  const msg = document.getElementById("userInput").value.trim();
  if(!msg){
    updateStatus("Enter a message", true);
    return;
  }

  isSending = true;
  setComposerSending(true);
  updateStatus("BEN is thinking…");

  try {
    showWelcomeScreen(false);
    appendUserMessage(msg);
    document.getElementById("userInput").value = "";

    const chat = document.getElementById("chatArea");
    const group = document.createElement("div");
    group.className = "message-group";
    group.id = "streaming-group";

    resetPipelineBar();
    mountLiveStreamShell(group);
    chat.appendChild(group);
    chat.scrollTop = chat.scrollHeight;

    const response = await fetch(SERVER + "/ensemble/stream", {
      method: "POST",
      headers: jsonHeadersWithAuth(),
      body: JSON.stringify({
        session_id: currentSession,
        question: msg,
        web_search: webSearchActive
      })
    });

    if (response.status === 401) {
      let detail = "Please login";
      try {
        const j = await response.json();
        if (typeof j.detail === "string") detail = j.detail;
      } catch (_x) {}
      group.remove();
      resetPipelineBar();
      updateStatus(detail, true);
      openAuthModal();
      return;
    }

    if (response.status === 403) {
      let errCode = null;
      try {
        const j = await response.json();
        let d = j.detail;
        if (typeof d === "string") {
          try { d = JSON.parse(d); } catch (_p) {}
        }
        if (d && typeof d === "object" && d.error) errCode = d.error;
      } catch (_x) {}
      group.remove();
      resetPipelineBar();
      if (errCode === "LIMIT_REACHED") {
        document.getElementById("limit-tier-modal").classList.add("visible");
        syncBodyScrollLock();
        updateStatus("Daily free limit reached.", true);
      } else {
        updateStatus("Request forbidden (403).", true);
      }
      return;
    }

    if (!response.ok) {
      const errTxt = await response.text().catch(() => "");
      group.remove();
      resetPipelineBar();
      updateStatus("Stream error (" + response.status + "): " + (errTxt || response.statusText), true);
      return;
    }

    updateStatus("Streaming ensemble…");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let partialLine = "";
    let finalData = null;
    let benGen = 0;
    let benText = "";

    function flushBenMarkdown(targetEl, raw){
      if(!targetEl) return;
      let processed = raw;
      processed = processed.replace(/✅/g, '<span class="text-success">✅</span>');
      processed = processed.replace(/⚠️/g, '<span class="text-warning">⚠️</span>');
      processed = processed.replace(/❌/g, '<span class="text-danger">❌</span>');
      try {
        targetEl.innerHTML = marked.parse(processed);
      } catch(_e){
        targetEl.textContent = raw;
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      partialLine += decoder.decode(value, { stream: true });
      const lines = partialLine.split("\n");
      partialLine = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          let data;
          try {
            const payload = line.startsWith("data:") ? line.slice(5).trimStart() : line;
            if (!payload || payload === "[DONE]") continue;
            data = JSON.parse(payload);
          } catch (parseErr){
            console.warn("[ndjson] skipping invalid line", parseErr, line);
            continue;
          }
          console.log("[stream event]", data);

          if (Object.prototype.hasOwnProperty.call(data, "success") && data.success === false) {
            updateStatus("Error: " + (data.error || "request failed"), true);
            if (data.trial_exceeded) document.getElementById("upgrade-modal").classList.add("visible");
            resetPipelineBar();
            return;
          }
          if (data.type === "error") {
            updateStatus("Error: " + (data.content || "stream error"), true);
            resetPipelineBar();
            continue;
          }
          if (data.type === "pipeline") {
            setPipelineBar(typeof data.step === "number" ? data.step : 0, data.label);
            continue;
          }
        if (data.type === "token_saver") {
          showTokenSaverBanner(group, data.message);
          continue;
        }
          if (data.type === "benchmark_card") {
            const bel = group.querySelector("#liveBenchmarkCard");
            if (bel) {
              bel.classList.remove("hidden");
              const md = data.markdown || "";
              try {
                bel.innerHTML = marked.parse(md);
              } catch (_e) {
                bel.textContent = md;
              }
            }
            continue;
          }
          if (data.type === "round1_start") {
            setPipelineBar(0, "Researching");
            applyRound1ToolsActive(group, data.tools_active || null);
            continue;
          }
          if (data.type === "fast_first_stream") {
            const sec = (Number(data.ms_since_start) || 0) / 1000;
            const who = String(data.model || "").toUpperCase();
            updateStatus(`Fast answer stream · ${who} first tokens in ${sec.toFixed(2)}s`, false);
            continue;
          }
          if (data.type === "round1_chunk") {
            const mid = data.model;
            if (mid && group._liveBuf){
              group._liveBuf[mid] = (group._liveBuf[mid] || "") + (data.content || "");
              flushLiveRound1Card(mid, group._liveBuf[mid], group);
            }
            continue;
          }
          if (data.type === "round1_complete") {
            const card = group.querySelector(`.stream-model-card[data-model="${data.model}"]`);
            if(card) card.classList.add("done");
            continue;
          }
          if (data.type === "ben_reset") {
            if(typeof data.generation === "number") benGen = data.generation;
            benText = "";
            const bodyEl = group.querySelector("#liveBenBody");
            if(bodyEl) bodyEl.innerHTML = "";
            revealLiveBenSection(group, {
              kicker: data.draft
                ? "BEN — live synthesis (updates as more analysts finish…)"
                : "BEN — final verdict"
            });
            continue;
          }
          if (data.type === "ben_chunk") {
            const g = data.generation;
            if(typeof g === "number" && g !== benGen) continue;
            benText += data.content || "";
            flushBenMarkdown(group.querySelector("#liveBenBody"), benText);
            continue;
          }
          /* Legacy NDJSON fallback */
          if (data.type === "status") {
            setPipelineBar(0, data.content);
            continue;
          }
          if (data.type === "chunk") {
            benText += data.content || "";
            revealLiveBenSection(group, { kicker: "BEN — Supreme Judge" });
            flushBenMarkdown(group.querySelector("#liveBenBody"), benText);
            continue;
          }
          if (data.type === "done") {
            finalData = data;
          if(data.session_cost && data.session_cost.usd_saved_vs_full != null){
            addMoneySaved(data.session_cost.usd_saved_vs_full);
          }
          }
        } catch (lineErr) {
          console.error("[stream line handler] error; skipping line", lineErr, line);
          continue;
        }
      }
      chat.scrollTop = chat.scrollHeight;
    }

    resetPipelineBar();
    if (finalData) {
      group.remove();
      appendAssistantMessage(
        finalData.final,
        finalData.round1,
        finalData.round2,
        finalData.consensus_data,
        finalData.category
      );
      updateStatus("Turbo synthesis complete");
      loadSessions();
    }
  } catch(err) {
    updateStatus("Error: " + err.message, true);
    resetPipelineBar();
    console.error("[stream] fatal fetch/read error", err);
  } finally {
    isSending = false;
    setComposerSending(false);
  }
}

// ========================
// INITIALIZATION
// ========================

function triggerFileUpload(){
    document.getElementById('fileInput').click();
}

function setDocumentBadge(visible){
  const el = document.getElementById('docLoadedBadge');
  if(el) el.classList.toggle('hidden', !visible);
}

async function handleFileUpload(input){
    const file = input.files[0];
    if(!file || !currentSession) {
      if(file && !currentSession) updateStatus("Select or create a chat first to attach a file.", true);
      return;
    }
    
    updateStatus(`Uploading ${file.name}…`);
    const formData = new FormData();
    formData.append('session_id', currentSession);
    formData.append('file', file, file.name);
    
    try {
        const response = await fetch(SERVER + "/upload", {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if(data.success){
            updateStatus(`Document indexed: ${file.name} (${data.extracted_length} chars)`, false);
            setDocumentBadge(true);
        } else {
            updateStatus(`Upload failed: ${data.error}`, true);
        }
    } catch(err) {
        updateStatus(`Upload error: ${err.message}`, true);
    } finally {
        input.value = "";
    }
}

function toggleWebSearch(){
    webSearchActive = !webSearchActive;
    document.getElementById('webSearchBtn').classList.toggle('active', webSearchActive);
    updateStatus(webSearchActive ? "Search: Web search enabled" : "Search: Web search disabled");
}

function showEngineerInfo(){
    updateStatus("Engineer: Python Code Interpreter is active in assistant responses.");
}

async function testSimpleAI(){
    try {
        const response = await fetch(SERVER + "/test-ai", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                prompt: "Reply with exactly: Test route is working."
            })
        });
        const data = await response.json();
        if(data.success){
            alert("Test Simple AI OK:\n\n" + (data.result || ""));
        } else {
            alert("Test Simple AI failed:\n\n" + (data.error || "Unknown error"));
        }
    } catch(err) {
        alert("Test Simple AI network error:\n\n" + err.message);
    }
}

async function runCreditCheck(){
    try {
        const response = await fetch(SERVER + "/credit-check", { method: "POST" });
        const data = await response.json();
        if(!data.success || !data.checks){
            alert("Credit Check failed: " + (data.error || "unknown error"));
            return;
        }
        const checks = data.checks;
        const line = (k, label) => {
            const c = checks[k] || {};
            return `${label}: ${c.status || "Unknown"}`;
        };
        alert(
            "Credit Check\n\n" +
            line("openai", "OpenAI") + "\n" +
            line("gemini", "Gemini") + "\n" +
            line("anthropic", "Anthropic")
        );
    } catch(err) {
        alert("Credit Check error:\n\n" + err.message);
    }
}

async function executeCode(code, container){
    const outputDiv = document.createElement('div');
    outputDiv.className = 'code-output';
    outputDiv.textContent = "Running code in sandbox...";
    container.appendChild(outputDiv);
    
    try {
        const response = await fetch(SERVER + "/execute", {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ code: code, language: 'python' })
        });
        const data = await response.json();
        if(data.success){
            outputDiv.textContent = data.stdout || (data.stderr ? "Error: " + data.stderr : "Code executed successfully (no output).");
            if(data.exit_code !== 0 && data.stderr) {
                outputDiv.style.color = "#f87171";
                outputDiv.textContent = data.stderr;
            }
        } else {
            outputDiv.textContent = "Execution Error: " + data.error;
            outputDiv.style.color = "#f87171";
        }
    } catch(err) {
        outputDiv.textContent = "System Error: " + err.message;
        outputDiv.style.color = "#f87171";
    }
}

function wireAuthModalControls() {
  /* Login uses onclick="loginUser()" on #authLoginBtn — do not add a second click listener or login runs twice. */
  const passEl = document.getElementById("authPassword");
  if (passEl && !passEl.dataset.authWired) {
    passEl.dataset.authWired = "1";
    passEl.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      loginUser();
    });
  }
}

window.addEventListener("load", () => {
  const inp = document.getElementById("userInput");
  wireAuthModalControls();
  if (inp) {
    inp.addEventListener("input", function resizeComposer() {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 200) + "px";
    });
    inp.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      if (e.shiftKey) return;
      e.preventDefault();
      sendMessage();
    });
  }
  refreshAuthBadge();
  const tokLoad = localStorage.getItem(AUTH_STORAGE_KEY);
  if (tokenLooksValid(tokLoad)) {
    hideAuthModal();
    showWelcomeScreen(false);
  }
  syncUrlWithAuthState();
  const path = normalizeAppPath();
  if (path === "/login") {
    openAuthModal();
  } else {
    loadSessions();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") syncUrlWithAuthState();
});

async function handleUpgrade() {
    // In a real app, this would fetch a session URL from the backend
    const checkoutUrl = "https://checkout.stripe.com/pay/placeholder"; 
    window.location.href = checkoutUrl;
}

async function verifyLicense() {
    const key = document.getElementById('licenseKey').value.trim();
    if(!key) return alert("Please enter a license key");
    
    // Placeholder for license verification
    updateStatus("Verifying license key...");
    setTimeout(() => {
        alert("License key verified! You are now a Pro user.");
        document.getElementById('upgrade-modal').classList.remove('visible');
        // In a real app, we would update the DB and reload
        location.reload(); 
    }, 1500);
}

async function runSmartTool(endpoint, context, container){
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'code-output';
    loadingDiv.textContent = `Processing ${endpoint.replace('_', ' ')}...`;
    container.parentElement.appendChild(loadingDiv);
    
    try {
        const response = await fetch(SERVER + "/ensemble/" + endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: context, session_id: currentSession })
        });
        const data = await response.json();
        if(data.success){
            if(endpoint === 'export_pdf'){
                loadingDiv.innerHTML = `✅ PDF Generated: <a href="${SERVER}/${data.filename}" target="_blank" style="color:#3b82f6">${data.filename}</a>`;
            } else {
                loadingDiv.innerHTML = marked.parse(data.code || data.examples);
            }
        } else {
            loadingDiv.textContent = "Error: " + data.error;
            loadingDiv.style.color = "#f87171";
        }
    } catch(err) {
        loadingDiv.textContent = "System Error: " + err.message;
        loadingDiv.style.color = "#f87171";
    }
}

document.addEventListener("keydown", (e) => {
  if(e.key === "Escape"){
    closeNavDrawer();
    const pm = document.getElementById("profileModal");
    if(pm && pm.classList.contains("visible")) hideProfileModal();
    const am = document.getElementById("authModal");
    if(am && am.classList.contains("visible")) hideAuthModal();
    const lm = document.getElementById("limit-tier-modal");
    if(lm && lm.classList.contains("visible")) closeLimitTierModal();
    return;
  }
});

/* Explicit globals for inline onclick / DevTools (classic script hoisting can fail in some bundlers). */
window.loginUser = loginUser;
window.submitAuthLogin = submitAuthLogin;
window.resetBenAppCache = resetBenAppCache;