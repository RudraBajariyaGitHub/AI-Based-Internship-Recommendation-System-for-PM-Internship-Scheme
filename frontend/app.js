const form = document.querySelector("#profileForm");
const submitBtn = document.querySelector("#submitBtn");
const loadingState = document.querySelector("#loadingState");
const resultsGrid = document.querySelector("#resultsGrid");
const resultsTitle = document.querySelector("#resultsTitle");
const pipelineList = document.querySelector("#pipelineList");
const systemStatus = document.querySelector("#systemStatus");
const isLocalHost = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
const servedByFastApi = isLocalHost && window.location.port === "8000";
const API_BASE =
  servedByFastApi ? "" : "http://127.0.0.1:8000";

const selectMap = {
  branches: document.querySelector("#branchSelect"),
  locations: document.querySelector("#locationSelect"),
  tiers: document.querySelector("#tierSelect"),
  regions: document.querySelector("#regionSelect"),
  work_modes: document.querySelector("#workModeSelect"),
};

let lastProfile = null;

const fallbackOptions = {
  branches: [
    "Computer Science",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Electronics & Communication",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology",
    "Other",
  ],
  locations: ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Noida", "Gurgaon", "Kolkata", "Remote", "Other"],
  tiers: ["Tier-1", "Tier-2", "Tier-3"],
  regions: ["Urban", "Rural"],
  work_modes: ["Remote", "Hybrid", "On-site", "No Preference"],
};

const fallbackPipeline = [
  {
    stage: "Candidate retrieval",
    description: "Compares student skills and interests with internship requirements.",
  },
  {
    stage: "ML ranking",
    description: "Ranks internships using skills, domain fit, location, CGPA, stipend, and popularity.",
  },
  {
    stage: "Fairness and explainability",
    description: "Balances visibility and explains why each internship was recommended.",
  },
];

function safeText(value, fallback = "Not specified") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function numberFormat(value) {
  return new Intl.NumberFormat("en-IN").format(Number(value || 0));
}

function populateSelect(select, values, preferred) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === preferred) option.selected = true;
    select.append(option);
  });
}

async function fetchJson(url, options) {
  const response = await fetch(`${API_BASE}${url}`, options);
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  return response.json();
}

function renderPipeline(items = []) {
  const visibleItems = items.filter((item) => safeText(item.stage).toLowerCase() !== "feedback learning");

  pipelineList.innerHTML = visibleItems
    .map(
      (item, index) => `
        <article class="pipeline-item">
          <span class="pipeline-step">${index + 1}</span>
          <div>
            <strong>${safeText(item.stage)}</strong>
            <p>${safeText(item.description)}</p>
          </div>
        </article>
      `
    )
    .join("");
}

async function boot() {
  try {
    const [health, options, modelInfo] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/options"),
      fetchJson("/api/model-info"),
    ]);

    document.querySelector("#studentCount").textContent = numberFormat(health.students);
    document.querySelector("#internshipCount").textContent = numberFormat(health.internships);

    populateSelect(selectMap.branches, options.branches || [], "Computer Science");
    populateSelect(selectMap.locations, options.locations || [], "Remote");
    populateSelect(selectMap.tiers, options.tiers || [], "Tier-2");
    populateSelect(selectMap.regions, options.regions || [], "Urban");
    populateSelect(selectMap.work_modes, options.work_modes || [], "No Preference");

    renderPipeline(modelInfo.pipeline || []);

    systemStatus.textContent = "Ready";
    systemStatus.classList.add("ready");
  } catch (error) {
    populateSelect(selectMap.branches, fallbackOptions.branches, "Computer Science");
    populateSelect(selectMap.locations, fallbackOptions.locations, "Remote");
    populateSelect(selectMap.tiers, fallbackOptions.tiers, "Tier-2");
    populateSelect(selectMap.regions, fallbackOptions.regions, "Urban");
    populateSelect(selectMap.work_modes, fallbackOptions.work_modes, "No Preference");
    renderPipeline(fallbackPipeline);
    systemStatus.textContent = "API offline";
    renderError(`Could not initialize the interface: ${error.message}`);
  }
}

function formPayload() {
  const data = new FormData(form);
  const payload = Object.fromEntries(data.entries());
  payload.cgpa = Number(payload.cgpa);
  payload.top_k = Number(payload.top_k);
  payload.objective_learning = Number(payload.objective_learning);
  payload.objective_career_fit = Number(payload.objective_career_fit);
  payload.objective_compensation = Number(payload.objective_compensation);
  if (!payload.student_id) {
    payload.student_id = `WEB-${Date.now()}`;
  }
  return payload;
}

function renderError(message) {
  resultsTitle.textContent = "Something needs attention";
  resultsGrid.innerHTML = `<div class="error-box">${safeText(message)}</div>`;
}

function scorePercent(rec) {
  const explanationPct = rec.explanation?.match_percentage;
  if (Number.isFinite(explanationPct) && explanationPct > 0) return explanationPct;
  return Math.round((rec.scores?.policy_score || 0) * 100);
}

function renderSkillChips(skills, type) {
  if (!skills || !skills.length) {
    return `<span class="skill-chip">None listed</span>`;
  }
  return skills
    .slice(0, 8)
    .map((skill) => `<span class="skill-chip ${type}">${safeText(skill)}</span>`)
    .join("");
}

function renderRecommendation(rec) {
  const pct = Math.max(0, Math.min(100, scorePercent(rec)));
  const reasons = rec.explanation?.reasons || [];
  const actions = rec.explanation?.improvement_actions || [];
  const objectives = rec.objective_breakdown || {};

  return `
    <article class="recommendation-card">
      <div class="card-top">
        <div>
          <p class="eyebrow">Recommendation ${rec.rank}</p>
          <h3>${safeText(rec.title)} at ${safeText(rec.company)}</h3>
          <div class="meta">
            <span>${safeText(rec.domain)}</span>
            <span>${safeText(rec.location)}</span>
            <span>${safeText(rec.work_type)}</span>
            <span>${safeText(rec.duration_weeks)} weeks</span>
          </div>
        </div>
        <div class="rank-badge">#${rec.rank}</div>
      </div>

      <div class="card-body">
        <div class="score-ring" style="--score: ${pct}%">
          <div class="score-ring-inner">
            <span>
              <strong>${pct}%</strong>
              <span class="score-label">skill match</span>
            </span>
          </div>
        </div>

        <div>
          <div class="detail-grid">
            <div class="detail-tile">
              <small>Stipend</small>
              <strong>INR ${numberFormat(rec.stipend)}/mo</strong>
            </div>
            <div class="detail-tile">
              <small>Location</small>
              <strong>${safeText(rec.location)}</strong>
            </div>
            <div class="detail-tile">
              <small>Work mode</small>
              <strong>${safeText(rec.work_type)}</strong>
            </div>
          </div>

          <p>${safeText(rec.description, "No description provided.")}</p>

          <strong>Why this fits</strong>
          <ul class="reason-list">
            ${reasons.slice(0, 4).map((reason) => `<li>${safeText(reason)}</li>`).join("")}
          </ul>

          <strong>Matched skills</strong>
          <div class="skill-row">${renderSkillChips(rec.explanation?.matched_skills, "matched")}</div>

          <strong>Growth suggestions</strong>
          <div class="skill-row">
            ${actions.length ? actions.slice(0, 3).map((action) => `<span class="skill-chip missing">${safeText(action)}</span>`).join("") : `<span class="skill-chip matched">Profile is already strong for this role</span>`}
          </div>

          <div class="detail-grid">
            <div class="detail-tile">
              <small>Learning</small>
              <strong>${Math.round((objectives.learning_score || 0) * 100)}%</strong>
            </div>
            <div class="detail-tile">
              <small>Career fit</small>
              <strong>${Math.round((objectives.career_fit_score || 0) * 100)}%</strong>
            </div>
            <div class="detail-tile">
              <small>Compensation</small>
              <strong>${Math.round((objectives.compensation_score || 0) * 100)}%</strong>
            </div>
          </div>

        </div>
      </div>
    </article>
  `;
}

function renderResults(data) {
  lastProfile = data.profile;
  const recommendations = data.recommendations || [];
  resultsTitle.textContent = recommendations.length
    ? `Top ${recommendations.length} internships for ${safeText(data.profile?.branch, "your profile")}`
    : "No recommendations found";
  resultsGrid.innerHTML = recommendations.length
    ? recommendations.map(renderRecommendation).join("")
    : `<div class="error-box">The engine could not find suitable matches. Try adding more skills or widening location/work-mode preferences.</div>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = formPayload();
  submitBtn.disabled = true;
  loadingState.hidden = false;
  resultsGrid.innerHTML = "";
  resultsTitle.textContent = "Ranking internships";

  try {
    const data = await fetchJson("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResults(data);
    document.querySelector("#resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    renderError(error.message);
  } finally {
    submitBtn.disabled = false;
    loadingState.hidden = true;
  }
});

boot();
