/**
 * AdSpark Flask Dashboard JavaScript Engine.
 * Handles Theme Toggling, Figure Lightbox, and Interactive CTR Predictor with Mathematical Trace.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initLightbox();
  initPredictor();
});

/* ---------- Theme Switcher ---------- */
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle');
  const storedTheme = localStorage.getItem('adspark-theme') || 'dark';

  document.documentElement.setAttribute('data-theme', storedTheme);
  updateThemeBtnText(storedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('adspark-theme', next);
      updateThemeBtnText(next);
    });
  }
}

function updateThemeBtnText(theme) {
  const btnText = document.getElementById('theme-toggle-text');
  if (btnText) {
    btnText.textContent = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
  }
}

/* ---------- Image Lightbox ---------- */
function initLightbox() {
  const images = document.querySelectorAll('.figure-card img, .card-body img');
  if (!images.length) return;

  const modal = document.createElement('div');
  modal.className = 'lightbox-modal';
  modal.innerHTML = '<img src="" alt="Expanded View">';
  document.body.appendChild(modal);

  const modalImg = modal.querySelector('img');

  images.forEach(img => {
    img.addEventListener('click', () => {
      modalImg.src = img.src;
      modal.classList.add('active');
    });
  });

  modal.addEventListener('click', () => {
    modal.classList.remove('active');
  });
}

/* ---------- CTR Predictor AJAX & Mathematical Trace ---------- */
function initPredictor() {
  const form = document.getElementById('ctr-predict-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    const resultBox = document.getElementById('prediction-output');
    if (resultBox) {
      resultBox.style.opacity = '0.5';
    }

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const res = await resp.json();

      if (res.success && resultBox) {
        const p = res.prediction;
        const elPct = document.getElementById('res-pct');
        const elLabel = document.getElementById('res-label');
        const elTier = document.getElementById('res-tier');
        const elModel = document.getElementById('res-model');
        const elCi = document.getElementById('res-ci');
        const elMeter = document.getElementById('res-meter');
        const elDrivers = document.getElementById('res-drivers');

        // Mathematical trace elements
        const trLinear = document.getElementById('trace-linear');
        const trInter = document.getElementById('trace-inter');
        const trLogOdds = document.getElementById('trace-logodds');
        const trShift = document.getElementById('trace-shift');
        const trBayes = document.getElementById('trace-bayes');

        if (elPct) elPct.textContent = p.ctr_percentage + '%';
        if (elLabel) {
          elLabel.textContent = p.prediction_label;
          elLabel.style.color = p.predicted_click === 1 ? 'var(--green)' : 'var(--red)';
        }
        if (elTier) {
          elTier.textContent = p.confidence_tier;
          elTier.style.color = p.tier_color === 'green' ? 'var(--green)' : (p.tier_color === 'red' ? 'var(--red)' : 'var(--accent)');
        }
        if (elModel && p.model_selected) {
          elModel.textContent = 'Model: ' + p.model_selected;
        }
        if (elCi && p.wilson_ci_95) {
          elCi.textContent = `95% Wilson CI: [${p.wilson_ci_95.lower}%, ${p.wilson_ci_95.upper}%]`;
        }

        if (trLinear) trLinear.textContent = (p.linear_component >= 0 ? '+' : '') + p.linear_component.toFixed(3);
        if (trInter) trInter.textContent = (p.interaction_component >= 0 ? '+' : '') + p.interaction_component.toFixed(3);
        if (trLogOdds) trLogOdds.textContent = (p.log_odds >= 0 ? '+' : '') + p.log_odds.toFixed(3);
        if (trShift) trShift.textContent = p.downsampling_log_odds_shift.toFixed(3);
        if (trBayes) trBayes.textContent = p.bayesian_smoothed_ctr + '%';

        if (elMeter) {
          elMeter.style.width = Math.min(100, Math.max(5, p.ctr_percentage * 2.5)) + '%';
          elMeter.style.background = p.predicted_click === 1 
            ? 'linear-gradient(90deg, var(--accent) 0%, var(--green) 100%)' 
            : 'linear-gradient(90deg, var(--orange) 0%, var(--red) 100%)';
        }

        if (elDrivers && p.key_drivers) {
          if (p.key_drivers.length === 0) {
            elDrivers.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">Standard Baseline Features</p>';
          } else {
            elDrivers.innerHTML = p.key_drivers.map(d => `
              <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-color); font-size: 0.88rem;">
                <span>${d.factor}</span>
                <span style="font-weight: 700; color: ${d.impact.includes('Positive') ? 'var(--green)' : (d.impact.includes('Negative') ? 'var(--red)' : 'var(--accent)')}">${d.weight}</span>
              </div>
            `).join('');
          }
        }

        resultBox.style.opacity = '1';
        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    } catch (err) {
      console.error('Prediction error:', err);
    }
  });
}
