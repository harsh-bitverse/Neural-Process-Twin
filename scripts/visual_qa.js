/**
 * Visual QA automation script using puppeteer-core and Microsoft Edge.
 * Captures representative application states for Milestone 07A.
 */

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SCREENSHOT_DIR = path.resolve('reports/screenshots');

async function runVisualQA() {
  console.log('Launching headless Microsoft Edge via puppeteer-core...');
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: true,
    args: [
      '--no-proxy-server',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1440,1080',
    ],
    defaultViewport: { width: 1440, height: 1080 },
  });

  const page = await browser.newPage();
  console.log('Navigating to http://127.0.0.1:8000/ ...');
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle0' });

  // 1. Initial State
  console.log('Capturing Screenshot 1: Initial Dashboard...');
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_initial_dashboard.png') });

  // 2. Provenance Modal
  console.log('Capturing Screenshot 2: Audited Provenance Modal...');
  await page.click('.btn-provenance');
  await page.waitForSelector('.modal-content', { visible: true });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '02_provenance_modal.png') });
  await page.click('.modal-close-btn');
  await page.waitForSelector('.modal-content', { hidden: true });

  // 3. Nominal Composite - Run Simulation
  console.log('Capturing Screenshot 3: Nominal Composite Simulation...');
  await page.click('.btn-run-simulation');
  await page.waitForSelector('.status-decision-card', { visible: true });
  await new Promise(r => setTimeout(r, 800));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '03_nominal_composite_simulation.png') });

  // 4. Uncertainty Tab
  console.log('Capturing Screenshot 4: Epistemic Uncertainty Field...');
  const tabs = await page.$$('.field-tab-btn');
  for (const tab of tabs) {
    const text = await (await tab.getProperty('innerText')).jsonValue();
    if (text.includes('Uncertainty')) {
      await tab.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '04_epistemic_uncertainty.png') });

  // 5. Hover Readout on Canvas
  console.log('Capturing Screenshot 5: Canvas Hover Readout...');
  for (const tab of tabs) {
    const text = await (await tab.getProperty('innerText')).jsonValue();
    if (text.includes('Temperature')) {
      await tab.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 500));
  const canvas = await page.waitForSelector('.field-canvas');
  if (canvas) {
    await canvas.hover();
    await page.waitForSelector('.field-hover-readout', { visible: true });
    await new Promise(r => setTimeout(r, 400));
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '05_canvas_hover_inspection.png') });
  }

  // 6. High Contrast (30:1) Fallback
  console.log('Capturing Screenshot 6: High Contrast Fallback Succeeded...');
  const pills = await page.$$('.preset-pill');
  for (const pill of pills) {
    const text = await (await pill.getProperty('innerText')).jsonValue();
    if (text.includes('High Contrast')) {
      await pill.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 400));
  await page.click('.btn-run-simulation');
  await page.waitForFunction(() => {
    const el = document.querySelector('.status-decision-card');
    return el && el.innerText.includes('FALLBACK SUCCEEDED');
  });
  await new Promise(r => setTimeout(r, 800));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '06_high_contrast_fallback.png') });

  // 7. Research Comparison Mode
  console.log('Capturing Screenshot 7: Research Comparison Mode...');
  const modeBtns = await page.$$('.mode-btn');
  for (const btn of modeBtns) {
    const text = await (await btn.getProperty('innerText')).jsonValue();
    if (text.includes('Comparison')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 400));
  await page.click('.btn-run-simulation');
  await page.waitForSelector('.comparison-view-card', { visible: true });
  await new Promise(r => setTimeout(r, 800));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '07_research_comparison_mode.png') });

  // 8. 64x64 Fine Mesh Direct Solve
  console.log('Capturing Screenshot 8: 64x64 Fine Mesh Direct Solve...');
  for (const btn of modeBtns) {
    const text = await (await btn.getProperty('innerText')).jsonValue();
    if (text.includes('Hybrid')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 400));
  const pills2 = await page.$$('.preset-pill');
  for (const pill of pills2) {
    const text = await (await pill.getProperty('innerText')).jsonValue();
    if (text.includes('64x64')) {
      await pill.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 400));
  await page.click('.btn-run-simulation');
  await page.waitForSelector('.status-decision-card', { visible: true });
  await new Promise(r => setTimeout(r, 800));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '08_grid_64_fine_mesh.png') });

  // 9. Mobile Responsive Viewport
  console.log('Capturing Screenshot 9: Mobile Responsive Viewport (768x1024)...');
  await page.setViewport({ width: 768, height: 1024 });
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '09_responsive_tablet_view.png') });

  console.log('Visual QA complete! All 9 screenshots saved to reports/screenshots/.');
  await browser.close();
}

runVisualQA().catch(err => {
  console.error('Visual QA failed with error:', err);
  process.exit(1);
});
