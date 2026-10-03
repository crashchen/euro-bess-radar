// Width matrix + resize persistence (checks 35-38, 48-51; Data Trust at 960).
// Usage: NODE_PATH=<dir with playwright> node width_matrix.js <label> <port> [forward-only]
// Populates every panel once at 1440 px (sidebar expanded), then resizes the
// SAME page through 1440/1280/960/390 and inspects each tab. Resizing must not
// rerun the script, re-solve, expose stale warnings or drop downloads (check 38).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const FIX = path.join(__dirname, '..', 'fixtures');
const WIDTHS = [1440, 1280, 960, 390];
const TABS = ['Market Overview', 'Revenue Estimation', 'Forward Scenarios', 'Renewable Correlation',
  'Data Trust', 'Simulation Cockpit'];
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function settle(page, extra = 600, timeout = 900000) {
  await page.waitForTimeout(extra);
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout },
  );
  await page.waitForTimeout(400);
}

async function sidebarOpen(page, open) {
  const sb = page.locator('[data-testid="stSidebar"]');
  const expanded = (await sb.getAttribute('aria-expanded')) === 'true';
  if (expanded === open) return;
  const toggle = open
    ? page.locator('[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] button').first()
    : page.locator('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarHeader"] button').first();
  await toggle.click();
  await page.waitForTimeout(800);
}

async function tab(page, name) {
  await page.locator('button[role="tab"]', { hasText: name }).first().click();
  await page.waitForTimeout(700);
}

function benchmarkChart(page) {
  return panel(page).locator('[data-testid="stExpander"]', { hasText: 'External trader revenue benchmark' })
    .locator('[data-testid="stPlotlyChart"]').first();
}

function panel(page) {
  return page.locator('[role="tabpanel"]:not([hidden])').first();
}

async function expand(scope, title) {
  const exp = scope.locator('[data-testid="stExpander"]', { hasText: title }).first();
  if ((await exp.locator('details').first().getAttribute('open')) === null) {
    await exp.locator('summary').first().click();
    await exp.page().waitForTimeout(700);
  }
  return exp;
}

async function expandAll(page) {
  // Open every closed expander in the active tab (nested ones appear as we go).
  for (let pass = 0; pass < 4; pass += 1) {
    const closed = panel(page).locator('[data-testid="stExpander"] details:not([open]) > summary');
    const n = await closed.count();
    if (!n) return;
    for (let i = n - 1; i >= 0; i -= 1) {
      await closed.nth(i).click().catch(() => {});
      await page.waitForTimeout(250);
    }
  }
}

async function pick(scope, label, option) {
  const box = scope.locator('[data-testid="stSelectbox"]', { hasText: label }).first();
  await box.locator('div[data-baseweb="select"]').click();
  await scope.page().getByRole('option', { name: option }).first().click();
  await settle(scope.page());
}

const PROBE = () => {
  const panel = [...document.querySelectorAll('[role="tabpanel"]')].find((p) => !p.hidden && p.offsetParent !== null)
    || document.querySelector('[data-testid="stMain"]');
  const vis = (el) => el.offsetParent !== null && el.getClientRects().length > 0;
  const cut = (el) => el.scrollWidth > el.clientWidth + 1;
  const W = innerWidth;
  const metrics = [...panel.querySelectorAll('[data-testid="stMetric"]')].filter(vis).map((m) => {
    const leaves = [...m.querySelectorAll('*')].filter((n) => n.children.length === 0 && n.textContent.trim());
    const r = m.getBoundingClientRect();
    return { text: m.innerText.replace(/\n/g, ' | ').slice(0, 90), truncated: leaves.filter(cut).map((n) => n.textContent.trim()),
      offscreen: r.right > W + 1, width: Math.round(r.width) };
  });
  const kpis = [...panel.querySelectorAll('.cockpit-kpi-label,.cockpit-kpi-value,.cockpit-kpi-help,.cockpit-health-value,.cockpit-health-label')]
    .filter(vis).map((n) => ({ cls: n.className, text: n.textContent.trim().slice(0, 70), truncated: cut(n),
      offscreen: n.getBoundingClientRect().right > W + 1 }));
  const text = panel.innerText;
  return {
    viewport: `${W}x${innerHeight}`,
    sidebar: document.querySelector('[data-testid="stSidebar"]').getAttribute('aria-expanded'),
    docOverflowX: document.documentElement.scrollWidth > W,
    metricCount: metrics.length,
    badMetrics: metrics.filter((m) => m.truncated.length || m.offscreen),
    kpiCount: kpis.length,
    badKpis: kpis.filter((k) => k.truncated || k.offscreen),
    minMetricWidth: metrics.length ? Math.min(...metrics.map((m) => m.width)) : null,
    stale: (text.match(/Inputs changed since the last run/g) || []).length,
    downloads: [...panel.querySelectorAll('[data-testid="stDownloadButton"]')].filter(vis).length,
    statusWidget: !!document.querySelector('[data-testid="stStatusWidget"]'),
    metricsSample: metrics.slice(0, 40).map((m) => m.text),
  };
};

async function cardShots(page, dir, prefix, limit = 30) {
  // One screenshot per visible metric row / cockpit card grid in the active tab.
  const handles = await panel(page).evaluateHandle((p) => {
    const vis = (el) => el.offsetParent !== null && el.getClientRects().length > 0;
    const groups = new Set();
    p.querySelectorAll('[data-testid="stMetric"]').forEach((m) => {
      if (!vis(m)) return;
      const g = m.closest('[data-testid="stHorizontalBlock"]') || m.closest('[data-testid="stVerticalBlock"]') || m;
      groups.add(g);
    });
    p.querySelectorAll('.cockpit-kpi-grid,.cockpit-health-grid').forEach((g) => vis(g) && groups.add(g));
    return [...groups];
  });
  const props = await handles.getProperties();
  let i = 0;
  for (const h of props.values()) {
    const el = h.asElement();
    if (!el || i >= limit) continue;
    try {
      await el.scrollIntoViewIfNeeded({ timeout: 5000 });
      await el.screenshot({ path: path.join(dir, `${prefix}-${String(i).padStart(2, '0')}.png`), timeout: 10000 });
      i += 1;
    } catch (_) { /* element detached or zero-size */ }
  }
  return i;
}

async function setDates(page, a, b) {
  const dates = page.locator('[data-testid="stSidebar"] [data-testid="stDateInput"] input');
  for (const [i, v] of [[0, a], [1, b]]) {
    await dates.nth(i).click();
    await dates.nth(i).press('ControlOrMeta+a');
    await dates.nth(i).pressSequentially(v);
    await dates.nth(i).press('Enter');
    await page.keyboard.press('Escape');
    await settle(page, 300);
  }
}

async function uploadMain(page, label, file) {
  const up = panel(page).locator('[data-testid="stFileUploader"]', { hasText: label }).first();
  await up.locator('input[type="file"]').setInputFiles(file);
  await settle(page, 1000);
}

async function populate(page, rec, forwardOnly) {
  await sidebarOpen(page, true);
  await setDates(page, '2026/06/22', '2026/06/28');
  const form = page.locator('[data-testid="stSidebar"] [data-testid="stForm"]').first();
  const capex = form.locator('[data-testid="stNumberInput"]', { hasText: 'CapEx' }).locator('input');
  await capex.fill('2000');
  await capex.press('Tab');
  await form.locator('button', { hasText: 'Apply BESS parameters' }).click();
  await settle(page);
  log('fetch data');
  await page.locator('[data-testid="stSidebar"] button', { hasText: 'Fetch Data' }).click();
  await page.waitForSelector('button[role="tab"]', { timeout: 300000 });
  await settle(page, 1500);
  rec.fetchAlerts = await page.locator('[data-testid="stSidebar"] [data-testid="stAlert"]').allInnerTexts();

  if (!forwardOnly) {
    await sidebarOpen(page, true);
    await expand(page.locator('[data-testid="stSidebar"]'), 'Auto-Fetch Ancillary Data');
    log('fetch ancillary');
    await page.locator('[data-testid="stSidebar"] button:visible', { hasText: 'Fetch ancillary data for' }).first().click();
    await settle(page, 1500);
    rec.ancillaryAlerts = await page.locator('[data-testid="stSidebar"] [data-testid="stAlert"]').allInnerTexts();
  }

  log('forward uploads');
  await tab(page, 'Forward Scenarios');
  await uploadMain(page, 'Forward-curve CSV', path.join(FIX, 'SYNTHETIC_forward_DE_LU_2y.csv'));
  await expand(panel(page), 'External trader revenue benchmark');
  await uploadMain(page, 'External annual revenue benchmark CSV', path.join(FIX, 'SYNTHETIC_benchmark_DE_LU_2y.csv'));
  if (forwardOnly) return;

  log('project case');
  await tab(page, 'Revenue Estimation');
  await page.locator('button', { hasText: 'Run Project Case' }).click();
  await settle(page, 2000);
  await panel(page).getByText('Run complete', { exact: false }).first().waitFor({ timeout: 5000 }).catch(() => {});

  await tab(page, 'Simulation Cockpit');
  log('multi-day');
  const md = await expand(panel(page), 'Multi-day replay summary');
  await md.locator('button', { hasText: 'Run multi-day replay' }).click();
  await settle(page, 1500);
  log('frontier');
  const fr = await expand(panel(page), 'net-revenue frontier');
  await fr.locator('button', { hasText: 'Run frontier sweep' }).click();
  await settle(page, 1500);
  log('floor');
  const fl = await expand(panel(page), 'Contracted floor versus merchant cash flow');
  await fl.locator('button', { hasText: 'Run contracted-floor comparison' }).click();
  await settle(page, 1500);
  log('forecast policy');
  const fp = await expand(panel(page), 'Forecast-driven IDA policy');
  const reserve = fp.locator('[data-testid="stSelectbox"]', { hasText: 'Reserve product' }).first();
  if (await reserve.count()) {
    await reserve.locator('div[data-baseweb="select"]').click();
    const opts = await page.getByRole('option').allInnerTexts();
    rec.reserveOptions = opts;
    const choice = opts.find((o) => /FCR/.test(o)) || opts.find((o) => !/none/i.test(o));
    if (choice) await page.getByRole('option', { name: choice, exact: true }).click();
    else await page.keyboard.press('Escape');
    await settle(page);
  }
  const stoch = fp.locator('[data-testid="stCheckbox"]', { hasText: 'Include stochastic policy' }).first();
  if (await stoch.count()) {
    await stoch.locator('input').check({ force: true }).catch(async () => stoch.click());
    await settle(page);
  }
  const t0 = Date.now();
  await fp.locator('button', { hasText: 'Run forecast policy' }).click();
  await settle(page, 3000, 1800000);
  rec.forecastPolicySeconds = Math.round((Date.now() - t0) / 1000);
  log('populated');
}

(async () => {
  const [label, port, mode] = process.argv.slice(2);
  const forwardOnly = mode === 'forward-only';
  const OUT = path.join(__dirname, 'matrix', label);
  fs.mkdirSync(OUT, { recursive: true });
  const rec = { label, port: Number(port), forwardOnly, populate: {}, widths: {} };
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, acceptDownloads: true });
    const page = await context.newPage();
    let scriptRuns = 0;
    page.on('websocket', (ws) => ws.on('framesent', () => { scriptRuns += 1; }));
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 60000 });
    await settle(page, 1500);
    await populate(page, rec.populate, forwardOnly);
    const tabs = forwardOnly ? ['Forward Scenarios'] : TABS;
    for (const t of tabs) { await tab(page, t); await expandAll(page); }
    await settle(page);
    for (const width of WIDTHS) {
      const framesBefore = scriptRuns;
      await page.setViewportSize({ width, height: 900 });
      await page.waitForTimeout(1500);
      if (width < 600) await sidebarOpen(page, false); else await sidebarOpen(page, true);
      const w = { framesSentDuringResize: scriptRuns - framesBefore, tabs: {} };
      for (const t of tabs) {
        await tab(page, t);
        const dir = path.join(OUT, String(width));
        fs.mkdirSync(dir, { recursive: true });
        const slug = t.toLowerCase().replace(/[^a-z]+/g, '-');
        const probe = await page.evaluate(PROBE);
        probe.cardShots = await cardShots(page, dir, slug);
        if (t === 'Forward Scenarios') {
          const chart = benchmarkChart(page);
          if (await chart.count()) {
            await chart.scrollIntoViewIfNeeded();
            await chart.screenshot({ path: path.join(dir, `${slug}-benchmark-chart.png`) });
          }
        }
        w.tabs[t] = probe;
        log(label, width, t, `metrics=${probe.metricCount} bad=${probe.badMetrics.length} kpis=${probe.kpiCount} badK=${probe.badKpis.length} stale=${probe.stale} dl=${probe.downloads} ovX=${probe.docOverflowX}`);
      }
      w.framesSentTotal = scriptRuns - framesBefore;
      rec.widths[width] = w;
      fs.writeFileSync(path.join(OUT, 'matrix.json'), JSON.stringify(rec, null, 1));
    }
    if (forwardOnly) {
      log('longer range');
      await page.setViewportSize({ width: 1280, height: 900 });
      await sidebarOpen(page, true);
      await tab(page, 'Forward Scenarios');
      await uploadMain(page, 'Forward-curve CSV', path.join(FIX, 'SYNTHETIC_forward_DE_LU_20y.csv'));
      await uploadMain(page, 'External annual revenue benchmark CSV', path.join(FIX, 'SYNTHETIC_benchmark_DE_LU_20y.csv'));
      for (const width of [1440, 1280, 390]) {
        await page.setViewportSize({ width, height: 900 });
        await page.waitForTimeout(1500);
        if (width < 600) await sidebarOpen(page, false); else await sidebarOpen(page, true);
        const chart = benchmarkChart(page);
        await chart.scrollIntoViewIfNeeded();
        await chart.screenshot({ path: path.join(OUT, `forward-20y-${width}-benchmark-chart.png`) });
        rec[`ticks20y_${width}`] = await chart.evaluate((c) => [...c.querySelectorAll('.xtick text')].map((t) => t.textContent));
      }
    }
  } finally {
    fs.writeFileSync(path.join(OUT, 'matrix.json'), JSON.stringify(rec, null, 1));
    await browser.close();
  }
  log('done');
})();
