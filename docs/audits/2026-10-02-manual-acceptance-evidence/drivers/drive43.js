// Check 43 driver: real app on the seeded DST tree; real downloads saved to disk.
// Usage: NODE_PATH=<dir with playwright> node drive43.js <port>
// Per target day: window [T, T], 1 MW / 1 h / 88% / capture 100%, DE_FCR
// session upload (Revenue joint MILP) + unified capacity (Project Case,
// cache-first), Project Case "DA + reserve co-optimised" FCR, availability 0.95.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'check43');
const FIX = path.join(__dirname, '..', 'fixtures');
const DAYS = [['ordinary', '2026/03/28'], ['spring', '2026/03/29'], ['autumn', '2025/10/26']];

async function settle(page, extra = 600) {
  await page.waitForTimeout(extra);
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 300000 },
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

async function expand(scope, title) {
  const exp = scope.locator('[data-testid="stExpander"]', { hasText: title }).first();
  const details = exp.locator('details');
  const open = await details.getAttribute('open');
  if (open === null) {
    await exp.locator('summary').first().click();
    await exp.page().waitForTimeout(700);
  }
  return exp;
}

async function setDates(page, value) {
  const dates = page.locator('[data-testid="stSidebar"] [data-testid="stDateInput"] input');
  for (const i of [0, 1]) {
    await dates.nth(i).click();
    await dates.nth(i).press('ControlOrMeta+a');
    await dates.nth(i).pressSequentially(value);
    await dates.nth(i).press('Enter');
    await page.keyboard.press('Escape');
    await settle(page, 300);
  }
}

async function download(page, locator, prefix) {
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), locator.click()]);
  const target = path.join(OUT, `${prefix}-${dl.suggestedFilename()}`);
  await dl.saveAs(target);
  await settle(page);
  return { file: path.basename(target), bytes: fs.statSync(target).size };
}

async function captionFit(page, needle) {
  return page.evaluate((needle) => {
    const nodes = [...document.querySelectorAll('[data-testid="stCaptionContainer"]')]
      .filter((n) => n.textContent.includes(needle) && n.offsetParent !== null);
    return {
      viewport: innerWidth,
      sidebar: document.querySelector('[data-testid="stSidebar"]').getAttribute('aria-expanded'),
      docOverflowX: document.documentElement.scrollWidth > innerWidth,
      captions: nodes.map((n) => {
        const r = n.getBoundingClientRect();
        return { text: n.textContent.slice(0, 400), width: Math.round(r.width), right: Math.round(r.right),
          clipped: n.scrollWidth > n.clientWidth + 1 || r.right > innerWidth + 1 };
      }),
    };
  }, needle);
}

async function metricText(page, label) {
  return page.evaluate((label) => {
    const m = [...document.querySelectorAll('[data-testid="stMetric"]')]
      .find((n) => n.offsetParent !== null && n.querySelector('[data-testid="stMetricLabel"]')?.textContent.trim() === label);
    return m ? m.querySelector('[data-testid="stMetricValue"]').textContent.trim() : null;
  }, label);
}

async function narrowShots(page, name, targets) {
  const out = {};
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForTimeout(1200);
  await sidebarOpen(page, false);
  out.statusWidgets = await page.locator('[data-testid="stStatusWidget"]').count();
  for (const [key, locator, needle] of targets) {
    await locator.scrollIntoViewIfNeeded();
    await locator.screenshot({ path: path.join(OUT, `${name}-390-${key}.png`) });
    out[key] = await captionFit(page, needle);
  }
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForTimeout(1200);
  await sidebarOpen(page, true);
  return out;
}

async function uploadAndImport(page, uploaderLabel, file, buttonText) {
  const up = page.locator('[data-testid="stSidebar"] [data-testid="stFileUploader"]', { hasText: uploaderLabel }).first();
  await up.locator('input[type="file"]').setInputFiles(file);
  await settle(page, 800);
  await page.locator('[data-testid="stSidebar"] button:visible', { hasText: buttonText }).first().click();
  await settle(page, 800);
  return page.locator('[data-testid="stSidebar"] [data-testid="stAlert"]').allInnerTexts();
}

async function runDay(page, name, value, first) {
  const res = { day: value };
  await sidebarOpen(page, true);
  await setDates(page, value);
  if (first) {
    const form = page.locator('[data-testid="stSidebar"] [data-testid="stForm"]').first();
    const power = form.locator('[data-testid="stNumberInput"]', { hasText: 'Power (MW)' }).locator('input');
    await power.fill('1');
    await power.press('Tab');
    const capture = form.locator('[data-testid="stSlider"]', { hasText: 'Capture (%)' }).locator('[role="slider"]');
    await capture.focus();
    await capture.press('End');
    await form.locator('button', { hasText: 'Apply BESS parameters' }).click();
    await settle(page);
  }
  await page.locator('[data-testid="stSidebar"] button', { hasText: 'Fetch Data' }).click();
  await page.waitForSelector('button[role="tab"]', { timeout: 180000 });
  await settle(page, 1000);
  await sidebarOpen(page, true);
  const anc = await expand(page.locator('[data-testid="stSidebar"]'), 'Ancillary Services Data');
  const tmpl = anc.locator('[data-testid="stSelectbox"]', { hasText: 'Template' }).first();
  await tmpl.locator('div[data-baseweb="select"]').click();
  await page.getByRole('option', { name: /^DE_FCR/ }).click();
  await settle(page);
  res.sessionUpload = await uploadAndImport(page, 'Upload per-country ancillary CSV',
    path.join(FIX, 'SYNTHETIC_dst_DE_FCR.csv'), 'Parse & Import');
  if (first) {
    res.capacityUpload = await uploadAndImport(page, 'Upload unified capacity CSV',
      path.join(FIX, 'SYNTHETIC_dst_unified_capacity.csv'), 'Parse & Import capacity');
  }

  await page.locator('button[role="tab"]', { hasText: 'Revenue Estimation' }).click();
  await settle(page);
  const joint = await expand(page, 'Joint MILP co-optimization estimate');
  await joint.scrollIntoViewIfNeeded();
  await joint.screenshot({ path: path.join(OUT, `${name}-1280-joint.png`) });
  res.joint = {
    capacity: await metricText(page, 'MILP Capacity Component'),
    da: await metricText(page, 'MILP DA Component'),
    caption: await captionFit(page, 'aggregate capacity'),
  };

  const strategy = page.locator('[data-testid="stSelectbox"]', { hasText: 'Cash-NPV dispatch strategy' }).first();
  await strategy.locator('div[data-baseweb="select"]').click();
  await page.getByRole('option', { name: 'DA + reserve co-optimised', exact: true }).click();
  await settle(page);
  const product = page.locator('[data-testid="stSelectbox"]', { hasText: 'Reserve capacity product and direction' }).first();
  res.reserveProduct = (await product.innerText()).replace(/\n/g, ' | ');
  await page.locator('button', { hasText: 'Run Project Case' }).click();
  await settle(page, 2000);
  const pcCaption = page.locator('[data-testid="stCaptionContainer"]', { hasText: 'Reserve capacity settlement basis' }).first();
  await pcCaption.waitFor({ timeout: 300000 });
  await pcCaption.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(OUT, `${name}-1280-pc.png`) });
  res.pcCaption = await captionFit(page, 'Reserve capacity settlement basis');
  res.handoff = await download(page,
    page.locator('[data-testid="stDownloadButton"] button:visible', { hasText: 'Handoff JSON' }).first(), name);
  res.xlsx = await download(page,
    page.locator('[data-testid="stDownloadButton"] button:visible', { hasText: 'Export to Excel' }).first(), name);
  res.pdf = await download(page,
    page.locator('[data-testid="stDownloadButton"] button:visible', { hasText: 'Export to PDF' }).first(), name);

  res.narrowRevenue = await narrowShots(page, name, [
    ['pc', pcCaption, 'Reserve capacity settlement basis'],
    ['joint', joint, 'aggregate capacity'],
  ]);

  await page.locator('button[role="tab"]', { hasText: 'Simulation Cockpit' }).click();
  await settle(page);
  const mirror = await expand(page, 'Project Case NPV — read-only Revenue-tab result');
  await mirror.scrollIntoViewIfNeeded();
  await mirror.screenshot({ path: path.join(OUT, `${name}-1280-mirror.png`) });
  res.mirror = await captionFit(page, 'nominal 4h');
  res.narrowMirror = await narrowShots(page, name, [['mirror', mirror, 'nominal 4h']]);
  return res;
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const port = Number(process.argv[2]);
  const only = process.argv[3];
  const browser = await chromium.launch();
  const result = {};
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1, acceptDownloads: true });
    const page = await context.newPage();
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 60000 });
    await settle(page, 1500);
    let first = true;
    for (const [name, value] of DAYS) {
      if (only && only !== name) continue;
      try {
        result[name] = await runDay(page, name, value, first);
      } catch (err) {
        await page.screenshot({ path: path.join(OUT, `FAIL-${name}.png`), fullPage: false });
        result[name] = { error: err.message.split('\n')[0] };
        console.log(name, 'ERROR', err.message.split('\n')[0]);
        break;
      }
      first = false;
      fs.writeFileSync(path.join(OUT, 'check43.json'), JSON.stringify(result, null, 1));
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(OUT, 'check43.json'), JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result, null, 1).slice(0, 6000));
})();
