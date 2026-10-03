// Check 42 driver: synthetic Revenue harness, real browser downloads to disk.
// Usage: NODE_PATH=<dir with playwright> node drive42.js <port>
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'check42');

async function settle(page, extra = 600) {
  await page.waitForTimeout(extra);
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 120000 },
  );
  await page.waitForTimeout(300);
}

async function captionFit(page, needle) {
  return page.evaluate((needle) => {
    const p = [...document.querySelectorAll('[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"]')]
      .find((n) => n.textContent.includes(needle));
    if (!p) return { missing: true };
    return { text: p.textContent, clientWidth: p.clientWidth, scrollWidth: p.scrollWidth,
      overflow: p.scrollWidth > p.clientWidth + 1, docOverflowX: document.documentElement.scrollWidth > innerWidth };
  }, needle);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const port = Number(process.argv[2]);
  const browser = await chromium.launch();
  const result = {};
  try {
    for (const width of [1280, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, deviceScaleFactor: 1, acceptDownloads: true });
      const page = await context.newPage();
      await page.goto(`http://127.0.0.1:${port}/`);
      await page.waitForSelector('[data-testid="stExpander"]', { timeout: 120000 });
      await settle(page, 1500);
      const sidebar = await page.locator('[data-testid="stSidebar"]').getAttribute('aria-expanded');
      if (width < 600 && sidebar === 'true') {
        await page.locator('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarHeader"] button').first().click();
        await page.waitForTimeout(800);
      }
      const joint = page.locator('[data-testid="stExpander"]', { hasText: 'Joint MILP co-optimization estimate' }).first();
      await joint.locator('summary').click();
      await page.waitForTimeout(800);
      await joint.scrollIntoViewIfNeeded();
      await joint.screenshot({ path: path.join(OUT, `joint-${width}.png`) });
      const entry = { sidebarAtLoad: sidebar, caption: await captionFit(page, 'aggregate capacity') };
      for (const label of ['Export to Excel', 'Export to PDF']) {
        const [download] = await Promise.all([
          page.waitForEvent('download', { timeout: 60000 }),
          page.locator('[data-testid="stDownloadButton"] button', { hasText: label }).first().click(),
        ]);
        const target = path.join(OUT, `${width}-${download.suggestedFilename()}`);
        await download.saveAs(target);
        entry[label] = { file: path.basename(target), bytes: fs.statSync(target).size };
      }
      await settle(page);
      entry.afterDownloads = {
        jointStillVisible: await joint.isVisible(),
        status: await page.locator('[data-testid="stStatusWidget"]').count(),
      };
      result[width] = entry;
      await context.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(OUT, 'check42.json'), JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result, null, 1));
})();
