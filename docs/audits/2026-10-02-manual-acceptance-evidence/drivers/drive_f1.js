// Check 5 recheck on merged code: unpublished activation window names the lag.
// Usage: NODE_PATH=<dir with playwright> node drive_f1.js <port>
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const OUT = path.join(__dirname, 'check05');
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
    const settle = async (extra = 600) => {
      await page.waitForTimeout(extra);
      await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 180000 });
      await page.waitForTimeout(400);
    };
    await page.goto(`http://127.0.0.1:${process.argv[2]}/`);
    await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 60000 });
    await settle(1500);
    const dates = page.locator('[data-testid="stSidebar"] [data-testid="stDateInput"] input');
    for (const [i, v] of [[0, '2026/09/30'], [1, '2026/10/03']]) {
      await dates.nth(i).click();
      await dates.nth(i).press('ControlOrMeta+a');
      await dates.nth(i).pressSequentially(v);
      await dates.nth(i).press('Enter');
      await page.keyboard.press('Escape');
      await settle(300);
    }
    const anc = page.locator('[data-testid="stSidebar"] [data-testid="stExpander"]', { hasText: 'Ancillary Services Data' }).first();
    await anc.locator('summary').first().click();
    await page.waitForTimeout(800);
    const btn = page.locator('[data-testid="stSidebar"] button:visible', { hasText: 'Fetch Netztransparenz + ENTSO-E activation energy' }).first();
    await btn.click();
    await settle(1500);
    const alert = page.locator('[data-testid="stSidebar"] [data-testid="stAlert"]', { hasText: 'Activation' }).first();
    await alert.scrollIntoViewIfNeeded();
    await alert.screenshot({ path: path.join(OUT, 'activation-lag-message-1280.png') });
    const text = await page.locator('[data-testid="stSidebar"] [data-testid="stAlert"]').allInnerTexts();
    fs.writeFileSync(path.join(OUT, 'check05.json'), JSON.stringify({ window: '2026-09-30..2026-10-03', alerts: text }, null, 1));
    console.log(JSON.stringify(text, null, 1));
  } finally {
    await browser.close();
  }
})();
