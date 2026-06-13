const fs = require('fs');
const { chromium } = require('playwright-core');
const FPS = 20, DIR = '/tmp/gif-demo';
function findChromium() {
  const root = `${process.env.HOME}/.cache/ms-playwright`;
  const d = fs.readdirSync(root).filter(x => x.startsWith('chromium-')).sort().pop();
  return `${root}/${d}/chrome-linux64/chrome`;
}
(async () => {
  const browser = await chromium.launch({ executablePath: findChromium(),
    args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
  const page = await browser.newPage({ viewport: { width: 1000, height: 600 }, deviceScaleFactor: 1 });
  await page.goto(`file://${DIR}/term.html`);
  await page.waitForFunction('window.__ready === true');
  const total = await page.evaluate('window.__total');
  const frames = Math.round(total * FPS);
  fs.mkdirSync(`${DIR}/frames`, { recursive: true });
  for (let i = 0; i < frames; i++) {
    await page.evaluate(t => seek(t), i / FPS);
    await page.screenshot({ path: `${DIR}/frames/f${String(i).padStart(4, '0')}.png` });
  }
  console.log('frames', frames, 'total', total.toFixed(1));
  await browser.close();
})();
