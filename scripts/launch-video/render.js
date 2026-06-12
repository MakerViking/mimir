// Frame renderer: scrubs video.html with seek(t) and screenshots each frame.
// Usage: node render.js              -> full render (frames/f00001.png ...)
//        node render.js 5 16 28 ...  -> test stills at given seconds (test_<t>.png)
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const FPS = 30, DURATION = 93.5;
const DIR = '/tmp/mimir-video';

function findChromium() {
  const root = `${process.env.HOME}/.cache/ms-playwright`;
  const dirs = fs.readdirSync(root).filter(d => d.startsWith('chromium-')).sort();
  const d = dirs[dirs.length - 1];
  return `${root}/${d}/chrome-linux64/chrome`;
}

(async () => {
  const browser = await chromium.launch({
    executablePath: findChromium(),
    args: ['--force-color-profile=srgb', '--disable-lcd-text', '--hide-scrollbars'],
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(`file://${DIR}/video.html`);
  await page.waitForFunction('window.__ready === true');

  const tests = process.argv.slice(2).map(Number);
  if (tests.length) {
    for (const t of tests) {
      await page.evaluate(t => seek(t), t);
      await page.screenshot({ path: `${DIR}/test_${t}.png` });
      console.log(`test_${t}.png`);
    }
  } else {
    const total = Math.round(DURATION * FPS);
    for (let i = 0; i < total; i++) {
      await page.evaluate(t => seek(t), i / FPS);
      await page.screenshot({ path: `${DIR}/frames/f${String(i).padStart(5, '0')}.png` });
      if (i % 300 === 0) console.log(`${i}/${total}`);
    }
    console.log('done', total);
  }
  await browser.close();
})();
