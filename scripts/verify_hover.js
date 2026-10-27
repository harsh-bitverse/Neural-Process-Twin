const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: ['--no-proxy-server', '--disable-gpu']
  });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle0' });
  await page.click('.btn-run-simulation');
  await page.waitForSelector('.status-decision-card');
  const canvas = await page.$('.field-canvas');
  const box = await canvas.boundingBox();
  // Move to center of canvas
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await new Promise(r => setTimeout(r, 400));
  const readout = await page.$eval('.field-hover-readout', el => el.innerText);
  console.log('Hover Readout text:', readout);

  // Move to bottom-left corner of canvas
  await page.mouse.move(box.x + 5, box.y + box.height - 5);
  await new Promise(r => setTimeout(r, 400));
  const readoutBL = await page.$eval('.field-hover-readout', el => el.innerText);
  console.log('Hover Readout Bottom-Left text:', readoutBL);

  // Move to top-right corner of canvas
  await page.mouse.move(box.x + box.width - 5, box.y + 5);
  await new Promise(r => setTimeout(r, 400));
  const readoutTR = await page.$eval('.field-hover-readout', el => el.innerText);
  console.log('Hover Readout Top-Right text:', readoutTR);

  await browser.close();
})();
