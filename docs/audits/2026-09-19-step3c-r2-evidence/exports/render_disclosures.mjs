import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

// Import and render the saved production output without modifying/exporting it.
const folder = path.dirname(fileURLToPath(import.meta.url));
for (const [name, sheetName, range] of [
  ['mixed', 'Summary', 'A8:N14'],
  ['comparison', 'Zone Comparison', 'A1:F3'],
]) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(folder, `${name}.xlsx`)));
  const image = await workbook.render({ sheetName, range, scale: 1.25, format: 'png' });
  await fs.writeFile(path.join(folder, `${name}-xlsx.png`), new Uint8Array(await image.arrayBuffer()));
}
console.log('Rendered saved Summary and Zone Comparison disclosure regions.');
