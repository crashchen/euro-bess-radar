import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

// Read-only import/render of the production generator's saved workbooks.
const folder = path.resolve('outputs/step3c-2026-09-16/export-check');
const results = JSON.parse(await fs.readFile(path.join(folder, 'results.json'), 'utf8'));
const inspections = {};
for (const name of ['mixed', 'partial-finite', 'gap', 'unknown-cadence']) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(folder, `${name}.xlsx`)));
  const row = Number(results[name].average_cell.coordinate.replace(/^[A-Z]+/, ''));
  const range = `A${row - 1}:N${row + 3}`;
  inspections[name] = (await workbook.inspect({kind: 'region', sheetId: 'Summary', range: `A${row}:B${row+1}`, maxChars: 2000})).ndjson;
  const image = await workbook.render({sheetName: 'Summary', range, scale: 1.25, format: 'png'});
  await fs.writeFile(path.join(folder, `${name}-xlsx.png`), new Uint8Array(await image.arrayBuffer()));
}
const comparison = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(folder, 'comparison.xlsx')));
inspections.comparison = (await comparison.inspect({kind: 'region', sheetId: 'Zone Comparison', range: 'A1:E4', maxChars: 3000})).ndjson;
const image = await comparison.render({sheetName: 'Zone Comparison', range: 'A1:H4', scale: 1.25, format: 'png'});
await fs.writeFile(path.join(folder, 'comparison-xlsx.png'), new Uint8Array(await image.arrayBuffer()));
await fs.writeFile(path.join(folder, 'xlsx-inspection.json'), JSON.stringify(inspections, null, 2));
console.log('Rendered 4 saved summary workbooks and 1 zone comparison workbook.');
