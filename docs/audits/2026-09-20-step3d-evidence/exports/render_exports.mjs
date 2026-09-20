// Read-only visual inspection of production exporter bytes. No workbook edits.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

if (process.argv.includes('--help')) {
  console.log('ARTIFACT_TOOL_NODE_MODULES=<bundled node_modules directory> node render_exports.mjs [artifact-directory]');
  process.exit(0);
}
const modules = process.env.ARTIFACT_TOOL_NODE_MODULES;
if (!modules) {
  throw new Error('Set ARTIFACT_TOOL_NODE_MODULES to the bundled node_modules directory returned by load_workspace_dependencies.');
}
const resolve = createRequire(path.join(path.resolve(modules), '..', 'package.json'));
const moduleUrl = pathToFileURL(resolve.resolve('@oai/artifact-tool')).href;
const { FileBlob, SpreadsheetFile } = await import(moduleUrl);
const out = process.argv[2] || path.join(os.tmpdir(), 'euro-bess-step3d-exports');
const manifest = JSON.parse(await fs.readFile(path.join(out, 'export-verification.json'), 'utf8'));
const books = new Map();
const results = [];
for (const task of manifest.renders) {
  let book = books.get(task.file);
  if (!book) {
    book = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out, task.file)));
    books.set(task.file, book);
  }
  const inspection = await book.inspect({
    kind: 'region', sheetId: task.sheet, range: task.range,
    maxChars: 12000, tableMaxRows: 20, tableMaxCols: 12,
  });
  const blob = await book.render({
    sheetName: task.sheet, range: task.range, scale: 1.5, format: 'png',
  });
  await fs.writeFile(path.join(out, task.png), new Uint8Array(await blob.arrayBuffer()));
  results.push({...task, inspection: inspection.ndjson});
  console.log(`${task.file}: ${task.sheet}!${task.range} -> ${task.png}`);
}
await fs.writeFile(path.join(out, 'xlsx-render-verification.json'), JSON.stringify(results, null, 2) + '\n');
