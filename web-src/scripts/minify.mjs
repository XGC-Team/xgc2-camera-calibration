import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { transform } from 'esbuild'

const page = process.argv[2]
if (page !== 'intrinsic' && page !== 'extrinsic') throw new Error(`invalid calibration page: ${page}`)
const target = resolve(import.meta.dirname, `../../xgc_camera_calibration/web/${page}/app.js`)
const source = await readFile(target, 'utf8')
const result = await transform(source, {
  charset: 'utf8',
  format: 'esm',
  legalComments: 'none',
  minify: true,
  target: 'es2022',
})
await writeFile(target, result.code)
