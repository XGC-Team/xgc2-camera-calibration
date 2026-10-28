import { createHash } from 'node:crypto'
import { readdir, readFile, rename, writeFile } from 'node:fs/promises'
import { relative, resolve } from 'node:path'

const webRoot = resolve(import.meta.dirname, '..')
const productRoot = resolve(webRoot, '..')
const receiptPath = resolve(productRoot, "xgc_camera_calibration/web/build-receipt.json")
const outputs = ["xgc_camera_calibration/web/intrinsic/app.js", "xgc_camera_calibration/web/intrinsic/styles.css", "xgc_camera_calibration/web/extrinsic/app.js", "xgc_camera_calibration/web/extrinsic/styles.css"]
const json = async path => JSON.parse(await readFile(path, 'utf8'))
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const fileHash = async path => digest(await readFile(path))

async function filesIn(path) {
  const result = []
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = resolve(path, entry.name)
    if (entry.isDirectory()) result.push(...await filesIn(child))
    else if (entry.isFile()) result.push(child)
    else throw new Error(`Unexpected non-file build input: ${child}`)
  }
  return result.sort()
}

async function fileRecords(paths, root) {
  return Promise.all(paths.sort().map(async path => ({ path: relative(root, path), sha256: await fileHash(path) })))
}

async function dependenciesSnapshot() {
  const manifest = await json(resolve(webRoot, 'package.json'))
  const lock = await json(resolve(webRoot, 'package-lock.json'))
  const dependencies = {}
  for (const name of ['@xgc2/ui-react', '@xgc2/ui-policy', 'react', 'react-dom']) {
    const root = resolve(webRoot, 'node_modules', name)
    const installed = await json(resolve(root, 'package.json'))
    const locked = lock.packages[`node_modules/${name}`]
    const wanted = manifest.dependencies?.[name] ?? manifest.devDependencies?.[name]
    const rootLock = lock.packages[''].dependencies?.[name] ?? lock.packages[''].devDependencies?.[name]
    if (!locked || !wanted || rootLock !== wanted || installed.version !== locked.version || !locked.integrity) {
      throw new Error(`${name}: installed package, manifest and lock do not agree`)
    }
    const files = await fileRecords(await filesIn(root), root)
    dependencies[name] = { specifier: wanted, lockedVersion: locked.version,
      installedVersion: installed.version, lockIntegrity: locked.integrity,
      installedFilesSha256: digest(JSON.stringify(files)) }
  }
  return dependencies
}

async function snapshot() {
  const sourcePaths = [
    'package.json', 'package-lock.json', 'tsconfig.json', 'vite.config.ts',
    'scripts/minify.mjs', 'scripts/build-receipt.mjs',
  ].map(path => resolve(webRoot, path))
  sourcePaths.push(...await filesIn(resolve(webRoot, 'src')))
  return { schemaVersion: 1, builder: 'npm run build',
    source: await fileRecords(sourcePaths, productRoot), dependencies: await dependenciesSnapshot(),
    artifacts: await fileRecords(outputs.map(path => resolve(productRoot, path)), productRoot) }
}

if (process.argv[2] === '--check-install') {
  const saved = await json(receiptPath)
  if (JSON.stringify(saved.dependencies) !== JSON.stringify(await dependenciesSnapshot())) throw new Error('Installed Web dependencies do not match the build receipt')
} else if (process.argv[2] === '--verify') {
  const actual = await snapshot()
  const saved = await json(receiptPath)
  if (JSON.stringify(saved) !== JSON.stringify(actual)) throw new Error('Web build receipt does not match current source, installed dependencies or built artifacts')
} else {
  if (process.argv.length !== 2) throw new Error('Usage: node scripts/build-receipt.mjs [--verify|--check-install]')
  const actual = await snapshot()
  const temporary = `${receiptPath}.tmp`
  await writeFile(temporary, JSON.stringify(actual, null, 2) + '\n')
  await rename(temporary, receiptPath)
}
