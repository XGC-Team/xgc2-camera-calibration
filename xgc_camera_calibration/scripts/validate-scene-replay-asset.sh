#!/usr/bin/env bash
set -euo pipefail

# Validate one sealed scene-replay asset directory against its manifest and
# print the absolute media file path on the last stdout line for the
# orchestration stdoutTail binding. Every failure exits non-zero on stderr.

usage() {
  echo "usage: $0 --asset-dir <dir> [--scene-file <path>]" >&2
  exit 2
}

asset_dir=""
scene_file=""
while (($#)); do
  case "$1" in
    --asset-dir)
      [[ $# -ge 2 ]] || usage
      asset_dir="$2"
      shift 2
      ;;
    --scene-file)
      [[ $# -ge 2 ]] || usage
      scene_file="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done
[[ -n "$asset_dir" ]] || usage

for command in jq sha256sum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "validate-scene-replay-asset: $command is required" >&2
    exit 1
  }
done

[[ -d "$asset_dir" ]] || {
  echo "validate-scene-replay-asset: replay asset directory does not exist: $asset_dir" >&2
  exit 1
}
asset_dir="$(cd -- "$asset_dir" && pwd)"
manifest_path="$asset_dir/manifest.json"
[[ -f "$manifest_path" ]] || {
  echo "validate-scene-replay-asset: manifest.json is missing in $asset_dir" >&2
  exit 1
}
manifest="$(cat -- "$manifest_path")"
jq -e '
  type == "object"
  and .kind == "xgc.scene-replay-asset.v1"
  and .schemaVersion == 1
' <<<"$manifest" >/dev/null || {
  echo "validate-scene-replay-asset: $manifest_path is not an xgc.scene-replay-asset.v1 manifest with schemaVersion 1" >&2
  exit 1
}

check_entry() {
  local key="$1" file digest actual path
  file="$(jq -er --arg key "$key" '.[$key].file // empty' <<<"$manifest")" || {
    echo "validate-scene-replay-asset: manifest.$key.file is missing" >&2
    exit 1
  }
  digest="$(jq -er --arg key "$key" '.[$key].sha256 // empty' <<<"$manifest")" || {
    echo "validate-scene-replay-asset: manifest.$key.sha256 is missing" >&2
    exit 1
  }
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "validate-scene-replay-asset: manifest.$key.sha256 is not a sha256 digest: $digest" >&2
    exit 1
  }
  case "$file" in
    /*|*..*|*$'\n'*|*$'\r'*)
      echo "validate-scene-replay-asset: manifest.$key.file must be a single-line relative path inside the asset directory: $file" >&2
      exit 1
      ;;
  esac
  path="$asset_dir/$file"
  [[ -f "$path" ]] || {
    echo "validate-scene-replay-asset: manifest.$key file is missing: $path" >&2
    exit 1
  }
  actual="$(sha256sum -- "$path" | cut -d' ' -f1)"
  [[ "$actual" == "$digest" ]] || {
    echo "validate-scene-replay-asset: manifest.$key digest mismatch for $path: manifest $digest, actual $actual" >&2
    exit 1
  }
}

for key in media cameraInfo extrinsic sceneDocument fieldSite; do
  check_entry "$key"
done

if [[ -n "$scene_file" ]]; then
  [[ -f "$scene_file" ]] || {
    echo "validate-scene-replay-asset: scene file does not exist: $scene_file" >&2
    exit 1
  }
  expected_scene="$(jq -er '.sceneDocument.sha256' <<<"$manifest")"
  actual_scene="$(sha256sum -- "$scene_file" | cut -d' ' -f1)"
  [[ "$actual_scene" == "$expected_scene" ]] || {
    echo "validate-scene-replay-asset: scene document is not the one pinned by the replay asset:" >&2
    echo "  asset manifest sceneDocument.sha256: $expected_scene" >&2
    echo "  scene file $scene_file sha256:       $actual_scene" >&2
    exit 1
  }
fi

media_file="$(jq -er '.media.file' <<<"$manifest")"
printf '%s/%s\n' "$asset_dir" "$media_file"
