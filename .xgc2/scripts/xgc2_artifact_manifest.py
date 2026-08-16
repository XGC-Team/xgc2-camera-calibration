#!/usr/bin/env python3
"""Create an XGC2 trusted build-artifact manifest for Debian outputs."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def field(path, name):
    return subprocess.check_output(
        ["dpkg-deb", "-f", str(path), name], text=True
    ).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    for option in (
        "deb-dir",
        "output-dir",
        "product",
        "product-version",
        "distribution",
        "architecture",
        "source-sha",
        "ci-run-id",
        "ci-workflow",
        "ci-workflow-ref",
    ):
        build.add_argument("--" + option, required=True)
    arguments = parser.parse_args()

    debs = sorted(Path(arguments.deb_dir).glob("*.deb"))
    if not debs:
        raise SystemExit("no Debian artifacts found")
    entries = []
    for deb in debs:
        architecture = field(deb, "Architecture")
        if architecture not in (arguments.architecture, "all"):
            raise SystemExit("artifact architecture mismatch: " + deb.name)
        entries.append(
            {
                "file": deb.name,
                "package": field(deb, "Package"),
                "version": field(deb, "Version"),
                "architecture": architecture,
                "sha256": sha256(deb),
                "size": deb.stat().st_size,
            }
        )

    payload = {
        "schema": "xgc2.build-artifact.v1",
        "product": arguments.product,
        "source_sha": arguments.source_sha,
        "version": arguments.product_version,
        "distribution": arguments.distribution,
        "architecture": arguments.architecture,
        "ci": {
            "run_id": str(arguments.ci_run_id),
            "workflow": arguments.ci_workflow,
            "workflow_ref": arguments.ci_workflow_ref,
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "debs": entries,
    }
    output = (
        Path(arguments.output_dir)
        / f"{arguments.product}_{arguments.distribution}_{arguments.architecture}.build.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
