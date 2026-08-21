#!/bin/bash
# Applies patches/<submodule>/*.patch onto gaussian-splatting/submodules/<submodule>.
#
# These fix latent bugs in the upstream submodules (e.g. missing includes
# that only surface on certain compiler/CUDA header combinations) that we
# can't push upstream. Idempotent: skips a patch if it's already applied.
set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patches_root="$repo_root/patches"
submodules_root="$repo_root/gaussian-splatting/submodules"

for submodule_dir in "$patches_root"/*/; do
    [ -d "$submodule_dir" ] || continue
    submodule_name="$(basename "$submodule_dir")"
    target="$submodules_root/$submodule_name"

    for patch in "$submodule_dir"*.patch; do
        [ -e "$patch" ] || continue
        if git -C "$target" apply --check "$patch" 2>/dev/null; then
            echo "Applying $(basename "$patch") to $submodule_name"
            git -C "$target" apply "$patch"
        else
            echo "Skipping $(basename "$patch") for $submodule_name (already applied)"
        fi
    done
done
