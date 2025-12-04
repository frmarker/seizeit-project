import os
import argparse
from pathlib import Path


def find_broken_symlinks(bids_root):
    bids_root = Path(bids_root)
    broken = []

    for root, dirs, files in os.walk(bids_root):
        for fname in files:
            if not fname.lower().endswith(".edf"):
                continue

            fpath = Path(root) / fname
            if fpath.is_symlink() and not fpath.exists():
                target = os.readlink(fpath)
                broken.append((fpath, target))

    return broken


import os
import argparse
from pathlib import Path


def find_broken_symlinks(bids_root):
    bids_root = Path(bids_root)
    broken = []

    for root, dirs, files in os.walk(bids_root):
        for fname in files:
            if not fname.lower().endswith('.edf'):
                continue

            fpath = Path(root) / fname
            if fpath.is_symlink() and not fpath.exists():
                target = os.readlink(fpath)
                broken.append((fpath, target))

    return broken


def search_target(target_root, basename):
    target_root = Path(target_root)
    matches = []
    for root, dirs, files in os.walk(target_root):
        for fname in files:
            if fname == basename:
                matches.append(Path(root) / fname)
    return matches


def build_edf_index(target_root):
    target_root = Path(target_root)
    index = {}
    for root, dirs, files in os.walk(target_root):
        for fname in files:
            if not fname.lower().endswith('.edf'):
                continue
            fpath = Path(root) / fname
            index.setdefault(fname, []).append(fpath)
    return index


def repair_symlink(link_path, new_target, dry_run=False):
    link_path = Path(link_path)
    new_target = Path(new_target)

    # Make path relative, so symlink is portable inside the dataset
    try:
        rel_target = os.path.relpath(new_target, start=link_path.parent)
    except ValueError:
        # Fallback: absolute path
        rel_target = str(new_target)

    print(f'  -> FIX: {link_path} -> {rel_target}')
    if not dry_run:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(rel_target, link_path)


def main():
    parser = argparse.ArgumentParser(
        description='Find and fix broken EDF symlinks in a BIDS dataset.'
    )
    parser.add_argument(
        '--bids-root',
        type=str,
        required=True,
        help='Path to BIDS dataset root (e.g. /.../dataset).',
    )
    parser.add_argument(
        '--target-root',
        type=str,
        required=True,
        help=(
            'Path where the REAL EDF files live. '
            'If they are inside the BIDS tree, you can set this equal to --bids-root.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print what would be changed; do NOT modify anything',
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    target_root = Path(args.target_root)

    print(f'BIDS root: {bids_root}')
    print(f'Target root: {target_root}')
    print(f'Dry run: {args.dry_run}')
    print('\nScanning for broken EDF symlinks...\n')

    # Build an index of all EDF files in the target tree once,
    # so we don't repeatedly walk the filesystem for each broken link.
    print('Indexing EDF files under target root...\n')
    edf_index = build_edf_index(target_root)
    total_indexed = sum(len(v) for v in edf_index.values())
    print(f'Indexed {total_indexed} EDF files under {target_root}.\n')

    broken = find_broken_symlinks(bids_root)
    if not broken:
        print('No broken EDF symlinks found. Nothing to do')
        return

    print(f'Found {len(broken)} broken EDF symlinks\n')

    fixed_count = 0
    multi_match = 0
    not_found = 0

    for link_path, old_target in broken:
        basename = link_path.name
        print(f'Broken link: {link_path}')
        print(f'old target: {old_target}')

        matches = edf_index.get(basename, [])

        if len(matches) == 0:
            print('No matching EDF file found under target-root.')
            not_found += 1
            continue
        elif len(matches) > 1:
            print('Multiple matches found:')
            for m in matches:
                print(f'     - {m}')
            print('Skipping (ambiguous).')
            multi_match += 1
            continue
        else:
            # Exactly one match – repair the link
            repair_symlink(link_path, matches[0], dry_run=args.dry_run)
            fixed_count += 1

        print()

    print('\n==== SUMMARY ====')
    print(f'Broken symlinks total: {len(broken)}')
    print(f'Fixed: {fixed_count}')
    print(f'Unresolved (no match): {not_found}')
    print(f'Ambiguous (multi): {multi_match}')
    if args.dry_run:
        print('\nNOTE: This was a DRY RUN. Re-run without --dry-run to apply fixes.')
    else:
        print('\nDone. Symlinks updated where a unique real EDF file was found.')


if __name__ == '__main__':
    main()