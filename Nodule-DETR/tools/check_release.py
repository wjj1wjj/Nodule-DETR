"""Check a clean source tree for accidental data, weights and generated binaries."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.npy', '.npz',
    '.onnx', '.engine', '.so', '.o', '.a', '.egg', '.zip', '.tar', '.gz', '.7z',
    '.docx', '.pdf', '.xlsx', '.csv', '.png', '.jpg', '.jpeg', '.dcm', '.nii', '.tif'}
FORBIDDEN_DIRS = {'data', 'outputs', 'checkpoints', 'logs', 'build', 'dist', '.venv'}
errors = []
files = []
for path in ROOT.rglob('*'):
    relative = path.relative_to(ROOT)
    if any(part in {'.git', '__pycache__'} for part in relative.parts):
        continue
    if path.is_symlink():
        errors.append(f'Symlink: {relative}')
    if not path.is_file():
        continue
    files.append(path)
    if 'copy' in path.stem.lower() or '副本' in path.stem:
        errors.append(f'Historical backup: {relative}')
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(p in FORBIDDEN_DIRS or p.endswith('.egg-info') for p in relative.parts):
        errors.append(f'Generated/data file: {relative}')
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(relative))
    if path.suffix in {'.py', '.md', '.txt', '.sh'}:
        text = path.read_text()
        if any(private in text for private in [chr(47) + name + chr(47) for name in ('hdd2', 'hdd', 'home')]):
            errors.append(f'Machine-specific path: {relative}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(files)} source/documentation files, {sum(p.stat().st_size for p in files):,} bytes; no bundled weights, data, binary builds, or private absolute paths.')
