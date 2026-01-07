#!/usr/bin/env python3
import json
import os
from pathlib import Path
import pickle

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if HAS_PANDAS:
                import pandas as pd
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if isinstance(obj, (pd.Series, pd.Index)):
                    return obj.tolist()
            return super().default(obj)
        except Exception:
            return repr(obj)


def safe_dump_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)


def load_pickle(path):
    if HAS_PANDAS:
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    with open(path, 'rb') as f:
        return pickle.load(f)


def convert_file(src_path, dst_path):
    suffix = src_path.suffix.lower()
    try:
        if suffix == '.json':
            # pretty copy
            with open(src_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            safe_dump_json(obj, dst_path)
            return True, None

        if suffix == '.pkl' or suffix == '.pickle':
            obj = load_pickle(src_path)
            # if pandas DataFrame, convert to records
            if HAS_PANDAS and isinstance(obj, pd.DataFrame):
                safe_dump_json(obj.to_dict(orient='records'), dst_path)
            elif HAS_PANDAS and isinstance(obj, pd.Series):
                safe_dump_json(obj.tolist(), dst_path)
            else:
                safe_dump_json(obj, dst_path)
            return True, None

        if suffix == '.csv':
            if not HAS_PANDAS:
                return False, 'pandas required for CSV'
            df = pd.read_csv(src_path)
            safe_dump_json(df.to_dict(orient='records'), dst_path)
            return True, None

        if suffix == '.npz':
            data = np.load(src_path, allow_pickle=True)
            out = {}
            for k in data.files:
                try:
                    out[k] = data[k].tolist()
                except Exception:
                    out[k] = repr(data[k])
            safe_dump_json(out, dst_path)
            return True, None

        if suffix == '.npy':
            arr = np.load(src_path, allow_pickle=True)
            try:
                safe_dump_json(arr.tolist(), dst_path)
            except Exception:
                safe_dump_json(repr(arr), dst_path)
            return True, None

        # fallback: try pickle, then text
        try:
            obj = load_pickle(src_path)
            safe_dump_json(obj, dst_path)
            return True, None
        except Exception:
            pass

        try:
            text = src_path.read_text(encoding='utf-8')
            # try parse JSON
            try:
                obj = json.loads(text)
                safe_dump_json(obj, dst_path)
                return True, None
            except Exception:
                safe_dump_json({'text': text}, dst_path)
                return True, None
        except Exception as e:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    src_dir = Path('graficos/arquivos/models')
    dst_dir = src_dir / 'json'
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print('Source directory not found:', src_dir)
        return 2

    files = [p for p in src_dir.iterdir() if p.is_file()]
    processed = []
    skipped = []

    for f in files:
        dst = dst_dir / (f.stem + '.json')
        ok, err = convert_file(f, dst)
        if ok:
            processed.append(str(dst))
            print('Converted:', f.name, '->', dst.name)
        else:
            skipped.append((f.name, err))
            print('Skipped:', f.name, 'reason:', err)

    print('\nSummary:')
    print('Processed:', len(processed))
    print('Skipped :', len(skipped))
    if skipped:
        print('Skipped files:')
        for name, reason in skipped:
            print('-', name, '->', reason)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
