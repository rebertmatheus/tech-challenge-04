#!/usr/bin/env python3
import argparse
import os
import pickle
import json
from pathlib import Path

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


def make_readable(obj, out_dir, base_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{base_name}_summary.txt"
    json_path = out_dir / f"{base_name}.json"
    csv_path = out_dir / f"{base_name}.csv"

    with open(summary_path, 'w', encoding='utf-8') as s:
        s.write('Type: ' + str(type(obj)) + '\n\n')

        # pandas DataFrame
        if HAS_PANDAS and isinstance(obj, pd.DataFrame):
            s.write('Pandas DataFrame detected\n')
            s.write('Shape: %s\n\n' % (obj.shape,))
            s.write('Head:\n')
            s.write(obj.head(10).to_string())
            s.write('\n\nDescribe:\n')
            try:
                s.write(obj.describe(include='all').to_string())
            except Exception:
                pass
            # save CSV and JSON
            obj.to_csv(csv_path, index=False)
            try:
                safe_dump_json(obj.to_dict(orient='records'), json_path)
            except Exception:
                pass
            return [str(summary_path), str(json_path), str(csv_path)]

        # pandas Series
        if HAS_PANDAS and isinstance(obj, pd.Series):
            s.write('Pandas Series detected\n')
            s.write('Length: %d\n\n' % (len(obj),))
            s.write('Head:\n')
            s.write(obj.head(20).to_string())
            try:
                safe_dump_json(obj.tolist(), json_path)
            except Exception:
                pass
            return [str(summary_path), str(json_path)]

        # dict-like
        if isinstance(obj, dict):
            s.write('Dict with keys:\n')
            for k in sorted(obj.keys()):
                s.write(f"- {k}: {type(obj[k])}\n")
            s.write('\nSample values (repr, truncated):\n')
            for k in sorted(obj.keys()):
                v = obj[k]
                s.write(f"[{k}] -> ")
                try:
                    rep = repr(v)
                    if len(rep) > 400:
                        rep = rep[:400] + '...'
                except Exception:
                    rep = str(type(v))
                s.write(rep.replace('\n',' ') + '\n')

            # try to save as JSON
            try:
                safe_dump_json(obj, json_path)
            except Exception:
                pass

            # if dict of arrays/records, try to make CSV
            # find if values are list-like and equal-length
            lengths = []
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    lengths.append(len(v))
            if lengths and len(set(lengths)) == 1:
                # convert to records
                try:
                    records = [{k: (np.asarray(v)[i].item() if isinstance(v, np.ndarray) and np.asarray(v).ndim==1 else (v[i] if isinstance(v, (list,tuple)) else v)) for k, v in obj.items()} for i in range(lengths[0])]
                    # write CSV
                    import csv
                    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                        writer = csv.DictWriter(cf, fieldnames=sorted(obj.keys()))
                        writer.writeheader()
                        for r in records:
                            writer.writerow(r)
                except Exception:
                    pass

            return [str(summary_path), str(json_path), str(csv_path)]

        # list-like
        if isinstance(obj, (list, tuple)):
            s.write('List/Tuple of length %d\n' % (len(obj),))
            s.write('Sample repr of first elements:\n')
            for i, v in enumerate(obj[:10]):
                s.write(f"[{i}] {type(v)} -> ")
                try:
                    rep = repr(v)
                    if len(rep) > 400:
                        rep = rep[:400] + '...'
                except Exception:
                    rep = str(type(v))
                s.write(rep.replace('\n',' ') + '\n')
            try:
                safe_dump_json(obj, json_path)
            except Exception:
                pass
            return [str(summary_path), str(json_path)]

        # fallback: try to JSON-serialize
        try:
            safe_dump_json(obj, json_path)
            s.write('\nJSON dump succeeded\n')
            return [str(summary_path), str(json_path)]
        except Exception:
            s.write('\nCould not JSON serialize object. Will write repr to file.\n')
            rep_path = out_dir / f"{base_name}_repr.txt"
            with open(rep_path, 'w', encoding='utf-8') as rf:
                rf.write(repr(obj))
            return [str(summary_path), str(rep_path)]


def load_pickle(path):
    # prefer pandas if available
    if HAS_PANDAS:
        try:
            import pandas as pd
            return pd.read_pickle(path)
        except Exception:
            pass
    # fallback to pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description='Dump a .pkl to readable formats (JSON/CSV/TXT)')
    p.add_argument('pkl', help='Path to .pkl file')
    p.add_argument('--out', '-o', default='graficos/output', help='Output directory')
    args = p.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print('File not found:', pkl_path)
        return 2

    base_name = pkl_path.stem
    print('Loading:', pkl_path)
    obj = load_pickle(pkl_path)
    print('Loaded type:', type(obj))

    outputs = make_readable(obj, args.out, base_name)
    print('Wrote:', '\n'.join([str(x) for x in outputs if x]))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
