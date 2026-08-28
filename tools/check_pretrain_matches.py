import sys
from pathlib import Path
import importlib.util

module_path = Path(__file__).parents[1] / 'src' / '6_train_model.py'
spec = importlib.util.spec_from_file_location('trainmod', str(module_path))
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain-root', required=True)
    parser.add_argument('--codebook', default='outputs/final_codebook.pkl')
    parser.add_argument('--pretrain-manifest', default=None)
    parser.add_argument('--max-classes', type=int, default=100)
    parser.add_argument('--max-samples', type=int, default=20)
    args = parser.parse_args()

    codebook = trainmod.load_codebook(Path(args.codebook))
    print(f'Loaded codebook entries: {len(codebook)}')
    pretrain_root = Path(args.pretrain_root)
    manifest_path = Path(args.pretrain_manifest) if args.pretrain_manifest else None
    class_to_index, entries = trainmod.collect_class_samples(pretrain_root, codebook, max_classes=args.max_classes, max_samples_per_class=args.max_samples, manifest_path=manifest_path, allow_empty=True)
    print(f'Found classes: {len(class_to_index)}, entries: {len(entries)}')
    # print sample mappings
    from collections import Counter
    if entries:
        counts = Counter([e['unicode'] for e in entries])
        for k,v in counts.most_common(20):
            print(k, v)

if __name__ == '__main__':
    main()
