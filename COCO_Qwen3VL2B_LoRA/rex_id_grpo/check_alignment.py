"""Inspect source-attributed settings; refuse launch until unknowns are confirmed."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-confirmation',action='store_true')
    args=parser.parse_args()
    settings=json.loads(Path(__file__).with_name('alignment.json').read_text())
    print(json.dumps(settings,ensure_ascii=False,indent=2),flush=True)
    if args.require_confirmation and not settings['unpublished_details_confirmed']:
        parser.exit(2,'Training disabled: unpublished GroundingPI details await user confirmation.\n')


if __name__=='__main__':
    main()
