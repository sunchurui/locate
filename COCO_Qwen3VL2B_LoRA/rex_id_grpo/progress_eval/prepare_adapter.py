"""Create a metadata view of a saved adapter without modifying checkpoints."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, metadata, output = map(lambda p: Path(p).resolve(), (args.checkpoint, args.metadata, args.output))
    config = json.loads((source / 'adapter_config.json').read_text())
    if config.get('trainable_token_indices') != list(range(150643, 151643)):
        raise ValueError('Expected the original Rex coordinate-ID adapter')
    if not (source / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Adapter weights have not finished saving')
    if source.name.startswith('checkpoint-'):
        state = json.loads((source / 'trainer_state.json').read_text())
        if state['global_step'] != int(source.name.removeprefix('checkpoint-')):
            raise ValueError('Checkpoint state does not match its step')
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            if not (source / name).is_file():
                raise FileNotFoundError(f'Checkpoint is still incomplete: {name}')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'adapter_config.json', output / 'adapter_config.json')
    weights = output / 'adapter_model.safetensors'
    if weights.exists():
        if not weights.is_symlink() or weights.resolve() != (source / weights.name).resolve():
            raise ValueError('Adapter view already points to different weights')
    else:
        weights.symlink_to(source / weights.name)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json',
                 'video_preprocessor_config.json', 'chat_template.json', 'chat_template.jinja',
                 'processor_config.json', 'special_tokens_map.json'):
        if (metadata / name).is_file():
            shutil.copy2(metadata / name, output / name)
    if not any((output / name).is_file() for name in ('chat_template.json', 'chat_template.jinja')):
        raise FileNotFoundError('Processor chat template is missing from the adapter view')
    print(output)


if __name__ == '__main__':
    main()
