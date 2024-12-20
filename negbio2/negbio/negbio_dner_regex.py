"""
Detect concepts from vocab

Usage:
    negbio_dner_regex [options] --output=<directory> <file> ...

Options:
    --suffix=<suffix>       Append an additional SUFFIX to file names. [default: .regex.xml]
    --output=<directory>    Specify the output directory.
    --verbose               Print more information about progress.
    --phrases=<file>        File containing phrases for each observation. [default: patterns/cxr14_phrases_v2.yml]
    --overwrite             Overwrite the output file.
    --workers=<n>           Number of threads [default: 1]
    --files_per_worker=<n>  Number of input files per worker [default: 32]
"""
from pathlib import Path

from negbio2.negbio.pipeline2.dner_regex import RegExExtractor
from negbio2.negbio.cli_utils import parse_args, calls_asynchronously
from negbio2.negbio.pipeline2.pipeline import NegBioPipeline


if __name__ == '__main__':
    argv = parse_args(__doc__)
    workers = int(argv['--workers'])
    if workers == 1:
        phrases_file = Path(argv['--phrases'])
        extractor = RegExExtractor(phrases_file, phrases_file.stem)
        pipeline = NegBioPipeline(pipeline=[('RegEx', extractor)])
        pipeline.scan(source=argv['<file>'], directory=argv['--output'], suffix=argv['--suffix'],
                      overwrite=argv['--overwrite'])
    else:
        calls_asynchronously(argv, 'python -m negbio.negbio_dner_regex')
