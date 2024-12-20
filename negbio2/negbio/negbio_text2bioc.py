"""
Convert text FILEs to the BioC output file

Usage:
    negbio_text2bioc [options] --output=<file> <file> ...

Options:
    --output=<file>     Specify the output file name.
    --verbose           Print more information about progress.
"""

import bioc

from negbio2.negbio.cli_utils import parse_args
from negbio2.negbio.ext.text2bioc import text2collection

if __name__ == '__main__':
    argv = parse_args(__doc__)
    collection = text2collection(*argv['<file>'])
    with open(argv['--output'], 'w') as fp:
        bioc.dump(collection, fp)
