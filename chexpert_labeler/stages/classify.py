"""Define mention classifier class."""
import logging
from pathlib import Path
# from NegBio.negbio.pipeline import parse, ptb2ud, negdetect
# from NegBio.negbio.neg import semgraph, propagator, neg_detector
# from NegBio.negbio import ngrex
from negbio2.negbio.pipeline2 import parse, ptb2ud, lemmatize
from negbio2.negbio.neg import semgraph, propagator, neg_detector
from negbio2.negbio.pipeline2.negdetect import NegBioNegDetector
from negbio2.negbio.pipeline2.negdetect2 import NegBioNegDetector2
from negbio2.negbio import ngrex
from tqdm import tqdm
from lavis.common.dist_utils import is_dist_avail_and_initialized
from chexpert_labeler.constants import *


class ModifiedDetector(neg_detector.Detector):
    """Child class of NegBio Detector class.

    Overrides parent methods __init__, detect, and match_uncertainty.
    """
    def __init__(self, pre_negation_uncertainty_path,
                 negation_path, post_negation_uncertainty_path):
        
        self.neg_patterns = ngrex.load(negation_path)
        self.uncertain_patterns = ngrex.load(post_negation_uncertainty_path)
        self.preneg_uncertain_patterns\
            = ngrex.load(pre_negation_uncertainty_path)
        self.total_patterns = {}
        for pattern in self.neg_patterns + self.uncertain_patterns + self.preneg_uncertain_patterns:
            self.total_patterns[pattern] = {
                'patternobj': pattern,
                'pattern': str(pattern)
            }

    def convert_graph_strings(self, g):
        """Convert Java strings to Python strings in graph while preserving existing Python strings."""
        def to_python_string(value):
            
            if isinstance(value, str):
                return value
            # If it is a Java string, convert it to a Python string
            if hasattr(value, 'toString'):
                return str(value.toString())
            
            return value
            
        
        for node in g.nodes():
            for key in g.nodes[node]:
                g.nodes[node][key] = to_python_string(g.nodes[node][key])
                
       
        for u, v, data in g.edges(data=True):
            for key in data:
                data[key] = to_python_string(data[key])
        
        return g

    def detect(self, sentence, locs):
        """Detect rules in report sentences.

        Args:
            sentence(BioCSentence): a sentence with universal dependencies
            locs(list): a list of (begin, end)

        Return:
            (str, MatcherObj, (begin, end)): negation or uncertainty,
            matcher, matched annotation
        """
        logger = logging.getLogger(__name__)

        try:
            g = semgraph.load(sentence)
            g = self.convert_graph_strings(g)
            propagator.propagate(g)
        except Exception:
            logger.exception('Cannot parse dependency graph ' +
                             f'[offset={sentence.offset}]')
            raise
        else:
            for loc in locs:
                for node in neg_detector.find_nodes(g, loc[0], loc[1]):
                    # Match pre-negation uncertainty rules first.
                    preneg_m = self.match_prenegation_uncertainty(g, node)
                    if preneg_m:
                        yield UNCERTAINTY, preneg_m, loc
                    else:
                        # Then match negation rules.
                        neg_m = self.match_neg(g, node)
                        if neg_m:
                            yield NEGATION, neg_m, loc
                        else:
                            # Finally match post-negation uncertainty rules.
                            postneg_m = self.match_uncertainty(g, node)
                            if postneg_m:
                                yield UNCERTAINTY, postneg_m, loc

    def match_uncertainty(self, graph, node):
        for pattern in self.uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m

    def match_prenegation_uncertainty(self, graph, node):
        for pattern in self.preneg_uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m


class Classifier(object):
    """Classify mentions of observations from radiology reports."""
    def __init__(self, pre_negation_uncertainty_path, negation_path,
                 post_negation_uncertainty_path, verbose=False):
        if is_dist_avail_and_initialized():
            self.parser = parse.NegBioParser(model_dir=PARSING_MODEL_DIR)
        else:
            self.parser = parse.NegBioParser()
        lemmatizer = lemmatize.Lemmatizer()
        # self.ptb2dep = ptb2ud.NegBioPtb2DepConverter(lemmatizer, universal=True)
        self.ptb2dep = ptb2ud.NegBioPtb2DepConverter(
            representation='CCprocessed',
            # lemmatizer,
            universal=True
        )
        self.all_dependencies = set()
        self.ptb2dep = ptb2ud.NegBioPtb2DepConverter(
            universal=True
        )
        

        self.verbose = verbose

        self.detector = ModifiedDetector(pre_negation_uncertainty_path,
                                         negation_path,
                                         post_negation_uncertainty_path)
        self.neg_detector = NegBioNegDetector2(self.detector)
    
    def collect_all_dependencies(self, document):
        dependencies = set()
        for passage in document.passages:
            for sentence in passage.sentences:
                for relation in sentence.relations:
                    if 'dependency' in relation.infons:
                        dependencies.add(relation.infons['dependency'])
        return dependencies

    def classify(self, collection):

        """Classify each mention into one of
        negative, uncertain, or positive."""
        documents = collection.documents
        if self.verbose:
            documents = tqdm(documents)
        for document in documents:
            # print("\n" + "="*50)
            # print("=== Document ID:", document.id, "===")
            # print("\nOriginal text:")
            # print(document.passages[0].text)
            
            # print("\nParser state:")
            # print(f"Model directory: {self.parser.model_dir}")
            # print(f"Parser initialized: {hasattr(self.parser, 'rrp')}")
            
            # # print("\nBefore parsing:")
            # for sentence in document.passages[0].sentences:
            #     print(f"Sentence: {sentence.text}")
            #     print(f"Current infons: {sentence.infons}")
            # print("\nApplying parser...")
            # Parse the impression text in place.
            
            self.parser(document)
            
            
            # print("\nAfter parsing:")
            # for sentence in document.passages[0].sentences:
            #     print(f"Sentence: {sentence.text}")
            #     print(f"Parse tree in infons: {sentence.infons.get('parse tree')}")
            # print("\nGenerating dependency graph...")
            # Add the universal dependency graph in place.
            self.ptb2dep(document)
            # print("\nDependency relations:")
            # for sentence in document.passages[0].sentences:
                # print(f"Sentence: {sentence.text}")
                # print(f"Annotations: {sentence.annotations}")  
                # print(f"Relations: {sentence.relations}")
                # for relation in sentence.relations:
                #     if 'dependency' in relation.infons:
                #         self.all_dependencies.add(relation.infons['dependency'])
            
            # print("\nDetecting negation...")
            # Detect the negation and uncertainty rules in place.
            self.neg_detector(document, self.detector)
            # print("\nNegation results:")
            # for passage in document.passages:
            #     for ann in passage.annotations:
            #         if ann.infons.get('negation') == 'True':
                        # print(f"Text: {ann.text}")
                        # print(f"Negation info: {ann.infons}")
                
            # To reduce memory consumption, remove sentences text.
            del document.passages[0].sentences[:]
        
        # print("\nAll dependency types used:")
        # for dep in sorted(self.all_dependencies):
        #     print(f"- {dep}")
