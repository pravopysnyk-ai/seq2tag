import sys
import pymorphy2
import spacy_udpipe
from helpers.classes.SpaceHandler import SpaceHandler
from helpers.classes.Inflector import Inflector
from helpers.classes.GrammarInterpreter import GrammarInterpreter
from helpers.classes.PunctInterpreter import PunctInterpreter
from helpers.classes.AntiTokenizer import AntiTokenizer
from helpers.classes.FormExtractorInflector import FormExtractorInflector
from helpers.classes.SurzhInterpreter import SurzhInterpreter

class Seq2TagPredict(object):
    """
    Predicts the applied tags for the Seq2Tag category.
    """
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer(lang='uk')
        self.space_handler = SpaceHandler()
        self.punct_interpreter = PunctInterpreter(space_handler=self.space_handler)
        self.inflector = Inflector(self.morph)
        self.grammar_interpreter = GrammarInterpreter(space_handler=self.space_handler,
                                                      inflector=self.inflector)
        self.spacy_model = spacy_udpipe.load_from_path(lang="uk",
                                                       path="./helpers/modules/ukrainian-iu-ud-2.5-191206.udpipe")
        self.surzh_interpreter = SurzhInterpreter(self.inflector, self.spacy_model)

    def load_antitokenizers(self):
        """
        Loads interpreters for each error type.
        Don't forget to specify your own models.
        """
        # punctuation
        punct_anti_tokenizer = AntiTokenizer(model_name='bavovna-BNR',
                                             checkpoint='checkpoint-25934',
                                             model_type='punct',
                                             space_handler = self.space_handler)
        # grammar
        grammar_anti_tokenizer = AntiTokenizer(model_name='chornobaivka-25',
                                               checkpoint='checkpoint-51588',
                                               model_type='grammar',
                                               space_handler=self.space_handler)
        return punct_anti_tokenizer, grammar_anti_tokenizer

    def main(self):
        # loading the antitokenizers
        punct_anti_tokenizer, grammar_anti_tokenizer = load_antitokenizers()
        # printing the use information on the screen
        print("Welcome to Pravopysnyk!\nPlease enter your sentence below.\nTo exit the program, press Enter with no input.\n\n\n")

        # while loop for running the program
        while True:
            errorful_sentence = input("Your sentence: ") # user input
            if len(errorful_sentence) == 0: # if empty, exit
                break
            else:
                words, tags = grammar_anti_tokenizer(errorful_sentence)
                out = grammar_interpreter(words, tags)
                words, tags = punct_anti_tokenizer(out)
                out = punct_interpreter(words, tags)
                print("Corrected sentence: " + out + "\n") # and print out
                continue
        return
