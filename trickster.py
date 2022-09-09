import json
import unittest
import torch
import torch.nn as nn
import nltk
from nltk.stem.snowball import GermanStemmer
import numpy as np
import random

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        self.out = self.l1(x)
        self.out = self.relu(self.out)
        self.out = self.l2(self.out)
        self.out = self.relu(self.out)
        self.out = self.l3(self.out)
        # no activation and no softmax at the end
        return self.out

class Trickster:
    """
    Intent chatbot that uses a torch trained model.
    """

    def __init__(self, path_to_model: str, path_to_intents: str) -> None:
        """
        Constructor.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = torch.load(path_to_model, map_location=self.device)
        with open(path_to_intents, encoding='utf-8')  as json_data:
            self.intents = json.load(json_data)

        self.stemmer = GermanStemmer()

        self.input_size: int = self.data["input_size"]
        self.hidden_size: int = self.data["hidden_size"]
        self.output_size: int = self.data["output_size"]
        self.all_words: list[str] = self.data['all_words']
        self.tags: list[str] = self.data['tags']
        self.model_state = self.data["model_state"]

        self.model: NeuralNet = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()


    def tokenize(self, sentence):
        """
        split sentence into array of words/tokens
        a token can be a word or punctuation character, or number
        """
        return nltk.word_tokenize(sentence, language='german')

    def stem(self, word: str) -> list[str]:
        """
        stemming = find the root form of the word
        examples:
        words = ["organize", "organizes", "organizing"]
        words = [stem(w) for w in words]
        -> ["organ", "organ", "organ"]
        """
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence: list[str], words: list[str]):
        """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
        # stem each word
        self.sentence_words: list[str] = [self.stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        self.bag = np.zeros(len(self.all_words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in self.sentence_words: 
                self.bag[idx] = 1

        return self.bag
    
    def greeting(self) -> str:
        """
        return reply from bot as if it's a 'Hello' message.
        """
        return self.response("Hallo")

    def response(self, sentence: str) -> str:
        """
        return text response from bot to user massage.
        """
        if sentence == "quit":
            #f.close()
            return "Danke, bis zum nächsten mal."

        self.sentence = self.tokenize(sentence)
        self.X = self.bag_of_words(self.sentence, self.all_words)
        self.X = self.X.reshape(1, self.X.shape[0])
        self.X = torch.from_numpy(self.X).to(self.device)

        self.output = self.model(self.X)
        _, self.predicted = torch.max(self.output, dim=1)

        self.tag = self.tags[self.predicted.item()]

        self.probs = torch.softmax(self.output, dim=1)
        self.prob = self.probs[0][self.predicted.item()]
        self.response_text = ""
        if self.prob.item() > 0.75:
            for intent in self.intents['intents']:
                if self.tag == "coffelecture":
                    pass #crawl lecture date
                if self.tag == intent["tag"]:
                    self.response_text = f"{random.choice(intent['responses'])}"
        else:
            self.response_text = "👀 Das habe ich nicht verststanden..."
        return self.response_text


class TricksterTestCase(unittest.TestCase):

    def setUp(self):
        self.trickster = Trickster("./data.pth", "./intents.json")

    def test_trickster_greetings(self):
        self.assertTrue(self.trickster.greeting() in self.trickster.intents['intents'][0]['responses'])
        self.assertFalse(self.trickster.greeting() in self.trickster.intents['intents'][1]['responses'])

    def test_trickster_response(self):
        self.assertTrue(self.trickster.response("Hey") in self.trickster.intents['intents'][0]['responses'])
        self.assertTrue(self.trickster.response("Tschüss") in self.trickster.intents['intents'][1]['responses'])
        self.assertTrue(self.trickster.response("Wie lange ist heute geöffnet") in self.trickster.intents['intents'][4]['responses'])
        self.assertIn(self.trickster.response("quit"), "Danke, bis zum nächsten mal.")
        self.assertIn(self.trickster.response("foo bar"), "Das habe ich nicht verststanden...")


if __name__ == '__main__':
    unittest.main()