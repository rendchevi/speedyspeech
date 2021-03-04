import re

from g2p_en import G2p
from functional import pad_batch


class TextProcessor:

    # only available for English - output from g2p
    phonemes = ["<pad>", "<unk>"] \
               + [
                   'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                   'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                   'B', 'CH', 'D', 'DH',
                   'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                   'F', 'G', 'HH',
                   'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                   'JH', 'K', 'L', 'M', 'N', 'NG',
                   'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
                   'P', 'R', 'S', 'SH', 'T', 'TH',
                   'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
                   'V', 'W', 'Y', 'Z', 'ZH', ' ', '.', ',', '?', '!', '\-'
               ]

    def __init__(self, graphemes_list, phonemize=False):
        """

        :param graphemes_list: list of graphemes starting with ['<pad>', '<unk>']
        :param phonemes: list of phonemes from gpt
        """

        assert graphemes_list[0] == '<pad>' and graphemes_list[1] == '<unk>', \
            'First two items must be <pad> and <unk>'

        self.graphemes = graphemes_list
        self.phonemize = phonemize

        if phonemize:
            self.g2p = G2p()
            self.phon2idx = {g: i for i, g in enumerate(self.phonemes)}
            self.idx2phon = {i: g for i, g in enumerate(self.phonemes)}
        else:
            self.text2idx = {g: i for i, g in enumerate(graphemes_list)}
            self.idx2text = {i: g for i, g in enumerate(graphemes_list)}

    def __call__(self, text):
        """

        :param text: list of sentences
        :return:
            zero-padded batch, (num_sentences, max_sentence_len)
            orig_lengths, list of int
        """

        text = [t.lower() for t in text]
        if not self.phonemize:
            text = [
                [self.text2idx.get(ch, 1) for ch in s]  # use <unk> if character not in grapheme_list
                for s in text
            ]
            return pad_batch(text)
        else:
            keep_re = "[^" + str(self.graphemes[2:]) +"]"
            text = [re.sub(keep_re, "", t) for t in text]  # remove
            phonemes = [self.g2p(t) for t in text]
            phonemes = [
                [self.phon2idx.get(ch, 1) for ch in s]
                for s in phonemes
            ]
            return pad_batch(phonemes)

class TextProcessorInd:

    phonemes = ['pad'] \
                + [
                    'ch', 'ng', 'ny', 'ai', 'ay', 'oi', 'oy', 'ei',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                ] \
                + [
                ' ', '!', '?', '.', ',', ',', '-', ':', ';', '(', ')'
                ]

    def __init__(self):

        self.char_list, self.punctuation, self.vocab_mapper, self.special = self.prepare_vocab()


    def __call__(self, texts):
        '''
        Input : list of strings ['januari yang basah', 'Hai!']
        Output: list of list of int [[1,2,4,0,0], [2,4,5]]
        '''

        # Filter texts
        filtered_texts = [''.join(c for c in text.lower() if c in self.char_list + self.punctuation) for text in texts]
        
        # Tokenize texts to phonemes
        sep_texts = []
        for text in filtered_texts:
            for c in self.char_list + self.punctuation:
                text = text.replace(c, str(self.vocab_mapper['phon2idx'][c]) + '-')
            sep_texts.append(text)

        tokenized_phonemes = [[int(t) for t in text.split('-')[:-1]] for text in sep_texts]

        return pad_batch(tokenized_phonemes)

    def prepare_vocab(self):

        char_list = [
            'ch', 'ng', 'ny', 'ai', 'ay', 'oi', 'oy', 'ei',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ]

        punctuation = [
            ' ', '!', '?', '.', ',', ','
        ]

        special = [
            'pad'
        ]

        vocab_mapper = {
            'phon2idx': {k: v for k, v in zip(special + char_list + punctuation, range(0, len(char_list + punctuation)))},
            'idx2phon': {k: v for k, v in zip(range(0, len(char_list + punctuation)), special + char_list + punctuation)}
        }

        return char_list, punctuation, vocab_mapper, special