#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk import sent_tokenize,word_tokenize
from nltk import RegexpParser,Tree
from nltk.tag.perceptron import PerceptronTagger
import re


class HearstPatterns(object):
    '''
    Manual hypernymy extraction method based on hearst patterns.
    Make use of 'nltk' for - sentence tokenization, word tokenization,
                           - pos tagging,
                           - pos chunking.
            're' for regular expression matching.
    To use this, only need to implement find_hyponyms() method.
    
    Methods skeleton:
        prepare(): preprocess, does NLTK default sentence segmenter, word tokenizer, POS tagger
        chunk(): return chunks for find_hyponyms()
        prepare_chunks(): merge consecutive NP chunks and tagged with 'NP_'
        find_hyponyms(): main entry point for this code.
                        takes as input the rawtext to process and returns a list of tuples (specific-term, general-term)
                        where each tuple represents a hypernym pair.
        clean_hyonym_term(): remove 'NP_' or '_' in hyponymys
        __str__(): return method name
    
    Major input: rawtext
    '''
    
    def __init__(self,extended=False):
        self.__chunk_patterns=r"""
                         NP: {<DT|PP\$>?<JJ>*<NN>+}
                             {<NNP>+}
                             {<NNS>+}
                 """  # implement sequentially   'NP: {(<J.*>*<N.*>)*}'
        
        self.__np_chunker=RegexpParser(self.__chunk_patterns)  # create a chunk parser 
        
        self.__pos_tagger=PerceptronTagger()
        
        # now define hearst patterns
        # format: <hearst-pattern>, <general-term> position label
        # where <general-term> is ragarded as hypernym in this project
        # the rest NPs are specifics, which are regarded as hyponyms in this project
        self.__hearst_patterns=[
            
                # Hearst, 1992
                ("(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)", "first"),  # NP such as {NP, NP, ..., (and|or)} NP
                ("(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)", "first"),  # such NP as {NP,}*{(or|and)} NP
                ("((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)", "last"),  # NP {,NP}*{,} or other NP; NP {,NP}*{,} and other NP
                ("(NP_\w+ (, )?including (NP_\w+ ?(, )?(and |or )?)+)", "first"),  # NP {,} including {NP,}*{or|and} NP
                ("(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)", "first"),  # NP {,} especially {NP,}*{or|and} NP
            
                # Facebook, 2018 added
                ("((NP_\w+ ?(, )?)+(and |or )?which is a(n)? (example|class|kind )?(of )?NP_\w+)", "last"),  # NP which is a (example|class|kind|...) of NP
                ("((NP_\w+ ?(, )?)+(and |or )?(any|some )?other NP_\w+)", "last"),  # NP (and|or)(any|some) other NP
                ("((NP_\w+ ?(, )?)+(and |or )?wich is called NP_\w+)", "last"),  # NP which is called NP
                ("((NP_\w+ ?(, )?)+ is a special case of NP_\w+)", "last"),  # NP is a special case of NP
                
            ]
        if extended:
            self.__hearst_patterns.extend([
                # remains to be modified
                ("((NP_\w+ ?(, )?)+(and |or )?any other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?some other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?is a(n)? NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?is NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?was a NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?were a NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?are a NP_\w+)", "last"),
                ("(NP_\w+ (, )?like (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("such (NP_\w+ (, )?as (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?like other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?one of the NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?one of these NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?one of those NP_\w+)", "last"),
                ("examples of (NP_\w+ (, )?is (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("examples of (NP_\w+ (, )?are (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?are examples of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?is example of NP_\w+)", "last"),
                ("(NP_\w+ (, )?for example (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?which is named NP_\w+)", "last"),
                ("(NP_\w+ (, )?mainly (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?mostly (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?notably (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?particularly (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?principally (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?in particular (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?except (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?other than (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?e.g. (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?i.e. (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?a kind of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?kinds of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?form of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?forms of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?which looks like NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?which sounds like NP_\w+)", "last"),
                ("(NP_\w+ (, )?which are similar to (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?which is similar to (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?examples of this is (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?examples of this are (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?types (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )? NP_\w+ types)", "last"),
                ("(NP_\w+ (, )?whether (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(compare (NP_\w+ ?(, )?)+(and |or )?with NP_\w+)", "last"),
                ("(NP_\w+ (, )?compared to (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?among them (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?as NP_\w+)", "last"),
                ("(NP_\w+ (, )? (NP_\w+ ? (, )?(and |or )?)+ for instance)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?sort of NP_\w+)", "last"),
            ])
    
    def prepare(self,rawtxt):
        sentences=sent_tokenize(rawtxt.strip()) # strip(): eliminate \n, \t before and after the string
        sentences=[word_tokenize(sent) for sent in sentences]
        sentences=[self.__pos_tagger.tag(sent) for sent in sentences]

        return sentences
    
    def chunk(self,rawtxt):
        sentences=self.prepare(rawtxt.strip())
        all_chunks=[]
        for sentence in sentences:
            chunks=self.__np_chunker.parse(sentence)  # parse sentences
            all_chunks.append(self.prepare_chunks(chunks))  # NP_phrase
        return all_chunks
    
    def prepare_chunks(self,chunks):
        tmp_sen=[]
        for chunk in chunks:
            label=None
            if isinstance(chunk,Tree):  # stackoverflow
                label=chunk.label()
            if label is None:
                token=chunk[0]
                pos=chunk[1]
                if pos in ['.',':','-','_']:  # remove punctuation
                    continue
                tmp_sen.append(token)
            else:
                tmp_sen.append('NP_'+'_'.join([a[0] for a in chunk]))
        return ' '.join(tmp_sen)
    
    '''
        This is the main entry for this code.
        It takes as input the rawtext to process and returns a list of tuples (specific-term, general-term)
        where each tuple represents a hypernym pair.
    '''
    def find_hyponyms(self,rawtxt):
        
        hyponyms=[]
        np_sentences=self.chunk(rawtxt)
        for sen in np_sentences:
            sentence = re.sub(r'(NP_\w+ NP_\w+)+', lambda m: m.expand(r'\1').replace('NP_', '_'), sen)
            
            for (hearst_pattern,parser) in self.__hearst_patterns:
                find=re.search(hearst_pattern,sentence)
                if find:
                    match_str=find.group(0)
                    
                    nps=[a for a in match_str.split() if a.startswith('NP_')]
                    
                    if parser=='first':
                        general=nps[0]
                        specifics=nps[1:]
                    else:
                        general=nps[-1]
                        specifics=nps[:-1]
                    
                    for i in range(len(specifics)):  # because specifics is a list
                        hyponyms.append((self.clean_hyponym_term(specifics[i]), self.clean_hyponym_term(general)))

        return hyponyms

    def clean_hyponym_term(self,term):
        return term.replace('NP_','').replace('_',' ')
    
    def __str__(self):
        return 'hearstPatterns'
        

