import nltk
# nltk.download()

#  Speech delivered by Swami Vivekananda on September 11, 1893 at the First World's Parliament of Religions in Chicago
message = """ 
It fills my heart with joy unspeakable to rise in response to the warm and cordial welcome 
which you have given us. I thank you in the name of the most ancient order of monks in the 
world; I thank you in the name of the mother of religions, and I thank you in the name of 
millions and millions of Hindu people of all classes and sects.
My thanks, also, to some of the speakers on this platform who, referring to the delegates 
from the Orient, have told you that these men from far-off nations may well claim the honor 
of bearing to different lands the idea of toleration. I am proud to belong to a religion 
which has taught the world both tolerance and universal acceptance. We believe not only in 
universal toleration, but we accept all religions as true. I am proud to belong to a nation 
which has sheltered the persecuted and the refugees of all religions and all nations of the 
earth. I am proud to tell you that we have gathered in our bosom the purest remnant of the 
Israelites, who came to Southern India and took refuge with us in the very year in which 
their holy temple was shattered to pieces by Roman tyranny. I am proud to belong to the 
religion which has sheltered and is still fostering the remnant of the grand Zoroastrian 
nation. I will quote to you, brethren, a few lines from a hymn which I remember to have 
repeated from my earliest boyhood, which is every day repeated by millions of human beings: 
    "As the different streams having their sources in different paths which men take through
    different tendencies, various though they appear, crooked or straight, all lead to Thee."
The present convention, which is one of the most august assemblies ever held, is in itself 
a vindication, a declaration to the world of the wonderful doctrine preached in the Gita: 
    "Whosoever comes to Me, through whatsoever form, I reach him; all men are struggling 
    through paths which in the end lead to me." 
Sectarianism, bigotry, and its horrible descendant, fanaticism, have long possessed this 
beautiful earth. They have filled the earth with violence, drenched it often and often with
human blood, destroyed civilization and sent whole nations to despair. Had it not been for 
these horrible demons, human society would be far more advanced than it is now. But their 
time is come; and I fervently hope that the bell that tolled this morning in honor of this 
convention may be the death-knell of all fanaticism, of all persecutions with the sword or
with the pen, and of all uncharitable feelings between persons wending their way to the 
same goal"""


# Breaking the message into sentences 
sentences = nltk.sent_tokenize(message)

# Stemming
"""
 With stemming, words are reduced to their word stems. A word stem need not be the same 
 root as a dictionary-based morphological root, it just is an equal to or smaller form of
 the word.
"""

# Porter Stemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
porter_sentences = []

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    porter_words = [stemmer.stem(word) for word in words]
    porter_sentences[i] = ' '.join(porter_words)
