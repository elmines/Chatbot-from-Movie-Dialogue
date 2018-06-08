import sys
import argparse

lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


_DEFAULT_MAX_WORDS = 12000


def create_parser():
    parser = argparse.ArgumentParser(description="Preprocess Cornell Movie Dialogue dataset")

    parser.add_argument("--lines", default="./movie_lines.txt", type=str, metavar="<path>", help="Path to file with lines text")
    parser.add_argumetn("--conversations", "--conv", default="./movie_conversations.txt", type=str, metavar="<path>", help="Path to file with conversations")

    parser.add_argument("--max_words", "--words", default=_DEFAULT_MAX_WORDS, type=int, metavar="n", help="Maximum size vocabulary, not including metacharacters")

    parser.add_argument("--vocab-out", default=None, type=str, metavar="<path>", help="Path to write text for 


# In[3]:

# The sentences that we will be using to train our model.
lines[:10]


# In[4]:

# The sentences' ids, which will be processed to become our input and target data.
conv_lines[:10]


# In[5]:

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# In[6]:

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))


# In[7]:

convs[:10]


# In[8]:

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# In[9]:

# Check if we have loaded the data correctly
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()


# In[10]:

# Compare lengths of questions and answers
print(len(questions))
print(len(answers))


# In[11]:

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# In[12]:

# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))


# In[13]:

# Take a look at some of the data to ensure that it has been cleaned well.
limit = 0
for i in range(limit, limit+5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()


# In[14]:

# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# In[15]:

lengths.describe()


# In[16]:

print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))


# In[17]:

# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1


# In[18]:

# Compare the number of lines we will use with the total number of lines.
print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))


# In[19]:

# Create a dictionary for the frequency of the vocabulary
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
            
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# In[20]:

# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1


# In[21]:

print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)


# In[22]:

# In case we want to use a different vocabulary sizes for the source and target text, 
# we can set different threshold values.
# Nonetheless, we will create dictionaries to provide a unique integer for each word.
questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1
        
answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1


# In[23]:

# Add the unique tokens to the vocabulary dictionaries.
codes = ['<PAD>','<EOS>','<UNK>','<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int)+1
    
for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int)+1


# In[24]:

# Create dictionaries to map the unique integers to their respective words.
# i.e. an inverse dictionary for vocab_to_int.
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}


# In[25]:

# Check the length of the dictionaries.
print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))


# In[26]:

# Add the end of sentence token to the end of every answer.
for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'


# In[27]:

# Convert the text to integers. 
# Replace any words that are not in the respective vocabulary with <UNK> 
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)
    
answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)


# In[28]:

# Check the lengths
print(len(questions_int))
print(len(answers_int))


# In[29]:

# Calculate what percentage of all words have been replaced with <UNK>
word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
unk_ratio = round(unk_count/word_count,4)*100
    
print("Total number of words:", word_count)
print("Number of times <UNK> is used:", unk_count)
print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))


# In[30]:

# Sort questions and answers by the length of questions.
# This will reduce the amount of padding during training
# Which should speed up training and help to reduce the loss

sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])

print(len(sorted_questions))
print(len(sorted_answers))
print()
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()
