#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import json


# ## Task 1: Vocabulary Creation

# In[2]:


trainData = []
trainDataWord = []
trainDataTag = ['start']
trainDataWordTag = []
with open('./data/train') as f:
    for line in f:
        ls = (line.rstrip('\n')).split("\t")
        if len(ls) == 3:
            trainDataTag.append(ls[2])
            word = ls[1]
            trainDataWord.append(word)
            trainDataWordTag.append((ls[2], word))
        else:
            trainDataTag.append('start')


# In[3]:


trainWordCount = collections.Counter(trainDataWord)


# In[4]:


unkCnt = 0
vocabDict = {}
unkWords = []
for key in trainWordCount:
    if trainWordCount[key] <= 2:
        unkCnt += trainWordCount[key]
        unkWords.append(key)
    else:
        vocabDict[key] = trainWordCount[key]


# In[5]:


vocabDict = dict(sorted(vocabDict.items(), key=lambda item: item[1], reverse=True))


# In[6]:


vocab = []
i = 2
for key in vocabDict:
    a =key+'\t'+str(i)+'\t'+str(vocabDict[key])
    i += 1
    vocab.append(a)
vocab.insert( 0, '<unk>'+'\t'+"1"+'\t'+str(unkCnt))


# In[7]:


file = open('vocab.txt','w')
for item in vocab:
    file.write(item+"\n")
file.close()


# ## Task 2: Model Learning

# In[9]:


trainWordTag = []
for word in trainDataWordTag:
    if word[1] not in vocabDict:
        trainWordTag.append((word[0], '<unk>'))
    else:
        trainWordTag.append((word[0], word[1]))


# In[10]:


trainWordTagCnt = collections.Counter(trainWordTag)


# In[11]:


trainDataWordTagCount = {}
trainDataTagCount = collections.defaultdict(int)
for tag, word in trainDataWordTag:
    trainDataTagCount[tag] += 1
    trainDataWordTagCount[(tag, word)] = trainDataWordTagCount.get((tag, word), 0) + 1


# In[12]:


transitionProb = {}
emissionProb = {}


# In[13]:


def transitionHelper(tags):
    getTwoTags = []
    for i in range(len(tags)): 
        getTwoTags.append(tuple(tags[i: i + 2]))
    for i in getTwoTags:
        transitionHelperCnt[i] = transitionHelperCnt.get(i, 0) + 1
    return transitionHelperCnt

def getTransitionProbabilty(tags):
    getTwoTags = []
    for i in range(len(tags)): 
        getTwoTags.append(tuple(tags[i: i + 2]))
    for i in getTwoTags:
        transitionProb[i] = transitionHelperCnt[i]  / trainTagStartCount[i[0]]
    return transitionProb

def getEmissionProbabilty(tagged_words):
    for tag, word in tagged_words:
        if word not in vocabDict:
            emissionProb[tag, '<unk>'] = trainWordTagCnt[tag, '<unk>'] / trainDataTagCount[tag]
        else:
            emissionProb[tag, word] = trainDataWordTagCount[tag, word] / trainDataTagCount[tag]
    return emissionProb


# In[14]:


transitionHelperCnt = {}
transitionHelper(trainDataTag)
trainTagStartCount = collections.Counter(trainDataTag)


# In[15]:


getTransitionProbabilty(trainDataTag)


# In[16]:


getEmissionProbabilty(trainDataWordTag)


# In[17]:


finalemissionProb = {}
tagSet = set(trainDataTag)
tagSet.remove('start')
trainDataWordset = set(trainDataWord)
for word in trainDataWordset:
    for ptwo in tagSet:
        if (word, ptwo) in emissionProb:
            finalemissionProb[(ptwo, word)] = emissionProb[(ptwo, word)]
        else:
            finalemissionProb[(ptwo, word)] = 0


# In[19]:


# print('Transition parameters:', len(transitionProb))
# print('Emission parameters:', len(emissionProb))


# In[20]:


transitionProbJson = {",".join(key): value for key, value in transitionProb.items()}
emissionProbJson = {",".join(key): value for key, value in emissionProb.items()}
hmm = {'transition':transitionProbJson, 'emission':emissionProbJson}
with open('hmm.json', 'w') as fp:
    json.dump(hmm, fp)
file.close()


# ## Task 3: Greedy Decoding withHMM

# In[21]:


def greedyDecoding(words):
    predTags = []
    iniTag = 'start'
    iniWord = words[0]
    maxVal = -1
    finalIniTag = ''
    posTags = set(trainDataTag)

    for p in posTags:
        if iniWord not in vocabDict:
            iniWord = '<unk>'
        if (iniTag,p) in transitionProb and (p, iniWord) in emissionProb:
            res = transitionProb[(iniTag,p)]*emissionProb[(p, iniWord)]
        else:
            res = 0
        if res > maxVal:
            maxVal = res
            finalIniTag = p
    predTags.append(finalIniTag)

    for idx, w in enumerate(words):
        if idx == 0:
            continue
        maxVal = -1
        if w not in vocabDict:
            w = '<unk>'
        prevIniTag = predTags[-1]
        finalIniTag = ""
        for p in posTags:
            if (prevIniTag,p) in transitionProb and (p, w) in emissionProb:
                res = transitionProb[(prevIniTag,p)]*emissionProb[(p, w)]
            else:
                res = 0
            if res > maxVal:
                maxVal = res
                finalIniTag = p
        predTags.append(finalIniTag)
    return predTags


# In[22]:


devSentence = []
devTagsList = []
devWords = []
devTags = []
predictedTags = []
with open('./data/dev') as f:
    for line in f:
        ls = (line.rstrip('\n')).split("\t")
        if len(ls) == 3:
            devWords.append(ls[1])
            devTags.append(ls[2])
        else:
            predictedTags.append(greedyDecoding(devWords))
            devTagsList.append(devTags)
            devSentence.append(devWords)
            devWords = []
            devTags = []


# In[23]:


pred = []
i = 0
cnt = 0
correct = 0
while i < len(devTagsList):
    j = 0
    while j < len(devTagsList[i])-2:
        if devTagsList[i][j] == predictedTags[i][j]:
            correct += 1
        cnt += 1
        j += 1
    i += 1
# print('Greedy decoding accuracy on dev data:',(correct/cnt)*100)


# In[24]:


testSentence = []
testWords = []
greedyPredictedTags = []
with open('./data/test') as f:
    for line in f:
        ls = (line.rstrip('\n')).split("\t")
        if len(ls) == 2:
            testWords.append(ls[1])
        else:
            greedyPredictedTags.append(greedyDecoding(testWords))
            testSentence.append(testWords)
            testWords = []


# In[25]:


i = 0
finalRes = []
while i < len(testSentence):
    c = 1
    j = 0
    while j < len(testSentence[i]):
        a = str(c)+'\t'+testSentence[i][j]+'\t'+greedyPredictedTags[i][j]+'\n'
        finalRes.append(a)
        j += 1
        c += 1
    finalRes.append('\n')
    i += 1


# In[26]:


finalRes = finalRes[:len(finalRes)-1]


# In[27]:


file = open('greedy.out','w')
for item in finalRes:
    file.write(item)
file.close()


# ## Task 4: Viterbi Decoding withHMM

# In[28]:


def viterbiDecoding(sentence):
        posTags = set(trainDataTag)
        posTags.remove("start")
        viterbiDP = {}
        for p in posTags:
            if sentence[0] not in vocabDict:
                if (p, '<unk>') not in emissionProb:
                    viterbiDP[p, 0] = 0
                else:
                    if ("start", p) not in transitionProb:
                        viterbiDP[p, 0] = 0
                    else:
                        viterbiDP[p, 0] = transitionProb["start", p] * emissionProb[p, '<unk>']
            elif (p, sentence[0]) not in emissionProb:
                viterbiDP[p, 0] = 0
            else:
                if ("start", p) not in transitionProb:
                    viterbiDP[p, 0] = 0
                else:
                    viterbiDP[p, 0] = transitionProb["start", p] * emissionProb[p, sentence[0]]
        
        
        for i in range(1, len(sentence)):
            getWord = sentence[i]
            if getWord not in vocabDict:
                getWord = '<unk>'
                
            for pone in posTags:
                probTag = []
                for ptwo in posTags:
                    if (ptwo,pone) not in transitionProb or (pone, getWord) not in emissionProb:
                        value = 0
                    else:
                        value = viterbiDP[ptwo, i - 1] * transitionProb[ptwo, pone] * emissionProb[pone, getWord]
                    probTag.append((value,ptwo))
                arr = sorted(probTag)[-1][1]
                if (arr,pone) not in transitionProb or (pone, getWord) not in emissionProb:
                    viterbiDP[pone, i] = 0
                else:
                    viterbiDP[pone, i] = viterbiDP[arr, i - 1] * transitionProb[arr, pone] * emissionProb[pone, getWord]

        predTags = []
        for i in range(len(sentence) - 1, -1, -1):
            finalProbTag = sorted([(viterbiDP[k, i], k) for k in posTags])[-1][1]
            predTags.append(finalProbTag)
        predTags.reverse()
        return predTags


# In[29]:


devSentence = []
devTagsList = []
devWords = []
devTags = []
predictedTags = []
with open('./data/dev') as f:
    for line in f:
        ls = (line.rstrip('\n')).split("\t")
        if len(ls) == 3:
            devWords.append(ls[1])
            devTags.append(ls[2])
        else:
            predictedTags.append(viterbiDecoding(devWords))
            devWords = []
            devTagsList.append(devTags)
            devTags = []


# In[30]:


pred = []
i = 0
cnt = 0
correct = 0
while i < len(devTagsList):
    j = 0
    while j < len(devTagsList[i]):
        if devTagsList[i][j] == predictedTags[i][j]:
            correct += 1
        cnt += 1
        j += 1
    i += 1

# print('Viterbi decoding accuracy on dev data:',(correct/cnt)*100)


# In[31]:


testSentence = []
testWords = []
viterbiPredictedTags = []
with open('./data/test') as f:
    for line in f:
        ls = (line.rstrip('\n')).split("\t")
        if len(ls) == 2:
            testWords.append(ls[1])
        else:
            viterbiPredictedTags.append(viterbiDecoding(testWords))
            testSentence.append(testWords)
            testWords = []


# In[32]:


i = 0
finalRes = []
while i < len(testSentence):
    c = 1
    j = 0
    while j < len(testSentence[i]):
        a = str(c)+'\t'+testSentence[i][j]+'\t'+viterbiPredictedTags[i][j]+'\n'
        finalRes.append(a)
        j += 1
        c += 1
    finalRes.append('\n')
    i += 1


# In[33]:


finalRes = finalRes[:len(finalRes)-1]


# In[34]:


file = open('veterbi.out','w')
for item in finalRes:
    file.write(item)
file.close()


# In[ ]:





# In[ ]:





# In[ ]:




