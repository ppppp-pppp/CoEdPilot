#这个脚本的特点是它允许对多个测试集进行高效的BLEU分数计算，通过将BLEU计算分为三个阶段：预处理参考句子、预处理测试句子和计算分数。这样可以避免重复处理相同的数据，提高计算效率。
#BLEU分数：是一种评估机器翻译质量的方法，特别是在机器翻译研究领域中广泛使用。它通过计算机器翻译输出与一组参考翻译（通常是人工翻译）之间的重叠来工作。BLEU分数的范围是0到1，分数越高表示机器翻译的质量越接近人工翻译。
#!/usr/bin/python

'''
This script was adapted from the original version by hieuhoang1972 which is part of MOSES. 
'''

# $Id: bleu.py 1307 2007-03-14 22:22:36Z hieuhoang1972 $

'''Provides:这些注释描述了脚本提供的四个主要函数及其作用

cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
score_cooked(alltest, n=4): Score a list of cooked test sentences.

score_set(s, testid, refids, n=4): Interface with dataset.py; calculate BLEU score of testid against refids.

The reason for breaking the BLEU computation into three phases cook_refs(), cook_test(), and score_cooked() is to allow the caller to calculate BLEU scores for multiple test sets as efficiently as possible.
'''

import sys, math, re, xml.sax.saxutils
import subprocess
import os
import json

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0 #这是一个变量，用于控制是否跳过NIST风格的预处理。NIST风格的预处理是指在自然语言处理（NLP）和机器翻译领域中，对文本数据进行标准化处理的一种方法，以提高机器翻译系统的性能和准确性。

preserve_case = False #这个变量控制是否保留原始大小写。
eff_ref_len = "shortest" # eff_ref_len用来确定在计算BLEU分数时应该使用多长的参考句子。这里表示使用最短的参考句子长度作为有效参考长度。这是默认设置，意味着在计算BP时，会取所有参考句子中最短的一个作为基准。

normalize1 = [
    ('<skipped>', ''),         # strip "skipped" tags 用于查找文本中的 <skipped> 标签，并将其替换为空字符串，也就是将其从文本中移
    (r'-\n', ''),              # strip end-of-line hyphenation and join lines 用于处理断行的连字符，确保文本的连贯性
    (r'\n', ' '),              # join lines 将多行文本合并为单行
#    (r'(\d)\s+(?=\d)', r'\1'), # join digits 用于将连续的数字连接起来，被注释掉了
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1] #这行代码将每个正则表达式模式编译成正则表达式对象，以提高匹配效率

normalize2 = [ #专注于对标点符号分词
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])',r' \1 '), # tokenize punctuation. apostrophe is missing 用于匹配西文标点符号（包括括号、引号、连字符、冒号、分号等），并在每个标点符号前添加一个空格，从而将其与相邻的词汇分开。这样做有助于后续的分词和处理，因为很多NLP任务将标点符号视为独立的标记。
    (r'([^0-9])([\.,])',r'\1 \2 '),              # tokenize period and comma unless preceded by a digit  #将字符都分开
    (r'([\.,])([^0-9])',r' \1 \2'),              # tokenize period and comma unless followed by a digit
    (r'([0-9])(-)',r'\1 \2 ')                    # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2] #进一步清理和标准化文本，特别是对标点符号进行处理，使其更适合后续的NLP任务，如分词、词性标注、句法分析等。

def normalize(s): #这个 normalize 函数通过这些步骤，确保了文本数据的一致性和清洁性，为后续的NLP任务（如机器翻译评估）提供了标准化的输入。
    '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if (nonorm):   #如果 nonorm 变量为真，则不进行任何预处理，直接将输入的字符串 s 分割成单词列表并返回。
        return s.split() 
    if type(s) is not str: #如果输入 s 不是字符串类型（例如列表），则将其转换为字符串，单词之间用空格分隔。
        s = " ".join(s)
    # language-independent part:
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;':'"'}) #对文本中的HTML实体进行解码，这里只解码了 &quot; 实体，将其转换为双引号 "。
    # language-dependent part (assuming Western languages):
    s = " %s " % s #在文本的前后各添加一个空格，这样做可以确保文本开头和结尾的标点符号能够被正确地分词
    if not preserve_case:
        s = s.lower()         # this might not be identical to the original
    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()  #最后，将预处理后的文本分割成单词列表并返回。

def count_ngrams(words, n=4): #四个连续（4个单词或4个字母）项目的出现概率
    counts = {} 
    for k in range(1,n+1):
        for i in range(len(words)-k+1): #len(words)-k+1 确保在每次迭代中都能形成一个长度为 k 的序列。
            ngram = tuple(words[i:i+k])
            counts[ngram] = counts.get(ngram, 0)+1
    return counts

def cook_refs(refs, n=4): #这个函数用于处理参考句子，为BLEU分数计算做准备。
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''
    
    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return ([len(ref) for ref in refs], maxcounts)

def cook_test(test, item, n=4): #这个函数用于处理测试句子，为BLEU分数计算做准备。
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflens, refmaxcounts)=item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.
    
    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens))/len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen-len(test)) < min_diff:
                min_diff = abs(reflen-len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test)-k+1,0) for k in range(1,n+1)]

    result['correct'] = [0]*n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result

def score_cooked(allcomps, n=4, ground=0, smooth=1): #这个函数计算经过预处理的测试句子的BLEU分数。它接受一个列表 allcomps，其中包含了多个测试句子的统计数据，以及n-gram的最大长度 n。函数计算每个n-gram的精确度，并应用平滑技术来处理数据稀疏问题。最后，它计算并返回BLEU分数。
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}
    for comps in allcomps:
        for key in ['testlen','reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess','correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
      correct = totalcomps['correct'][k]
      guess = totalcomps['guess'][k]
      addsmooth = 0
      if smooth == 1 and k > 0:
        addsmooth = 1
      logbleu += math.log(correct + addsmooth + sys.float_info.min)-math.log(guess + addsmooth+ sys.float_info.min)
      if guess == 0:
        all_bleus.append(-10000000) #避免使用魔术数字，例如在score_cooked函数中的-10000000，应该定义为一个常量。BLEU_MIN_VALUE = -10000000
      else:
        all_bleus.append(math.log(correct + sys.float_info.min)-math.log( guess ))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0,1-float(totalcomps['reflen'] + 1)/(totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
      if i ==0:
        all_bleus[i] += brevPenalty
      all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus

def bleu(refs,  candidate, ground=0, smooth=1): #这个函数是一个简化的接口，直接接受参考句子列表 refs 和候选句子 candidate，然后计算BLEU分数。
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)

def splitPuncts(line): #这个函数用于将输入的字符串 line 分割成单词和标点符号。它使用正则表达式来匹配单词和非单词字符，并将它们以空格分隔的方式重新组合成一个字符串。
  return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))

def computeMaps(predictions, goldfile): #这个函数用于从预测结果和参考文件中构建两个映射 predictionMap 和 goldMap。每个映射都以唯一的ID作为键，对应的值是预处理后的翻译句子列表。
  predictionMap = {}
  goldMap = {}
  
  for row in predictions:
    cols = row.strip().split('\t')
    if len(cols) == 1:
      (rid, pred) = (cols[0], '') 
    else:
      (rid, pred) = (cols[0], cols[1]) 
    predictionMap[rid] = [splitPuncts(pred.strip().lower())]

  with open(goldfile, 'r') as f:
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) == 1:
        (rid, pred) = (cols[0], '')
      else:
        (rid, pred) = (cols[0], cols[1])
      if rid in predictionMap: # Only insert if the id exists for the method
        if rid not in goldMap:
          goldMap[rid] = []
        goldMap[rid].append(splitPuncts(pred.strip().lower()))

  # sys.stderr.write('Total: ' + str(len(goldMap)) + '\n')
  return (goldMap, predictionMap)

def computeMaps_2list(predictions, gold): #这个函数也是用于构建映射，但它接受两个列表作为输入，一个是预测结果列表，另一个是参考结果列表。
  predictionMap = {}
  goldMap = {}
  
  for row in predictions:
    cols = row.strip().split('\t')
    if len(cols) == 1:
      (rid, pred) = (cols[0], '') 
    else:
      (rid, pred) = (cols[0], cols[1]) 
    predictionMap[rid] = [splitPuncts(pred.strip().lower())]

  for row in gold:
    split_row = row.split('\t')
    if len(split_row) == 1:
      (rid, pred) = (split_row[0], '')
    elif len(split_row) == 2:
      (rid, pred) = (split_row[0], split_row[1])
    else:
      (rid, pred) = (split_row[0], '\t'.join(split_row[1:])) 
    if rid in predictionMap: # Only insert if the id exists for the method
      if rid not in goldMap:
        goldMap[rid] = []
      goldMap[rid].append(splitPuncts(pred.strip().lower()))

  # sys.stderr.write('Total: ' + str(len(goldMap)) + '\n')
  return (goldMap, predictionMap)

def direct_computeMaps(pred, gold): #这个函数用于直接从预测句子和参考句子构建映射，适用于单个句子的比较。
  predictionMap = {}
  goldMap = {}
  
  predictionMap['0'] = [splitPuncts(pred.strip().lower())]

  goldMap['0'] = []
  goldMap['0'].append(splitPuncts(gold.strip().lower()))

  # sys.stderr.write('Total: ' + str(len(goldMap)) + '\n')
  return (goldMap, predictionMap)

def computeMaps_multiple(jsonfile, k): #这个函数用于处理多个预测结果，通常用于处理JSON格式的文件，其中包含了多个预测和参考句子对。
  predictionMap = {}
  goldMap = {}
  
  with open(jsonfile, 'r') as f:
    data = json.load(f)
  for idx in data:
    predictions = data[idx][0]
    gold = data[idx][1]

    for pred in predictions[:k]:
      if idx not in predictionMap:
        predictionMap[idx] = [splitPuncts(pred.strip().lower())]
      else:
        predictionMap[idx].append(splitPuncts(pred.strip().lower()))

    if idx not in goldMap:
      goldMap[idx] = [splitPuncts(gold.strip().lower())]
    else:
      goldMap[idx].append(splitPuncts(gold.strip().lower()))

  return (goldMap, predictionMap)

#m1 is the reference map
#m2 is the prediction map
def bleuFromMaps(m1, m2): #这个函数接受两个映射 m1 和 m2，分别代表参考句子和预测句子，然后计算它们的BLEU分数。
  score = [0] * 5
  num = 0.0
  
  for key in m1:
    if key in m2:
      if len(m2[key]) == 1:
        bl = bleu(m1[key], m2[key][0])
      else:
        bls = []
        for i in range(0, len(m2[key])):
          bls.append(bleu(m1[key], m2[key][i]))
        bl = [max([bls[j][i] for j in range(0, len(bls))]) for i in range(0, len(bls[0]))]
      score = [ score[i] + bl[i] for i in range(0, len(bl))]
      num += 1
  return [s * 100.0 / num for s in score]

if __name__ == '__main__': #当脚本作为主程序运行时，它从命令行参数中读取参考文件路径，从标准输入中读取预测结果，然后调用 computeMaps 函数来构建映射，并最终调用 bleuFromMaps 函数来计算并打印BLEU分数。v
  reference_file = sys.argv[1]
  predictions = []
  for row in sys.stdin:
    predictions.append(row)
  (goldMap, predictionMap) = computeMaps(predictions, reference_file) 
  print (bleuFromMaps(goldMap, predictionMap)[0])

