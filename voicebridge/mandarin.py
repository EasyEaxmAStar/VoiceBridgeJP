"""Final-ASR-only Mandarin sentence grouping; never waits for future speech."""
import re


def sentences(text):
    if not isinstance(text,str) or len(text)>20000:raise ValueError('Invalid committed Mandarin text')
    # Commas, conditional clauses, decimals, technical tokens and quoted
    # punctuation are not split. Sentence groups preserve every original char.
    groups=[];start=0;stack=[];pairs={'“':'”','「':'」','（':'）','(' : ')','《':'》'}
    for i,char in enumerate(text):
        if char in pairs:stack.append(pairs[char])
        elif stack and char==stack[-1]:stack.pop()
        if char in '。！？!?' and not stack and (i+1==len(text) or text[i+1] not in '。！？!?'):
            part=text[start:i+1]
            if part.strip():groups.append(part)
            start=i+1
    if text[start:].strip():groups.append(text[start:])
    elif start<len(text) and groups:groups[-1]+=text[start:]
    # Tiny fragments stay with the adjacent sentence, avoiding isolated fillers.
    result=[]
    for group in groups:
        if result and len(re.sub(r'\W','',group))<4:result[-1]+=group
        else:result.append(group)
    return result or [text]
