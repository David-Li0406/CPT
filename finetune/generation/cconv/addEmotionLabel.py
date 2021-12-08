import json

data_files = ['train.json', 'test.json', 'valid.json']
emotionData = json.load(open('./AllEmotionLabel.json', encoding='utf8'))

emotionDict = {}
for dialog in emotionData:
    for utterance in dialog:
        emotionDict[utterance['sens']] = {'emotion':utterance['label'],'intent':utterance['intent']}

for file in data_files:
    data = json.load(open(file,'r',encoding='utf8'))
    for dialog in data['data']:
        emotion_list = []
        intent_list = []
        for utterance in dialog['content']:
            if utterance in emotionDict.keys():
                emotion_list.append(emotionDict[utterance]['emotion'])
                intent_list.append(emotionDict[utterance]['intent'])
            else:
                emotion_list.append(0)
                intent_list.append(0)
        dialog['emotion'] = emotion_list
        dialog['intent'] = intent_list
    json.dump(data,open(file,'w',encoding='utf8'),ensure_ascii=False)