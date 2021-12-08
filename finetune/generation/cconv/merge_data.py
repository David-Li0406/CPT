origin_data_dir = './origin_data/'
label_data_dir = './label_data/'
origin_data = '情感试标{}-{}.txt'
cat = ['类别','意图']
dirname = '情感{}{}-三人结果/'
label_data = '情感{}试标{}-{}.json'
label_person = ['王','吴','徐']
import json
import os
import re
pattern = re.compile('【对话人.】')
all_data = []
count = 0

if __name__ == '__main__':
    for index in range(1,18):
        for person in label_person:
            o_data = origin_data.format(index,person)
            if not os.path.exists(os.path.join(origin_data_dir,o_data)):
                continue
            with open(os.path.join(origin_data_dir,o_data), 'r',encoding='utf8')as f1:
                e_data = json.load(open(os.path.join(label_data_dir,dirname.format('类别',index),label_data.format('类别',index,person)),encoding='utf8'))
                i_data = json.load(open(os.path.join(label_data_dir,dirname.format('意图',index),label_data.format('意图',index,person)),encoding='utf8'))
                for line, e_l, i_l in zip(f1, e_data,i_data):
                    dialogue_list = []
                    utterance_list = re.split(pattern,line.strip())
                    utterance_list = utterance_list[1:]
                    emotion_dic = dict(zip([d1['content'] for d1 in e_l['tags']],[d1['tag'] for d1 in e_l['tags']]))
                    intent_dic = dict(zip([d1['content'] for d1 in i_l['tags']],[d1['tag'] for d1 in i_l['tags']]))
                    for utterance in utterance_list:
                        count += 1
                        utterance_dic = {}
                        utterance_dic['sens'] = utterance
                        for k,v in emotion_dic.items():
                            if k in utterance:
                                utterance_dic['label'] = v
                        if 'label' not in utterance_dic.keys():
                            utterance_dic['label'] = 'others'
                        for k,v in intent_dic.items():
                            if k in utterance:
                                utterance_dic['intent'] = v
                        if 'intent' not in utterance_dic.keys():
                            utterance_dic['intent'] = 'others'
                        dialogue_list.append(utterance_dic)
                    all_data.append(dialogue_list)

    print(len(all_data))
    json.dump(all_data,open('./AllEmotionLabel.json','w',encoding='utf8'),ensure_ascii=False)

