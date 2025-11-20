import json

with open('pref_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

filt_data = []
for i in data:
    ent={
        'idx':i['idx'],
        'image':i['image'],
        'question':i['chosen'][0]['content']
    }
    filt_data.append(ent)

with open('filter_pref_data.json','w',encoding='utf-8') as f1:
    json.dump(filt_data,f1,ensure_ascii=False)