import csv


EMOTION = 2
CONTEXT = 3
UTTERANCE = 5

def replace_comma(text):
    p = {'_comma_':' ,', '..':'.', '...':'.'}
    for k, v in p.items():
        text = text.replace(k,v)
    return text

def read_from_csv(csv_file):
    data = []
    i = 0
    last_emotion = None
    last_context = None
    # csv_file.seek(0)
    reader = csv.reader(csv_file)
    
    for row in reader:
        if i == 0:
            i += 1
            continue
        # print(row)
        if row[EMOTION] == last_emotion:
            history.append(replace_comma(row[UTTERANCE]))            
        else:
            # append dictionary to data and reinitialize it
            if last_emotion != None:
                entry = {"emotion": last_emotion, "context": last_context, "history": history}
                data.append(entry)
            
            # create data for a new entry in dataset
            last_emotion = row[EMOTION]
            last_context = replace_comma(row[CONTEXT])
            history = [replace_comma(row[UTTERANCE])]
    return data

with open('./dataset/curated_train.csv') as csv_file:
    data = read_from_csv(csv_file)
    lens = []
    for d in data:
        lung = 0
        for ut in d["history"]:
            lung += len(ut.split(' '))
        
        lens.append(lung)
        if lung > 350 and lung <380:
            print('aoleu', lung, d["history"])
