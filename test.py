import json

with open('class_dict.json', 'r') as fp:
    classes = json.load(fp)
chairs = 0
tables = 0
for _, value in classes.items(): #iterates over all keys and values
    if value == '03001627':
        chairs = chairs +1
    else:
        tables = tables +1

print('number chairs:', chairs)
print('number tables:', tables)

